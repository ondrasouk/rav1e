use crate::context::TileBlockOffset;
use crate::encoder::{FrameInvariants, FrameState};
use crate::partition::{BlockSize, PartitionType};
use crate::tiling::TileStateMut;
use fast_slic_rust::{
  arrays::{Array2D, LABImage},
  assign::assign,
  atomic_arrays::AtomicArray2D,
  cielab::srgb_to_cielab_pixel,
  cluster::Cluster,
  slic::Clusters,
  slic::{compute_spatial_path, update},
};
use itertools::Itertools;
use rayon::{current_num_threads, prelude::*, ThreadPoolBuilder};
use std::ops::Range;
use std::sync::atomic::{AtomicU16, Ordering};
use std::sync::Arc;
use v_frame::pixel::Pixel;

/// Do not confuse with "segmentation" which does estimation of QP based on spatio-temporal scores
#[profiling::function]
pub(crate) fn slic_segmentation_compute<T: Pixel>(
  fi: &FrameInvariants<T>, fs: &mut FrameState<T>,
) {
  let num_threads = current_num_threads(); // TODO: Use set number of threads in CLI
  let pool =
    ThreadPoolBuilder::new().num_threads(num_threads).build().unwrap();

  let lab_image = yuv_planes_to_lab(fi, fs);

  let pixels_per_superpixel = 300;
  // Config setup
  // TODO: maybe spatio-temporal info could be helpful for determining the right amount of clusters and compactness
  let slic_config = fast_slic_rust::common::Config {
    num_of_clusters: ((lab_image.width * lab_image.height)
      / pixels_per_superpixel) as u16,
    max_iterations: 10,
    compactness: 2.0,
    subsample_stride: 3,
    ..fast_slic_rust::common::Config::default()
  };

  // clusters
  let mut clusters = Clusters::initialize_clusters(&lab_image, &slic_config);

  let search_region_size = ((lab_image.width * lab_image.height) as f32
    / slic_config.num_of_clusters as f32)
    .sqrt() as u16;
  let spatial_distance_lut =
    compute_spatial_path(&slic_config, &search_region_size);
  let mut min_distances =
    AtomicArray2D::from_fill(0xFFFFu16, lab_image.width, lab_image.height);
  let mut subsaple_start = 0;

  pool.install(|| {
    for _ in 0..slic_config.max_iterations {
      assign(
        &lab_image,
        &slic_config,
        &mut clusters,
        &mut min_distances,
        &spatial_distance_lut,
        search_region_size,
        subsaple_start,
      );
      update(&mut clusters, &lab_image, &slic_config, subsaple_start);
      subsaple_start = (subsaple_start + 1) % slic_config.subsample_stride;
    }
  });
  let mut no_subsample_config = slic_config.clone();
  no_subsample_config.subsample_stride = 1;
  pool.install(|| {
    assign(
      &lab_image,
      &no_subsample_config,
      &mut clusters,
      &mut min_distances,
      &spatial_distance_lut,
      search_region_size,
      0,
    );
  });
  pool.install(|| {
    let ranges = split_length_to_ranges(
      clusters.assignments.height,
      current_num_threads(),
    );
    ranges.into_par_iter().for_each(|range| {
      range.into_iter().for_each(|row| {
        let row = clusters.assignments.get_row(row);
        let mut prev = row[0].load(Ordering::Relaxed);
        if prev == u16::MAX {
          prev = row
            .iter()
            .find(|x| x.load(Ordering::Relaxed) != u16::MAX)
            .unwrap()
            .load(Ordering::Relaxed);
        }
        row.iter().for_each(|x| {
          let n = x.load(Ordering::Relaxed);
          if n == u16::MAX {
            x.store(prev, Ordering::Relaxed);
          } else {
            prev = n;
          }
        })
      })
    });
  });

  pool.install(|| {
    update(&mut clusters, &lab_image, &no_subsample_config, 0);
    compute_color_distance(&lab_image, &clusters, &mut min_distances, 2);
  });

  let variances = Array2D::from_slice(
    &fi.coded_frame_data.clone().unwrap().activity_mask.variances,
    lab_image.width.div_ceil(8),
    lab_image.height.div_ceil(8),
  )
  .unwrap();
  let variances_sum: u32 = variances.data.iter().sum();
  let variances_mean: f32 = variances_sum as f32 / variances.data.len() as f32;

  let mut scores = compute_8x8_regions_scores(
    &clusters.assignments,
    &min_distances,
    &clusters.clusters,
  );
  let scores_sum: f32 = scores.data.iter().filter(|x| x.is_finite()).sum();
  let scores_mean: f32 = scores_sum / scores.data.len() as f32;

  scores.data.iter_mut().zip(variances.data.iter()).for_each(
    |(score, var)| {
      let variance_score = ((*var as f32) - variances_mean)
        .min(6.0 * variances_mean)
        / (variances_mean / scores_mean);
      let var_log = variance_score.log2().max(0.1);
      let score_log = score.log2().max(0.1);
      if var_log.is_finite() && score_log.is_finite() {
        *score = score_log * var_log
      } else {
        *score = 0.0
      }
    },
  );

  let scores_sum: f32 = scores.data.iter().filter(|x| x.is_finite()).sum();
  let scores_mean: f32 = scores_sum / scores.data.len() as f32;
  fs.superpixels_split_score = Arc::new(scores);
  fs.superpixels_split_score_mean = scores_mean;
}

#[profiling::function]
fn yuv_planes_to_lab<T: Pixel>(
  fi: &FrameInvariants<T>, fs: &mut FrameState<T>,
) -> LABImage {
  // FIXME: Cs400 (monochrome) not supported now
  // FIXME: Parallelization
  let frame = &fs.input;
  let plane_y = &frame.planes[0];
  let plane_u = &frame.planes[1];
  let plane_v = &frame.planes[2];
  debug_assert_eq!(plane_u.cfg.xdec, plane_v.cfg.xdec);
  debug_assert_eq!(plane_u.cfg.ydec, plane_v.cfg.ydec);
  let chroma_xdec = plane_u.cfg.xdec;
  let chroma_ydec = plane_u.cfg.ydec;

  let bit_depth = fi.config.bit_depth;
  let coeff_shift = bit_depth - 8;

  let mut lab_data: Vec<u8> =
    Vec::from_iter((0..plane_y.cfg.width * plane_y.cfg.height * 4).map(|_| 0));

  let y_iter = plane_y.rows_iter();

  for (luma_row_num, luma_row) in y_iter.enumerate() {
    let (u_row, v_row) = if chroma_ydec == 1 {
      (
        plane_u.row((luma_row_num / 2) as isize),
        plane_v.row((luma_row_num / 2) as isize),
      )
    } else if chroma_ydec == 0 {
      (plane_u.row(luma_row_num as isize), plane_v.row(luma_row_num as isize))
    } else {
      unimplemented!()
    };

    if chroma_xdec == 0 {
      luma_row.iter().enumerate().zip(u_row).zip(v_row).for_each(
        |(((x, y), u), v)| {
          let labp_ind = (luma_row_num * plane_y.cfg.width + x) * 4;
          lab_data[labp_ind..labp_ind + 4]
            .copy_from_slice(yuv_to_labp(*y, *u, *v, coeff_shift).as_slice());
        },
      );
    } else if chroma_xdec == 1 {
      luma_row
        .iter()
        .enumerate()
        .zip(u_row.into_iter().flat_map(|n| std::iter::repeat(n).take(2)))
        .zip(v_row.into_iter().flat_map(|n| std::iter::repeat(n).take(2)))
        .for_each(|(((x, y), u), v)| {
          let labp_ind = (luma_row_num * plane_y.cfg.width + x) * 4;
          lab_data[labp_ind..labp_ind + 4]
            .copy_from_slice(yuv_to_labp(*y, *u, *v, coeff_shift).as_slice());
        });
    } else {
      unimplemented!()
    }
  }

  /// coef_shift is for 8-bit 0, for 10-bit depth 2, for 12-bit 4, etc.
  fn yuv_to_labp<T: Pixel>(y: T, u: T, v: T, coeff_shift: usize) -> [u8; 4] {
    let mut yp = (y >> coeff_shift).to_f32().unwrap();
    let mut up = (u >> coeff_shift).to_f32().unwrap();
    let mut vp = (v >> coeff_shift).to_f32().unwrap();
    debug_assert!(yp <= 255.0 || up <= 255.0 || vp <= 255.0);
    // YUV to RGB conversion
    // https://stackoverflow.com/a/17934865
    // TODO: faster conversion
    yp -= 16.0;
    up -= 128.0;
    vp -= 128.0;
    let r = (1.164 * yp + 1.596 * vp).clamp(0.0, 255.0) as u8;
    let g = (1.164 * yp - 0.392 * up - 0.813 * vp).clamp(0.0, 255.0) as u8;
    let b = (1.164 * yp + 2.017 * up).clamp(0.0, 255.0) as u8;
    // RGB to LAB
    let lab = srgb_to_cielab_pixel(&[r, g, b]);
    [lab[0], lab[1], lab[2], 0]
  }

  LABImage::from_raw_slice(&lab_data, plane_y.cfg.width, plane_y.cfg.height)
}

fn split_length_to_ranges(length: usize, splits: usize) -> Vec<Range<usize>> {
  let chunk_size = length / splits;
  let rem = length % splits;
  (0..splits)
    .scan((rem, 0usize), |(r, acc), _split| {
      let mut size = chunk_size;
      if *r > 0 {
        *r -= 1;
        size += 1;
      }
      let out = (*acc, *acc + size);
      *acc += size;
      Some(out.0..out.1)
    })
    .collect()
}

/// This is not an assign function, but it computes the color difference of pixel from its cluster,
/// and it has multiplier on luma channel difference (larger than 1).
#[profiling::function]
fn compute_color_distance(
  image: &LABImage, clusters: &Clusters,
  color_distance: &mut AtomicArray2D<AtomicU16>, luma_distance_multiplier: u8,
) {
  let ranges =
    split_length_to_ranges(image.height * image.width, current_num_threads());
  ranges.into_par_iter().for_each(|range| {
    let image_range = range.start * 4..range.end * 4;
    if luma_distance_multiplier < 2 {
      compute_color_distance_primitive_generic::<false, false>(
        &image.lab_data[image_range],
        &clusters.clusters,
        &clusters.assignments.data[range.clone()],
        &color_distance.data[range],
        luma_distance_multiplier,
      );
    } else if luma_distance_multiplier == 2 {
      compute_color_distance_primitive_generic::<false, true>(
        &image.lab_data[image_range],
        &clusters.clusters,
        &clusters.assignments.data[range.clone()],
        &color_distance.data[range],
        luma_distance_multiplier,
      );
    } else {
      compute_color_distance_primitive_generic::<true, false>(
        &image.lab_data[image_range],
        &clusters.clusters,
        &clusters.assignments.data[range.clone()],
        &color_distance.data[range],
        luma_distance_multiplier,
      );
    }
  })
}

#[inline(always)]
fn compute_color_distance_primitive_generic<
  const USE_LUMA_MULTIPLIER: bool,
  const USE_LUMA_DOUBLE: bool,
>(
  image: &[u8], clusters: &[Cluster], assignments: &[AtomicU16],
  color_distance: &[AtomicU16], luma_distance_multiplier: u8,
) {
  color_distance.iter().zip(assignments).zip(image.chunks(4)).for_each(
    |((distance, assignment), pix)| {
      let assign = assignment.load(Ordering::Relaxed);
      let cluster = &clusters[assign as usize];
      let l_dist = pix[0].abs_diff(cluster.l) as u16;
      let a_dist = pix[1].abs_diff(cluster.a) as u16;
      let b_dist = pix[2].abs_diff(cluster.b) as u16;
      if USE_LUMA_MULTIPLIER {
        distance.store(
          (l_dist * luma_distance_multiplier as u16) + a_dist + b_dist,
          Ordering::Relaxed,
        );
      } else if USE_LUMA_DOUBLE {
        distance.store(l_dist + l_dist + a_dist + b_dist, Ordering::Relaxed);
      } else {
        distance.store(l_dist + a_dist + b_dist, Ordering::Relaxed);
      }
    },
  )
}

#[profiling::function]
fn compute_8x8_regions_scores(
  assignments: &AtomicArray2D<AtomicU16>,
  min_distances: &AtomicArray2D<AtomicU16>, clusters: &[Cluster],
) -> Array2D<f32> {
  let num_of_region_rows = assignments.height.div_ceil(8);
  let num_of_region_columns = assignments.width.div_ceil(8);
  let mut regions_scores_array =
    Array2D::from_fill(0f32, num_of_region_columns, num_of_region_rows);

  let mut region_row_distance_acc = vec![0u32; num_of_region_columns];
  let mut region_max_distance = vec![0u16; num_of_region_columns];

  let mut region_clusters_num =
    vec![Vec::with_capacity(28); num_of_region_columns];

  (0..assignments.height - 1).for_each(|y| {
    let assignments_row = assignments.get_row(y);
    let assignments_row_bellow = assignments.get_row(y + 1);
    let min_distance_row = min_distances.get_row(y);
    let min_distance_row_bellow = min_distances.get_row(y + 1);

    // 2x2 kernel detecting edges in assignments
    let mut num_bef = assignments_row[0].load(Ordering::Relaxed);
    let mut num_b_bef = assignments_row_bellow[0].load(Ordering::Relaxed);
    let mut dist_bef =
      min_distance_row[0].load(Ordering::Relaxed).saturating_sub(10);
    let mut dist_b_bef =
      min_distance_row_bellow[0].load(Ordering::Relaxed).saturating_sub(10);

    if num_bef != num_b_bef {
      region_clusters_num[0].push(num_bef);
      region_clusters_num[0].push(num_b_bef);
    }
    region_row_distance_acc[0] += dist_bef as u32;
    region_max_distance[0] = region_max_distance[0].max(dist_bef);

    (1usize..)
      .zip(assignments_row[1..].iter())
      .zip(assignments_row_bellow[1..].iter())
      .zip(min_distance_row[1..].iter())
      .zip(min_distance_row_bellow[1..].iter())
      .for_each(|((((x, num_a), num_b_a), dist_a), dist_b_a)| {
        let num = num_a.load(Ordering::Relaxed);
        let num_b = num_b_a.load(Ordering::Relaxed);
        let dist = dist_a.load(Ordering::Relaxed).saturating_sub(10);
        let dist_b = dist_b_a.load(Ordering::Relaxed).saturating_sub(10);
        let x_r = x / 8;
        region_row_distance_acc[x_r] += dist as u32;
        region_max_distance[x_r] = region_max_distance[x_r].max(dist);
        if region_clusters_num[x_r].is_empty() {
          region_clusters_num[x_r].push(num);
        }

        // edge in assignments
        if !((num == num_b) && (num_bef == num_b_bef) && (num_bef == num)) {
          // Add unique cluster numbers to vector
          let mut b1 = true;
          let mut b2 = true;
          for n in region_clusters_num[x_r].iter() {
            if *n == num {
              b1 = false;
            }
            if *n == num_b {
              b2 = false;
            }
          }
          if b1 {
            region_clusters_num[x_r].push(num);
          }
          if b2 {
            region_clusters_num[x_r].push(num_b);
          }
        }
        num_bef = num;
        num_b_bef = num_b;
        dist_bef = dist;
        dist_b_bef = dist_b;
      });

    if (y % 8) == 7 {
      let y_r = y / 8;
      for x_r in 0..num_of_region_columns {
        let mean_distance = (region_row_distance_acc[x_r] / 64) as u16;
        let max_distance = region_max_distance[x_r];
        let num_of_superpixels = region_clusters_num[x_r].len() as u8;
        debug_assert_ne!(num_of_superpixels, 0);

        // High difference means, that there is some peak. No difference is basically, that it's perfect.
        // In this context it means, that for no edges in assignment, there is an small detail.
        // If there is an edge in assignment, then this is not the only thing. We need to take into an account.
        let distance_metric =
          (mean_distance.abs_diff(max_distance) as f32).powf(1.2);

        if num_of_superpixels == 1 {
          regions_scores_array[(x_r, y_r)] = distance_metric;
        } else {
          let cluster_diff: f32 = region_clusters_num[x_r]
            .iter()
            .combinations(2)
            .map(|ab| {
              let diff_l = clusters[*ab[0] as usize]
                .l
                .abs_diff(clusters[*ab[1] as usize].l)
                as f32;
              let diff_a = clusters[*ab[0] as usize]
                .a
                .abs_diff(clusters[*ab[1] as usize].a)
                as f32;
              let diff_b = clusters[*ab[0] as usize]
                .b
                .abs_diff(clusters[*ab[1] as usize].b)
                as f32;
              (diff_l).hypot(diff_a).hypot(diff_b)
            })
            .max_by(|a, b| a.partial_cmp(b).expect("Tried to compare a NaN"))
            .unwrap_or(0.0);

          regions_scores_array[(x_r, y_r)] =
            (cluster_diff - 10.0).max(0.0).powf(1.2);
        }
      }

      region_row_distance_acc.fill(0);
      region_max_distance.fill(0);
      region_clusters_num.iter_mut().for_each(|v| v.clear());
    }
  });
  regions_scores_array
}

#[profiling::function]
pub(crate) fn superpixels_partition_decision<T: Pixel>(
  ts: &TileStateMut<T>, tile_bo: TileBlockOffset, bsize: BlockSize,
) -> PartitionType {
  // scores have 8x8 pixels blocks
  let scores = ts.superpixels_split_score;
  let scores_mean: f32 = ts.superpixels_split_score_mean;

  let x_r = tile_bo.0.x / 2;
  let y_r = tile_bo.0.y / 2;
  let width_r = bsize.width() / 8;
  let height_r = bsize.height() / 8;

  // 0 - top left, 1 - top right, 2 - bottom left, 3 - bottom right
  let mut scores_block_max = [0f32; 4];
  let mut scores_block_sum = [0f32; 4];
  let mut scores_block_means = [0f32; 4];
  let ranges_y = [
    y_r..(y_r + height_r / 2),
    y_r..(y_r + height_r / 2),
    (y_r + height_r / 2)..(y_r + height_r),
    (y_r + height_r / 2)..(y_r + height_r),
  ];
  let ranges_x = [
    x_r..(x_r + width_r / 2),
    (x_r + width_r / 2)..(x_r + width_r),
    x_r..(x_r + width_r / 2),
    (x_r + width_r / 2)..(x_r + width_r),
  ];

  (0..4).into_iter().for_each(|i| {
    ranges_y[i].clone().into_iter().for_each(|row| {
      let scores_row =
        scores.get_row_part(row, ranges_x[i].start, ranges_x[i].end - 1);
      scores_row.iter().filter(|x| x.is_finite()).for_each(|x| {
        scores_block_max[i] = scores_block_max[i].max(*x);
        scores_block_sum[i] += *x;
      });
    });
    scores_block_means[i] =
      scores_block_sum[i] / ((width_r * height_r) / 4) as f32;
  });

  // TODO: check other possible thresholding methods
  let mut need_split_max = [false; 4];
  let mut need_split_mean = [false; 4];
  let mut need_split_and = [false; 4];
  let mut need_split_or = [false; 4];
  for (i, (max, mean)) in
    scores_block_max.iter().zip(&scores_block_means).enumerate()
  {
    need_split_max[i] = *max > (1.8 * (*mean).max(scores_mean));
    if *max > (5.0 * *mean) {
      return PartitionType::PARTITION_SPLIT;
    }
    if bsize <= BlockSize::BLOCK_32X32 {
      need_split_mean[i] = *mean > (6.5 * scores_mean);
    } else {
      need_split_mean[i] = *mean > (4.5 * scores_mean);
    }
    need_split_and[i] = need_split_max[i] && need_split_mean[i];
    need_split_or[i] = need_split_max[i] || need_split_mean[i];
  }

  if need_split_or.iter().all(|x| !x) {
    return PartitionType::PARTITION_NONE;
  } else if need_split_or.iter().filter(|x| **x).count() > 2 {
    return PartitionType::PARTITION_SPLIT;
  } else if need_split_and.iter().filter(|x| **x).count() > 0 {
    return PartitionType::PARTITION_SPLIT;
  } else if (need_split_or[0] && need_split_or[1])
    || (need_split_or[2] && need_split_or[3])
  {
    return PartitionType::PARTITION_SPLIT;
  } else if (need_split_or[0] && need_split_or[2])
    || (need_split_or[1] && need_split_or[3])
  {
    return PartitionType::PARTITION_SPLIT;
  }

  // undecided
  return PartitionType::PARTITION_INVALID;
}
