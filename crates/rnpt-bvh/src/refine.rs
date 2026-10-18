//! Regrouping triangles into chunks, guided by rays actually traced (experiment).
//!
//! A chunk's box is a filter: a ray that enters it without finding its answer there is a
//! false positive, the waste a tree should minimize. Given rays observed with the distance of
//! their answer (closest hit, or the end of the segment), neighbouring chunks re-share their
//! triangles so that fewer of these rays enter boxes they have nothing to do in. Rays are
//! counted only up to their answer: beyond it, traversal prunes. A surface-area term stands
//! for the rays not observed, so that boxes do not grow where the sample is blind.

use crate::builder::{triangle_boxes, triangle_offsets};
use crate::bvh::BvhInner;
use crate::math::{InternalRay, AABB};
use crate::{to_internal, Geometry, Ray, RaySample};

const PAIR: usize = 16;

/// What the refinement did: ray entries into chunk boxes over the sample, before and after,
/// and the surface-area weight used.
#[derive(Clone, Copy, Debug)]
pub struct RefineReport {
    pub samples: usize,
    pub entries_before: u64,
    pub entries_after: u64,
    pub area_weight: f32,
}

pub(crate) fn refine(
    geometries: &[Geometry],
    inner: &BvhInner,
    samples: &[RaySample],
    prior: f32,
) -> (Vec<[u32; 8]>, RefineReport) {
    let offsets = triangle_offsets(geometries);
    let tri_boxes = triangle_boxes(geometries);
    let mut groups: Vec<[u32; 8]> = inner
        .tri_slots
        .chunks_exact(8)
        .map(|s| {
            std::array::from_fn(|i| {
                if s[i].orig_idx == u32::MAX { u32::MAX } else { offsets[s[i].geom_id as usize] + s[i].orig_idx }
            })
        })
        .collect();
    let group_box = |g: &[u32]| {
        let mut b = AABB::invalid();
        for &t in g.iter().filter(|&&t| t != u32::MAX) {
            b.extend_aabb(&tri_boxes[t as usize]);
        }
        b
    };

    // Which rays enter which chunk box before their answer.
    let rays: Vec<InternalRay> = samples
        .iter()
        .map(|s| to_internal(&Ray::new_with_minmax(s.org, s.dir, s.tmin, s.t_end)))
        .collect();
    let mut rays_of: Vec<Vec<u32>> = vec![Vec::new(); groups.len()];
    let mut candidates = Vec::new();
    for (i, ray) in rays.iter().enumerate() {
        candidates.clear();
        inner.leaf_chunks_along(ray, &mut candidates);
        for &c in &candidates {
            if enters(ray, &group_box(&groups[c as usize])) {
                rays_of[c as usize].push(i as u32);
            }
        }
    }
    let entries_before: u64 = rays_of.iter().map(|r| r.len() as u64).sum();
    let total_area: f64 = groups.iter().map(|g| group_box(g).surface_area() as f64).sum();
    let area_weight = (prior as f64 * entries_before as f64 / total_area.max(1e-30)) as f32;

    // Pairs of neighbouring chunks (siblings' chunks are contiguous), disjoint within a pass.
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    for (dist, parity) in [(1, 0), (1, 1), (2, 0), (2, 1)] {
        let pairs: Vec<(usize, usize)> = (0..groups.len())
            .filter(|&a| (a / dist) % 2 == parity && a + dist < groups.len())
            .map(|a| (a, a + dist))
            .filter(|&(a, b)| !rays_of[a].is_empty() || !rays_of[b].is_empty())
            .filter(|&(a, b)| groups[a].iter().chain(&groups[b]).all(|&t| t != u32::MAX))
            .collect();
        let mut work: Vec<[u32; PAIR]> = pairs
            .iter()
            .map(|&(a, b)| {
                let mut w = [0u32; PAIR];
                w[..8].copy_from_slice(&groups[a]);
                w[8..].copy_from_slice(&groups[b]);
                w
            })
            .collect();
        let per_thread = work.len().div_ceil(threads).max(1);
        std::thread::scope(|scope| {
            for (part, part_pairs) in work.chunks_mut(per_thread).zip(pairs.chunks(per_thread)) {
                let (rays, rays_of, tri_boxes) = (&rays, &rays_of, &tri_boxes);
                scope.spawn(move || {
                    let mut seen = Vec::new();
                    for (w, &(a, b)) in part.iter_mut().zip(part_pairs) {
                        seen.clear();
                        seen.extend_from_slice(&rays_of[a]);
                        seen.extend_from_slice(&rays_of[b]);
                        seen.sort_unstable();
                        seen.dedup();
                        improve_pair(w, &seen, rays, tri_boxes, area_weight);
                    }
                });
            }
        });
        for (&(a, b), w) in pairs.iter().zip(&work) {
            groups[a].copy_from_slice(&w[..8]);
            groups[b].copy_from_slice(&w[8..]);
        }
    }

    // Entries after, over the same sample (boxes have changed: recount through the new groups).
    let mut entries_after = 0u64;
    for (c, g) in groups.iter().enumerate() {
        let b = group_box(g);
        // Rays that entered this chunk or a neighbour before; others are counted by the prior.
        let near: Vec<u32> = (c.saturating_sub(2)..(c + 3).min(groups.len()))
            .flat_map(|n| rays_of[n].iter().copied())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        entries_after += near.iter().filter(|&&r| enters(&rays[r as usize], &b)).count() as u64;
    }
    let report = RefineReport { samples: samples.len(), entries_before, entries_after, area_weight };
    (groups, report)
}

/// Re-shares 16 triangles between two chunks: the observed rays entering each box, plus the
/// surface-area prior. First the best cut along an axis, then single swaps while one helps.
fn improve_pair(w: &mut [u32; PAIR], rays_seen: &[u32], rays: &[InternalRay], tri_boxes: &[AABB], area_weight: f32) {
    let cost = |w: &[u32; PAIR]| {
        let mut total = 0.0f32;
        for half in [&w[..8], &w[8..]] {
            let mut b = AABB::invalid();
            for &t in half {
                b.extend_aabb(&tri_boxes[t as usize]);
            }
            let entering = rays_seen.iter().filter(|&&r| enters(&rays[r as usize], &b)).count();
            total += entering as f32 + area_weight * b.surface_area();
        }
        total
    };
    let mut best = cost(w);
    for axis in 0..3 {
        let mut sorted = *w;
        sorted.sort_unstable_by(|&a, &b| {
            tri_boxes[a as usize].center()[axis].total_cmp(&tri_boxes[b as usize].center()[axis])
        });
        let c = cost(&sorted);
        if c < best {
            best = c;
            *w = sorted;
        }
    }
    for _ in 0..16 {
        let mut best_swap = None;
        for i in 0..8 {
            for j in 8..PAIR {
                w.swap(i, j);
                let c = cost(w);
                w.swap(i, j);
                if c < best * (1.0 - 1e-6) {
                    best = c;
                    best_swap = Some((i, j));
                }
            }
        }
        match best_swap {
            Some((i, j)) => w.swap(i, j),
            None => break,
        }
    }
}

/// Does the ray enter the box within its segment?
fn enters(ray: &InternalRay, b: &AABB) -> bool {
    let (mut t0, mut t1) = (ray.tmin, ray.tmax);
    for k in 0..3 {
        let (o, inv) = (ray.origin[k], ray.inv_direction[k]);
        let (a, c) = ((b.min[k] - o) * inv, (b.max[k] - o) * inv);
        t0 = t0.max(a.min(c));
        t1 = t1.min(a.max(c));
    }
    t0 <= t1
}
