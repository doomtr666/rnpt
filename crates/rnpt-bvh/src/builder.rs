use nalgebra::Point3;

use crate::{
    bvh::{BvhInner, Bvh8Node, FlatTriangles, TriSlot},
    math::AABB,
    Geometry,
};

const SAH_BINS: usize = 16;
const MAX_LEAF_CHUNKS: u32 = 1;
const TRAVERSAL_COST: f32 = 1.0;
const INTERSECT_COST: f32 = 1.0;

#[derive(Clone, Copy)]
struct BuildTri {
    v0: u32,
    v1: u32,
    v2: u32,
    orig_idx: u32,
    geom_id: u32,
    double_sided: bool,
}

#[derive(Clone, Copy)]
struct BvhNode {
    aabb: AABB,
    left_first: u32,
    chunk_count: u32,
}

impl BvhNode {
    fn is_leaf(&self) -> bool {
        self.chunk_count > 0
    }
}

/// Triangles of all geometries in one list, with their vertices in world space.
fn gather(geometries: &[Geometry]) -> (Vec<Point3<f32>>, Vec<BuildTri>) {
    let mut world_vertices: Vec<Point3<f32>> = Vec::new();
    let mut build_tris: Vec<BuildTri> = Vec::new();

    for (geom_id, geom) in geometries.iter().enumerate() {
        let vert_offset = world_vertices.len() as u32;
        for &v in &geom.verts {
            world_vertices.push(Point3::new(v[0], v[1], v[2]));
        }
        for (orig_idx, &t) in geom.tris.iter().enumerate() {
            let double_sided = geom.double_sided.get(orig_idx).copied().unwrap_or(false);
            build_tris.push(BuildTri {
                v0: t[0] + vert_offset,
                v1: t[1] + vert_offset,
                v2: t[2] + vert_offset,
                orig_idx: orig_idx as u32,
                geom_id: geom_id as u32,
                double_sided,
            });
        }
    }

    (world_vertices, build_tris)
}

/// Builds with the triangles grouped as given: every group becomes one chunk. A group lists
/// triangles as indices into the concatenated triangles of all geometries (`u32::MAX` pads).
pub(crate) fn build_grouped(geometries: &[Geometry], groups: &[[u32; 8]]) -> BvhInner {
    let (world_vertices, all) = gather(geometries);
    let pad = BuildTri { v0: 0, v1: 0, v2: 0, orig_idx: u32::MAX, geom_id: u32::MAX, double_sided: false };
    let build_tris: Vec<BuildTri> =
        groups.iter().flatten().map(|&t| if t == u32::MAX { pad } else { all[t as usize] }).collect();
    build_chunks_and_tree(&world_vertices, build_tris)
}

/// Global index (into the concatenated triangles of all geometries) of every geometry's first triangle.
pub(crate) fn triangle_offsets(geometries: &[Geometry]) -> Vec<u32> {
    let mut offsets = Vec::with_capacity(geometries.len());
    let mut n = 0u32;
    for g in geometries {
        offsets.push(n);
        n += g.tris.len() as u32;
    }
    offsets
}

/// World-space box of every triangle, in the concatenated order.
pub(crate) fn triangle_boxes(geometries: &[Geometry]) -> Vec<AABB> {
    let (world_vertices, all) = gather(geometries);
    all.iter()
        .map(|t| {
            let mut b = AABB::invalid();
            for v in [t.v0, t.v1, t.v2] {
                b.extend(world_vertices[v as usize]);
            }
            b
        })
        .collect()
}

pub(crate) fn build(geometries: &[Geometry]) -> BvhInner {
    let (world_vertices, mut build_tris) = gather(geometries);

    cluster_triangles(&world_vertices, &mut build_tris);

    // Experiment E4: "e4" re-shares triangles between neighbouring chunks so that the sum of
    // the chunks' box areas drops; "e4-split" also gives every chunk its own leaf box.
    let mode = std::env::var("RNPT_BVH_BUILD").unwrap_or_default();
    if mode.starts_with("e4") {
        let tri_boxes: Vec<AABB> = build_tris
            .iter()
            .map(|t| {
                let mut b = AABB::invalid();
                for v in [t.v0, t.v1, t.v2] {
                    b.extend(world_vertices[v as usize]);
                }
                b
            })
            .collect();
        let mut order: Vec<usize> = (0..build_tris.len()).collect();
        optimize_groups(&mut order, &tri_boxes);
        build_tris = order.iter().map(|&i| build_tris[i]).collect();
    }

    build_chunks_and_tree(&world_vertices, build_tris)
}

/// Every 8 consecutive triangles make a chunk; then the tree over the chunks.
fn build_chunks_and_tree(world_vertices: &[Point3<f32>], mut build_tris: Vec<BuildTri>) -> BvhInner {
    let num_triangles = build_tris.len();
    let remainder = num_triangles % 8;
    if remainder != 0 {
        for _ in 0..(8 - remainder) {
            build_tris.push(BuildTri { v0: 0, v1: 0, v2: 0, orig_idx: u32::MAX, geom_id: u32::MAX, double_sided: false });
        }
    }

    let mut soa_chunks: Vec<FlatTriangles> = Vec::new();
    let mut chunk_aabbs: Vec<AABB> = Vec::new();
    let mut chunk_centroids: Vec<Point3<f32>> = Vec::new();

    for chunk_start in (0..build_tris.len()).step_by(8) {
        let mut soa = FlatTriangles::default();
        let mut chunk_aabb = AABB::invalid();

        for i in 0..8 {
            let tri = &build_tris[chunk_start + i];
            let v0 = world_vertices[tri.v0 as usize];
            let v1 = world_vertices[tri.v1 as usize];
            let v2 = world_vertices[tri.v2 as usize];
            let e1 = v1 - v0;
            let e2 = v2 - v0;

            soa.v0_x[i] = v0.x; soa.v0_y[i] = v0.y; soa.v0_z[i] = v0.z;
            soa.e1_x[i] = e1.x; soa.e1_y[i] = e1.y; soa.e1_z[i] = e1.z;
            soa.e2_x[i] = e2.x; soa.e2_y[i] = e2.y; soa.e2_z[i] = e2.z;

            if tri.orig_idx != u32::MAX {
                chunk_aabb.extend(v0);
                chunk_aabb.extend(v1);
                chunk_aabb.extend(v2);
            }
        }

        let eps = nalgebra::Vector3::new(1e-5, 1e-5, 1e-5);
        chunk_aabb.min -= eps;
        chunk_aabb.max += eps;

        soa_chunks.push(soa);
        chunk_aabbs.push(chunk_aabb);
        chunk_centroids.push(chunk_aabb.center());
    }

    let mut nodes: Vec<BvhNode> = Vec::new();
    let mut chunk_indices: Vec<usize> = (0..soa_chunks.len()).collect();

    nodes.push(BvhNode {
        aabb: AABB::invalid(),
        left_first: 0,
        chunk_count: chunk_indices.len() as u32,
    });

    if !chunk_indices.is_empty() {
        update_node_bounds(0, &mut nodes, &chunk_indices, &chunk_aabbs);
        subdivide_sah(0, &mut nodes, &mut chunk_indices, &chunk_aabbs, &chunk_centroids);
    }

    let mut ordered_soa: Vec<FlatTriangles> = Vec::with_capacity(soa_chunks.len());
    let mut ordered_slots: Vec<TriSlot> = Vec::with_capacity(build_tris.len());

    let mut double_sided_masks: Vec<u8> = Vec::with_capacity(soa_chunks.len());
    for &idx in &chunk_indices {
        ordered_soa.push(soa_chunks[idx]);
        let mut ds_mask = 0u8;
        for i in 0..8 {
            let bt = &build_tris[idx * 8 + i];
            ordered_slots.push(TriSlot { orig_idx: bt.orig_idx, geom_id: bt.geom_id });
            if bt.double_sided {
                ds_mask |= 1 << i;
            }
        }
        double_sided_masks.push(ds_mask);
    }

    let mut bvh8_nodes: Vec<Bvh8Node> = Vec::new();
    if !chunk_indices.is_empty() {
        collapse_to_bvh8(0, &nodes, &mut bvh8_nodes);
    }

    BvhInner::new(bvh8_nodes, ordered_slots, ordered_soa, double_sided_masks)
}

fn cluster_triangles(vertices: &[Point3<f32>], tris: &mut Vec<BuildTri>) {
    let mut indices: Vec<usize> = (0..tris.len()).collect();

    let centroids: Vec<Point3<f32>> = tris
        .iter()
        .map(|tri| {
            let v0 = vertices[tri.v0 as usize];
            let v1 = vertices[tri.v1 as usize];
            let v2 = vertices[tri.v2 as usize];
            Point3::from((v0.coords + v1.coords + v2.coords) / 3.0)
        })
        .collect();

    let len = indices.len();
    split_clusters(&mut indices, &centroids, 0, len);

    let orig = tris.clone();
    for (dst, &src) in tris.iter_mut().zip(indices.iter()) {
        *dst = orig[src];
    }
}

fn split_clusters(indices: &mut [usize], centroids: &[Point3<f32>], start: usize, end: usize) {
    let count = end - start;
    if count as u32 <= MAX_LEAF_CHUNKS {
        return;
    }

    let mut centroid_bounds = AABB::invalid();
    for i in start..end {
        centroid_bounds.extend(centroids[indices[i]]);
    }

    let extent = centroid_bounds.max - centroid_bounds.min;
    let mut axis = 0usize;
    if extent.y > extent.x { axis = 1; }
    if extent.z > extent[axis] { axis = 2; }

    let mid = count / 2;
    let mut split_offset = (mid / 8) * 8;
    if split_offset == 0 {
        split_offset = 8.min(count - 1);
    }

    indices[start..end].select_nth_unstable_by(split_offset, |&a, &b| {
        centroids[a][axis].partial_cmp(&centroids[b][axis]).unwrap()
    });

    let split_idx = start + split_offset;
    split_clusters(indices, centroids, start, split_idx);
    split_clusters(indices, centroids, split_idx, end);
}

fn update_node_bounds(
    node_idx: usize,
    nodes: &mut Vec<BvhNode>,
    chunk_indices: &[usize],
    chunk_aabbs: &[AABB],
) {
    let first = nodes[node_idx].left_first as usize;
    let count = nodes[node_idx].chunk_count as usize;
    let mut aabb = AABB::invalid();
    for i in 0..count {
        aabb.extend_aabb(&chunk_aabbs[chunk_indices[first + i]]);
    }
    nodes[node_idx].aabb = aabb;
}

fn subdivide_sah(
    node_idx: usize,
    nodes: &mut Vec<BvhNode>,
    chunk_indices: &mut Vec<usize>,
    chunk_aabbs: &[AABB],
    chunk_centroids: &[Point3<f32>],
) {
    let first = nodes[node_idx].left_first as usize;
    let count = nodes[node_idx].chunk_count as usize;

    if count <= 1 {
        return;
    }

    let mut best_cost = f32::MAX;
    let mut best_axis = 0usize;
    let mut best_split_bin = 0usize;

    let mut centroid_bounds = AABB::invalid();
    for i in 0..count {
        centroid_bounds.extend(chunk_centroids[chunk_indices[first + i]]);
    }

    let extent = centroid_bounds.max - centroid_bounds.min;
    // A leaf's chunk count must fit its 7-bit field (Bvh8Node::COUNT_MASK).
    let too_big = count as u32 > 127;
    if extent.x == 0.0 && extent.y == 0.0 && extent.z == 0.0 && !too_big {
        return;
    }

    for axis in 0..3 {
        let bounds_min = centroid_bounds.min[axis];
        let bounds_max = centroid_bounds.max[axis];
        let bounds_extent = bounds_max - bounds_min;
        if bounds_extent == 0.0 { continue; }

        #[derive(Clone, Copy)]
        struct Bin { aabb: AABB, count: usize }
        let mut bins = [Bin { aabb: AABB::invalid(), count: 0 }; SAH_BINS];

        for i in 0..count {
            let chunk_idx = chunk_indices[first + i];
            let centroid = chunk_centroids[chunk_idx];
            let t = (centroid[axis] - bounds_min) / bounds_extent;
            let bin_idx = (SAH_BINS as f32 * t).min(SAH_BINS as f32 - 1.0) as usize;
            bins[bin_idx].aabb.extend_aabb(&chunk_aabbs[chunk_idx]);
            bins[bin_idx].count += 1;
        }

        let mut left_aabbs  = [AABB::invalid(); SAH_BINS - 1];
        let mut left_counts  = [0usize; SAH_BINS - 1];
        let mut right_aabbs = [AABB::invalid(); SAH_BINS - 1];
        let mut right_counts = [0usize; SAH_BINS - 1];

        let mut left_box = AABB::invalid();
        let mut left_count = 0;
        for i in 0..SAH_BINS - 1 {
            left_count += bins[i].count;
            left_box.extend_aabb(&bins[i].aabb);
            left_counts[i] = left_count;
            left_aabbs[i] = left_box;
        }

        let mut right_box = AABB::invalid();
        let mut right_count = 0;
        for i in (1..SAH_BINS).rev() {
            right_count += bins[i].count;
            right_box.extend_aabb(&bins[i].aabb);
            right_counts[i - 1] = right_count;
            right_aabbs[i - 1] = right_box;
        }

        let inv_total_area = 1.0 / nodes[node_idx].aabb.surface_area();

        for i in 0..SAH_BINS - 1 {
            if left_counts[i] > 0 && right_counts[i] > 0 {
                let cost = TRAVERSAL_COST
                    + INTERSECT_COST
                        * (left_aabbs[i].surface_area() * left_counts[i] as f32
                            + right_aabbs[i].surface_area() * right_counts[i] as f32)
                        * inv_total_area;

                if cost < best_cost {
                    best_cost = cost;
                    best_axis = axis;
                    best_split_bin = i;
                }
            }
        }
    }

    let leaf_cost = count as f32 * INTERSECT_COST;
    let split_all = std::env::var("RNPT_BVH_BUILD").is_ok_and(|v| v == "e4-split");
    if best_cost >= leaf_cost && !too_big && !split_all {
        return;
    }

    let mut left_count = 0;
    if best_cost < f32::MAX {
        let bounds_min    = centroid_bounds.min[best_axis];
        let bounds_extent = centroid_bounds.max[best_axis] - centroid_bounds.min[best_axis];

        let mut left  = first;
        let mut right = first + count - 1;

        while left <= right {
            let chunk_idx = chunk_indices[left];
            let centroid = chunk_centroids[chunk_idx];
            let mut bin_idx =
                (((centroid[best_axis] - bounds_min) / bounds_extent) * (SAH_BINS as f32)) as usize;
            bin_idx = bin_idx.min(SAH_BINS - 1);

            if bin_idx <= best_split_bin {
                left += 1;
            } else {
                chunk_indices.swap(left, right);
                if right == 0 { break; }
                right -= 1;
            }
        }
        left_count = left - first;
    }
    if left_count == 0 || left_count == count {
        if !too_big {
            return;
        }
        left_count = count / 2;
    }
    let left = first + left_count;

    let left_child_idx = nodes.len();
    nodes.push(BvhNode {
        aabb: AABB::invalid(),
        left_first: first as u32,
        chunk_count: left_count as u32,
    });

    let right_child_idx = nodes.len();
    nodes.push(BvhNode {
        aabb: AABB::invalid(),
        left_first: left as u32,
        chunk_count: (count - left_count) as u32,
    });

    nodes[node_idx].left_first = left_child_idx as u32;
    nodes[node_idx].chunk_count = 0;

    update_node_bounds(left_child_idx, nodes, chunk_indices, chunk_aabbs);
    update_node_bounds(right_child_idx, nodes, chunk_indices, chunk_aabbs);
    subdivide_sah(left_child_idx, nodes, chunk_indices, chunk_aabbs, chunk_centroids);
    subdivide_sah(right_child_idx, nodes, chunk_indices, chunk_aabbs, chunk_centroids);
}

fn collapse_to_bvh8(node_idx: usize, bvh2: &[BvhNode], bvh8: &mut Vec<Bvh8Node>) -> u32 {
    let bvh8_idx = bvh8.len() as u32;
    bvh8.push(Bvh8Node::default());

    let mut children = [0usize; 8];
    let mut n_children = 1usize;
    children[0] = node_idx;

    while n_children < 8 {
        let mut best_idx = None;
        let mut best_area = -1.0f32;
        for i in 0..n_children {
            let node = &bvh2[children[i]];
            if !node.is_leaf() {
                let area = node.aabb.surface_area();
                if area > best_area {
                    best_area = area;
                    best_idx = Some(i);
                }
            }
        }
        if let Some(i) = best_idx {
            let left = bvh2[children[i]].left_first as usize;
            children[i] = children[n_children - 1];
            n_children -= 1;
            children[n_children] = left;
            n_children += 1;
            children[n_children] = left + 1;
            n_children += 1;
        } else {
            break;
        }
    }

    // Children go to slots in the order they were found.
    let mut slot_to_child = [usize::MAX; 8];
    slot_to_child[..n_children].copy_from_slice(&children[..n_children]);

    let mut node8 = Bvh8Node::default();
    for slot in 0..8 {
        let c_idx = slot_to_child[slot];
        if c_idx == usize::MAX { continue; }
        let bvh2_node = &bvh2[c_idx];
        node8.p_min_x[slot] = bvh2_node.aabb.min.x;
        node8.p_min_y[slot] = bvh2_node.aabb.min.y;
        node8.p_min_z[slot] = bvh2_node.aabb.min.z;
        node8.p_max_x[slot] = bvh2_node.aabb.max.x;
        node8.p_max_y[slot] = bvh2_node.aabb.max.y;
        node8.p_max_z[slot] = bvh2_node.aabb.max.z;
        node8.children[slot] = if bvh2_node.is_leaf() {
            Bvh8Node::encode_leaf(bvh2_node.left_first, bvh2_node.chunk_count)
        } else {
            collapse_to_bvh8(c_idx, bvh2, bvh8)
        };
    }

    bvh8[bvh8_idx as usize] = node8;
    bvh8_idx
}

// ── Group optimization ───────────────────────────────────────────────────────
//
// Every 8 consecutive items of `order` form a group. Passes over pairs of groups
// re-share their 16 items so that the sum of the two groups' box areas drops: first
// the best cut along an axis, then single swaps while one helps. The pairs of a pass
// are disjoint, so they are processed in parallel.

const GROUP: usize = 8;
const MAX_PASSES: usize = 8;

fn optimize_groups(order: &mut [usize], boxes: &[AABB]) {
    let groups = order.len() / GROUP; // a trailing partial group stays as it is
    if groups < 2 {
        return;
    }
    for _ in 0..MAX_PASSES {
        let mut gain = 0.0f64;
        // Neighbours 1 apart then 2 apart, each in two disjoint pairings.
        for (dist, parity) in [(1, 0), (1, 1), (2, 0), (2, 1)] {
            let pairs: Vec<(usize, usize)> = (0..groups)
                .filter(|&g| (g / dist) % 2 == parity && g + dist < groups)
                .map(|g| (g, g + dist))
                .collect();
            gain += improve_pairs(order, boxes, &pairs);
        }
        let total: f64 = (0..groups).map(|g| group_area(&order[g * GROUP..(g + 1) * GROUP], boxes) as f64).sum();
        if gain < 1e-3 * total {
            break;
        }
    }
}

fn improve_pairs(order: &mut [usize], boxes: &[AABB], pairs: &[(usize, usize)]) -> f64 {
    let mut work: Vec<[usize; 2 * GROUP]> = pairs
        .iter()
        .map(|&(a, b)| {
            let mut w = [0usize; 2 * GROUP];
            w[..GROUP].copy_from_slice(&order[a * GROUP..(a + 1) * GROUP]);
            w[GROUP..].copy_from_slice(&order[b * GROUP..(b + 1) * GROUP]);
            w
        })
        .collect();

    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let per_thread = work.len().div_ceil(threads).max(1);
    let gain: f64 = std::thread::scope(|scope| {
        let handles: Vec<_> = work
            .chunks_mut(per_thread)
            .map(|part| scope.spawn(move || part.iter_mut().map(|w| improve_pair(w, boxes) as f64).sum::<f64>()))
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).sum()
    });

    for (&(a, b), w) in pairs.iter().zip(&work) {
        order[a * GROUP..(a + 1) * GROUP].copy_from_slice(&w[..GROUP]);
        order[b * GROUP..(b + 1) * GROUP].copy_from_slice(&w[GROUP..]);
    }
    gain
}

/// Re-shares 16 items between two groups of 8; returns the area saved.
fn improve_pair(w: &mut [usize; 2 * GROUP], boxes: &[AABB]) -> f32 {
    let cost = |w: &[usize; 2 * GROUP]| group_area(&w[..GROUP], boxes) + group_area(&w[GROUP..], boxes);
    let before = cost(w);
    let mut best = before;

    // Best cut along an axis.
    for axis in 0..3 {
        let mut sorted = *w;
        sorted.sort_unstable_by(|&a, &b| boxes[a].center()[axis].total_cmp(&boxes[b].center()[axis]));
        let c = cost(&sorted);
        if c < best {
            best = c;
            *w = sorted;
        }
    }

    // Then single swaps, best first, while one helps.
    loop {
        let mut best_swap = None;
        for i in 0..GROUP {
            for j in GROUP..2 * GROUP {
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
    before - best
}

fn group_area(items: &[usize], boxes: &[AABB]) -> f32 {
    let mut b = AABB::invalid();
    for &i in items {
        b.extend_aabb(&boxes[i]);
    }
    b.surface_area()
}
