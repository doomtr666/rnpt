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

pub(crate) fn build(geometries: &[Geometry]) -> BvhInner {
    let mut world_vertices: Vec<Point3<f32>> = Vec::new();
    let mut build_tris: Vec<BuildTri> = Vec::new();

    for (geom_id, geom) in geometries.iter().enumerate() {
        let vert_offset = world_vertices.len() as u32;
        for &v in &geom.verts {
            world_vertices.push(Point3::new(v[0], v[1], v[2]));
        }
        for (orig_idx, &t) in geom.tris.iter().enumerate() {
            build_tris.push(BuildTri {
                v0: t[0] + vert_offset,
                v1: t[1] + vert_offset,
                v2: t[2] + vert_offset,
                orig_idx: orig_idx as u32,
                geom_id: geom_id as u32,
            });
        }
    }

    cluster_triangles(&world_vertices, &mut build_tris);

    let num_triangles = build_tris.len();
    let remainder = num_triangles % 8;
    if remainder != 0 {
        for _ in 0..(8 - remainder) {
            build_tris.push(BuildTri { v0: 0, v1: 0, v2: 0, orig_idx: u32::MAX, geom_id: u32::MAX });
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

            if chunk_start + i < num_triangles {
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

    for &idx in &chunk_indices {
        ordered_soa.push(soa_chunks[idx]);
        for i in 0..8 {
            let bt = &build_tris[idx * 8 + i];
            ordered_slots.push(TriSlot { orig_idx: bt.orig_idx, geom_id: bt.geom_id });
        }
    }

    let mut bvh8_nodes: Vec<Bvh8Node> = Vec::new();
    if !chunk_indices.is_empty() {
        collapse_to_bvh8(0, &nodes, &mut bvh8_nodes);
    }

    BvhInner { nodes: bvh8_nodes, tri_slots: ordered_slots, soa_chunks: ordered_soa }
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
    if extent.x == 0.0 && extent.y == 0.0 && extent.z == 0.0 {
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
    if best_cost >= leaf_cost {
        return;
    }

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

    let left_count = left - first;
    if left_count == 0 || left_count == count {
        return;
    }

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

    let mut min_x = f32::MAX; let mut max_x = f32::NEG_INFINITY;
    let mut min_y = f32::MAX; let mut max_y = f32::NEG_INFINITY;
    let mut min_z = f32::MAX; let mut max_z = f32::NEG_INFINITY;
    for i in 0..n_children {
        let a = &bvh2[children[i]].aabb;
        if a.min.x < min_x { min_x = a.min.x } if a.max.x > max_x { max_x = a.max.x }
        if a.min.y < min_y { min_y = a.min.y } if a.max.y > max_y { max_y = a.max.y }
        if a.min.z < min_z { min_z = a.min.z } if a.max.z > max_z { max_z = a.max.z }
    }
    let cx = (min_x + max_x) * 0.5;
    let cy = (min_y + max_y) * 0.5;
    let cz = (min_z + max_z) * 0.5;

    let mut slot_to_child = [usize::MAX; 8];
    let mut slot_used = [false; 8];
    let mut pending = [0usize; 8];
    let mut n_pending = 0usize;

    for i in 0..n_children {
        let c_idx = children[i];
        let a = &bvh2[c_idx].aabb;
        let pcx = (a.min.x + a.max.x) * 0.5;
        let pcy = (a.min.y + a.max.y) * 0.5;
        let pcz = (a.min.z + a.max.z) * 0.5;
        let oct = ((pcx >= cx) as usize)
                | (((pcy >= cy) as usize) << 1)
                | (((pcz >= cz) as usize) << 2);
        if !slot_used[oct] {
            slot_used[oct] = true;
            slot_to_child[oct] = c_idx;
        } else {
            pending[n_pending] = c_idx;
            n_pending += 1;
        }
    }
    for i in 0..n_pending {
        let mut slot = 0;
        while slot_used[slot] { slot += 1; }
        slot_used[slot] = true;
        slot_to_child[slot] = pending[i];
    }

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
