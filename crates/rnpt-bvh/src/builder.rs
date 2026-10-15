use nalgebra::Point3;

use crate::{
    bvh::{BvhInner, BvhNode, FlatTriangle, FlatTriangles},
    math::AABB,
    Geometry,
};

const SAH_BINS: usize = 8;

pub(crate) fn build(geometries: &[Geometry]) -> BvhInner {
    let mut vertices: Vec<Point3<f32>> = Vec::new();
    let mut triangles: Vec<FlatTriangle> = Vec::new();

    for (geom_id, geom) in geometries.iter().enumerate() {
        let vert_offset = vertices.len() as u32;
        vertices.extend(geom.verts.iter().map(|v| Point3::new(v[0], v[1], v[2])));
        for (orig_idx, t) in geom.tris.iter().enumerate() {
            triangles.push(FlatTriangle {
                v0: t[0] + vert_offset,
                v1: t[1] + vert_offset,
                v2: t[2] + vert_offset,
                orig_idx: orig_idx as u32,
                geom_id: geom_id as u32,
                double_sided: geom.double_sided.get(orig_idx).copied().unwrap_or(false),
            });
        }
    }

    cluster_triangles(&vertices, &mut triangles);

    // Pad to a multiple of 8 with degenerate triangles that never hit.
    let num_triangles = triangles.len();
    let remainder = num_triangles % 8;
    if remainder != 0 {
        for _ in 0..(8 - remainder) {
            triangles.push(FlatTriangle { v0: 0, v1: 0, v2: 0, orig_idx: u32::MAX, geom_id: u32::MAX, double_sided: false });
        }
    }

    // Chunks first: every 8 consecutive triangles become one SoA chunk; the
    // tree is then built over chunks, not triangles.
    let mut soa_chunks = Vec::new();
    let mut chunk_aabbs = Vec::new();
    let mut chunk_centroids = Vec::new();
    for chunk_start in (0..triangles.len()).step_by(8) {
        let mut soa = FlatTriangles::default();
        let mut chunk_aabb = AABB::invalid();

        for i in 0..8 {
            let tri = &triangles[chunk_start + i];
            let v0 = vertices[tri.v0 as usize];
            let v1 = vertices[tri.v1 as usize];
            let v2 = vertices[tri.v2 as usize];
            let e1 = v1 - v0;
            let e2 = v2 - v0;
            soa.v0_x[i] = v0.x; soa.v0_y[i] = v0.y; soa.v0_z[i] = v0.z;
            soa.e1_x[i] = e1.x; soa.e1_y[i] = e1.y; soa.e1_z[i] = e1.z;
            soa.e2_x[i] = e2.x; soa.e2_y[i] = e2.y; soa.e2_z[i] = e2.z;
            if tri.double_sided {
                soa.double_sided_mask |= 1 << i;
            }
            // Padding must not grow the bounds.
            if chunk_start + i < num_triangles {
                chunk_aabb.extend(v0);
                chunk_aabb.extend(v1);
                chunk_aabb.extend(v2);
            }
        }

        // Expand slightly to avoid precision issues with flat triangles.
        let eps = nalgebra::Vector3::new(1e-5, 1e-5, 1e-5);
        chunk_aabb.min -= eps;
        chunk_aabb.max += eps;

        soa_chunks.push(soa);
        chunk_aabbs.push(chunk_aabb);
        chunk_centroids.push(chunk_aabb.center());
    }

    let mut chunk_indices: Vec<usize> = (0..soa_chunks.len()).collect();
    let mut nodes = vec![BvhNode { aabb: AABB::invalid(), left_first: 0, tri_count: chunk_indices.len() as u32 }];
    if !chunk_indices.is_empty() {
        update_node_bounds(0, &mut nodes, &chunk_indices, &chunk_aabbs);
        subdivide_sah(0, &mut nodes, &mut chunk_indices, &chunk_aabbs, &chunk_centroids);
    }

    // Store chunks (and their triangles) in leaf order.
    let ordered_chunks = chunk_indices.iter().map(|&idx| soa_chunks[idx]).collect();
    let ordered_triangles = chunk_indices.iter().flat_map(|&idx| triangles[idx * 8..idx * 8 + 8].iter().copied()).collect();

    BvhInner { nodes, triangles: ordered_triangles, soa_chunks: ordered_chunks }
}

/// Group triangles into compact clusters of 8: recursive median splits along
/// the largest centroid extent, with split points on multiples of 8, so that
/// each chunk holds 8 spatially close triangles.
fn cluster_triangles(vertices: &[Point3<f32>], triangles: &mut Vec<FlatTriangle>) {
    let centroids: Vec<Point3<f32>> = triangles
        .iter()
        .map(|tri| {
            Point3::from(
                (vertices[tri.v0 as usize].coords + vertices[tri.v1 as usize].coords + vertices[tri.v2 as usize].coords)
                    / 3.0,
            )
        })
        .collect();

    let mut indices: Vec<usize> = (0..triangles.len()).collect();
    let len = indices.len();
    split_clusters(&mut indices, &centroids, 0, len);

    let orig = triangles.clone();
    for (dst, &src) in triangles.iter_mut().zip(&indices) {
        *dst = orig[src];
    }
}

fn split_clusters(indices: &mut [usize], centroids: &[Point3<f32>], start: usize, end: usize) {
    let count = end - start;
    if count <= 8 {
        return;
    }

    let mut centroid_bounds = AABB::invalid();
    for &i in &indices[start..end] {
        centroid_bounds.extend(centroids[i]);
    }

    let extent = centroid_bounds.max - centroid_bounds.min;
    let mut axis = 0usize;
    if extent.y > extent.x {
        axis = 1;
    }
    if extent.z > extent[axis] {
        axis = 2;
    }

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

fn update_node_bounds(node_idx: usize, nodes: &mut [BvhNode], chunk_indices: &[usize], chunk_aabbs: &[AABB]) {
    let first = nodes[node_idx].left_first as usize;
    let count = nodes[node_idx].tri_count as usize;

    let mut aabb = AABB::invalid();
    for &chunk_idx in &chunk_indices[first..first + count] {
        aabb.extend_aabb(&chunk_aabbs[chunk_idx]);
    }
    nodes[node_idx].aabb = aabb;
}

/// Binned SAH over chunks. A chunk costs 1 to intersect whatever it holds:
/// that is the whole point of testing 8 triangles at once.
fn subdivide_sah(
    node_idx: usize,
    nodes: &mut Vec<BvhNode>,
    chunk_indices: &mut [usize],
    chunk_aabbs: &[AABB],
    chunk_centroids: &[Point3<f32>],
) {
    let first = nodes[node_idx].left_first as usize;
    let count = nodes[node_idx].tri_count as usize;

    if count <= 1 {
        return;
    }

    let mut best_cost = f32::MAX;
    let mut best_axis = 0usize;
    let mut best_split_bin = 0usize;

    let mut centroid_bounds = AABB::invalid();
    for &chunk_idx in &chunk_indices[first..first + count] {
        centroid_bounds.extend(chunk_centroids[chunk_idx]);
    }

    let extent = centroid_bounds.max - centroid_bounds.min;
    if extent.x == 0.0 && extent.y == 0.0 && extent.z == 0.0 {
        return; // All centroids are identical.
    }

    for axis in 0..3 {
        let bounds_min = centroid_bounds.min[axis];
        let bounds_extent = centroid_bounds.max[axis] - bounds_min;
        if bounds_extent == 0.0 {
            continue;
        }

        #[derive(Clone, Copy)]
        struct Bin {
            aabb: AABB,
            count: usize,
        }
        let mut bins = [Bin { aabb: AABB::invalid(), count: 0 }; SAH_BINS];

        for &chunk_idx in &chunk_indices[first..first + count] {
            let centroid = chunk_centroids[chunk_idx];
            let bin_idx = ((((centroid[axis] - bounds_min) / bounds_extent) * SAH_BINS as f32) as usize).min(SAH_BINS - 1);
            bins[bin_idx].aabb.extend_aabb(&chunk_aabbs[chunk_idx]);
            bins[bin_idx].count += 1;
        }

        let mut left_aabbs = [AABB::invalid(); SAH_BINS - 1];
        let mut left_counts = [0usize; SAH_BINS - 1];
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
                let cost = 1.0
                    + (left_aabbs[i].surface_area() * left_counts[i] as f32
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

    let leaf_cost = count as f32;
    if best_cost >= leaf_cost {
        return;
    }

    let bounds_min = centroid_bounds.min[best_axis];
    let bounds_extent = centroid_bounds.max[best_axis] - bounds_min;

    let mut left = first;
    let mut right = first + count - 1;
    while left <= right {
        let centroid = chunk_centroids[chunk_indices[left]];
        let bin_idx = ((((centroid[best_axis] - bounds_min) / bounds_extent) * SAH_BINS as f32) as usize).min(SAH_BINS - 1);
        if bin_idx <= best_split_bin {
            left += 1;
        } else {
            chunk_indices.swap(left, right);
            if right == 0 {
                break;
            }
            right -= 1;
        }
    }

    let left_count = left - first;
    if left_count == 0 || left_count == count {
        return;
    }

    let left_child_idx = nodes.len();
    nodes.push(BvhNode { aabb: AABB::invalid(), left_first: first as u32, tri_count: left_count as u32 });
    let right_child_idx = nodes.len();
    nodes.push(BvhNode { aabb: AABB::invalid(), left_first: left as u32, tri_count: (count - left_count) as u32 });

    nodes[node_idx].left_first = left_child_idx as u32;
    nodes[node_idx].tri_count = 0;

    update_node_bounds(left_child_idx, nodes, chunk_indices, chunk_aabbs);
    update_node_bounds(right_child_idx, nodes, chunk_indices, chunk_aabbs);
    subdivide_sah(left_child_idx, nodes, chunk_indices, chunk_aabbs, chunk_centroids);
    subdivide_sah(right_child_idx, nodes, chunk_indices, chunk_aabbs, chunk_centroids);
}
