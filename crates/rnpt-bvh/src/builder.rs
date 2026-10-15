use nalgebra::Point3;

use crate::{
    bvh::{BvhInner, BvhNode, FlatTriangle, FlatTriangles},
    math::AABB,
    Geometry,
};

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

    sort_by_morton_code(&vertices, &mut triangles);

    // Pad to a multiple of 8 with degenerate triangles that never hit.
    let num_triangles = triangles.len();
    let remainder = num_triangles % 8;
    if remainder != 0 {
        for _ in 0..(8 - remainder) {
            triangles.push(FlatTriangle { v0: 0, v1: 0, v2: 0, orig_idx: u32::MAX, geom_id: u32::MAX, double_sided: false });
        }
    }

    // Chunks first: every 8 consecutive triangles in Morton order become one
    // SoA chunk; the tree is then built over chunks, not triangles.
    let mut soa_chunks = Vec::new();
    let mut chunk_aabbs = Vec::new();
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
    }

    let chunk_count = soa_chunks.len() as u32;
    let mut nodes = vec![BvhNode { aabb: AABB::invalid(), left_first: 0, tri_count: chunk_count }];
    if chunk_count > 0 {
        update_node_bounds(0, &mut nodes, &chunk_aabbs);
        subdivide(0, &mut nodes, &chunk_aabbs);
    }

    BvhInner { nodes, triangles, soa_chunks }
}

/// Order triangles along a Z-order curve of their centroids, so that a plain
/// median split of the array gives spatially coherent children.
fn sort_by_morton_code(vertices: &[Point3<f32>], triangles: &mut [FlatTriangle]) {
    let mut world_aabb = AABB::invalid();
    for &v in vertices {
        world_aabb.extend(v);
    }

    triangles.sort_by_cached_key(|tri| {
        let centroid = Point3::from(
            (vertices[tri.v0 as usize].coords + vertices[tri.v1 as usize].coords + vertices[tri.v2 as usize].coords)
                / 3.0,
        );
        let c = world_aabb.normalize(centroid);
        morton_3d(c.x, c.y, c.z)
    });
}

fn update_node_bounds(node_idx: usize, nodes: &mut [BvhNode], chunk_aabbs: &[AABB]) {
    let first = nodes[node_idx].left_first as usize;
    let count = nodes[node_idx].tri_count as usize;

    let mut aabb = AABB::invalid();
    for chunk_aabb in &chunk_aabbs[first..first + count] {
        aabb.extend_aabb(chunk_aabb);
    }
    nodes[node_idx].aabb = aabb;
}

fn subdivide(node_idx: usize, nodes: &mut Vec<BvhNode>, chunk_aabbs: &[AABB]) {
    let node = nodes[node_idx];
    if node.tri_count <= 1 {
        return;
    }

    // Split in the middle: chunks are already in Morton order.
    let first = node.left_first as usize;
    let count = node.tri_count as usize;
    let split_idx = first + count / 2;

    let left_child_idx = nodes.len();
    nodes.push(BvhNode { aabb: AABB::invalid(), left_first: first as u32, tri_count: (split_idx - first) as u32 });
    let right_child_idx = nodes.len();
    nodes.push(BvhNode {
        aabb: AABB::invalid(),
        left_first: split_idx as u32,
        tri_count: (count - (split_idx - first)) as u32,
    });

    nodes[node_idx].left_first = left_child_idx as u32;
    nodes[node_idx].tri_count = 0;

    update_node_bounds(left_child_idx, nodes, chunk_aabbs);
    update_node_bounds(right_child_idx, nodes, chunk_aabbs);
    subdivide(left_child_idx, nodes, chunk_aabbs);
    subdivide(right_child_idx, nodes, chunk_aabbs);
}

/// Spread the 21 low bits of `v` so that two zeros sit between each bit.
#[inline(always)]
fn expand_bits(mut v: u64) -> u64 {
    v &= 0x0000_0000_001f_ffff;
    v = (v | (v << 32)) & 0x001f_0000_0000_ffff;
    v = (v | (v << 16)) & 0x001f_0000_ff00_00ff;
    v = (v | (v << 8)) & 0x100f_00f0_0f00_f00f;
    v = (v | (v << 4)) & 0x10c3_0c30_c30c_30c3;
    v = (v | (v << 2)) & 0x1249_2492_4924_9249;
    v
}

/// 3D Morton code of a point normalized to [0, 1].
#[inline]
fn morton_3d(x: f32, y: f32, z: f32) -> u64 {
    let q = |c: f32| (c.clamp(0.0, 1.0) * 2_097_151.0) as u64;
    expand_bits(q(x)) | (expand_bits(q(y)) << 1) | (expand_bits(q(z)) << 2)
}
