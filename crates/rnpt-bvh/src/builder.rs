use nalgebra::Point3;

use crate::{
    bvh::{BvhInner, BvhNode, FlatTriangle, FlatTriangles},
    math::AABB,
    Geometry,
};

/// A leaf holds at most one SIMD chunk.
const MAX_LEAF_TRIANGLES: u32 = 8;

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

    let tri_count = triangles.len() as u32;
    let mut nodes = vec![BvhNode { aabb: AABB::invalid(), left_first: 0, tri_count }];
    if tri_count > 0 {
        update_node_bounds(0, &mut nodes, &vertices, &triangles);
        subdivide(0, &mut nodes, &vertices, &triangles);
    }

    // Pack each leaf's triangles into one SoA chunk; the leaf now points to it.
    let mut soa_chunks = Vec::new();
    for node in nodes.iter_mut().filter(|n| n.is_leaf()) {
        let first = node.left_first as usize;
        let mut soa = FlatTriangles { first_triangle_idx: first as u32, ..Default::default() };
        for i in 0..node.tri_count as usize {
            let tri = &triangles[first + i];
            let v0 = vertices[tri.v0 as usize];
            let e1 = vertices[tri.v1 as usize] - v0;
            let e2 = vertices[tri.v2 as usize] - v0;
            soa.v0_x[i] = v0.x; soa.v0_y[i] = v0.y; soa.v0_z[i] = v0.z;
            soa.e1_x[i] = e1.x; soa.e1_y[i] = e1.y; soa.e1_z[i] = e1.z;
            soa.e2_x[i] = e2.x; soa.e2_y[i] = e2.y; soa.e2_z[i] = e2.z;
            if tri.double_sided {
                soa.double_sided_mask |= 1 << i;
            }
        }
        node.left_first = soa_chunks.len() as u32;
        soa_chunks.push(soa);
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

fn update_node_bounds(node_idx: usize, nodes: &mut [BvhNode], vertices: &[Point3<f32>], triangles: &[FlatTriangle]) {
    let first = nodes[node_idx].left_first as usize;
    let count = nodes[node_idx].tri_count as usize;

    let mut aabb = AABB::invalid();
    for tri in &triangles[first..first + count] {
        aabb.extend(vertices[tri.v0 as usize]);
        aabb.extend(vertices[tri.v1 as usize]);
        aabb.extend(vertices[tri.v2 as usize]);
    }
    // Expand slightly to avoid precision issues with flat triangles.
    let eps = nalgebra::Vector3::new(1e-5, 1e-5, 1e-5);
    aabb.min -= eps;
    aabb.max += eps;

    nodes[node_idx].aabb = aabb;
}

fn subdivide(node_idx: usize, nodes: &mut Vec<BvhNode>, vertices: &[Point3<f32>], triangles: &[FlatTriangle]) {
    let node = nodes[node_idx];
    if node.tri_count <= MAX_LEAF_TRIANGLES {
        return;
    }

    // Split in the middle: the array is already sorted by Morton code.
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

    update_node_bounds(left_child_idx, nodes, vertices, triangles);
    update_node_bounds(right_child_idx, nodes, vertices, triangles);
    subdivide(left_child_idx, nodes, vertices, triangles);
    subdivide(right_child_idx, nodes, vertices, triangles);
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
