use std::collections::HashMap;
use std::mem::MaybeUninit;

use crate::Color;
use crate::emitters::{EmitterTri, MeshEmitter};
use crate::{AABB, Ray, Scene, TriangleHit};
use nalgebra::{Point3, Transform3, UnitVector3, Vector2};

const MAX_TRAVERSAL_DEPTH: usize = 64;
const SAH_BINS: usize = 16;
const MAX_LEAF_CHUNKS: u32 = 1; // 1 chunk = 8 triangles

// SAH cost model. Only the ratio matters. Raising TRAVERSAL_COST relative to
// INTERSECT_COST biases the builder toward shallower trees with bigger leaves
// (fewer internal nodes visited, more SIMD chunk tests per leaf).
const TRAVERSAL_COST: f32 = 1.0; // Ct: cost of visiting a node
const INTERSECT_COST: f32 = 1.0; // Ci: cost of one SIMD-8 chunk intersection

#[derive(Clone, Copy)]
pub struct FlatTriangle {
    pub v0: u32,
    pub v1: u32,
    pub v2: u32,
    pub material: u32,
}

#[repr(C, align(32))]
#[derive(Clone, Copy)]
pub struct FlatTriangles {
    pub v0_x: [f32; 8],
    pub v0_y: [f32; 8],
    pub v0_z: [f32; 8],
    pub e1_x: [f32; 8],
    pub e1_y: [f32; 8],
    pub e1_z: [f32; 8],
    pub e2_x: [f32; 8],
    pub e2_y: [f32; 8],
    pub e2_z: [f32; 8],
}

impl Default for FlatTriangles {
    fn default() -> Self {
        Self {
            v0_x: [0.0; 8],
            v0_y: [0.0; 8],
            v0_z: [0.0; 8],
            e1_x: [0.0; 8],
            e1_y: [0.0; 8],
            e1_z: [0.0; 8],
            e2_x: [0.0; 8],
            e2_y: [0.0; 8],
            e2_z: [0.0; 8],
        }
    }
}

pub struct BvhBuilder {
    world_vertices: Vec<Point3<f32>>,
    world_normals: Vec<UnitVector3<f32>>,
    world_uvs: Vec<Vector2<f32>>,
    world_colors: Vec<Color>,
    flat_triangles: Vec<FlatTriangle>,
    emitter_meshes: Vec<MeshEmitter>,
}

impl BvhBuilder {
    pub fn new(scene: &Scene) -> Self {
        let world_vertices = Vec::new();
        let world_normals = Vec::new();
        let world_uvs = Vec::new();
        let world_colors = Vec::new();
        let flat_triangles = Vec::new();
        let emitter_meshes = Vec::new();

        let mut builder = Self {
            world_vertices,
            world_normals,
            world_uvs,
            world_colors,
            flat_triangles,
            emitter_meshes,
        };

        builder.flatten_scene(scene);
        builder.cluster_triangles();

        builder
    }

    fn cluster_triangles(&mut self) {
        let mut indices: Vec<usize> = (0..self.flat_triangles.len()).collect();
        let vertices = &self.world_vertices;
        let triangles = &self.flat_triangles;

        let centroids: Vec<Point3<f32>> = triangles
            .iter()
            .map(|tri| {
                let v0 = vertices[tri.v0 as usize];
                let v1 = vertices[tri.v1 as usize];
                let v2 = vertices[tri.v2 as usize];
                Point3::from((v0.coords + v1.coords + v2.coords) / 3.0)
            })
            .collect();

        let len = indices.len();
        Self::split_clusters(&mut indices, &centroids, 0, len);

        // Reorder triangles based on clustered indices
        let mut new_triangles = Vec::with_capacity(triangles.len());
        for &idx in &indices {
            new_triangles.push(triangles[idx]);
        }
        self.flat_triangles = new_triangles;
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
        let mut axis = 0;
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

        Self::split_clusters(indices, centroids, start, split_idx);
        Self::split_clusters(indices, centroids, split_idx, end);
    }

    fn flatten_scene(&mut self, scene: &Scene) {
        let mut stack = Vec::new();

        let mut vertex_map: HashMap<(usize, u32, u32), usize> = HashMap::new();

        for &root_idx in &scene.roots {
            stack.push((root_idx as usize, Transform3::identity()));
        }

        while let Some((node_idx, parent_transform)) = stack.pop() {
            let node = &scene.nodes[node_idx];
            let world_transform = parent_transform * node.transform;

            let normal_matrix = world_transform
                .matrix()
                .fixed_view::<3, 3>(0, 0)
                .try_inverse()
                .unwrap_or_else(nalgebra::Matrix3::identity)
                .transpose();

            for &mesh_idx in &node.meshes {
                let mesh = &scene.meshes[mesh_idx as usize];
                let mat = &scene.materials[mesh.material as usize];
                let is_emissive =
                    mat.emissive.x > 0.0 || mat.emissive.y > 0.0 || mat.emissive.z > 0.0;
                // One area light per emissive mesh instance.
                let mut emitter_tris: Vec<EmitterTri> = Vec::new();

                for &tri in &mesh.triangles {
                    let mut get_or_add_vertex = |local_idx: u32| -> usize {
                        let key = (node_idx, mesh_idx, local_idx);
                        *vertex_map.entry(key).or_insert_with(|| {
                            let p =
                                world_transform.transform_point(&mesh.vertices[local_idx as usize]);
                            let normal =
                                normal_matrix * mesh.normals[local_idx as usize].into_inner();
                            self.world_normals.push(UnitVector3::new_normalize(normal));

                            let uv = if !mesh.uvs.is_empty() {
                                mesh.uvs[local_idx as usize]
                            } else {
                                Vector2::new(0.0, 0.0)
                            };
                            self.world_uvs.push(uv);

                            let color = if !mesh.colors.is_empty() {
                                mesh.colors[local_idx as usize]
                            } else {
                                Color::new(1.0, 1.0, 1.0)
                            };
                            self.world_colors.push(color);

                            let idx = self.world_vertices.len();
                            self.world_vertices.push(p);
                            idx
                        })
                    };

                    let v0 = get_or_add_vertex(tri.v0);
                    let v1 = get_or_add_vertex(tri.v1);
                    let v2 = get_or_add_vertex(tri.v2);

                    self.flat_triangles.push(FlatTriangle {
                        v0: v0 as u32,
                        v1: v1 as u32,
                        v2: v2 as u32,
                        material: mesh.material,
                    });

                    if is_emissive {
                        // World-space triangle for the area light (same transform
                        // as the geometry above, so positions match exactly).
                        let p0 = world_transform.transform_point(&mesh.vertices[tri.v0 as usize]);
                        let p1 = world_transform.transform_point(&mesh.vertices[tri.v1 as usize]);
                        let p2 = world_transform.transform_point(&mesh.vertices[tri.v2 as usize]);
                        let n = (p1 - p0).cross(&(p2 - p0));
                        let len = n.norm();
                        if len > 0.0 {
                            let uv = |li: u32| {
                                if !mesh.uvs.is_empty() {
                                    mesh.uvs[li as usize]
                                } else {
                                    Vector2::new(0.0, 0.0)
                                }
                            };
                            emitter_tris.push(EmitterTri {
                                v0: p0,
                                v1: p1,
                                v2: p2,
                                uv0: uv(tri.v0),
                                uv1: uv(tri.v1),
                                uv2: uv(tri.v2),
                                normal: UnitVector3::new_unchecked(n / len),
                            });
                        }
                    }
                }

                if let Some(emitter) =
                    MeshEmitter::new(emitter_tris, mat.emissive, mat.emissive_texture)
                {
                    self.emitter_meshes.push(emitter);
                }
            }

            for &child_idx in &node.children {
                stack.push((child_idx as usize, world_transform));
            }
        }
    }

    pub fn build(mut self) -> (Bvh, Vec<MeshEmitter>) {
        let num_triangles = self.flat_triangles.len();

        let remainder = num_triangles % 8;
        if remainder != 0 {
            for _ in 0..(8 - remainder) {
                self.flat_triangles.push(FlatTriangle {
                    v0: 0,
                    v1: 0,
                    v2: 0,
                    material: 0,
                });
            }
        }

        let mut soa_chunks = Vec::new();
        let mut chunk_aabbs = Vec::new();
        let mut chunk_centroids = Vec::new();

        for chunk_start in (0..self.flat_triangles.len()).step_by(8) {
            let mut soa = FlatTriangles::default();
            let mut chunk_aabb = AABB::invalid();

            for i in 0..8 {
                let tri = &self.flat_triangles[chunk_start + i];

                let v0 = self.world_vertices[tri.v0 as usize];
                let v1 = self.world_vertices[tri.v1 as usize];
                let v2 = self.world_vertices[tri.v2 as usize];
                let e1 = v1 - v0;
                let e2 = v2 - v0;

                soa.v0_x[i] = v0.x;
                soa.v0_y[i] = v0.y;
                soa.v0_z[i] = v0.z;
                soa.e1_x[i] = e1.x;
                soa.e1_y[i] = e1.y;
                soa.e1_z[i] = e1.z;
                soa.e2_x[i] = e2.x;
                soa.e2_y[i] = e2.y;
                soa.e2_z[i] = e2.z;

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

        let mut nodes = Vec::new();
        let mut chunk_indices: Vec<usize> = (0..soa_chunks.len()).collect();

        nodes.push(BvhNode {
            aabb: AABB::invalid(),
            left_first: 0,
            chunk_count: chunk_indices.len() as u32,
        });

        if !chunk_indices.is_empty() {
            self.update_node_bounds(0, &mut nodes, &chunk_indices, &chunk_aabbs);
            self.subdivide_sah(
                0,
                &mut nodes,
                &mut chunk_indices,
                &chunk_aabbs,
                &chunk_centroids,
            );
        }

        let mut ordered_soa_chunks = Vec::with_capacity(soa_chunks.len());
        let mut ordered_triangles = Vec::with_capacity(self.flat_triangles.len());

        for &idx in &chunk_indices {
            ordered_soa_chunks.push(soa_chunks[idx]);
            for i in 0..8 {
                ordered_triangles.push(self.flat_triangles[idx * 8 + i]);
            }
        }

        let mut bvh8_nodes = Vec::new();
        Self::collapse_to_bvh8(0, &nodes, &mut bvh8_nodes);

        let bvh = Bvh {
            nodes: bvh8_nodes,
            vertices: self.world_vertices,
            normals: self.world_normals,
            uvs: self.world_uvs,
            colors: self.world_colors,
            triangles: ordered_triangles,
            soa_chunks: ordered_soa_chunks,
        };
        (bvh, self.emitter_meshes)
    }

    fn update_node_bounds(
        &self,
        node_idx: usize,
        nodes: &mut Vec<BvhNode>,
        chunk_indices: &[usize],
        chunk_aabbs: &[AABB],
    ) {
        let node = &mut nodes[node_idx];
        let mut aabb = AABB::invalid();
        let first = node.left_first as usize;
        let count = node.chunk_count as usize;

        for i in 0..count {
            let chunk_idx = chunk_indices[first + i];
            aabb.extend_aabb(&chunk_aabbs[chunk_idx]);
        }

        node.aabb = aabb;
    }

    fn subdivide_sah(
        &self,
        node_idx: usize,
        nodes: &mut Vec<BvhNode>,
        chunk_indices: &mut [usize],
        chunk_aabbs: &[AABB],
        chunk_centroids: &[Point3<f32>],
    ) {
        let node = &nodes[node_idx];
        let first = node.left_first as usize;
        let count = node.chunk_count as usize;

        if count <= 1 {
            return;
        }

        let mut best_cost = f32::MAX;
        let mut best_axis = 0;
        let mut best_split_bin = 0;

        let mut centroid_bounds = AABB::invalid();
        for i in 0..count {
            centroid_bounds.extend(chunk_centroids[chunk_indices[first + i]]);
        }

        let extent = centroid_bounds.max - centroid_bounds.min;
        if extent.x == 0.0 && extent.y == 0.0 && extent.z == 0.0 {
            return; // All centroids are identical
        }

        for axis in 0..3 {
            let bounds_min = centroid_bounds.min[axis];
            let bounds_max = centroid_bounds.max[axis];
            let bounds_extent = bounds_max - bounds_min;
            if bounds_extent == 0.0 {
                continue;
            }

            #[derive(Clone, Copy)]
            struct Bin {
                aabb: AABB,
                count: usize,
            }
            let mut bins = [Bin {
                aabb: AABB::invalid(),
                count: 0,
            }; SAH_BINS];

            for i in 0..count {
                let chunk_idx = chunk_indices[first + i];
                let centroid = chunk_centroids[chunk_idx];
                let t = (centroid[axis] - bounds_min) / bounds_extent;
                let bin_idx = (SAH_BINS as f32 * t).min(SAH_BINS as f32 - 1.0) as usize;

                bins[bin_idx].aabb.extend_aabb(&chunk_aabbs[chunk_idx]);
                bins[bin_idx].count += 1;
            }

            let mut left_aabbs = [AABB::invalid(); SAH_BINS - 1];
            let mut left_counts = [0; SAH_BINS - 1];
            let mut right_aabbs = [AABB::invalid(); SAH_BINS - 1];
            let mut right_counts = [0; SAH_BINS - 1];

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

        let leaf_cost = count as f32 * INTERSECT_COST; // count chunks, each one SIMD test

        if best_cost >= leaf_cost {
            return;
        }

        let bounds_min = centroid_bounds.min[best_axis];
        let bounds_extent = centroid_bounds.max[best_axis] - centroid_bounds.min[best_axis];

        let mut left = first;
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
                if right == 0 {
                    break;
                }
                right -= 1;
            }
        }

        let left_count = left - first;
        if left_count == 0 || left_count == count {
            return; // Fallback
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

        self.update_node_bounds(left_child_idx, nodes, chunk_indices, chunk_aabbs);
        self.update_node_bounds(right_child_idx, nodes, chunk_indices, chunk_aabbs);

        self.subdivide_sah(
            left_child_idx,
            nodes,
            chunk_indices,
            chunk_aabbs,
            chunk_centroids,
        );
        self.subdivide_sah(
            right_child_idx,
            nodes,
            chunk_indices,
            chunk_aabbs,
            chunk_centroids,
        );
    }

    fn collapse_to_bvh8(node_idx: usize, bvh2: &[BvhNode], bvh8: &mut Vec<Bvh8Node>) -> u32 {
        let bvh8_idx = bvh8.len() as u32;
        bvh8.push(Bvh8Node::default());

        let mut children = vec![node_idx];

        while children.len() < 8 {
            let mut best_idx = None;
            let mut best_area = -1.0;

            for (i, &c_idx) in children.iter().enumerate() {
                let node = &bvh2[c_idx];
                if !node.is_leaf() {
                    let area = node.aabb.surface_area();
                    if area > best_area {
                        best_area = area;
                        best_idx = Some(i);
                    }
                }
            }

            if let Some(i) = best_idx {
                let node = &bvh2[children[i]];
                let left = node.left_first as usize;
                let right = left + 1;
                children.swap_remove(i);
                children.push(left);
                children.push(right);
            } else {
                break;
            }
        }

        let mut node8 = Bvh8Node::default();

        for (i, &c_idx) in children.iter().enumerate() {
            let bvh2_node = &bvh2[c_idx];
            node8.p_min_x[i] = bvh2_node.aabb.min.x;
            node8.p_min_y[i] = bvh2_node.aabb.min.y;
            node8.p_min_z[i] = bvh2_node.aabb.min.z;
            node8.p_max_x[i] = bvh2_node.aabb.max.x;
            node8.p_max_y[i] = bvh2_node.aabb.max.y;
            node8.p_max_z[i] = bvh2_node.aabb.max.z;

            if bvh2_node.is_leaf() {
                // count is always > 0 for a leaf, so the leaf flag is always set.
                node8.children[i] =
                    Bvh8Node::encode_leaf(bvh2_node.left_first, bvh2_node.chunk_count);
            } else {
                node8.children[i] = Self::collapse_to_bvh8(c_idx, bvh2, bvh8);
            }
        }

        bvh8[bvh8_idx as usize] = node8;
        bvh8_idx
    }
}

#[repr(C, align(32))]
#[derive(Clone, Copy)]
pub struct Bvh8Node {
    pub p_min_x: [f32; 8],
    pub p_min_y: [f32; 8],
    pub p_min_z: [f32; 8],
    pub p_max_x: [f32; 8],
    pub p_max_y: [f32; 8],
    pub p_max_z: [f32; 8],
    // Pre-encoded child references (computed once at build): for a leaf,
    // `idx | (chunk_count << 24) | 0x8000_0000`; for an internal node, the
    // plain Bvh8 node index. `u32::MAX` marks an empty lane.
    pub children: [u32; 8],
}

impl Default for Bvh8Node {
    fn default() -> Self {
        Self {
            p_min_x: [f32::NAN; 8],
            p_min_y: [f32::NAN; 8],
            p_min_z: [f32::NAN; 8],
            p_max_x: [f32::NAN; 8],
            p_max_y: [f32::NAN; 8],
            p_max_z: [f32::NAN; 8],
            children: [u32::MAX; 8],
        }
    }
}

impl Bvh8Node {
    // Child reference encoding: leaves set the top bit, store the chunk count
    // in bits [24,30] and the chunk-range start in bits [0,23]. Internal nodes
    // store a plain Bvh8 node index.
    const LEAF_FLAG: u32 = 0x8000_0000;
    const COUNT_SHIFT: u32 = 24;
    const COUNT_MASK: u32 = 0x7F;
    const START_MASK: u32 = 0x00FF_FFFF;

    #[inline(always)]
    fn encode_leaf(start_chunk: u32, chunk_count: u32) -> u32 {
        start_chunk | (chunk_count << Self::COUNT_SHIFT) | Self::LEAF_FLAG
    }

    #[inline(always)]
    fn is_leaf(encoded: u32) -> bool {
        encoded & Self::LEAF_FLAG != 0
    }

    #[inline(always)]
    fn leaf_start(encoded: u32) -> u32 {
        encoded & Self::START_MASK
    }

    #[inline(always)]
    fn leaf_count(encoded: u32) -> u32 {
        (encoded >> Self::COUNT_SHIFT) & Self::COUNT_MASK
    }
}

#[derive(Clone, Copy)]
pub struct BvhNode {
    pub aabb: AABB,
    pub left_first: u32,
    pub chunk_count: u32,
}

impl BvhNode {
    pub fn is_leaf(&self) -> bool {
        self.chunk_count > 0
    }
}

pub struct BvhHit {
    pub hit: TriangleHit,
    pub material: u32,
    pub v0: u32,
    pub v1: u32,
    pub v2: u32,
    pub chunk_idx: u32,
}

pub struct Bvh {
    pub nodes: Vec<Bvh8Node>,
    pub vertices: Vec<Point3<f32>>,
    pub normals: Vec<UnitVector3<f32>>,
    pub uvs: Vec<Vector2<f32>>,
    pub colors: Vec<Color>,
    pub triangles: Vec<FlatTriangle>,
    pub soa_chunks: Vec<FlatTriangles>,
}

struct BvhStack {
    // Uninitialized on purpose: every slot is written by `push` before it can
    // be read by `pop`, so zero-initializing it per ray was a dead 512-byte
    // memset (one per Bvh::intersect call).
    data: [MaybeUninit<(u32, f32)>; MAX_TRAVERSAL_DEPTH],
    ptr: usize,
}

impl BvhStack {
    #[inline(always)]
    fn new() -> Self {
        Self {
            data: [const { MaybeUninit::uninit() }; MAX_TRAVERSAL_DEPTH],
            ptr: 0,
        }
    }

    #[inline(always)]
    fn push(&mut self, node_idx: u32, t: f32) {
        unsafe {
            self.data.get_unchecked_mut(self.ptr).write((node_idx, t));
        }
        self.ptr += 1;
    }

    #[inline(always)]
    fn pop(&mut self) -> (u32, f32) {
        self.ptr -= 1;
        unsafe { self.data.get_unchecked(self.ptr).assume_init() }
    }

    #[inline(always)]
    fn is_empty(&self) -> bool {
        self.ptr == 0
    }
}

impl Bvh {
    // Unchecked accessors for the traversal hot path. Indices are always valid
    // by construction (encoded at build time / derived from in-range lanes), so
    // we skip bounds checks while keeping `intersect` readable.
    #[inline(always)]
    fn get_node(&self, idx: u32) -> &Bvh8Node {
        unsafe { self.nodes.get_unchecked(idx as usize) }
    }

    #[inline(always)]
    fn get_chunk(&self, idx: u32) -> &FlatTriangles {
        unsafe { self.soa_chunks.get_unchecked(idx as usize) }
    }

    #[inline(always)]
    fn get_triangle(&self, idx: u32) -> &FlatTriangle {
        unsafe { self.triangles.get_unchecked(idx as usize) }
    }

    pub fn intersect(&self, ray: &Ray) -> Option<BvhHit> {
        let mut closest_hit: Option<BvhHit> = None;
        let mut t_max = ray.tmax;

        if self.nodes.is_empty() {
            return None;
        }

        let mut stack = BvhStack::new();
        stack.push(0, 0.0); // push root

        while !stack.is_empty() {
            let (encoded_idx, node_t) = stack.pop();

            if node_t >= t_max {
                continue;
            }

            if Bvh8Node::is_leaf(encoded_idx) {
                // Leaf node
                let start_chunk = Bvh8Node::leaf_start(encoded_idx);
                let chunk_count = Bvh8Node::leaf_count(encoded_idx);

                for i in 0..chunk_count {
                    let chunk_idx = start_chunk + i;
                    let chunk = self.get_chunk(chunk_idx);

                    if let Some(simd_hit) = ray.closest_triangle_simd8(chunk, t_max) {
                        t_max = simd_hit.hit.t;
                        let tri_global_idx = chunk_idx * 8 + simd_hit.lane;
                        let tri = self.get_triangle(tri_global_idx);

                        closest_hit = Some(BvhHit {
                            hit: simd_hit.hit,
                            material: tri.material,
                            v0: tri.v0,
                            v1: tri.v1,
                            v2: tri.v2,
                            chunk_idx,
                        });
                    }
                }
            } else {
                // Internal Bvh8Node
                let node = self.get_node(encoded_idx);
                let (mut bitmask, t_arr) = ray.intersect_bvh8(node, t_max);

                let mut hits = [(0u32, 0.0f32); 8];
                let mut hit_count = 0;

                while bitmask != 0 {
                    let lane = bitmask.trailing_zeros() as usize;
                    bitmask &= bitmask - 1;

                    // Encoded once at build time. Guard against empty lanes
                    // (u32::MAX) in case a NaN slips through min/max in the mask.
                    let encoded = node.children[lane];
                    let t = t_arr[lane];
                    if encoded != u32::MAX && t < t_max {
                        hits[hit_count] = (encoded, t);
                        hit_count += 1;
                    }
                }

                if hit_count > 0 {
                    // Insertion sort (descending) so the furthest is pushed first
                    for i in 1..hit_count {
                        let mut j = i;
                        while j > 0 && hits[j - 1].1 < hits[j].1 {
                            hits.swap(j - 1, j);
                            j -= 1;
                        }
                    }

                    for i in 0..hit_count {
                        stack.push(hits[i].0, hits[i].1);
                    }
                }
            }
        }

        closest_hit
    }

    /// Any-hit traversal for shadow rays: returns true as soon as any triangle
    /// is hit within the ray's range. Cheaper than `intersect`: no closest
    /// tracking, no `t_max` tightening, no child sorting, and it bails out at
    /// the first hit. `t_max` is fixed to the ray's end (e.g. light distance).
    pub fn is_occluded(&self, ray: &Ray) -> bool {
        if self.nodes.is_empty() {
            return false;
        }

        let t_max = ray.tmax;
        let mut stack = BvhStack::new();
        stack.push(0, 0.0);

        while !stack.is_empty() {
            let (encoded_idx, _) = stack.pop();

            if Bvh8Node::is_leaf(encoded_idx) {
                // Leaf node
                let start_chunk = Bvh8Node::leaf_start(encoded_idx);
                let chunk_count = Bvh8Node::leaf_count(encoded_idx);

                for i in 0..chunk_count {
                    let chunk_idx = start_chunk + i;
                    if ray.any_triangle_simd8(self.get_chunk(chunk_idx), t_max) {
                        return true; // early-out: occluder found
                    }
                }
            } else {
                // Internal node: push hit children unsorted (order is irrelevant
                // for any-hit, and t_max never tightens).
                let node = self.get_node(encoded_idx);
                let (mut bitmask, t_arr) = ray.intersect_bvh8(node, t_max);

                while bitmask != 0 {
                    let lane = bitmask.trailing_zeros() as usize;
                    bitmask &= bitmask - 1;

                    let encoded = node.children[lane];
                    if encoded != u32::MAX && t_arr[lane] < t_max {
                        stack.push(encoded, 0.0);
                    }
                }
            }
        }

        false
    }
}
