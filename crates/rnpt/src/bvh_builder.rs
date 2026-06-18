use std::collections::HashMap;

use crate::emitters::{EmitterTri, MeshEmitter};
use crate::{AABB, Bvh, Bvh8Node, Color, FlatTriangle, FlatTriangles, Scene};
use nalgebra::{Point3, Transform3, UnitVector3, Vector2};

const SAH_BINS: usize = 16;
const MAX_LEAF_CHUNKS: u32 = 1; // 1 chunk = 8 triangles

// SAH cost model. Only the ratio matters. Raising TRAVERSAL_COST relative to
// INTERSECT_COST biases the builder toward shallower trees with bigger leaves
// (fewer internal nodes visited, more SIMD chunk tests per leaf).
const TRAVERSAL_COST: f32 = 1.0; // Ct: cost of visiting a node
const INTERSECT_COST: f32 = 1.0; // Ci: cost of one SIMD-8 chunk intersection

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
                // One area light per emissive mesh instance. Its unified-light
                // index = (punctual lights) + (this emitter's position), matching
                // `build_lights` (punctual ++ emitters). `u32::MAX` for non-emitters.
                let light_idx = if is_emissive {
                    scene.lights.len() as u32 + self.emitter_meshes.len() as u32
                } else {
                    u32::MAX
                };
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
                        light: light_idx,
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
                    light: u32::MAX,
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

        // Expand the best (largest-area) non-leaf child until we have up to 8 children.
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
                children[i] = children[n_children - 1]; // swap_remove
                n_children -= 1;
                children[n_children] = left;
                n_children += 1;
                children[n_children] = left + 1;
                n_children += 1;
            } else {
                break;
            }
        }

        // Compute node center for octant slot assignment (all stack, no heap).
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

        // Assign each child to its octant slot; collisions fall back to first free slot.
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

        // Fill BVH8 node; recursive calls may grow bvh8, bvh8_idx stays valid.
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
                Self::collapse_to_bvh8(c_idx, bvh2, bvh8)
            };
        }

        bvh8[bvh8_idx as usize] = node8;
        bvh8_idx
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
