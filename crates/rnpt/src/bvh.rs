use std::collections::HashMap;

use crate::{AABB, Scene};
use nalgebra::{Point3, Transform3, UnitVector3};

pub struct FlatTriangle {
    v0: u32,
    v1: u32,
    v2: u32,
    material: u32,
    morton_code: u64,
}

pub struct BvhBuilder {
    world_vertices: Vec<Point3<f32>>,
    world_normals: Vec<UnitVector3<f32>>,
    flat_triangles: Vec<FlatTriangle>,
}

impl BvhBuilder {
    pub fn new(scene: &Scene) -> Self {
        let world_vertices = Vec::new();
        let world_normals = Vec::new();
        let flat_triangles = Vec::new();

        let mut builder = Self {
            world_vertices,
            world_normals,
            flat_triangles,
        };

        builder.flatten_scene(scene);
        builder.reorder_flat_triangles();

        builder
    }

    fn reorder_flat_triangles(&mut self) {
        // compute world aabb
        let mut world_aabb = AABB::invalid();

        for &vertex in &self.world_vertices {
            world_aabb.extend(vertex);
        }

        // compute morton conde for each flat triangle
        for flat_tri in &mut self.flat_triangles {
            let v0_idx = flat_tri.v0 as usize;
            let v1_idx = flat_tri.v1 as usize;
            let v2_idx = flat_tri.v2 as usize;

            // compute centroid
            let mut centroid = Point3::from(
                (self.world_vertices[v0_idx].coords
                    + self.world_vertices[v1_idx].coords
                    + self.world_vertices[v2_idx].coords)
                    / 3.0,
            );

            // normalize centroid
            centroid = world_aabb.normalize(centroid);

            // compute morton code
            let morton_code = Self::morton_3d(centroid.x, centroid.y, centroid.z);

            flat_tri.morton_code = morton_code;
        }

        // sort
        self.flat_triangles.sort_by_key(|a| a.morton_code);
    }

    fn flatten_scene(&mut self, scene: &Scene) {
        let mut stack = [(0usize, Transform3::identity()); 4096];
        let mut stack_ptr = 0;

        let mut vertex_map: HashMap<(usize, u32, u32), usize> = HashMap::new();

        for &root_idx in &scene.roots {
            stack[stack_ptr] = (root_idx as usize, Transform3::identity());
            stack_ptr += 1;
        }

        while stack_ptr > 0 {
            stack_ptr -= 1;
            let (node_idx, parent_transform) = stack[stack_ptr];
            let node = &scene.nodes[node_idx];
            let world_transform = parent_transform * node.transform;

            // inverse transform nomal matrix to avoid non-uniform scaling distortion
            let normal_matrix = world_transform
                .matrix()
                .fixed_view::<3, 3>(0, 0)
                .try_inverse()
                .unwrap_or_else(nalgebra::Matrix3::identity)
                .transpose();

            for &mesh_idx in &node.meshes {
                let mesh = &scene.meshes[mesh_idx as usize];

                for &tri in &mesh.triangles {
                    // Helper function to get or add a vertex to the global list
                    let mut get_or_add_vertex = |local_idx: u32| -> usize {
                        let key = (node_idx, mesh_idx, local_idx);
                        *vertex_map.entry(key).or_insert_with(|| {
                            let p =
                                world_transform.transform_point(&mesh.vertices[local_idx as usize]);
                            let n = UnitVector3::new_normalize(
                                normal_matrix * mesh.normals[local_idx as usize].into_inner(),
                            );

                            let global_idx = self.world_vertices.len();
                            self.world_vertices.push(p);
                            self.world_normals.push(n);
                            global_idx
                        })
                    };

                    let v0_idx = get_or_add_vertex(tri.v0);
                    let v1_idx = get_or_add_vertex(tri.v1);
                    let v2_idx = get_or_add_vertex(tri.v2);

                    self.flat_triangles.push(FlatTriangle {
                        v0: v0_idx as u32,
                        v1: v1_idx as u32,
                        v2: v2_idx as u32,
                        material: mesh.material,
                        morton_code: 0,
                    });
                }
            }

            for &child_idx in &node.children {
                if stack_ptr < 64 {
                    stack[stack_ptr] = (child_idx as usize, world_transform);
                    stack_ptr += 1;
                }
            }
        }
    }

    pub fn build(mut self) -> Bvh {
        let mut nodes = Vec::new();

        let tri_count = self.flat_triangles.len() as u32;

        // Node 0 is the root
        nodes.push(BvhNode {
            aabb: AABB::invalid(),
            left_first: 0,
            tri_count,
        });

        if tri_count > 0 {
            self.update_node_bounds(0, &mut nodes);
            self.subdivide(0, &mut nodes);
        }

        Bvh {
            nodes,
            vertices: self.world_vertices,
            normals: self.world_normals,
            triangles: self.flat_triangles,
        }
    }

    fn update_node_bounds(&self, node_idx: usize, nodes: &mut Vec<BvhNode>) {
        let node = &mut nodes[node_idx];
        let mut aabb = AABB::invalid();
        let first = node.left_first as usize;
        let count = node.tri_count as usize;

        for i in 0..count {
            let tri = &self.flat_triangles[first + i];
            let v0 = self.world_vertices[tri.v0 as usize];
            let v1 = self.world_vertices[tri.v1 as usize];
            let v2 = self.world_vertices[tri.v2 as usize];
            aabb.extend(v0);
            aabb.extend(v1);
            aabb.extend(v2);
        }
        // Expand AABB slightly to avoid precision issues with flat triangles
        let eps = nalgebra::Vector3::new(1e-5, 1e-5, 1e-5);
        aabb.min -= eps;
        aabb.max += eps;

        node.aabb = aabb;
    }

    fn subdivide(&mut self, node_idx: usize, nodes: &mut Vec<BvhNode>) {
        let node = &nodes[node_idx];
        if node.tri_count <= 2 {
            return;
        }

        let first = node.left_first as usize;
        let count = node.tri_count as usize;

        // Split in the middle (array is already sorted by morton code)
        let split_idx = first + count / 2;
        let left_count = split_idx - first;
        let right_count = count - left_count;

        let left_child_idx = nodes.len();
        nodes.push(BvhNode {
            aabb: AABB::invalid(),
            left_first: first as u32,
            tri_count: left_count as u32,
        });

        let right_child_idx = nodes.len();
        nodes.push(BvhNode {
            aabb: AABB::invalid(),
            left_first: split_idx as u32,
            tri_count: right_count as u32,
        });

        // Mutate current node to become internal
        nodes[node_idx].left_first = left_child_idx as u32;
        nodes[node_idx].tri_count = 0;

        self.update_node_bounds(left_child_idx, nodes);
        self.update_node_bounds(right_child_idx, nodes);

        self.subdivide(left_child_idx, nodes);
        self.subdivide(right_child_idx, nodes);
    }

    /// Expands a 21-bit integer into 64 bits, inserting two zeros between each bit.
    #[inline(always)]
    fn expand_bits(mut v: u64) -> u64 {
        v &= 0x00000000001fffff;
        v = (v | (v << 32)) & 0x001f00000000ffff;
        v = (v | (v << 16)) & 0x001f0000ff0000ff;
        v = (v | (v << 8)) & 0x100f00f00f00f00f;
        v = (v | (v << 4)) & 0x10c30c30c30c30c3;
        v = (v | (v << 2)) & 0x1249249249249249;
        v
    }

    /// Calculates the 3D Morton code for a normalized point (values must be in [0, 1]).
    #[inline]
    fn morton_3d(normalized_x: f32, normalized_y: f32, normalized_z: f32) -> u64 {
        // 2^21 - 1 = 2097151.0
        // Clamp to ensure no overflow from floating point inaccuracies
        let x = (normalized_x.max(0.0).min(1.0) * 2097151.0) as u64;
        let y = (normalized_y.max(0.0).min(1.0) * 2097151.0) as u64;
        let z = (normalized_z.max(0.0).min(1.0) * 2097151.0) as u64;

        Self::expand_bits(x) | (Self::expand_bits(y) << 1) | (Self::expand_bits(z) << 2)
    }
}

#[derive(Clone, Copy)]
pub struct BvhNode {
    pub aabb: AABB,
    pub left_first: u32,
    pub tri_count: u32,
}

impl BvhNode {
    pub fn is_leaf(&self) -> bool {
        self.tri_count > 0
    }
}

pub struct BvhHit {
    pub hit: crate::math::TriangleHit,
    pub material: u32,
    pub v0: u32,
    pub v1: u32,
    pub v2: u32,
}

pub struct Bvh {
    pub nodes: Vec<BvhNode>,
    pub vertices: Vec<Point3<f32>>,
    pub normals: Vec<UnitVector3<f32>>,
    pub triangles: Vec<FlatTriangle>,
}

impl Bvh {
    pub fn intersect(&self, ray: &crate::Ray) -> Option<BvhHit> {
        let mut closest_hit: Option<BvhHit> = None;
        let mut t_max = ray.tmax;

        if self.nodes.is_empty() {
            return None;
        }

        let mut stack = [(0usize, 0.0f32); 64];
        let mut stack_ptr = 0;

        let root_t = ray.intersect_aabb(&self.nodes[0].aabb, t_max);
        let Some(root_t) = root_t else {
            return None;
        };
        stack[stack_ptr] = (0, root_t);
        stack_ptr += 1;

        while stack_ptr > 0 {
            stack_ptr -= 1;
            let (node_idx, node_t) = stack[stack_ptr];

            if node_t >= t_max {
                continue;
            }

            let node = &self.nodes[node_idx];

            if node.is_leaf() {
                let first = node.left_first as usize;
                let count = node.tri_count as usize;

                for i in 0..count {
                    let tri = &self.triangles[first + i];
                    let v0 = self.vertices[tri.v0 as usize];
                    let v1 = self.vertices[tri.v1 as usize];
                    let v2 = self.vertices[tri.v2 as usize];

                    let mut test_ray = ray.clone();
                    test_ray.tmax = t_max;

                    if let Some(hit) = test_ray.intersect_triangle(&v0, &v1, &v2) {
                        t_max = hit.t;
                        closest_hit = Some(BvhHit {
                            hit,
                            material: tri.material,
                            v0: tri.v0,
                            v1: tri.v1,
                            v2: tri.v2,
                        });
                    }
                }
            } else {
                if stack_ptr + 1 < 64 {
                    let left_idx = node.left_first as usize;
                    let right_idx = left_idx + 1;
                    let t_left_opt = ray.intersect_aabb(&self.nodes[left_idx].aabb, t_max);
                    let t_right_opt = ray.intersect_aabb(&self.nodes[right_idx].aabb, t_max);
                    
                    match (t_left_opt, t_right_opt) {
                        (Some(t_left), Some(t_right)) => {
                            // Les deux boîtes sont touchées par le rayon.
                            // ORDRE FRONT-TO-BACK : On empile la plus lointaine d'abord.
                            // Ainsi, la plus proche se retrouve au sommet de la pile et sera dépilée en premier !
                            if t_left < t_right {
                                stack[stack_ptr] = (right_idx, t_right);
                                stack[stack_ptr + 1] = (left_idx, t_left);
                            } else {
                                stack[stack_ptr] = (left_idx, t_left);
                                stack[stack_ptr + 1] = (right_idx, t_right);
                            }
                            stack_ptr += 2;
                        }
                        (Some(t_left), None) => {
                            stack[stack_ptr] = (left_idx, t_left);
                            stack_ptr += 1;
                        }
                        (None, Some(t_right)) => {
                            stack[stack_ptr] = (right_idx, t_right);
                            stack_ptr += 1;
                        }
                        (None, None) => {}
                    }
                }
            }
        }

        closest_hit
    }
}
