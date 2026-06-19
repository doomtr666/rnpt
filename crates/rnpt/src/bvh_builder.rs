use std::collections::HashMap;

use crate::emitters::{EmitterTri, MeshEmitter};
use crate::{Bvh, Color, Scene, TriangleMeta};
use nalgebra::{Point3, Transform3, UnitVector3, Vector2, Vector3, Vector4};

pub struct BvhBuilder {
    world_vertices:  Vec<Point3<f32>>,
    world_normals:   Vec<UnitVector3<f32>>,
    world_uvs:       Vec<Vector2<f32>>,
    world_colors:    Vec<Color>,
    world_tangents:  Vec<Vector4<f32>>,
    /// Flat triangles: (v0, v1, v2, material, light) — one per input triangle.
    flat_meta:       Vec<TriangleMeta>,
    emitter_meshes:  Vec<MeshEmitter>,
}

impl BvhBuilder {
    pub fn new(scene: &Scene) -> Self {
        let mut builder = Self {
            world_vertices:  Vec::new(),
            world_normals:   Vec::new(),
            world_uvs:       Vec::new(),
            world_colors:    Vec::new(),
            world_tangents:  Vec::new(),
            flat_meta:       Vec::new(),
            emitter_meshes:  Vec::new(),
        };
        builder.flatten_scene(scene);
        builder
    }

    fn flatten_scene(&mut self, scene: &Scene) {
        let mut vertex_map: HashMap<(usize, u32, u32), usize> = HashMap::new();

        let mut stack = Vec::new();
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
                let light_idx = if is_emissive {
                    scene.lights.len() as u32 + self.emitter_meshes.len() as u32
                } else {
                    u32::MAX
                };
                let mut emitter_tris: Vec<EmitterTri> = Vec::new();

                for &tri in &mesh.triangles {
                    // Rotation-scale 3×3 for transforming tangent direction vectors.
                    let m3 = world_transform.matrix().fixed_view::<3, 3>(0, 0);
                    let m3_det = m3.determinant();

                    let mut get_or_add_vertex = |local_idx: u32| -> usize {
                        let key = (node_idx, mesh_idx, local_idx);
                        *vertex_map.entry(key).or_insert_with(|| {
                            let p = world_transform
                                .transform_point(&mesh.vertices[local_idx as usize]);
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

                            // Tangent: transform xyz as a direction, keep W but flip when
                            // the world transform has a negative determinant (mirror).
                            let tangent = if !mesh.tangents.is_empty() {
                                let t = mesh.tangents[local_idx as usize];
                                let t3 = (m3 * Vector3::new(t.x, t.y, t.z)).normalize();
                                let w = t.w * if m3_det < 0.0 { -1.0 } else { 1.0 };
                                Vector4::new(t3.x, t3.y, t3.z, w)
                            } else {
                                Vector4::zeros()
                            };
                            self.world_tangents.push(tangent);

                            let idx = self.world_vertices.len();
                            self.world_vertices.push(p);
                            idx
                        })
                    };

                    let v0 = get_or_add_vertex(tri.v0) as u32;
                    let v1 = get_or_add_vertex(tri.v1) as u32;
                    let v2 = get_or_add_vertex(tri.v2) as u32;

                    self.flat_meta.push(TriangleMeta {
                        material: mesh.material,
                        light:    light_idx,
                        v0,
                        v1,
                        v2,
                    });

                    if is_emissive {
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
                                v0: p0, v1: p1, v2: p2,
                                uv0: uv(tri.v0), uv1: uv(tri.v1), uv2: uv(tri.v2),
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

    pub fn build(self) -> (Bvh, Vec<MeshEmitter>) {
        // Convert vertex positions to the [f32;3] format required by rnpt-bvh.
        let verts: Vec<[f32; 3]> = self.world_vertices.iter()
            .map(|p| [p.x, p.y, p.z])
            .collect();

        // Build the triangle index list.
        let tris: Vec<[u32; 3]> = self.flat_meta.iter()
            .map(|m| [m.v0, m.v1, m.v2])
            .collect();

        let mut accel = rnpt_bvh::Scene::new();
        accel.attach_geometry(rnpt_bvh::Geometry::triangle_mesh(&verts, &tris));
        accel.commit();

        let bvh = Bvh {
            accel,
            vertices:      self.world_vertices,
            normals:       self.world_normals,
            uvs:           self.world_uvs,
            colors:        self.world_colors,
            tangents:      self.world_tangents,
            triangle_meta: self.flat_meta,
        };

        (bvh, self.emitter_meshes)
    }
}
