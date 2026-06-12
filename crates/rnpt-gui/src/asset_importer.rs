use nalgebra::{Matrix4, Point3, Transform3, UnitVector3, Vector3};
use rnpt::{Camera, Color, Light, LightType, Material, Mesh, Node, Scene, Triangle};
use std::fs;
use std::path::{Path, PathBuf};

/// Scans the directory for any .glb or .gltf files and returns their paths.
pub fn list_assets<P: AsRef<Path>>(dir: P) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                if let Some(ext) = path.extension() {
                    if ext == "glb" || ext == "gltf" {
                        files.push(path);
                    }
                }
            }
        }
    }
    files.sort();
    files
}

fn gltf_matrix_to_nalgebra(matrix: [[f32; 4]; 4]) -> Matrix4<f32> {
    Matrix4::new(
        matrix[0][0],
        matrix[1][0],
        matrix[2][0],
        matrix[3][0],
        matrix[0][1],
        matrix[1][1],
        matrix[2][1],
        matrix[3][1],
        matrix[0][2],
        matrix[1][2],
        matrix[2][2],
        matrix[3][2],
        matrix[0][3],
        matrix[1][3],
        matrix[2][3],
        matrix[3][3],
    )
}

/// Imports a GLB or glTF file and parses it into an rnpt::Scene.
pub fn import_scene<P: AsRef<Path>>(path: P) -> Result<Scene, Box<dyn std::error::Error>> {
    let (document, buffers, _images) = gltf::import(path)?;

    // 1. Load Materials
    let mut materials = Vec::new();
    for (_idx, mat) in document.materials().enumerate() {
        let pbr = mat.pbr_metallic_roughness();
        let base_color = pbr.base_color_factor(); // [f32; 4]
        let emissive_color = mat.emissive_factor(); // [f32; 3]
        // Extract strength from KHR_materials_emissive_strength if available
        let emissive_strength = mat.emissive_strength().unwrap_or(1.0);

        let mut emissive = Color::from(emissive_color);
        emissive *= emissive_strength;

        materials.push(Material {
            albedo: Color::from([base_color[0], base_color[1], base_color[2]]),
            emissive,
        });
    }

    // 2. Load Meshes
    // In glTF, a single Mesh can contain multiple Primitives. Since each Primitive
    // can have a different material, we split them into individual rnpt::Mesh instances.
    let mut meshes = Vec::new();
    let mut gltf_mesh_to_rnpt_meshes = Vec::new();
    for mesh in document.meshes() {
        let mut rnpt_mesh_indices = Vec::new();
        for primitive in mesh.primitives() {
            rnpt_mesh_indices.push(meshes.len() as u32);
            let mut vertices = Vec::new();
            let mut normals = Vec::new();
            let mut triangles = Vec::new();
            let mut material_idx = 0;

            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

            // Read positions
            if let Some(pos_iter) = reader.read_positions() {
                for pos in pos_iter {
                    vertices.push(Point3::new(pos[0], pos[1], pos[2]));
                }
            }

            // Read normals
            if let Some(norm_iter) = reader.read_normals() {
                for norm in norm_iter {
                    normals.push(UnitVector3::new_normalize(Vector3::new(
                        norm[0], norm[1], norm[2],
                    )));
                }
            }

            // Read indices
            if let Some(indices_iter) = reader.read_indices() {
                let indices: Vec<u32> = indices_iter.into_u32().collect();
                for chunk in indices.chunks_exact(3) {
                    triangles.push(Triangle {
                        v0: chunk[0],
                        v1: chunk[1],
                        v2: chunk[2],
                    });
                }
            }

            // Material index
            if let Some(mat) = primitive.material().index() {
                material_idx = mat as u32;
            }

            meshes.push(Mesh {
                vertices,
                normals,
                triangles,
                material: material_idx,
            });
        }
        gltf_mesh_to_rnpt_meshes.push(rnpt_mesh_indices);
    }

    // 3. Load Nodes, Cameras and Lights
    let scene = document
        .default_scene()
        .or_else(|| document.scenes().next())
        .ok_or("No scene found in glTF file")?;

    let mut node_world_transforms = vec![Matrix4::identity(); document.nodes().count()];
    let mut cameras = Vec::new();
    let mut lights = Vec::new();
    let mut nodes_list = Vec::new();

    // Traverses node hierarchy to compute world transforms and extract cameras/lights
    fn traverse_node(
        node: &gltf::Node,
        parent_transform: &Matrix4<f32>,
        node_world_transforms: &mut [Matrix4<f32>],
        cameras: &mut Vec<Camera>,
        lights: &mut Vec<Light>,
        buffers: &[gltf::buffer::Data],
    ) {
        let local_matrix = node.transform().matrix(); // [[f32; 4]; 4]
        let local_transform = gltf_matrix_to_nalgebra(local_matrix);
        let world_transform = parent_transform * local_transform;
        node_world_transforms[node.index()] = world_transform;

        // Camera extraction
        if let Some(gltf_camera) = node.camera() {
            let pos = Point3::new(
                world_transform[(0, 3)],
                world_transform[(1, 3)],
                world_transform[(2, 3)],
            );
            let dir_vec = world_transform.transform_vector(&Vector3::new(0.0, 0.0, -1.0));
            let target = pos + dir_vec;

            let fov = match gltf_camera.projection() {
                gltf::camera::Projection::Perspective(p) => p.yfov().to_degrees(),
                gltf::camera::Projection::Orthographic(_) => 60.0,
            };

            cameras.push(Camera {
                position: pos,
                target,
                fov,
            });
        }

        // Light extraction (KHR_lights_punctual)
        if let Some(gltf_light) = node.light() {
            let pos = Point3::new(
                world_transform[(0, 3)],
                world_transform[(1, 3)],
                world_transform[(2, 3)],
            );
            let dir = world_transform
                .transform_vector(&Vector3::new(0.0, 0.0, -1.0))
                .normalize();

            let light_type = match gltf_light.kind() {
                gltf::khr_lights_punctual::Kind::Directional => LightType::Directional,
                gltf::khr_lights_punctual::Kind::Point => LightType::Point,
                gltf::khr_lights_punctual::Kind::Spot { .. } => LightType::Spot,
            };

            // Convert glTF intensity (Candelas/Lux) back to Blender's intuitive Watts/Strength
            let intensity = match light_type {
                LightType::Directional => gltf_light.intensity() / 683.0,
                _ => gltf_light.intensity() / (683.0 / (4.0 * std::f32::consts::PI)),
            };

            lights.push(Light {
                position: pos,
                direction: dir,
                color: Color::from(gltf_light.color()),
                intensity,
                light_type,
            });
        }

        // Children traversal
        for child in node.children() {
            traverse_node(
                &child,
                &world_transform,
                node_world_transforms,
                cameras,
                lights,
                buffers,
            );
        }
    }

    for root in scene.nodes() {
        traverse_node(
            &root,
            &Matrix4::identity(),
            &mut node_world_transforms,
            &mut cameras,
            &mut lights,
            &buffers,
        );
    }

    // Construct the node list
    for node in document.nodes() {
        let matrix = node.transform().matrix();
        let m = gltf_matrix_to_nalgebra(matrix);
        let transform = Transform3::from_matrix_unchecked(m);
        let children = node.children().map(|c| c.index() as u32).collect();

        let mut node_meshes = Vec::new();
        if let Some(m) = node.mesh() {
            node_meshes = gltf_mesh_to_rnpt_meshes[m.index()].clone();
        }

        nodes_list.push(Node {
            transform,
            children,
            meshes: node_meshes,
        });
    }

    // Automatic default camera if none found
    if cameras.is_empty() {
        cameras.push(Camera {
            position: Point3::new(0.0, 0.0, 3.0),
            target: Point3::new(0.0, 0.0, 0.0),
            fov: 60.0,
        });
    }

    Ok(Scene {
        meshes,
        materials,
        lights,
        nodes: nodes_list,
        cameras,
    })
}
