use nalgebra::{Matrix4, Point3, Transform3, UnitVector3, Vector2, Vector3, Vector4};
use rnpt::{Camera, Color, Light, Material, Mesh, Node, Scene, Triangle};
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

/// IEC 61966-2-1 sRGB → linear (exact, matches Blender/OpenGL).
#[inline]
fn srgb_u8_to_linear(v: u8) -> f32 {
    let c = v as f32 / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Scans the directory for any .glb or .gltf files and returns their paths.
pub fn list_assets<P: AsRef<Path>>(dir: P) -> Vec<PathBuf> {
    list_by_ext(dir, &["glb", "gltf"])
}

/// Scans the directory for equirectangular HDRIs (.hdr).
pub fn list_hdris<P: AsRef<Path>>(dir: P) -> Vec<PathBuf> {
    list_by_ext(dir, &["hdr"])
}

fn list_by_ext<P: AsRef<Path>>(dir: P, exts: &[&str]) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                    if exts.iter().any(|&e| ext.eq_ignore_ascii_case(e)) {
                        files.push(path);
                    }
                }
            }
        }
    }
    files.sort();
    files
}

/// Load an equirectangular `.hdr` as linear RGB pixels (row-major) + dimensions.
pub fn load_hdr(path: &Path) -> Option<(Vec<Color>, usize, usize)> {
    let img = image::ImageReader::open(path).ok()?.decode().ok()?;
    let rgb = img.to_rgb32f();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    let pixels = rgb.pixels().map(|p| Color::new(p[0], p[1], p[2])).collect();
    Some((pixels, w, h))
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
    let (document, buffers, images) = gltf::import(path)?;

    // Scan materials first to determine which image indices are linear data
    // (normal maps, metallic/roughness, occlusion) vs sRGB color data (albedo,
    // emissive). This drives the per-image decode below.
    let mut linear_images: HashSet<usize> = HashSet::new();
    for mat in document.materials() {
        let pbr = mat.pbr_metallic_roughness();
        if let Some(info) = pbr.metallic_roughness_texture() {
            linear_images.insert(info.texture().source().index());
        }
        if let Some(info) = mat.normal_texture() {
            linear_images.insert(info.texture().source().index());
        }
        if let Some(info) = mat.occlusion_texture() {
            linear_images.insert(info.texture().source().index());
        }
    }

    // Load textures — sRGB → linear for color maps, raw /255 for linear maps.
    // sample_bilinear is pure interpolation with no per-sample color-space work.
    let mut textures = Vec::new();
    for (idx, img) in images.iter().enumerate() {
        let n = (img.width * img.height) as usize;
        let bytes = &img.pixels;
        let is_linear = linear_images.contains(&idx);
        let decode = |b: u8| -> f32 {
            if is_linear { b as f32 / 255.0 } else { srgb_u8_to_linear(b) }
        };

        let pixels: Vec<rnpt::Color> = match img.format {
            gltf::image::Format::R8G8B8 => (0..n)
                .map(|i| rnpt::Color::new(
                    decode(bytes[i * 3]),
                    decode(bytes[i * 3 + 1]),
                    decode(bytes[i * 3 + 2]),
                ))
                .collect(),
            gltf::image::Format::R8G8B8A8 => (0..n)
                .map(|i| rnpt::Color::new(
                    decode(bytes[i * 4]),
                    decode(bytes[i * 4 + 1]),
                    decode(bytes[i * 4 + 2]),
                ))
                .collect(),
            gltf::image::Format::R8 => (0..n)
                .map(|i| { let l = decode(bytes[i]); rnpt::Color::new(l, l, l) })
                .collect(),
            gltf::image::Format::R8G8 => (0..n)
                .map(|i| rnpt::Color::new(
                    decode(bytes[i * 2]),
                    decode(bytes[i * 2 + 1]),
                    0.0,
                ))
                .collect(),
            _ => {
                eprintln!("Unsupported texture format: {:?}", img.format);
                vec![rnpt::Color::new(1.0, 0.0, 1.0); n]
            }
        };
        textures.push(rnpt::Texture {
            width: img.width,
            height: img.height,
            pixels,
        });
    }

    // Load Materials
    let mut materials = Vec::new();
    for mat in document.materials() {
        let pbr = mat.pbr_metallic_roughness();
        let base_color = pbr.base_color_factor();
        let emissive_color = mat.emissive_factor();
        let emissive_strength = mat.emissive_strength().unwrap_or(1.0);
        let mut emissive = Color::from(emissive_color);
        emissive *= emissive_strength;

        // gltf crate returns Some(cutoff) for Mask mode, None for Opaque/Blend.
        let alpha_cutoff = mat.alpha_cutoff();

        let transmission = mat.transmission()
            .map(|t| t.transmission_factor())
            .unwrap_or(0.0);
        let ior = mat.ior().unwrap_or(1.5);

        let (thickness_factor, attenuation_distance, attenuation_color) =
            if let Some(vol) = mat.volume() {
                let ac = vol.attenuation_color();
                (
                    vol.thickness_factor(),
                    vol.attenuation_distance(),
                    Color::new(ac[0], ac[1], ac[2]),
                )
            } else {
                (0.0, f32::INFINITY, Color::new(1.0, 1.0, 1.0))
            };

        materials.push(Material {
            albedo: Color::from([base_color[0], base_color[1], base_color[2]]),
            emissive,
            albedo_texture: pbr.base_color_texture()
                .map(|info| info.texture().source().index() as u32),
            emissive_texture: mat.emissive_texture()
                .map(|info| info.texture().source().index() as u32),
            metallic: pbr.metallic_factor(),
            roughness: pbr.roughness_factor(),
            metallic_roughness_texture: pbr.metallic_roughness_texture()
                .map(|info| info.texture().source().index() as u32),
            normal_texture: mat.normal_texture()
                .map(|info| info.texture().source().index() as u32),
            normal_scale: mat.normal_texture().map(|info| info.scale()).unwrap_or(1.0),
            alpha_cutoff,
            double_sided: mat.double_sided(),
            transmission,
            ior,
            thickness_factor,
            attenuation_distance,
            attenuation_color,
        });
    }

    // Load Meshes
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
            let mut uvs = Vec::new();
            let mut mesh_colors = Vec::new();
            let mut triangles = Vec::new();

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

            // Read UVs
            if let Some(tex_coords) = reader.read_tex_coords(0).map(|v| v.into_f32()) {
                for uv in tex_coords {
                    uvs.push(Vector2::new(uv[0], uv[1]));
                }
            }

            // Colors
            if let Some(colors) = reader.read_colors(0) {
                let colors_f32 = colors.into_rgba_f32();
                for c in colors_f32 {
                    mesh_colors.push(rnpt::Color::new(c[0], c[1], c[2]));
                }
            }

            // Tangents (vec4: xyz = tangent, w = bitangent sign ±1, Mikktspace convention)
            let mut mesh_tangents = Vec::new();
            if let Some(tangents_iter) = reader.read_tangents() {
                for t in tangents_iter {
                    mesh_tangents.push(Vector4::new(t[0], t[1], t[2], t[3]));
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

            meshes.push(Mesh {
                vertices,
                normals,
                uvs,
                colors: mesh_colors,
                tangents: mesh_tangents,
                triangles,
                material: primitive.material().index().unwrap_or(0) as u32,
            });
        }
        gltf_mesh_to_rnpt_meshes.push(rnpt_mesh_indices);
    }

    // Load Nodes, Cameras and Lights
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

            // Convert glTF intensity (Candelas/Lux) back to Blender's intuitive Watts/Strength
            let color = Color::from(gltf_light.color());
            let punctual_intensity = gltf_light.intensity() / (683.0 / (4.0 * std::f32::consts::PI));
            let light = match gltf_light.kind() {
                gltf::khr_lights_punctual::Kind::Directional => Light::Directional {
                    direction: dir,
                    color,
                    intensity: gltf_light.intensity() / 683.0,
                },
                gltf::khr_lights_punctual::Kind::Point => Light::Point {
                    position: pos,
                    color,
                    intensity: punctual_intensity,
                },
                gltf::khr_lights_punctual::Kind::Spot { .. } => Light::Spot {
                    position: pos,
                    direction: dir,
                    color,
                    intensity: punctual_intensity,
                },
            };
            lights.push(light);
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

    // Extract roots
    let roots = scene.nodes().map(|n| n.index() as u32).collect();

    Ok(Scene {
        meshes,
        materials,
        textures,
        lights,
        nodes: nodes_list,
        roots,
        cameras,
    })
}
