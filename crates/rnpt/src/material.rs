use crate::{Brdf, Bvh, BvhHit, Color, Ray, Scene};
use nalgebra::{Point3, UnitVector3, Vector2, Vector3, Vector4};

/// Geometry + material evaluated at a ray hit. Produced by `evaluate_surface`;
/// consumed by the integrator for emission, direct lighting and BRDF sampling.
pub struct SurfaceInteraction {
    pub position: Point3<f32>,
    pub normal: UnitVector3<f32>,     // shading normal (vertex-interpolated + normal map)
    pub geo_normal: UnitVector3<f32>, // geometric normal (triangle winding)
    pub albedo: Color,
    pub emissive: Color,
    pub metallic: f32,
    pub roughness: f32,
}

impl SurfaceInteraction {
    /// Build the GGX Cook-Torrance BRDF for this surface.
    pub fn brdf(&self) -> Brdf {
        Brdf::CookTorrance {
            albedo: self.albedo,
            metallic: self.metallic,
            roughness: self.roughness,
        }
    }
}

/// Interpolate geometry and evaluate all material inputs (albedo, emissive,
/// metallic, roughness, normal map) at a ray hit.
pub fn evaluate_surface(hit: &BvhHit, ray: &Ray, bvh: &Bvh, scene: &Scene) -> SurfaceInteraction {
    let mat = &scene.materials[hit.material as usize];
    let (u, v) = (hit.hit.u, hit.hit.v);
    let w = 1.0 - u - v;

    // Shading normal (barycentric-interpolated vertex normals).
    let n0 = bvh.normals[hit.v0 as usize];
    let n1 = bvh.normals[hit.v1 as usize];
    let n2 = bvh.normals[hit.v2 as usize];
    let mut shading_normal = UnitVector3::new_normalize(n0.scale(w) + n1.scale(u) + n2.scale(v));

    // Geometric normal from the triangle winding (raw, to match the emitter
    // normal used by NEE — needed for the MIS cos_l on BRDF hits).
    let p0 = bvh.vertices[hit.v0 as usize];
    let p1 = bvh.vertices[hit.v1 as usize];
    let p2 = bvh.vertices[hit.v2 as usize];
    let geo_normal = UnitVector3::new_normalize((p1 - p0).cross(&(p2 - p0)));

    let position = ray.at(hit.hit.t);

    // Per-vertex color.
    let c0 = bvh.colors[hit.v0 as usize];
    let c1 = bvh.colors[hit.v1 as usize];
    let c2 = bvh.colors[hit.v2 as usize];
    let vertex_color = c0 * w + c1 * u + c2 * v;

    // UVs — read individual vertex UVs (needed for TBN) whenever any texture is bound.
    let has_textures = mat.albedo_texture.is_some()
        || mat.emissive_texture.is_some()
        || mat.metallic_roughness_texture.is_some()
        || mat.normal_texture.is_some();
    let (uv0, uv1, uv2) = if has_textures {
        (bvh.uvs[hit.v0 as usize], bvh.uvs[hit.v1 as usize], bvh.uvs[hit.v2 as usize])
    } else {
        let z = Vector2::zeros();
        (z, z, z)
    };
    let uv = uv0 * w + uv1 * u + uv2 * v;

    // Modulate a base color by its (optional) texture.
    let modulate = |tex: Option<u32>, base: Color| -> Color {
        match tex {
            Some(i) if (i as usize) < scene.textures.len() => {
                base.component_mul(&scene.textures[i as usize].sample_bilinear(uv))
            }
            _ => base,
        }
    };

    // Metallic/roughness: scalar × texture (glTF spec: G = roughness, B = metallic).
    let (metallic, roughness) = match mat.metallic_roughness_texture {
        Some(i) if (i as usize) < scene.textures.len() => {
            let mr = scene.textures[i as usize].sample_bilinear(uv);
            (mat.metallic * mr.z, mat.roughness * mr.y)
        }
        _ => (mat.metallic, mat.roughness),
    };

    // Normal map: decode tangent-space normal, build TBN, transform to world space.
    if let Some(i) = mat.normal_texture {
        if (i as usize) < scene.textures.len() {
            let nm = scene.textures[i as usize].sample_bilinear(uv);
            let ts = Vector3::new(
                (nm.x * 2.0 - 1.0) * mat.normal_scale,
                (nm.y * 2.0 - 1.0) * mat.normal_scale,
                nm.z * 2.0 - 1.0,
            );

            // Build TBN. Prefer pre-computed Mikktspace tangents (W ≠ 0) from the mesh,
            // which handle mirrored UVs correctly via the bitangent sign. Fall back to
            // runtime UV-delta derivation for meshes without a TANGENT attribute.
            let (tangent, bitangent) = {
                // Barycentric-interpolate the stored tangents.
                let get_tan = |idx: u32| -> Vector4<f32> {
                    bvh.tangents.get(idx as usize).copied().unwrap_or(Vector4::zeros())
                };
                let t0 = get_tan(hit.v0);
                let t1 = get_tan(hit.v1);
                let t2 = get_tan(hit.v2);

                // W is ±1 for valid Mikktspace tangents, 0 when missing.
                let has_mikkt = t0.w.abs() > 0.5;

                if has_mikkt {
                    // Interpolate and re-orthogonalize; W sign comes from any vertex (all ±1).
                    let t_raw = t0.xyz() * w + t1.xyz() * u + t2.xyz() * v;
                    let t_orth = t_raw - shading_normal.scale(shading_normal.dot(&t_raw));
                    let tangent = if t_orth.norm_squared() > 1e-7 {
                        t_orth.normalize()
                    } else {
                        t_raw.normalize()
                    };
                    // glTF spec: B = W * (N × T).
                    let bitangent = shading_normal.cross(&tangent) * t0.w;
                    (tangent, bitangent)
                } else {
                    // Runtime derivation from UV deltas.
                    let dp1 = p1 - p0;
                    let dp2 = p2 - p0;
                    let duv1 = uv1 - uv0;
                    let duv2 = uv2 - uv0;
                    let det = duv1.x * duv2.y - duv1.y * duv2.x;
                    let inv_det = if det.abs() > 1e-7 { 1.0 / det } else { 0.0 };
                    let raw_t = (dp1 * duv2.y - dp2 * duv1.y) * inv_det;
                    let raw_b = (dp2 * duv1.x - dp1 * duv2.x) * inv_det;
                    let t_orth = raw_t - shading_normal.scale(shading_normal.dot(&raw_t));
                    let tangent = if t_orth.norm_squared() > 1e-7 {
                        t_orth.normalize()
                    } else {
                        let fallback = if shading_normal.x.abs() < 0.9 {
                            Vector3::new(1.0_f32, 0.0, 0.0)
                        } else {
                            Vector3::new(0.0_f32, 1.0, 0.0)
                        };
                        fallback.cross(&shading_normal).normalize()
                    };
                    // B from UV deltas gives the direction of increasing V in the stored
                    // (glTF, V-top) UV.  Blender bakes normal maps in V-up convention,
                    // so B_baked = N×T (Mikktspace W=+1).  The glTF V-flip inverts B,
                    // meaning UV-delta B = -(N×T) = -B_baked.  Negate to match the bake.
                    let b_orth = raw_b - shading_normal.scale(shading_normal.dot(&raw_b));
                    let bitangent = if b_orth.norm_squared() > 1e-7 {
                        -b_orth.normalize()
                    } else {
                        // Degenerate UV: N×T is the correct Blender-baked B direction.
                        shading_normal.cross(&tangent)
                    };
                    (tangent, bitangent)
                }
            };

            let world = tangent * ts.x + bitangent * ts.y + shading_normal.scale(ts.z);
            shading_normal = UnitVector3::new_normalize(world);

            // Guard: normal-map perturbation can push the shading normal below the
            // incoming ray hemisphere, causing cos_n_o ≤ 0 in the BRDF → black pixels.
            // Compare against wo (not geo_normal) — that's the actual relevant horizon.
            // geo_normal is guaranteed to face wo (by back-face culling), so it's a safe fallback.
            let wo = -ray.direction.into_inner();
            if shading_normal.as_ref().dot(&wo) <= 0.0 {
                shading_normal = geo_normal;
            }
        }
    }

    SurfaceInteraction {
        position,
        normal: shading_normal,
        geo_normal,
        albedo: modulate(mat.albedo_texture, mat.albedo.component_mul(&vertex_color)),
        emissive: modulate(mat.emissive_texture, mat.emissive),
        metallic,
        roughness,
    }
}
