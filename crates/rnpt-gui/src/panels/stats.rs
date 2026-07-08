use eframe::egui;
use crate::app::RnptGuiApp;
use crate::tonemap::{format_paths_per_sec, format_rays_per_sec};

pub fn show(app: &mut RnptGuiApp, ui: &mut egui::Ui) {
    ui.heading("Rendering Stats");
    ui.add_space(4.0);
    ui.label(format!("Resolution: {}x{}", app.local_width, app.local_height));

    let avg_samples = if app.local_pixels.is_empty() {
        0.0
    } else {
        let total: u64 = app.local_pixels.iter().map(|p| p.samples as u64).sum();
        total as f64 / app.local_pixels.len() as f64
    };
    ui.label(format!("Samples/Pixel (avg): {:.1}", avg_samples));
    ui.label(format!("Performance: {}", format_paths_per_sec(app.local_rays_per_sec)));

    let total_rays = app.local_real_rays_per_sec + app.local_shadow_rays_per_sec;
    ui.label(format!("Rays: {}", format_rays_per_sec(total_rays)));
    let shadow_pct = if total_rays > 0.0 {
        100.0 * app.local_shadow_rays_per_sec / total_rays
    } else {
        0.0
    };
    ui.label(format!("  ├ closest: {}", format_rays_per_sec(app.local_real_rays_per_sec)));
    ui.label(format!(
        "  └ shadow:  {} ({:.0}%)",
        format_rays_per_sec(app.local_shadow_rays_per_sec),
        shadow_pct
    ));
    let rays_per_path = if app.local_rays_per_sec > 0.0 {
        total_rays / app.local_rays_per_sec
    } else {
        0.0
    };
    ui.label(format!("Rays/Path (avg): {:.1}", rays_per_path));

    if app.strategy == rnpt::SamplingStrategy::Nirc {
        ui.add_space(4.0);
        super::nirc::show(app, ui);
    }

    ui.add_space(6.0);
    ui.collapsing("Scene Stats", |ui| {
        if let Some(ref scene) = app.current_scene {
            ui.label(format!("Meshes: {}", scene.meshes.len()));
            ui.label(format!("Materials: {}", scene.materials.len()));
            ui.label(format!("Triangles: {}", app.bvh_triangle_count));
            ui.label(format!("Lights: {}", scene.lights.len()));
            ui.label(format!("Cameras: {}", scene.cameras.len()));
        } else {
            ui.label("No scene loaded");
        }
    });

    ui.collapsing("Load Times", |ui| {
        if let Some(ref s) = app.scene_load_stats {
            ui.label(format!("Total import:    {:>6} ms", s.total_ms));
            ui.label(format!("  gltf parse:   {:>6} ms", s.gltf_parse_ms));
            ui.label(format!(
                "  tex decode:   {:>6} ms  ({} tex, {}M px)",
                s.texture_decode_ms,
                s.texture_count,
                s.total_texture_pixels / 1_000_000,
            ));
            ui.label(format!("  mesh/mat:     {:>6} ms", s.mesh_process_ms));
            ui.label(format!("BVH build:       {:>6} ms", app.bvh_build_ms));
        } else {
            ui.label("No scene loaded");
        }
    });

    ui.collapsing("Debug", |ui| {
        ui.checkbox(&mut app.show_debug, "Show materials");
        if app.show_debug {
            if let Some(ref scene) = app.current_scene {
                egui::ScrollArea::vertical()
                    .max_height(300.0)
                    .show(ui, |ui| {
                        for (i, mat) in scene.materials.iter().enumerate() {
                            ui.collapsing(format!("Mat {i}"), |ui| {
                                ui.label(format!(
                                    "albedo: [{:.2}, {:.2}, {:.2}]",
                                    mat.albedo.x, mat.albedo.y, mat.albedo.z
                                ));
                                ui.label(format!(
                                    "roughness: {:.3}  metallic: {:.3}",
                                    mat.roughness, mat.metallic
                                ));
                                ui.label(format!(
                                    "transmission: {:.3}  ior: {:.3}",
                                    mat.transmission, mat.ior
                                ));
                                ui.label(format!(
                                    "thickness: {:.3}  att_dist: {:.3}",
                                    mat.thickness_factor, mat.attenuation_distance
                                ));
                                ui.label(format!(
                                    "att_color: [{:.2}, {:.2}, {:.2}]",
                                    mat.attenuation_color.x,
                                    mat.attenuation_color.y,
                                    mat.attenuation_color.z
                                ));
                                ui.label(format!(
                                    "emissive: [{:.2}, {:.2}, {:.2}]",
                                    mat.emissive.x, mat.emissive.y, mat.emissive.z
                                ));
                                ui.label(format!(
                                    "double_sided: {}  alpha_cut: {:?}",
                                    mat.double_sided, mat.alpha_cutoff
                                ));
                            });
                        }
                    });
            }
        }
    });
}
