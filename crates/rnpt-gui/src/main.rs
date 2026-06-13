use eframe::egui;
use std::time::{Duration, Instant};

mod asset_importer;

#[derive(Clone, Copy, PartialEq)]
pub enum TonemapOperator {
    Reinhard,
    Aces,
}

struct RnptGuiApp {
    camera: rnpt::Camera,
    resolution: [usize; 2],
    exposure: f32,
    tonemapper: TonemapOperator,

    tracer: Option<rnpt::ParallelTracer>,
    last_fps_time: Instant,
    rays_since_last_fps: u64,

    local_pixels: Vec<rnpt::Pixel>,
    local_width: usize,
    local_height: usize,
    local_rays_per_sec: f64,

    texture_handle: Option<egui::TextureHandle>,
    last_exposure: f32,
    last_tonemapper: TonemapOperator,

    // New fields
    asset_files: Vec<std::path::PathBuf>,
    selected_asset_index: usize,
    current_scene: Option<rnpt::Scene>,

    auto_fit: bool,
    resize_timeout: Option<Instant>,
}

impl RnptGuiApp {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let width = 800;
        let height = 600;
        let mut camera = rnpt::Camera::default();

        // Scan assets folder
        let asset_files = asset_importer::list_assets("assets");
        let selected_asset_index = asset_files
            .iter()
            .position(|p| p.file_name().map_or(false, |name| name == "cornell.glb"))
            .unwrap_or(0);

        let mut current_scene = None;
        if !asset_files.is_empty() {
            if let Ok(scene) = asset_importer::import_scene(&asset_files[selected_asset_index]) {
                if !scene.cameras.is_empty() {
                    let first_cam = &scene.cameras[0];
                    camera.position = first_cam.position;
                    camera.target = first_cam.target;
                    camera.fov = first_cam.fov;
                }
                current_scene = Some(scene);
            }
        }

        let scene = current_scene.clone().unwrap_or_else(|| rnpt::Scene {
            meshes: vec![],
            materials: vec![],
            lights: vec![],
            nodes: vec![],
            roots: vec![],
            cameras: vec![],
        });

        let config = rnpt::PathTracerConfig {
            width,
            height,
            camera: camera.clone(),
            scene,
        };

        let tracer = Some(rnpt::ParallelTracer::new(config));

        Self {
            camera,
            resolution: [width, height],
            exposure: 1.0,
            tonemapper: TonemapOperator::Aces,
            tracer,
            last_fps_time: Instant::now(),
            rays_since_last_fps: 0,
            local_pixels: vec![rnpt::Pixel::default(); width * height],
            local_width: width,
            local_height: height,
            local_rays_per_sec: 0.0,
            texture_handle: None,
            last_exposure: 1.0,
            last_tonemapper: TonemapOperator::Aces,
            asset_files,
            selected_asset_index,
            current_scene,
            auto_fit: true,
            resize_timeout: None,
        }
    }

    fn trigger_reset(&mut self) {
        let scene = self.current_scene.clone().unwrap_or_else(|| rnpt::Scene {
            meshes: vec![],
            materials: vec![],
            lights: vec![],
            nodes: vec![],
            roots: vec![],
            cameras: vec![],
        });

        let config = rnpt::PathTracerConfig {
            width: self.resolution[0],
            height: self.resolution[1],
            camera: self.camera.clone(),
            scene,
        };

        if self.local_width != self.resolution[0] || self.local_height != self.resolution[1] {
            // Recreate tracer and buffer if resolution changed
            self.tracer = None; // Drop old threads
            self.tracer = Some(rnpt::ParallelTracer::new(config));
            self.local_width = self.resolution[0];
            self.local_height = self.resolution[1];
            self.local_pixels = vec![rnpt::Pixel::default(); self.local_width * self.local_height];
        } else if let Some(tracer) = &mut self.tracer {
            // Fast reset via epoch
            tracer.update_scene(config);
        }

        self.rays_since_last_fps = 0;
        self.local_rays_per_sec = 0.0;
    }
}

impl eframe::App for RnptGuiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if let Some(timeout) = self.resize_timeout {
            if Instant::now() >= timeout {
                self.resize_timeout = None;
                self.trigger_reset();
            }
        }

        // Fetch new pixels from tracer
        let mut pixels_updated = false;
        if let Some(tracer) = &self.tracer {
            tracer.fetch_pixels(&mut self.local_pixels);

            let rays = tracer.pop_rays_traced();
            self.rays_since_last_fps += rays;
            pixels_updated = true;

            let now = Instant::now();
            let elapsed_fps = now.duration_since(self.last_fps_time).as_secs_f64();
            if elapsed_fps >= 0.5 {
                self.local_rays_per_sec = self.rays_since_last_fps as f64 / elapsed_fps;
                self.rays_since_last_fps = 0;
                self.last_fps_time = now;
            }
        }

        ctx.request_repaint();

        // If pixels updated, or exposure changed, regenerate the texture
        let exposure_changed = self.exposure != self.last_exposure;
        let tonemapper_changed = self.tonemapper != self.last_tonemapper;
        if pixels_updated
            || exposure_changed
            || tonemapper_changed
            || self.texture_handle.is_none()
        {
            let mut raw_rgba = vec![0u8; self.local_width * self.local_height * 4];

            tonemap_and_convert(
                &self.local_pixels,
                self.exposure,
                self.tonemapper,
                &mut raw_rgba,
            );

            let color_image = egui::ColorImage::from_rgba_unmultiplied(
                [self.local_width, self.local_height],
                &raw_rgba,
            );

            if let Some(ref mut texture) = self.texture_handle {
                texture.set(color_image, egui::TextureOptions::LINEAR);
            } else {
                self.texture_handle = Some(ctx.load_texture(
                    "render_texture",
                    color_image,
                    egui::TextureOptions::LINEAR,
                ));
            }

            self.last_exposure = self.exposure;
            self.last_tonemapper = self.tonemapper;
        }

        // UI Layout
        egui::SidePanel::left("controls_panel")
            .resizable(true)
            .default_width(260.0)
            .show(ctx, |ui| {
                ui.add_space(8.0);
                ui.heading("RNPT Controls");
                ui.add_space(10.0);

                if !self.asset_files.is_empty() {
                    ui.group(|ui| {
                        ui.label("Active Scene:");
                        let selected_name = self.asset_files[self.selected_asset_index]
                            .file_name()
                            .unwrap_or_default()
                            .to_string_lossy()
                            .into_owned();

                        let mut scene_changed = false;
                        egui::ComboBox::from_id_source("scene_selector")
                            .selected_text(&selected_name)
                            .show_ui(ui, |ui| {
                                for (idx, path) in self.asset_files.iter().enumerate() {
                                    let name =
                                        path.file_name().unwrap_or_default().to_string_lossy();
                                    scene_changed |= ui
                                        .selectable_value(&mut self.selected_asset_index, idx, name)
                                        .changed();
                                }
                            });

                        if scene_changed {
                            if let Ok(scene) = asset_importer::import_scene(
                                &self.asset_files[self.selected_asset_index],
                            ) {
                                if !scene.cameras.is_empty() {
                                    let first_cam = &scene.cameras[0];
                                    self.camera.position = first_cam.position;
                                    self.camera.target = first_cam.target;
                                    self.camera.fov = first_cam.fov;
                                }
                                self.current_scene = Some(scene);
                                self.trigger_reset();
                            }
                        }
                    });
                    ui.add_space(10.0);
                }

                let mut changed = false;

                // Camera section
                ui.collapsing("Camera Parameters", |ui| {
                    ui.label("Position:");
                    ui.horizontal(|ui| {
                        changed |= ui
                            .add(
                                egui::DragValue::new(&mut self.camera.position.x)
                                    .speed(0.1)
                                    .prefix("X: "),
                            )
                            .changed();
                        changed |= ui
                            .add(
                                egui::DragValue::new(&mut self.camera.position.y)
                                    .speed(0.1)
                                    .prefix("Y: "),
                            )
                            .changed();
                        changed |= ui
                            .add(
                                egui::DragValue::new(&mut self.camera.position.z)
                                    .speed(0.1)
                                    .prefix("Z: "),
                            )
                            .changed();
                    });

                    ui.add_space(4.0);
                    ui.label("Target:");
                    ui.horizontal(|ui| {
                        changed |= ui
                            .add(
                                egui::DragValue::new(&mut self.camera.target.x)
                                    .speed(0.1)
                                    .prefix("X: "),
                            )
                            .changed();
                        changed |= ui
                            .add(
                                egui::DragValue::new(&mut self.camera.target.y)
                                    .speed(0.1)
                                    .prefix("Y: "),
                            )
                            .changed();
                        changed |= ui
                            .add(
                                egui::DragValue::new(&mut self.camera.target.z)
                                    .speed(0.1)
                                    .prefix("Z: "),
                            )
                            .changed();
                    });

                    ui.add_space(4.0);
                    changed |= ui
                        .add(egui::Slider::new(&mut self.camera.fov, 10.0..=120.0).text("FOV"))
                        .changed();
                });

                ui.add_space(10.0);

                let mut resolution_changed = false;
                // Resolution section
                ui.collapsing("Resolution", |ui| {
                    let prev_auto_fit = self.auto_fit;
                    ui.checkbox(&mut self.auto_fit, "Auto-fit to viewport");
                    if prev_auto_fit != self.auto_fit {
                        resolution_changed = true;
                    }

                    ui.add_space(4.0);

                    ui.add_enabled_ui(!self.auto_fit, |ui| {
                        ui.horizontal(|ui| {
                            ui.label("W:");
                            let r1 = ui.add(
                                egui::DragValue::new(&mut self.resolution[0]).range(64..=2048),
                            );
                            ui.label("H:");
                            let r2 = ui.add(
                                egui::DragValue::new(&mut self.resolution[1]).range(64..=2048),
                            );

                            let r1_changed_finished = r1.drag_stopped()
                                || r1.lost_focus()
                                || (r1.changed() && !r1.dragged());
                            let r2_changed_finished = r2.drag_stopped()
                                || r2.lost_focus()
                                || (r2.changed() && !r2.dragged());
                            if r1_changed_finished || r2_changed_finished {
                                resolution_changed = true;
                            }
                        });
                    });
                });

                if changed || resolution_changed {
                    self.trigger_reset();
                }

                ui.add_space(10.0);
                ui.separator();
                ui.add_space(10.0);

                // Post Processing section
                ui.heading("Post-Processing");
                ui.add_space(4.0);
                ui.add(egui::Slider::new(&mut self.exposure, 0.1..=10.0).text("Exposure"));

                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.label("Tonemapper:");
                    egui::ComboBox::from_id_source("tonemapper_selector")
                        .selected_text(match self.tonemapper {
                            TonemapOperator::Reinhard => "Reinhard",
                            TonemapOperator::Aces => "ACES",
                        })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut self.tonemapper,
                                TonemapOperator::Aces,
                                "ACES",
                            );
                            ui.selectable_value(
                                &mut self.tonemapper,
                                TonemapOperator::Reinhard,
                                "Reinhard",
                            );
                        });
                });

                ui.add_space(10.0);
                ui.separator();
                ui.add_space(10.0);

                // Render Stats
                ui.heading("Rendering Stats");
                ui.add_space(4.0);
                ui.label(format!(
                    "Resolution: {}x{}",
                    self.local_width, self.local_height
                ));

                // Calculate average samples
                let avg_samples = if self.local_pixels.is_empty() {
                    0.0
                } else {
                    let total_samples: u64 =
                        self.local_pixels.iter().map(|p| p.samples as u64).sum();
                    total_samples as f64 / self.local_pixels.len() as f64
                };
                ui.label(format!("Samples/Pixel (avg): {:.1}", avg_samples));
                ui.label(format!(
                    "Performance: {}",
                    format_rays_per_sec(self.local_rays_per_sec)
                ));

                ui.add_space(6.0);
                ui.collapsing("Scene Stats", |ui| {
                    if let Some(ref scene) = self.current_scene {
                        ui.label(format!("Meshes: {}", scene.meshes.len()));
                        ui.label(format!("Materials: {}", scene.materials.len()));
                        ui.label(format!("Lights: {}", scene.lights.len()));
                        ui.label(format!("Cameras: {}", scene.cameras.len()));
                    } else {
                        ui.label("No scene loaded");
                    }
                });
            });

        let mut viewport_size = None;
        egui::CentralPanel::default().show(ctx, |ui| {
            let available = ui.available_size();
            viewport_size = Some([available.x, available.y]);

            if let Some(ref texture) = self.texture_handle {
                let size = texture.size_vec2();
                let aspect_ratio = size.x / size.y;

                let desired_size = if available.x / available.y > aspect_ratio {
                    egui::vec2(available.y * aspect_ratio, available.y)
                } else {
                    egui::vec2(available.x, available.x / aspect_ratio)
                };

                // Centering the image in the panel
                let rect = egui::Rect::from_min_size(
                    ui.min_rect().min + (available - desired_size) * 0.5,
                    desired_size,
                );

                ui.put(rect, egui::Image::new(texture));
            } else {
                ui.centered_and_justified(|ui| {
                    ui.spinner();
                });
            }
        });

        if self.auto_fit {
            if let Some(size) = viewport_size {
                let rounded_w = size[0].max(64.0) as usize;
                let rounded_h = size[1].max(64.0) as usize;

                if rounded_w != self.resolution[0] || rounded_h != self.resolution[1] {
                    self.resolution = [rounded_w, rounded_h];
                    self.resize_timeout = Some(Instant::now() + Duration::from_millis(150));
                }
            }
        }
    }
}

fn tonemap_and_convert(
    pixels: &[rnpt::Pixel],
    exposure: f32,
    operator: TonemapOperator,
    output_rgba: &mut [u8],
) {
    output_rgba
        .chunks_exact_mut(4)
        .zip(pixels.iter())
        .for_each(|(rgba, pixel)| {
            if pixel.samples == 0 {
                rgba[0] = 0;
                rgba[1] = 0;
                rgba[2] = 0;
                rgba[3] = 255;
                return;
            }

            // Average radiance
            let scale = 1.0 / pixel.samples as f32;
            let r_linear = pixel.accumulated_radiance[0] * scale;
            let g_linear = pixel.accumulated_radiance[1] * scale;
            let b_linear = pixel.accumulated_radiance[2] * scale;

            // Exposure
            let r_exp = r_linear * exposure;
            let g_exp = g_linear * exposure;
            let b_exp = b_linear * exposure;

            // Tonemapping
            let (r_tone, g_tone, b_tone) = match operator {
                TonemapOperator::Reinhard => (
                    r_exp / (r_exp + 1.0),
                    g_exp / (g_exp + 1.0),
                    b_exp / (b_exp + 1.0),
                ),
                TonemapOperator::Aces => {
                    // Narkowicz ACES fit
                    let a = 2.51f32;
                    let b = 0.03f32;
                    let c = 2.43f32;
                    let d = 0.59f32;
                    let e = 0.14f32;
                    let aces = |v: f32| (v * (a * v + b)) / (v * (c * v + d) + e);
                    (aces(r_exp), aces(g_exp), aces(b_exp))
                }
            };

            // Gamma correction (gamma = 2.2)
            let r_gamma = r_tone.powf(1.0 / 2.2);
            let g_gamma = g_tone.powf(1.0 / 2.2);
            let b_gamma = b_tone.powf(1.0 / 2.2);

            rgba[0] = (r_gamma.clamp(0.0, 1.0) * 255.0) as u8;
            rgba[1] = (g_gamma.clamp(0.0, 1.0) * 255.0) as u8;
            rgba[2] = (b_gamma.clamp(0.0, 1.0) * 255.0) as u8;
            rgba[3] = 255;
        });
}

fn format_rays_per_sec(rays_per_sec: f64) -> String {
    if rays_per_sec >= 1_000_000.0 {
        format!("{:.2} Mrays/s", rays_per_sec / 1_000_000.0)
    } else if rays_per_sec >= 1_000.0 {
        format!("{:.1} Krays/s", rays_per_sec / 1_000.0)
    } else {
        format!("{:.0} rays/s", rays_per_sec)
    }
}

// The background thread logic has been entirely encapsulated within rnpt::ParallelTracer!

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Rust Neural Path Tracer")
            .with_inner_size([1100.0, 700.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Rust Neural Path Tracer",
        options,
        Box::new(|cc| {
            cc.egui_ctx.set_visuals(egui::Visuals::dark());
            Ok(Box::new(RnptGuiApp::new(cc)))
        }),
    )
}
