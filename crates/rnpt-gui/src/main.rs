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
    real_rays_since_last_fps: u64,
    shadow_rays_since_last_fps: u64,

    local_pixels: Vec<rnpt::Pixel>,
    local_width: usize,
    local_height: usize,
    local_rays_per_sec: f64,
    local_real_rays_per_sec: f64,
    local_shadow_rays_per_sec: f64,

    texture_handle: Option<egui::TextureHandle>,
    last_exposure: f32,
    last_tonemapper: TonemapOperator,

    // New fields
    asset_files: Vec<std::path::PathBuf>,
    selected_asset_index: usize,
    current_scene: Option<std::sync::Arc<rnpt::Scene>>,
    current_bvh: Option<std::sync::Arc<rnpt::Bvh>>,
    current_lights: Option<std::sync::Arc<Vec<rnpt::Light>>>,

    auto_fit: bool,
    resize_timeout: Option<Instant>,

    strategy: rnpt::SamplingStrategy,
    selected_camera_index: usize,

    // Environment (HDRI) lighting.
    hdr_files: Vec<std::path::PathBuf>,
    selected_env: usize, // 0 = none, else hdr_files[selected_env - 1]
    env: Option<std::sync::Arc<rnpt::EnvLight>>,
    env_raw: Option<(Vec<rnpt::Color>, usize, usize)>, // cached pixels for intensity rebuild
    env_intensity: f32,
    env_rotation: f32, // degrees, [0, 360)

    // Load timings
    scene_load_stats: Option<asset_importer::SceneLoadStats>,
    bvh_build_ms: u64,
    bvh_triangle_count: usize,

    // Debug mode
    show_debug: bool,
}

/// Append the environment light (if any) to the unified light list and report
/// its index for the ray-escape hook.
fn with_env(
    lights: &std::sync::Arc<Vec<rnpt::Light>>,
    env: &Option<std::sync::Arc<rnpt::EnvLight>>,
) -> (std::sync::Arc<Vec<rnpt::Light>>, Option<usize>) {
    match env {
        Some(e) => {
            let mut v = (**lights).clone();
            v.push(rnpt::Light::Environment(e.clone()));
            let idx = v.len() - 1;
            (std::sync::Arc::new(v), Some(idx))
        }
        None => (lights.clone(), None),
    }
}

fn empty_scene() -> rnpt::Scene {
    rnpt::Scene {
        meshes: Vec::new(),
        materials: Vec::new(),
        textures: Vec::new(),
        lights: Vec::new(),
        nodes: Vec::new(),
        roots: Vec::new(),
        cameras: vec![rnpt::Camera::default()],
    }
}

/// Build the BVH and the unified light list (scene punctual lights + emissive
/// meshes collected during the BVH flatten) for a scene.
/// Returns (bvh, lights, build_ms, triangle_count).
fn build_bvh_and_lights(
    scene: &rnpt::Scene,
) -> (std::sync::Arc<rnpt::Bvh>, std::sync::Arc<Vec<rnpt::Light>>, u64, usize) {
    let t0 = Instant::now();
    let (bvh, emitters) = rnpt::BvhBuilder::new(scene).build();
    let build_ms = t0.elapsed().as_millis() as u64;
    let tri_count = bvh.triangle_meta.len();
    let lights = rnpt::build_lights(&scene.lights, emitters);
    (std::sync::Arc::new(bvh), std::sync::Arc::new(lights), build_ms, tri_count)
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
        let mut current_bvh = None;
        let mut current_lights = None;
        let mut scene_load_stats: Option<asset_importer::SceneLoadStats> = None;
        let mut bvh_build_ms = 0u64;
        let mut bvh_triangle_count = 0usize;
        if !asset_files.is_empty() {
            if let Ok((scene, stats)) =
                asset_importer::import_scene(&asset_files[selected_asset_index])
            {
                if !scene.cameras.is_empty() {
                    let first_cam = &scene.cameras[0];
                    camera.position = first_cam.position;
                    camera.target = first_cam.target;
                    camera.fov = first_cam.fov;
                }
                let scene_arc = std::sync::Arc::new(scene);
                let (bvh_arc, lights_arc, bms, tris) = build_bvh_and_lights(&scene_arc);
                scene_load_stats = Some(stats);
                bvh_build_ms = bms;
                bvh_triangle_count = tris;
                current_scene = Some(scene_arc);
                current_bvh = Some(bvh_arc);
                current_lights = Some(lights_arc);
            }
        }

        let scene = current_scene
            .clone()
            .unwrap_or_else(|| std::sync::Arc::new(empty_scene()));
        let (bvh, lights) = match (current_bvh.clone(), current_lights.clone()) {
            (Some(b), Some(l)) => (b, l),
            _ => {
                let (b, l, bms, tris) = build_bvh_and_lights(&scene);
                bvh_build_ms = bms;
                bvh_triangle_count = tris;
                (b, l)
            }
        };

        let strategy = rnpt::SamplingStrategy::Mis;
        let (restir_lights_vec, restir_alias) = rnpt::build_restir_lights(&lights);
        let config = rnpt::PathTracerConfig {
            width,
            height,
            camera: camera.clone(),
            scene,
            bvh,
            lights,
            env: None,
            strategy,
            restir_lights: std::sync::Arc::new(restir_lights_vec),
            restir_alias: std::sync::Arc::new(restir_alias),
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
            real_rays_since_last_fps: 0,
            shadow_rays_since_last_fps: 0,
            local_pixels: vec![rnpt::Pixel::default(); width * height],
            local_width: width,
            local_height: height,
            local_rays_per_sec: 0.0,
            local_real_rays_per_sec: 0.0,
            local_shadow_rays_per_sec: 0.0,
            texture_handle: None,
            last_exposure: 1.0,
            last_tonemapper: TonemapOperator::Aces,
            asset_files,
            selected_asset_index,
            current_scene,
            current_bvh,
            current_lights,
            auto_fit: true,
            resize_timeout: None,
            strategy,
            selected_camera_index: 0,
            hdr_files: asset_importer::list_hdris("assets"),
            selected_env: 0,
            env: None,
            env_raw: None,
            env_intensity: 1.0,
            env_rotation: 0.0,
            scene_load_stats,
            bvh_build_ms,
            bvh_triangle_count,
            show_debug: false,
        }
    }

    /// (Re)load the selected HDRI into `env` (clearing it if "None").
    fn load_env(&mut self) {
        if self.selected_env == 0 {
            self.env = None;
            self.env_raw = None;
            return;
        }
        let path = self.hdr_files[self.selected_env - 1].clone();
        self.env_raw = asset_importer::load_hdr(&path);
        self.rebuild_env();
    }

    /// Rebuild the `EnvLight` from the cached pixels at the current intensity
    /// (the importance-sampling distribution is intensity-independent).
    fn rebuild_env(&mut self) {
        let rot_rad = self.env_rotation.to_radians();
        self.env = self.env_raw.as_ref().map(|(pixels, w, h)| {
            std::sync::Arc::new(rnpt::EnvLight::new(pixels.clone(), *w, *h, self.env_intensity, rot_rad))
        });
    }

    fn trigger_reset(&mut self) {
        let scene = self
            .current_scene
            .clone()
            .unwrap_or_else(|| std::sync::Arc::new(empty_scene()));
        let (bvh, lights) = match (self.current_bvh.clone(), self.current_lights.clone()) {
            (Some(b), Some(l)) => (b, l),
            _ => {
                let (b, l, bms, tris) = build_bvh_and_lights(&scene);
                self.bvh_build_ms = bms;
                self.bvh_triangle_count = tris;
                (b, l)
            }
        };
        let (lights, env) = with_env(&lights, &self.env);
        let (restir_lights_vec, restir_alias) = rnpt::build_restir_lights(&lights);

        let config = rnpt::PathTracerConfig {
            width: self.resolution[0],
            height: self.resolution[1],
            camera: self.camera.clone(),
            scene,
            bvh,
            lights,
            env,
            strategy: self.strategy,
            restir_lights: std::sync::Arc::new(restir_lights_vec),
            restir_alias: std::sync::Arc::new(restir_alias),
        };

        // Always recreate tracer to ensure clean memory state and no race conditions
        self.tracer = None;
        self.tracer = Some(rnpt::ParallelTracer::new(config));
        self.local_width = self.resolution[0];
        self.local_height = self.resolution[1];
        self.local_pixels = vec![rnpt::Pixel::default(); self.local_width * self.local_height];

        self.rays_since_last_fps = 0;
        self.real_rays_since_last_fps = 0;
        self.shadow_rays_since_last_fps = 0;
        self.local_rays_per_sec = 0.0;
        self.local_real_rays_per_sec = 0.0;
        self.local_shadow_rays_per_sec = 0.0;
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

            self.rays_since_last_fps += tracer.pop_rays_traced();
            self.real_rays_since_last_fps += tracer.pop_real_rays_traced();
            self.shadow_rays_since_last_fps += tracer.pop_shadow_rays_traced();
            pixels_updated = true;

            let now = Instant::now();
            let elapsed_fps = now.duration_since(self.last_fps_time).as_secs_f64();
            if elapsed_fps >= 0.5 {
                self.local_rays_per_sec = self.rays_since_last_fps as f64 / elapsed_fps;
                self.local_real_rays_per_sec = self.real_rays_since_last_fps as f64 / elapsed_fps;
                self.local_shadow_rays_per_sec =
                    self.shadow_rays_since_last_fps as f64 / elapsed_fps;
                self.rays_since_last_fps = 0;
                self.real_rays_since_last_fps = 0;
                self.shadow_rays_since_last_fps = 0;
                self.last_fps_time = now;
            }
        }

        ctx.request_repaint();

        // If pixels updated, or exposure changed, regenerate the texture
        let exposure_changed = self.exposure != self.last_exposure;
        let tonemapper_changed = self.tonemapper != self.last_tonemapper;
        if pixels_updated || exposure_changed || tonemapper_changed || self.texture_handle.is_none()
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
                            match asset_importer::import_scene(
                                &self.asset_files[self.selected_asset_index],
                            ) {
                                Ok((scene, stats)) => {
                                    self.selected_camera_index = 0;
                                    if !scene.cameras.is_empty() {
                                        let first_cam = &scene.cameras[0];
                                        self.camera.position = first_cam.position;
                                        self.camera.target = first_cam.target;
                                        self.camera.fov = first_cam.fov;
                                    }
                                    let scene_arc = std::sync::Arc::new(scene);
                                    let (bvh_arc, lights_arc, bms, tris) =
                                        build_bvh_and_lights(&scene_arc);
                                    self.scene_load_stats = Some(stats);
                                    self.bvh_build_ms = bms;
                                    self.bvh_triangle_count = tris;
                                    self.current_scene = Some(scene_arc);
                                    self.current_bvh = Some(bvh_arc);
                                    self.current_lights = Some(lights_arc);
                                    self.trigger_reset();
                                }
                                Err(e) => {
                                    eprintln!("Failed to load scene: {}", e);
                                }
                            }
                        }
                    });
                    ui.add_space(10.0);
                }

                let mut changed = false;

                // Camera section
                ui.collapsing("Camera Parameters", |ui| {
                    // Camera selector (only when the scene has more than one camera)
                    let n_cams = self.current_scene.as_ref().map_or(0, |s| s.cameras.len());
                    if n_cams > 1 {
                        let prev = self.selected_camera_index;
                        egui::ComboBox::from_id_source("camera_selector")
                            .selected_text(format!("Camera {}", self.selected_camera_index))
                            .show_ui(ui, |ui| {
                                for i in 0..n_cams {
                                    ui.selectable_value(
                                        &mut self.selected_camera_index,
                                        i,
                                        format!("Camera {}", i),
                                    );
                                }
                            });
                        if self.selected_camera_index != prev {
                            let cam = self.current_scene.as_ref()
                                .and_then(|s| s.cameras.get(self.selected_camera_index))
                                .cloned();
                            if let Some(cam) = cam {
                                self.camera = cam;
                                changed = true;
                            }
                        }
                        ui.add_space(4.0);
                    }

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
                // Lighting method (A/B comparison: resets accumulation on change)
                ui.heading("Lighting");
                ui.add_space(4.0);
                let prev_strategy = self.strategy;
                ui.horizontal(|ui| {
                    ui.selectable_value(
                        &mut self.strategy,
                        rnpt::SamplingStrategy::ReStirDi,
                        "ReSTIR",
                    );
                    ui.selectable_value(&mut self.strategy, rnpt::SamplingStrategy::Mis, "MIS");
                    ui.selectable_value(&mut self.strategy, rnpt::SamplingStrategy::NeeOnly, "NEE");
                    ui.selectable_value(
                        &mut self.strategy,
                        rnpt::SamplingStrategy::BrdfOnly,
                        "BRDF",
                    );
                });
                if self.strategy != prev_strategy {
                    self.trigger_reset();
                }

                ui.add_space(10.0);
                ui.separator();
                ui.add_space(10.0);

                // Environment (HDRI)
                ui.heading("Environment");
                ui.add_space(4.0);
                let prev_env = self.selected_env;
                let env_label = if self.selected_env == 0 {
                    "None".to_string()
                } else {
                    self.hdr_files[self.selected_env - 1]
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("?")
                        .to_string()
                };
                egui::ComboBox::from_id_source("env_selector")
                    .selected_text(env_label)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.selected_env, 0, "None");
                        for (i, p) in self.hdr_files.iter().enumerate() {
                            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("?");
                            ui.selectable_value(&mut self.selected_env, i + 1, name);
                        }
                    });
                if self.selected_env != prev_env {
                    self.load_env();
                    self.trigger_reset();
                }
                let mut env_changed = false;
                ui.add_enabled_ui(self.env.is_some(), |ui| {
                    let prev_i = self.env_intensity;
                    ui.add(egui::Slider::new(&mut self.env_intensity, 0.0..=10.0).text("Intensity"));
                    if (self.env_intensity - prev_i).abs() > f32::EPSILON {
                        env_changed = true;
                    }
                    let prev_r = self.env_rotation;
                    ui.add(egui::Slider::new(&mut self.env_rotation, 0.0..=360.0).text("Rotation°"));
                    if (self.env_rotation - prev_r).abs() > f32::EPSILON {
                        env_changed = true;
                    }
                });
                if env_changed {
                    self.rebuild_env();
                    self.trigger_reset();
                }

                ui.add_space(10.0);
                ui.separator();
                ui.add_space(10.0);

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
                    format_paths_per_sec(self.local_rays_per_sec)
                ));

                let total_rays_per_sec =
                    self.local_real_rays_per_sec + self.local_shadow_rays_per_sec;
                ui.label(format!("Rays: {}", format_rays_per_sec(total_rays_per_sec)));

                let shadow_pct = if total_rays_per_sec > 0.0 {
                    100.0 * self.local_shadow_rays_per_sec / total_rays_per_sec
                } else {
                    0.0
                };
                ui.label(format!(
                    "  ├ closest: {}",
                    format_rays_per_sec(self.local_real_rays_per_sec)
                ));
                ui.label(format!(
                    "  └ shadow:  {} ({:.0}%)",
                    format_rays_per_sec(self.local_shadow_rays_per_sec),
                    shadow_pct
                ));

                let rays_per_path = if self.local_rays_per_sec > 0.0 {
                    total_rays_per_sec / self.local_rays_per_sec
                } else {
                    0.0
                };
                ui.label(format!("Rays/Path (avg): {:.1}", rays_per_path));

                ui.add_space(6.0);
                ui.collapsing("Scene Stats", |ui| {
                    if let Some(ref scene) = self.current_scene {
                        ui.label(format!("Meshes: {}", scene.meshes.len()));
                        ui.label(format!("Materials: {}", scene.materials.len()));
                        ui.label(format!("Triangles: {}", self.bvh_triangle_count));
                        ui.label(format!("Lights: {}", scene.lights.len()));
                        ui.label(format!("Cameras: {}", scene.cameras.len()));
                    } else {
                        ui.label("No scene loaded");
                    }
                });

                ui.collapsing("Load Times", |ui| {
                    if let Some(ref s) = self.scene_load_stats {
                        ui.label(format!("Total import:    {:>6} ms", s.total_ms));
                        ui.label(format!("  gltf parse:   {:>6} ms", s.gltf_parse_ms));
                        ui.label(format!("  tex decode:   {:>6} ms  ({} tex, {}M px)",
                            s.texture_decode_ms,
                            s.texture_count,
                            s.total_texture_pixels / 1_000_000,
                        ));
                        ui.label(format!("  mesh/mat:     {:>6} ms", s.mesh_process_ms));
                        ui.label(format!("BVH build:       {:>6} ms", self.bvh_build_ms));
                    } else {
                        ui.label("No scene loaded");
                    }
                });

                ui.collapsing("Debug", |ui| {
                    ui.checkbox(&mut self.show_debug, "Show materials");
                    if self.show_debug {
                        if let Some(ref scene) = self.current_scene {
                            egui::ScrollArea::vertical()
                                .max_height(300.0)
                                .show(ui, |ui| {
                                    for (i, mat) in scene.materials.iter().enumerate() {
                                        ui.collapsing(format!("Mat {i}"), |ui| {
                                            ui.label(format!("albedo: [{:.2}, {:.2}, {:.2}]",
                                                mat.albedo.x, mat.albedo.y, mat.albedo.z));
                                            ui.label(format!("roughness: {:.3}  metallic: {:.3}",
                                                mat.roughness, mat.metallic));
                                            ui.label(format!("transmission: {:.3}  ior: {:.3}",
                                                mat.transmission, mat.ior));
                                            ui.label(format!("thickness: {:.3}  att_dist: {:.3}",
                                                mat.thickness_factor, mat.attenuation_distance));
                                            ui.label(format!("att_color: [{:.2}, {:.2}, {:.2}]",
                                                mat.attenuation_color.x,
                                                mat.attenuation_color.y,
                                                mat.attenuation_color.z));
                                            ui.label(format!("emissive: [{:.2}, {:.2}, {:.2}]",
                                                mat.emissive.x, mat.emissive.y, mat.emissive.z));
                                            ui.label(format!(
                                                "double_sided: {}  alpha_cut: {:?}",
                                                mat.double_sided, mat.alpha_cutoff));
                                        });
                                    }
                                });
                        }
                    }
                });
            });

        let mut viewport_size = None;
        let mut nav_orbit = [0.0f32; 2];
        let mut nav_pan = [0.0f32; 2];
        let mut nav_scroll = 0.0f32;
        let mut nav_changed = false;
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

                // Camera navigation
                let response = ui.interact(rect, ui.id().with("nav"), egui::Sense::drag());
                if response.dragged_by(egui::PointerButton::Primary) {
                    let d = response.drag_delta();
                    if d.x != 0.0 || d.y != 0.0 {
                        nav_orbit = [d.x, d.y];
                        nav_changed = true;
                    }
                }
                if response.dragged_by(egui::PointerButton::Middle)
                    || response.dragged_by(egui::PointerButton::Secondary)
                {
                    let d = response.drag_delta();
                    if d.x != 0.0 || d.y != 0.0 {
                        nav_pan = [d.x, d.y];
                        nav_changed = true;
                    }
                }
                if response.hovered() {
                    let scroll = ctx.input(|i| i.smooth_scroll_delta.y);
                    if scroll.abs() > 0.5 {
                        nav_scroll = scroll;
                        nav_changed = true;
                    }
                }
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

        if nav_changed {
            orbit_camera(&mut self.camera, nav_orbit[0], nav_orbit[1]);
            pan_camera(&mut self.camera, nav_pan[0], nav_pan[1]);
            dolly_camera(&mut self.camera, nav_scroll);
            self.trigger_reset();
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

// paths/s: one path per `sample_pixel` (camera ray + bounces + shadow rays).
fn format_paths_per_sec(paths_per_sec: f64) -> String {
    if paths_per_sec >= 1_000_000.0 {
        format!("{:.2} Mpaths/s", paths_per_sec / 1_000_000.0)
    } else if paths_per_sec >= 1_000.0 {
        format!("{:.1} Kpaths/s", paths_per_sec / 1_000.0)
    } else {
        format!("{:.0} paths/s", paths_per_sec)
    }
}

// rays/s: actual rays traced (primary + bounces + shadow), the real BVH throughput.
fn format_rays_per_sec(rays_per_sec: f64) -> String {
    if rays_per_sec >= 1_000_000.0 {
        format!("{:.2} Mrays/s", rays_per_sec / 1_000_000.0)
    } else if rays_per_sec >= 1_000.0 {
        format!("{:.1} Krays/s", rays_per_sec / 1_000.0)
    } else {
        format!("{:.0} rays/s", rays_per_sec)
    }
}

/// Orbit the camera around its target in spherical coordinates.
/// Left-drag: dx rotates azimuth, dy rotates elevation.
fn orbit_camera(camera: &mut rnpt::Camera, dx: f32, dy: f32) {
    use std::f32::consts::PI;
    let to_cam = camera.position - camera.target;
    let r = to_cam.norm();
    if r < 1e-6 {
        return;
    }
    let theta = (to_cam.y / r).clamp(-1.0, 1.0).acos(); // polar [0, PI]
    let phi = to_cam.z.atan2(to_cam.x);                  // azimuth [-PI, PI]
    let s = 0.005f32;
    let new_phi = phi - dx * s;
    let new_theta = (theta + dy * s).clamp(0.005, PI - 0.005);
    camera.position = camera.target
        + nalgebra::Vector3::new(
            r * new_theta.sin() * new_phi.cos(),
            r * new_theta.cos(),
            r * new_theta.sin() * new_phi.sin(),
        );
}

/// Pan (translate) the camera and its target together in the view plane.
/// Middle/right-drag: move sideways and up/down without changing the look direction.
fn pan_camera(camera: &mut rnpt::Camera, dx: f32, dy: f32) {
    let v = camera.target - camera.position;
    let r = v.norm();
    if r < 1e-6 {
        return;
    }
    let forward = v / r;
    let world_up = nalgebra::Vector3::new(0.0f32, 1.0, 0.0);
    let right = forward.cross(&world_up);
    let right_norm = right.norm();
    if right_norm < 1e-6 {
        return;
    }
    let right = right / right_norm;
    let up = right.cross(&forward);
    let s = r * 0.001;
    let offset = right * (-dx * s) + up * (dy * s);
    camera.position += offset;
    camera.target += offset;
}

/// Dolly (zoom) the camera toward or away from the target.
/// Scroll up → closer, scroll down → farther.
fn dolly_camera(camera: &mut rnpt::Camera, scroll: f32) {
    let to_cam = camera.position - camera.target;
    let r = to_cam.norm();
    if r < 1e-6 {
        return;
    }
    let new_r = (r * (-scroll * 0.005f32).exp()).max(0.01);
    camera.position = camera.target + to_cam / r * new_r;
}

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
