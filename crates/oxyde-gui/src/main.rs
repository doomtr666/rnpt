use eframe::egui;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

enum RenderCommand {
    Reset {
        width: usize,
        height: usize,
        camera: oxyde::CameraConfig,
    },
    Stop,
}

struct SharedRenderState {
    width: usize,
    height: usize,
    pixels: Vec<oxyde::Pixel>,
    is_dirty: bool,
}

struct OxydeGuiApp {
    camera: oxyde::CameraConfig,
    resolution: [usize; 2],
    exposure: f32,
    
    cmd_tx: std::sync::mpsc::Sender<RenderCommand>,
    shared_state: Arc<Mutex<SharedRenderState>>,
    
    local_pixels: Vec<oxyde::Pixel>,
    local_width: usize,
    local_height: usize,
    
    texture_handle: Option<egui::TextureHandle>,
    last_exposure: f32,
}

impl OxydeGuiApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let width = 800;
        let height = 600;
        let camera = oxyde::CameraConfig::default();
        
        let shared_state = Arc::new(Mutex::new(SharedRenderState {
            width,
            height,
            pixels: vec![oxyde::Pixel::default(); width * height],
            is_dirty: false,
        }));
        
        let (cmd_tx, cmd_rx) = std::sync::mpsc::channel();
        
        // Spawn background rendering thread
        let shared_state_clone = shared_state.clone();
        let ctx_clone = cc.egui_ctx.clone();
        thread::spawn(move || {
            run_renderer_thread(cmd_rx, shared_state_clone, ctx_clone);
        });
        
        // Send initial reset command to start rendering
        let _ = cmd_tx.send(RenderCommand::Reset {
            width,
            height,
            camera: camera.clone(),
        });
        
        Self {
            camera,
            resolution: [width, height],
            exposure: 1.0,
            cmd_tx,
            shared_state,
            local_pixels: vec![oxyde::Pixel::default(); width * height],
            local_width: width,
            local_height: height,
            texture_handle: None,
            last_exposure: 1.0,
        }
    }
    
    fn trigger_reset(&self) {
        let _ = self.cmd_tx.send(RenderCommand::Reset {
            width: self.resolution[0],
            height: self.resolution[1],
            camera: self.camera.clone(),
        });
    }
}

impl eframe::App for OxydeGuiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let mut pixels_updated = false;
        
        // 1. Check if the renderer thread has produced new pixels
        {
            if let Ok(mut state) = self.shared_state.try_lock() {
                if state.is_dirty {
                    self.local_width = state.width;
                    self.local_height = state.height;
                    
                    // Resize local buffer if needed
                    if self.local_pixels.len() != state.pixels.len() {
                        self.local_pixels.resize(state.pixels.len(), oxyde::Pixel::default());
                    }
                    self.local_pixels.copy_from_slice(&state.pixels);
                    state.is_dirty = false;
                    pixels_updated = true;
                }
            }
        }
        
        // 2. If pixels updated, or exposure changed, regenerate the texture
        let exposure_changed = self.exposure != self.last_exposure;
        if pixels_updated || exposure_changed || self.texture_handle.is_none() {
            let mut raw_rgba = vec![0u8; self.local_width * self.local_height * 4];
            
            tonemap_and_convert(
                &self.local_pixels,
                self.exposure,
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
        }
        
        // 3. UI Layout
        egui::SidePanel::left("controls_panel")
            .resizable(true)
            .default_width(260.0)
            .show(ctx, |ui| {
                ui.add_space(8.0);
                ui.heading("Oxyde RT Controls");
                ui.add_space(10.0);
                
                let mut changed = false;
                
                // Camera section
                ui.collapsing("Camera Parameters", |ui| {
                    ui.label("Position:");
                    ui.horizontal(|ui| {
                        changed |= ui.add(egui::DragValue::new(&mut self.camera.position.x).speed(0.1).prefix("X: ")).changed();
                        changed |= ui.add(egui::DragValue::new(&mut self.camera.position.y).speed(0.1).prefix("Y: ")).changed();
                        changed |= ui.add(egui::DragValue::new(&mut self.camera.position.z).speed(0.1).prefix("Z: ")).changed();
                    });
                    
                    ui.add_space(4.0);
                    ui.label("Target:");
                    ui.horizontal(|ui| {
                        changed |= ui.add(egui::DragValue::new(&mut self.camera.target.x).speed(0.1).prefix("X: ")).changed();
                        changed |= ui.add(egui::DragValue::new(&mut self.camera.target.y).speed(0.1).prefix("Y: ")).changed();
                        changed |= ui.add(egui::DragValue::new(&mut self.camera.target.z).speed(0.1).prefix("Z: ")).changed();
                    });
                    
                    ui.add_space(4.0);
                    changed |= ui.add(egui::Slider::new(&mut self.camera.fov, 10.0..=120.0).text("FOV")).changed();
                });
                
                ui.add_space(10.0);
                
                // Resolution section
                ui.collapsing("Resolution", |ui| {
                    ui.horizontal(|ui| {
                        ui.label("W:");
                        changed |= ui.add(egui::DragValue::new(&mut self.resolution[0]).range(64..=2048)).changed();
                        ui.label("H:");
                        changed |= ui.add(egui::DragValue::new(&mut self.resolution[1]).range(64..=2048)).changed();
                    });
                });
                
                if changed {
                    self.trigger_reset();
                }
                
                ui.add_space(10.0);
                ui.separator();
                ui.add_space(10.0);
                
                // Post Processing section
                ui.heading("Post-Processing");
                ui.add_space(4.0);
                ui.add(egui::Slider::new(&mut self.exposure, 0.1..=10.0).text("Exposure"));
                
                ui.add_space(10.0);
                ui.separator();
                ui.add_space(10.0);
                
                // Render Stats
                ui.heading("Rendering Stats");
                ui.add_space(4.0);
                ui.label(format!("Resolution: {}x{}", self.local_width, self.local_height));
                
                // Calculate average samples
                let avg_samples = if self.local_pixels.is_empty() {
                    0.0
                } else {
                    let total_samples: u64 = self.local_pixels.iter().map(|p| p.samples as u64).sum();
                    total_samples as f64 / self.local_pixels.len() as f64
                };
                ui.label(format!("Samples/Pixel (avg): {:.1}", avg_samples));
            });
            
        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(ref texture) = self.texture_handle {
                let size = texture.size_vec2();
                let max_size = ui.available_size();
                let aspect_ratio = size.x / size.y;
                
                let desired_size = if max_size.x / max_size.y > aspect_ratio {
                    egui::vec2(max_size.y * aspect_ratio, max_size.y)
                } else {
                    egui::vec2(max_size.x, max_size.x / aspect_ratio)
                };
                
                // Centering the image in the panel
                let rect = egui::Rect::from_min_size(
                    ui.min_rect().min + (max_size - desired_size) * 0.5,
                    desired_size,
                );
                
                ui.put(rect, egui::Image::new(texture));
            } else {
                ui.centered_and_justified(|ui| {
                    ui.spinner();
                });
            }
        });
    }
    
    // Cleanup background thread when dropping
    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        let _ = self.cmd_tx.send(RenderCommand::Stop);
    }
}

fn tonemap_and_convert(
    pixels: &[oxyde::Pixel],
    exposure: f32,
    output_rgba: &mut [u8],
) {
    use rayon::prelude::*;
    
    output_rgba
        .par_chunks_exact_mut(4)
        .zip(pixels.par_iter())
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
            let r_linear = pixel.r * scale;
            let g_linear = pixel.g * scale;
            let b_linear = pixel.b * scale;
            
            // Exposure
            let r_exp = r_linear * exposure;
            let g_exp = g_linear * exposure;
            let b_exp = b_linear * exposure;
            
            // Reinhard tonemapping
            let r_tone = r_exp / (r_exp + 1.0);
            let g_tone = g_exp / (g_exp + 1.0);
            let b_tone = b_exp / (b_exp + 1.0);
            
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

fn run_renderer_thread(
    rx: std::sync::mpsc::Receiver<RenderCommand>,
    shared_state: Arc<Mutex<SharedRenderState>>,
    ctx: egui::Context,
) {
    let mut width = 800;
    let mut height = 600;
    let mut camera = oxyde::CameraConfig::default();
    let mut pixels = vec![oxyde::Pixel::default(); width * height];
    let mut running = true;
    let mut last_update_time = Instant::now();
    
    while running {
        // Process commands
        let mut got_command = true;
        while got_command {
            match rx.try_recv() {
                Ok(RenderCommand::Reset { width: w, height: h, camera: cam }) => {
                    width = w;
                    height = h;
                    camera = cam;
                    pixels = vec![oxyde::Pixel::default(); width * height];
                    last_update_time = Instant::now(); // Force update on reset
                }
                Ok(RenderCommand::Stop) => {
                    running = false;
                    got_command = false;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    got_command = false;
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    running = false;
                    got_command = false;
                }
            }
        }
        
        if !running {
            break;
        }
        
        if pixels.is_empty() {
            std::thread::sleep(Duration::from_millis(16));
            continue;
        }
        
        // Perform one sample pass over the whole image
        use rayon::prelude::*;
        pixels
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, pixel)| {
                let x = idx % width;
                let y = idx / width;
                oxyde::sample_pixel(x, y, width, height, &camera, pixel);
            });
            
        // Rate-limit updates to the GUI (e.g. 30 FPS / every 33 ms)
        let now = Instant::now();
        if now.duration_since(last_update_time) >= Duration::from_millis(33) {
            {
                let mut state = shared_state.lock().unwrap();
                state.width = width;
                state.height = height;
                state.pixels.clone_from(&pixels);
                state.is_dirty = true;
            }
            ctx.request_repaint();
            last_update_time = now;
        }
        
        // Sleep slightly to prevent 100% spin and keep the GUI responsive
        std::thread::sleep(Duration::from_millis(8));
    }
}

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Oxyde RT - Path Tracer")
            .with_inner_size([1100.0, 700.0]),
        ..Default::default()
    };
    
    eframe::run_native(
        "oxyde_rt",
        options,
        Box::new(|cc| {
            cc.egui_ctx.set_visuals(egui::Visuals::dark());
            Ok(Box::new(OxydeGuiApp::new(cc)))
        }),
    )
}
