use eframe::egui;
use crate::app::RnptGuiApp;

pub fn show(app: &mut RnptGuiApp, ui: &mut egui::Ui) {
    if app.asset_files.is_empty() {
        return;
    }

    ui.group(|ui| {
        ui.label("Active Scene:");
        let selected_name = app.asset_files[app.selected_asset_index]
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned();

        let mut scene_changed = false;
        egui::ComboBox::from_id_source("scene_selector")
            .selected_text(&selected_name)
            .show_ui(ui, |ui| {
                for (idx, path) in app.asset_files.iter().enumerate() {
                    let name = path.file_name().unwrap_or_default().to_string_lossy();
                    scene_changed |= ui
                        .selectable_value(&mut app.selected_asset_index, idx, name)
                        .changed();
                }
            });

        if scene_changed {
            match crate::asset_importer::import_scene(&app.asset_files[app.selected_asset_index]) {
                Ok((scene, stats)) => {
                    app.selected_camera_index = 0;
                    if !scene.cameras.is_empty() {
                        let first_cam = &scene.cameras[0];
                        app.camera.position = first_cam.position;
                        app.camera.target = first_cam.target;
                        app.camera.fov = first_cam.fov;
                    }
                    let scene_arc = std::sync::Arc::new(scene);
                    let (bvh_arc, lights_arc, bms, tris) =
                        crate::app::build_bvh_and_lights(&scene_arc);
                    app.scene_load_stats = Some(stats);
                    app.bvh_build_ms = bms;
                    app.bvh_triangle_count = tris;
                    app.current_scene = Some(scene_arc);
                    app.current_bvh = Some(bvh_arc);
                    app.current_lights = Some(lights_arc);
                    app.trigger_reset();
                }
                Err(e) => eprintln!("Failed to load scene: {}", e),
            }
        }
    });
    ui.add_space(10.0);
}
