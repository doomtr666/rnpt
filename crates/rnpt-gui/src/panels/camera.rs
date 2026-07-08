use eframe::egui;
use crate::app::RnptGuiApp;

pub fn show(app: &mut RnptGuiApp, ui: &mut egui::Ui) {
    let mut changed = false;

    ui.collapsing("Camera Parameters", |ui| {
        let n_cams = app.current_scene.as_ref().map_or(0, |s| s.cameras.len());
        if n_cams > 1 {
            let prev = app.selected_camera_index;
            egui::ComboBox::from_id_source("camera_selector")
                .selected_text(format!("Camera {}", app.selected_camera_index))
                .show_ui(ui, |ui| {
                    for i in 0..n_cams {
                        ui.selectable_value(
                            &mut app.selected_camera_index,
                            i,
                            format!("Camera {}", i),
                        );
                    }
                });
            if app.selected_camera_index != prev {
                if let Some(cam) = app
                    .current_scene
                    .as_ref()
                    .and_then(|s| s.cameras.get(app.selected_camera_index))
                    .cloned()
                {
                    app.camera = cam;
                    changed = true;
                }
            }
            ui.add_space(4.0);
        }

        ui.label("Position:");
        ui.horizontal(|ui| {
            changed |= ui.add(egui::DragValue::new(&mut app.camera.position.x).speed(0.1).prefix("X: ")).changed();
            changed |= ui.add(egui::DragValue::new(&mut app.camera.position.y).speed(0.1).prefix("Y: ")).changed();
            changed |= ui.add(egui::DragValue::new(&mut app.camera.position.z).speed(0.1).prefix("Z: ")).changed();
        });

        ui.add_space(4.0);
        ui.label("Target:");
        ui.horizontal(|ui| {
            changed |= ui.add(egui::DragValue::new(&mut app.camera.target.x).speed(0.1).prefix("X: ")).changed();
            changed |= ui.add(egui::DragValue::new(&mut app.camera.target.y).speed(0.1).prefix("Y: ")).changed();
            changed |= ui.add(egui::DragValue::new(&mut app.camera.target.z).speed(0.1).prefix("Z: ")).changed();
        });

        ui.add_space(4.0);
        changed |= ui.add(egui::Slider::new(&mut app.camera.fov, 10.0..=120.0).text("FOV")).changed();
    });

    ui.add_space(10.0);

    ui.collapsing("Resolution", |ui| {
        let prev_auto_fit = app.auto_fit;
        ui.checkbox(&mut app.auto_fit, "Auto-fit to viewport");
        if prev_auto_fit != app.auto_fit {
            changed = true;
        }
        ui.add_space(4.0);
        ui.add_enabled_ui(!app.auto_fit, |ui| {
            ui.horizontal(|ui| {
                ui.label("W:");
                let r1 = ui.add(egui::DragValue::new(&mut app.resolution[0]).range(64..=2048));
                ui.label("H:");
                let r2 = ui.add(egui::DragValue::new(&mut app.resolution[1]).range(64..=2048));
                let done1 = r1.drag_stopped() || r1.lost_focus() || (r1.changed() && !r1.dragged());
                let done2 = r2.drag_stopped() || r2.lost_focus() || (r2.changed() && !r2.dragged());
                if done1 || done2 {
                    changed = true;
                }
            });
        });
    });

    if changed {
        app.trigger_reset();
    }

    ui.add_space(10.0);
    ui.separator();
    ui.add_space(10.0);
}
