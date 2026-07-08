use eframe::egui;
use crate::app::RnptGuiApp;

pub fn show(app: &mut RnptGuiApp, ui: &mut egui::Ui) {
    ui.heading("Environment");
    ui.add_space(4.0);

    let prev_env = app.selected_env;
    let env_label = if app.selected_env == 0 {
        "None".to_string()
    } else {
        app.hdr_files[app.selected_env - 1]
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("?")
            .to_string()
    };
    egui::ComboBox::from_id_source("env_selector")
        .selected_text(env_label)
        .show_ui(ui, |ui| {
            ui.selectable_value(&mut app.selected_env, 0, "None");
            for (i, p) in app.hdr_files.iter().enumerate() {
                let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("?");
                ui.selectable_value(&mut app.selected_env, i + 1, name);
            }
        });
    if app.selected_env != prev_env {
        app.load_env();
        app.trigger_reset();
    }

    let mut env_changed = false;
    ui.add_enabled_ui(app.env.is_some(), |ui| {
        let prev_i = app.env_intensity;
        ui.add(egui::Slider::new(&mut app.env_intensity, 0.0..=10.0).text("Intensity"));
        if (app.env_intensity - prev_i).abs() > f32::EPSILON {
            env_changed = true;
        }
        let prev_r = app.env_rotation;
        ui.add(egui::Slider::new(&mut app.env_rotation, 0.0..=360.0).text("Rotation°"));
        if (app.env_rotation - prev_r).abs() > f32::EPSILON {
            env_changed = true;
        }
    });
    if env_changed {
        app.rebuild_env();
        app.trigger_reset();
    }

    ui.add_space(10.0);
    ui.separator();
    ui.add_space(10.0);
}
