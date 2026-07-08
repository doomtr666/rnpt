use eframe::egui;
use crate::app::RnptGuiApp;

/// Camera navigation accumulated from one GUI frame.
#[derive(Default)]
pub struct NavInput {
    pub orbit: [f32; 2],
    pub pan:   [f32; 2],
    pub scroll: f32,
    pub changed: bool,
}

/// Show the central render viewport. Returns any navigation input captured this frame.
pub fn show(app: &mut RnptGuiApp, ctx: &egui::Context) -> NavInput {
    let mut nav = NavInput::default();

    egui::CentralPanel::default().show(ctx, |ui| {
        let available = ui.available_size();

        if let Some(ref texture) = app.texture_handle {
            let size = texture.size_vec2();
            let aspect = size.x / size.y;
            let desired = if available.x / available.y > aspect {
                egui::vec2(available.y * aspect, available.y)
            } else {
                egui::vec2(available.x, available.x / aspect)
            };
            let rect = egui::Rect::from_min_size(
                ui.min_rect().min + (available - desired) * 0.5,
                desired,
            );

            ui.put(rect, egui::Image::new(texture));

            let ctrl_held = ctx.input(|i| i.modifiers.ctrl);

            // Camera navigation — disabled while Ctrl is held (probe placement).
            let response = ui.interact(rect, ui.id().with("nav"), egui::Sense::drag());
            if !ctrl_held {
                if response.dragged_by(egui::PointerButton::Primary) {
                    let d = response.drag_delta();
                    if d.x != 0.0 || d.y != 0.0 {
                        nav.orbit = [d.x, d.y];
                        nav.changed = true;
                    }
                }
                if response.dragged_by(egui::PointerButton::Middle)
                    || response.dragged_by(egui::PointerButton::Secondary)
                {
                    let d = response.drag_delta();
                    if d.x != 0.0 || d.y != 0.0 {
                        nav.pan = [d.x, d.y];
                        nav.changed = true;
                    }
                }
                if response.hovered() {
                    let scroll = ctx.input(|i| i.smooth_scroll_delta.y);
                    if scroll.abs() > 0.5 {
                        nav.scroll = scroll;
                        nav.changed = true;
                    }
                }
            }

            // Ctrl+click: place a directional NIRC probe at the clicked surface point.
            if ctrl_held
                && ctx.input(|i| i.pointer.button_clicked(egui::PointerButton::Primary))
            {
                if let Some(screen_pos) = ctx.input(|i| i.pointer.interact_pos()) {
                    if rect.contains(screen_pos) {
                        let px = (screen_pos.x - rect.min.x) / rect.width()
                            * app.local_width as f32;
                        let py = (screen_pos.y - rect.min.y) / rect.height()
                            * app.local_height as f32;
                        app.update_probe(ctx, px, py);
                    }
                }
            }
        } else {
            ui.centered_and_justified(|ui| {
                ui.spinner();
            });
        }
    });

    nav
}
