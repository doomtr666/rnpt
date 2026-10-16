use eframe::egui;
use crate::app::RnptGuiApp;

pub fn show(app: &mut RnptGuiApp, ui: &mut egui::Ui) {
    ui.heading("Lighting");
    ui.add_space(4.0);
    let prev = app.strategy;
    ui.horizontal(|ui| {
        ui.selectable_value(&mut app.strategy, rnpt::SamplingStrategy::Nirc, "NIRC");
        ui.selectable_value(&mut app.strategy, rnpt::SamplingStrategy::Mis, "MIS");
        ui.selectable_value(&mut app.strategy, rnpt::SamplingStrategy::NeeOnly, "NEE");
        ui.selectable_value(&mut app.strategy, rnpt::SamplingStrategy::BrdfOnly, "BRDF");
        ui.selectable_value(&mut app.strategy, rnpt::SamplingStrategy::DirectOnly, "Direct Only");
    });
    if app.strategy != prev {
        app.trigger_reset();
    }
    if ui.checkbox(&mut app.indirect_only, "Indirect light only").changed() {
        app.trigger_reset();
    }

    // Stop each pixel after a fixed cost, to compare strategies at equal work
    // rather than equal time. 0 = unlimited.
    ui.add_space(6.0);
    ui.label("Budget per pixel (0 = unlimited)");
    let mut changed = false;
    ui.horizontal(|ui| {
        changed |= ui
            .add(egui::DragValue::new(&mut app.budget.rays_per_pixel).speed(8.0).range(0..=u32::MAX).suffix(" rays"))
            .changed();
        changed |= ui
            .add(egui::DragValue::new(&mut app.budget.samples_per_pixel).speed(1.0).range(0..=u32::MAX).suffix(" spp"))
            .changed();
    });
    if changed {
        app.trigger_reset();
    }

    ui.add_space(10.0);
    ui.separator();
    ui.add_space(10.0);
}
