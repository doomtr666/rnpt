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

    ui.add_space(10.0);
    ui.separator();
    ui.add_space(10.0);
}
