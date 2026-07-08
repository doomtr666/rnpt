/// Orbit the camera around its target in spherical coordinates.
/// Left-drag: dx rotates azimuth, dy rotates elevation.
pub fn orbit(camera: &mut rnpt::Camera, dx: f32, dy: f32) {
    use std::f32::consts::PI;
    let to_cam = camera.position - camera.target;
    let r = to_cam.norm();
    if r < 1e-6 {
        return;
    }
    let theta = (to_cam.y / r).clamp(-1.0, 1.0).acos();
    let phi = to_cam.z.atan2(to_cam.x);
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
pub fn pan(camera: &mut rnpt::Camera, dx: f32, dy: f32) {
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
pub fn dolly(camera: &mut rnpt::Camera, scroll: f32) {
    let to_cam = camera.position - camera.target;
    let r = to_cam.norm();
    if r < 1e-6 {
        return;
    }
    let new_r = (r * (-scroll * 0.005f32).exp()).max(0.01);
    camera.position = camera.target + to_cam / r * new_r;
}
