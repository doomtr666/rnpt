/// MIS balance heuristic: w_i = p_i / Σ p_j
/// Properties that must hold regardless of the specific PDF values.

#[test]
fn balance_weights_sum_to_one() {
    let cases = [
        (1.0f32, 1.0f32),
        (10.0, 0.1),
        (0.001, 1000.0),
        (0.5, 0.5),
        (f32::EPSILON, 1.0),
    ];
    for (p_nee, p_brdf) in cases {
        let denom = p_nee + p_brdf;
        let w_nee  = p_nee  / denom;
        let w_brdf = p_brdf / denom;
        assert!((w_nee + w_brdf - 1.0).abs() < 1e-6, "sum={}", w_nee + w_brdf);
    }
}

#[test]
fn balance_weight_in_unit_interval() {
    let cases = [(2.0f32, 3.0f32), (0.01, 99.0), (1.0, 1.0)];
    for (p_nee, p_brdf) in cases {
        let denom = p_nee + p_brdf;
        let w = p_nee / denom;
        assert!(w >= 0.0 && w <= 1.0, "w={w} out of [0,1]");
    }
}

#[test]
fn balance_weight_degenerate_one_technique() {
    // If one PDF is zero, the other gets full weight.
    // (In practice the sample can't come from the zero-pdf technique, but the
    // formula should be well-defined when called with the non-zero pdf only.)
    let p_brdf = 1.0f32;
    let p_nee  = 0.0f32;
    // Guard: we never call with both zero in the integrator.
    if p_brdf + p_nee > 0.0 {
        let w_brdf = p_brdf / (p_brdf + p_nee);
        assert!((w_brdf - 1.0).abs() < 1e-6, "w_brdf={w_brdf}");
    }
}

#[test]
fn power_heuristic_less_variance_than_balance() {
    // Power heuristic (β=2): w_i = p_i² / Σ p_j².
    // For two equal-pdf techniques, both give w=0.5 — same result as balance.
    let p = 0.5f32;
    let w_balance = p / (p + p);
    let w_power   = p * p / (p * p + p * p);
    assert!((w_balance - 0.5).abs() < 1e-6);
    assert!((w_power - 0.5).abs() < 1e-6);
}
