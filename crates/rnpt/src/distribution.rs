//! Inverse-CDF sampling of piecewise-constant distributions (the same trick as
//! `MeshEmitter`'s area CDF, generalized to 1D/2D). Used for HDRI importance
//! sampling: a 2D distribution = a marginal over rows + one conditional per row.
//!
//! `AliasTable` provides O(1) *discrete* sampling via Walker's alias method and
//! is used for triangle selection in `MeshEmitter`.

/// O(1) discrete sampler — Walker's alias method (Vose 1991).
///
/// Build cost: O(n). Sample cost: exactly 2 array reads, no branching in the
/// hot path. Replaces a CDF `partition_point` (O(log n)) for discrete distributions.
#[derive(Clone, Debug)]
pub struct AliasTable {
    prob: Vec<f32>,       // probability of choosing index i directly (vs alias[i])
    alias: Vec<u32>,      // alternative index when U > prob[i]
    source_pdf: Vec<f32>, // normalized weight[i]/sum — true sampling probability per entry
}

impl AliasTable {
    /// Build from non-negative weights. Panics if `weights` is empty.
    pub fn new(weights: &[f32]) -> Self {
        let n = weights.len();
        assert!(n > 0, "AliasTable requires at least one weight");

        let sum: f32 = weights.iter().sum();
        let source_pdf: Vec<f32> = if sum > 0.0 {
            weights.iter().map(|&w| w / sum).collect()
        } else {
            vec![1.0 / n as f32; n]
        };
        // Scale weights so that each bin's expected share is 1.0.
        let mut scaled: Vec<f32> = if sum > 0.0 {
            weights.iter().map(|&w| w * n as f32 / sum).collect()
        } else {
            vec![1.0; n] // degenerate: uniform fallback
        };

        let mut prob = vec![0.0f32; n];
        let mut alias = vec![0u32; n];

        // Partition into under-full (< 1) and over-full (>= 1) bins.
        let mut small: Vec<usize> = Vec::with_capacity(n);
        let mut large: Vec<usize> = Vec::with_capacity(n);
        for (i, &s) in scaled.iter().enumerate() {
            if s < 1.0 { small.push(i); } else { large.push(i); }
        }

        while !small.is_empty() && !large.is_empty() {
            let l = small.pop().unwrap();
            let g = *large.last().unwrap();
            prob[l] = scaled[l];
            alias[l] = g as u32;
            scaled[g] -= 1.0 - scaled[l];
            if scaled[g] < 1.0 {
                large.pop();
                small.push(g);
            }
        }
        // Remaining bins have probability 1 (floating-point rounding residuals).
        for l in small { prob[l] = 1.0; }
        for g in large { prob[g] = 1.0; }

        Self { prob, alias, source_pdf }
    }

    /// O(1) discrete sample. `u` must be in `[0, 1)`.
    #[inline]
    pub fn sample(&self, u: f32) -> usize {
        let n = self.prob.len();
        let scaled = u * n as f32;
        let i = (scaled as usize).min(n - 1);
        let frac = scaled - i as f32;
        if frac < self.prob[i] { i } else { self.alias[i] as usize }
    }

    /// True selection probability for entry `i` (= weight[i] / sum(weights)).
    #[inline]
    pub fn entry_pdf(&self, i: usize) -> f32 {
        self.source_pdf[i]
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.prob.len()
    }
}

/// 1D piecewise-constant distribution over `n` bins. The CDF is normalized
/// (`cdf[n] == 1`); `integral` is the un-normalized sum of the input function.
#[derive(Clone, Debug)]
pub struct Distribution1D {
    cdf: Vec<f32>,
    integral: f32,
}

impl Distribution1D {
    pub fn new(f: &[f32]) -> Self {
        let n = f.len().max(1);
        let mut cdf = Vec::with_capacity(n + 1);
        cdf.push(0.0);
        let mut acc = 0.0f32;
        for &v in f {
            acc += v.max(0.0);
            cdf.push(acc);
        }
        let integral = acc;
        if integral > 0.0 {
            for c in cdf.iter_mut() {
                *c /= integral;
            }
        } else {
            // Degenerate (all-zero) function → fall back to uniform.
            for (i, c) in cdf.iter_mut().enumerate() {
                *c = i as f32 / n as f32;
            }
        }
        Self { cdf, integral }
    }

    /// The integral of the input function (un-normalized total weight).
    #[inline]
    pub fn integral(&self) -> f32 {
        self.integral
    }

    /// Sample a continuous position in `[0,1)` from a uniform `u`. Returns
    /// `(x, pdf, bin)` where `pdf` is the (piecewise-constant) density at `x`.
    pub fn sample(&self, u: f32) -> (f32, f32, usize) {
        let n = self.cdf.len() - 1;
        // Last index with cdf[bin] <= u  (FindInterval).
        let bin = self
            .cdf
            .partition_point(|&c| c <= u)
            .saturating_sub(1)
            .min(n - 1);
        let c0 = self.cdf[bin];
        let dc = self.cdf[bin + 1] - c0;
        let frac = if dc > 0.0 { (u - c0) / dc } else { 0.0 };
        let x = (bin as f32 + frac) / n as f32;
        (x, dc * n as f32, bin)
    }

    /// Piecewise-constant density at a continuous position `x` in `[0,1)`.
    pub fn pdf(&self, x: f32) -> f32 {
        let n = self.cdf.len() - 1;
        let bin = ((x * n as f32) as usize).min(n - 1);
        (self.cdf[bin + 1] - self.cdf[bin]) * n as f32
    }
}

/// 2D piecewise-constant distribution over a `w × h` grid (row-major), sampled
/// via a marginal over rows then a per-row conditional over columns.
#[derive(Clone, Debug)]
pub struct Distribution2D {
    conditional: Vec<Distribution1D>, // one per row (over the `w` columns)
    marginal: Distribution1D,         // over the `h` rows (weights = row integrals)
    height: usize,
}

impl Distribution2D {
    pub fn new(func: &[f32], w: usize, h: usize) -> Self {
        let mut conditional = Vec::with_capacity(h);
        let mut row_integrals = Vec::with_capacity(h);
        for v in 0..h {
            let d = Distribution1D::new(&func[v * w..(v + 1) * w]);
            row_integrals.push(d.integral());
            conditional.push(d);
        }
        let marginal = Distribution1D::new(&row_integrals);
        Self {
            conditional,
            marginal,
            height: h,
        }
    }

    /// Sample `(u, v)` in `[0,1)²` from two uniforms. Returns `((u, v), pdf)`
    /// where `pdf` is the joint density in the unit square.
    pub fn sample(&self, u1: f32, u2: f32) -> ((f32, f32), f32) {
        let (v, pdf_v, row) = self.marginal.sample(u1);
        let (u, pdf_u, _) = self.conditional[row].sample(u2);
        ((u, v), pdf_v * pdf_u)
    }

    /// Joint density of `(u, v)` in the unit square.
    pub fn pdf(&self, u: f32, v: f32) -> f32 {
        let row = ((v * self.height as f32) as usize).min(self.height - 1);
        self.marginal.pdf(v) * self.conditional[row].pdf(u)
    }
}
