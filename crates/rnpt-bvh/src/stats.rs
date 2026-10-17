//! Measuring the tree without a stopwatch.
//!
//! `TraversalStats` counts what traversal does (node tests, triangle chunk tests, ...)
//! when the `stats` feature is on; without it the counters compile to nothing.
//! `TreeStats` describes the built tree, including the terms of its SAH cost.
//! Both count operations, not time, so they do not depend on the machine's load.

/// Operations done by traversal, summed over the rays traced since the last `take`.
#[derive(Clone, Copy, Default, Debug)]
pub struct TraversalStats {
    pub rays: u64,
    /// Inner nodes visited: one 8-box SIMD test each.
    pub nodes: u64,
    /// Children whose box the ray entered (pushed on the stack).
    pub boxes_hit: u64,
    /// Stack entries discarded on pop: a closer hit was found meanwhile.
    pub culled: u64,
    /// Triangle chunks tested: one 8-triangle SIMD test each.
    pub chunks: u64,
    /// Real triangles in the tested chunks (the other lanes are padding).
    pub tri_lanes: u64,
}

#[cfg(feature = "stats")]
thread_local! {
    static COUNTERS: std::cell::Cell<TraversalStats> = std::cell::Cell::new(TraversalStats::default());
}

#[inline(always)]
pub(crate) fn record(f: impl FnOnce(&mut TraversalStats)) {
    #[cfg(feature = "stats")]
    COUNTERS.with(|c| {
        let mut s = c.get();
        f(&mut s);
        c.set(s);
    });
    #[cfg(not(feature = "stats"))]
    let _ = f;
}

/// Counters of the calling thread since the last call, then reset.
#[cfg(feature = "stats")]
pub fn take() -> TraversalStats {
    COUNTERS.with(|c| c.replace(TraversalStats::default()))
}

/// The built tree, and the two terms of its SAH cost.
///
/// For rays uniformly distributed in space, the probability of hitting a box knowing that
/// the ray hits the root's is the ratio of their surface areas. The expected cost of a
/// ray is then `c_node * sah_nodes + c_chunk * sah_chunks`, for any choice of the costs
/// of an 8-box test and of an 8-triangle test.
#[derive(Clone, Copy, Default, Debug)]
pub struct TreeStats {
    pub inner_nodes: u64,
    pub leaves: u64,
    pub chunks: u64,
    /// Real triangles, padding excluded.
    pub triangles: u64,
    /// Children slots in use, over all inner nodes (8 per node at most).
    pub used_slots: u64,
    pub max_depth: u32,
    pub node_bytes: u64,
    /// Triangle chunks, their ids and double-sided masks.
    pub chunk_bytes: u64,
    /// Expected inner nodes visited per ray: sum of area(node) / area(root).
    pub sah_nodes: f64,
    /// Expected chunks tested per ray: sum of chunks(leaf) * area(leaf) / area(root).
    pub sah_chunks: f64,
}
