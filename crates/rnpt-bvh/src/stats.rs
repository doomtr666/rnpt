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

/// What one ray touched: an entry per inner node visited and per chunk tested, in order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Touch {
    Node(u32),
    Chunk(u32),
    /// The chunk that holds the answer so far: the closest hit, or the occluder found.
    Answer(u32),
}

#[cfg(feature = "stats")]
thread_local! {
    static TRACE: std::cell::RefCell<Option<Vec<Touch>>> = const { std::cell::RefCell::new(None) };
}

#[inline(always)]
pub(crate) fn trace(t: Touch) {
    #[cfg(feature = "stats")]
    TRACE.with(|tr| {
        if let Some(v) = tr.borrow_mut().as_mut() {
            v.push(t);
        }
    });
    #[cfg(not(feature = "stats"))]
    let _ = t;
}

/// Starts recording what the calling thread's rays touch.
#[cfg(feature = "stats")]
pub fn trace_start() {
    TRACE.with(|tr| *tr.borrow_mut() = Some(Vec::new()));
}

/// Stops recording.
#[cfg(feature = "stats")]
pub fn trace_stop() {
    TRACE.with(|tr| *tr.borrow_mut() = None);
}

/// What was touched since `trace_start` or the last call; recording goes on.
#[cfg(feature = "stats")]
pub fn trace_take() -> Vec<Touch> {
    TRACE.with(|tr| tr.borrow_mut().as_mut().map(std::mem::take).unwrap_or_default())
}

/// Adds `s` to the calling thread's counters (counts taken per ray, given back in bulk).
#[cfg(feature = "stats")]
pub fn put_back(s: TraversalStats) {
    COUNTERS.with(|c| {
        let mut t = c.get();
        t.rays += s.rays;
        t.nodes += s.nodes;
        t.boxes_hit += s.boxes_hit;
        t.culled += s.culled;
        t.chunks += s.chunks;
        t.tri_lanes += s.tri_lanes;
        c.set(t);
    });
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
    /// Most chunks in one leaf.
    pub max_leaf_chunks: u32,
    pub node_bytes: u64,
    /// Triangle chunks, their ids and double-sided masks.
    pub chunk_bytes: u64,
    /// Expected inner nodes visited per ray: sum of area(node) / area(root).
    pub sah_nodes: f64,
    /// Expected chunks tested per ray: sum of chunks(leaf) * area(leaf) / area(root).
    pub sah_chunks: f64,
}
