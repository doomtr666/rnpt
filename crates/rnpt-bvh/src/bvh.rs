use std::mem::MaybeUninit;

use crate::math::InternalRay;
use crate::stats::{record, trace, Touch, TreeStats};
use std::sync::atomic::{AtomicU16, Ordering::Relaxed};
use crate::Hit;

const MAX_TRAVERSAL_DEPTH: usize = 64;

#[repr(C, align(32))]
#[derive(Clone, Copy)]
pub struct FlatTriangles {
    pub v0_x: [f32; 8],
    pub v0_y: [f32; 8],
    pub v0_z: [f32; 8],
    pub e1_x: [f32; 8],
    pub e1_y: [f32; 8],
    pub e1_z: [f32; 8],
    pub e2_x: [f32; 8],
    pub e2_y: [f32; 8],
    pub e2_z: [f32; 8],
}

impl Default for FlatTriangles {
    fn default() -> Self {
        Self {
            v0_x: [0.0; 8], v0_y: [0.0; 8], v0_z: [0.0; 8],
            e1_x: [0.0; 8], e1_y: [0.0; 8], e1_z: [0.0; 8],
            e2_x: [0.0; 8], e2_y: [0.0; 8], e2_z: [0.0; 8],
        }
    }
}

#[repr(C, align(32))]
#[derive(Clone, Copy)]
pub struct Bvh8Node {
    pub p_min_x: [f32; 8],
    pub p_min_y: [f32; 8],
    pub p_min_z: [f32; 8],
    pub p_max_x: [f32; 8],
    pub p_max_y: [f32; 8],
    pub p_max_z: [f32; 8],
    pub children: [u32; 8],
}

impl Default for Bvh8Node {
    fn default() -> Self {
        Self {
            p_min_x: [f32::NAN; 8], p_min_y: [f32::NAN; 8], p_min_z: [f32::NAN; 8],
            p_max_x: [f32::NAN; 8], p_max_y: [f32::NAN; 8], p_max_z: [f32::NAN; 8],
            children: [u32::MAX; 8],
        }
    }
}

impl Bvh8Node {
    const LEAF_FLAG: u32 = 0x8000_0000;
    const COUNT_SHIFT: u32 = 24;
    const COUNT_MASK: u32 = 0x7F;
    const START_MASK: u32 = 0x00FF_FFFF;

    #[inline(always)]
    pub(crate) fn encode_leaf(start_chunk: u32, chunk_count: u32) -> u32 {
        start_chunk | (chunk_count << Self::COUNT_SHIFT) | Self::LEAF_FLAG
    }

    #[inline(always)]
    fn is_leaf(encoded: u32) -> bool {
        encoded & Self::LEAF_FLAG != 0
    }

    #[inline(always)]
    fn leaf_start(encoded: u32) -> u32 {
        encoded & Self::START_MASK
    }

    #[inline(always)]
    fn leaf_count(encoded: u32) -> u32 {
        (encoded >> Self::COUNT_SHIFT) & Self::COUNT_MASK
    }
}

/// Per-triangle metadata stored at traversal runtime. Only identity fields,
/// no geometry (positions live in FlatTriangles SoA).
#[derive(Clone, Copy)]
pub struct TriSlot {
    pub orig_idx: u32,
    pub geom_id: u32,
}

pub struct BvhInner {
    pub nodes: Vec<Bvh8Node>,
    pub tri_slots: Vec<TriSlot>,
    pub soa_chunks: Vec<FlatTriangles>,
    /// One bitmask per SoA chunk (8 triangles per chunk). Bit i = 1 means triangle i
    /// in that chunk accepts backface hits (no culling). Self-hit avoidance for these
    /// triangles relies on `tmin = RAY_EPSILON` in the caller, not on det sign.
    pub double_sided_masks: Vec<u8>,
    /// Shadow rays: per node and slot, occluders found through that child (learning only).
    pub occluder_hits: Vec<[AtomicU16; 8]>,
    /// Shadow rays count where occluders are (see `apply_occluder_order`).
    pub learn_occluders: bool,
}

struct BvhStack {
    data: [MaybeUninit<(u32, f32)>; MAX_TRAVERSAL_DEPTH],
    ptr: usize,
}

impl BvhStack {
    #[inline(always)]
    fn new() -> Self {
        Self {
            data: [const { MaybeUninit::uninit() }; MAX_TRAVERSAL_DEPTH],
            ptr: 0,
        }
    }

    #[inline(always)]
    fn push(&mut self, node_idx: u32, t: f32) {
        unsafe { self.data.get_unchecked_mut(self.ptr).write((node_idx, t)); }
        self.ptr += 1;
    }

    #[inline(always)]
    fn pop(&mut self) -> (u32, f32) {
        self.ptr -= 1;
        unsafe { self.data.get_unchecked(self.ptr).assume_init() }
    }

    #[inline(always)]
    fn is_empty(&self) -> bool {
        self.ptr == 0
    }
}

impl BvhInner {
    pub(crate) fn new(
        nodes: Vec<Bvh8Node>,
        tri_slots: Vec<TriSlot>,
        soa_chunks: Vec<FlatTriangles>,
        double_sided_masks: Vec<u8>,
    ) -> Self {
        let occluder_hits = (0..nodes.len()).map(|_| Default::default()).collect();
        Self {
            nodes,
            tri_slots,
            soa_chunks,
            double_sided_masks,
            occluder_hits,
            learn_occluders: false,
        }
    }

    /// Makes the learned order physical: in every node, slots are sorted by the occluders
    /// found through them (stable: ties keep their order), so that plain traversal, in slot
    /// order, tries the likely occluder first. Children are referenced explicitly, so each
    /// node is permuted alone, nothing else moves.
    pub fn apply_occluder_order(&mut self) {
        for (node, hits) in self.nodes.iter_mut().zip(&self.occluder_hits) {
            let h: [u16; 8] = std::array::from_fn(|s| hits[s].load(Relaxed));
            let mut perm: [usize; 8] = std::array::from_fn(|k| k);
            perm.sort_by_key(|&s| std::cmp::Reverse(h[s]));
            let old = *node;
            for (k, &o) in perm.iter().enumerate() {
                node.p_min_x[k] = old.p_min_x[o];
                node.p_min_y[k] = old.p_min_y[o];
                node.p_min_z[k] = old.p_min_z[o];
                node.p_max_x[k] = old.p_max_x[o];
                node.p_max_y[k] = old.p_max_y[o];
                node.p_max_z[k] = old.p_max_z[o];
                node.children[k] = old.children[o];
            }
            for (k, &o) in perm.iter().enumerate() {
                hits[k].store(h[o], Relaxed);
            }
        }
    }

    /// Who holds whom: the parent of every node (`u32::MAX` for the root) and the node
    /// owning every chunk. For walking from an answer back to the root.
    pub fn parent_maps(&self) -> (Vec<u32>, Vec<u32>) {
        let mut node_parent = vec![u32::MAX; self.nodes.len()];
        let mut chunk_owner = vec![u32::MAX; self.soa_chunks.len()];
        for (n, q) in self.nodes.iter().enumerate() {
            for &c in q.children.iter().filter(|&&c| c != u32::MAX) {
                if Bvh8Node::is_leaf(c) {
                    let start = Bvh8Node::leaf_start(c);
                    for k in start..start + Bvh8Node::leaf_count(c) {
                        chunk_owner[k as usize] = n as u32;
                    }
                } else {
                    node_parent[c as usize] = n as u32;
                }
            }
        }
        (node_parent, chunk_owner)
    }

    /// Chunks of every leaf whose box the ray's segment `[tmin, tmax]` enters.
    pub(crate) fn leaf_chunks_along(&self, ray: &InternalRay, out: &mut Vec<u32>) {
        if self.nodes.is_empty() {
            return;
        }
        let mut stack = vec![0u32];
        while let Some(code) = stack.pop() {
            if Bvh8Node::is_leaf(code) {
                out.extend(Bvh8Node::leaf_start(code)..Bvh8Node::leaf_start(code) + Bvh8Node::leaf_count(code));
                continue;
            }
            let node = self.get_node(code);
            let (mut mask, t) = ray.intersect_bvh8(node, ray.tmax);
            while mask != 0 {
                let lane = mask.trailing_zeros() as usize;
                mask &= mask - 1;
                if node.children[lane] != u32::MAX && t[lane] <= ray.tmax {
                    stack.push(node.children[lane]);
                }
            }
        }
    }

    #[inline(always)]
    fn get_node(&self, idx: u32) -> &Bvh8Node {
        unsafe { self.nodes.get_unchecked(idx as usize) }
    }

    #[inline(always)]
    fn get_chunk(&self, idx: u32) -> &FlatTriangles {
        unsafe { self.soa_chunks.get_unchecked(idx as usize) }
    }

    #[inline(always)]
    fn get_slot(&self, idx: u32) -> &TriSlot {
        unsafe { self.tri_slots.get_unchecked(idx as usize) }
    }

    /// Real triangles in a chunk: padding slots have no original index.
    #[cfg_attr(not(feature = "stats"), allow(dead_code))]
    fn real_lanes(&self, chunk_idx: u32) -> u64 {
        let first = chunk_idx as usize * 8;
        self.tri_slots[first..first + 8].iter().filter(|t| t.orig_idx != u32::MAX).count() as u64
    }

    pub fn tree_stats(&self) -> TreeStats {
        let mut st = TreeStats {
            chunks: self.soa_chunks.len() as u64,
            triangles: self.tri_slots.iter().filter(|t| t.orig_idx != u32::MAX).count() as u64,
            node_bytes: (self.nodes.len() * std::mem::size_of::<Bvh8Node>()) as u64,
            chunk_bytes: (self.soa_chunks.len() * std::mem::size_of::<FlatTriangles>()
                + self.tri_slots.len() * std::mem::size_of::<TriSlot>()
                + self.double_sided_masks.len()) as u64,
            ..Default::default()
        };
        if self.nodes.is_empty() {
            return st;
        }
        // The root has no box of its own: it is the union of its children's.
        let root = self.get_node(0);
        let (mut lo, mut hi) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
        for slot in (0..8).filter(|&i| root.children[i] != u32::MAX) {
            let (a, b) = Self::slot_box(root, slot);
            for k in 0..3 {
                lo[k] = lo[k].min(a[k]);
                hi[k] = hi[k].max(b[k]);
            }
        }
        let root_area = area(lo, hi) as f64;
        st.inner_nodes = 1;
        st.sah_nodes = 1.0;
        self.walk(0, 1, root_area, &mut st);
        st
    }

    fn slot_box(node: &Bvh8Node, slot: usize) -> ([f32; 3], [f32; 3]) {
        (
            [node.p_min_x[slot], node.p_min_y[slot], node.p_min_z[slot]],
            [node.p_max_x[slot], node.p_max_y[slot], node.p_max_z[slot]],
        )
    }

    fn walk(&self, node_idx: u32, depth: u32, root_area: f64, st: &mut TreeStats) {
        st.max_depth = st.max_depth.max(depth);
        let node = self.get_node(node_idx);
        for slot in 0..8 {
            let encoded = node.children[slot];
            if encoded == u32::MAX {
                continue;
            }
            st.used_slots += 1;
            let (lo, hi) = Self::slot_box(node, slot);
            let p = area(lo, hi) as f64 / root_area;
            if Bvh8Node::is_leaf(encoded) {
                st.leaves += 1;
                st.max_leaf_chunks = st.max_leaf_chunks.max(Bvh8Node::leaf_count(encoded));
                st.sah_chunks += p * Bvh8Node::leaf_count(encoded) as f64;
            } else {
                st.inner_nodes += 1;
                st.sah_nodes += p;
                self.walk(encoded, depth + 1, root_area, st);
            }
        }
    }

    pub fn closest_hit(&self, ray: &InternalRay) -> Option<Hit> {
        if self.nodes.is_empty() {
            return None;
        }

        record(|s| s.rays += 1);
        let mut result: Option<Hit> = None;
        let mut t_max = ray.tmax;

        let mut stack = BvhStack::new();
        stack.push(0, 0.0);

        while !stack.is_empty() {
            let (encoded_idx, node_t) = stack.pop();

            if node_t >= t_max {
                record(|s| s.culled += 1);
                continue;
            }

            if Bvh8Node::is_leaf(encoded_idx) {
                let start_chunk = Bvh8Node::leaf_start(encoded_idx);
                let chunk_count = Bvh8Node::leaf_count(encoded_idx);

                for i in 0..chunk_count {
                    let chunk_idx = start_chunk + i;
                    let chunk = self.get_chunk(chunk_idx);
                    record(|s| {
                        s.chunks += 1;
                        s.tri_lanes += self.real_lanes(chunk_idx);
                    });
                    trace(Touch::Chunk(chunk_idx));
                    let ds_mask = self.double_sided_masks.get(chunk_idx as usize).copied().unwrap_or(0);

                    if let Some(simd_hit) = ray.closest_triangle_simd8(chunk, t_max, ds_mask) {
                        t_max = simd_hit.hit.t;
                        trace(Touch::Answer(chunk_idx));
                        let tri_global_idx = chunk_idx * 8 + simd_hit.lane;
                        let slot = self.get_slot(tri_global_idx);

                        result = Some(Hit {
                            t:       simd_hit.hit.t,
                            u:       simd_hit.hit.u,
                            v:       simd_hit.hit.v,
                            prim_id: slot.orig_idx,
                            geom_id: slot.geom_id,
                        });
                    }
                }
            } else {
                let node = self.get_node(encoded_idx);
                let (mut bitmask, t_arr) = ray.intersect_bvh8(node, t_max);
                record(|s| s.nodes += 1);
                trace(Touch::Node(encoded_idx));

                let mut hits = [(0u32, 0.0f32); 8];
                let mut hit_count = 0;

                while bitmask != 0 {
                    let lane = bitmask.trailing_zeros() as usize;
                    bitmask &= bitmask - 1;

                    // Guard against empty slots in case a NaN slips through the mask.
                    let encoded = node.children[lane];
                    let t = t_arr[lane];
                    if encoded != u32::MAX && t < t_max {
                        hits[hit_count] = (encoded, t);
                        hit_count += 1;
                    }
                }

                // Insertion sort, farthest first, so the nearest child is popped first.
                for i in 1..hit_count {
                    let mut j = i;
                    while j > 0 && hits[j - 1].1 < hits[j].1 {
                        hits.swap(j - 1, j);
                        j -= 1;
                    }
                }
                record(|s| s.boxes_hit += hit_count as u64);
                for &(encoded, t) in &hits[..hit_count] {
                    stack.push(encoded, t);
                }
            }
        }

        result
    }

    pub fn any_hit(&self, ray: &InternalRay) -> bool {
        if self.nodes.is_empty() {
            return false;
        }
        if self.learn_occluders {
            return self.any_hit_learning(ray);
        }

        record(|s| s.rays += 1);
        let t_max = ray.tmax;
        let mut stack = BvhStack::new();
        stack.push(0, 0.0);

        while !stack.is_empty() {
            let (encoded_idx, _) = stack.pop();

            if Bvh8Node::is_leaf(encoded_idx) {
                let start_chunk = Bvh8Node::leaf_start(encoded_idx);
                let chunk_count = Bvh8Node::leaf_count(encoded_idx);

                for i in 0..chunk_count {
                    let chunk_idx = start_chunk + i;
                    record(|s| {
                        s.chunks += 1;
                        s.tri_lanes += self.real_lanes(chunk_idx);
                    });
                    trace(Touch::Chunk(chunk_idx));
                    let ds_mask = self.double_sided_masks.get(chunk_idx as usize).copied().unwrap_or(0);
                    if ray.any_triangle_simd8(self.get_chunk(chunk_idx), t_max, ds_mask) {
                        trace(Touch::Answer(chunk_idx));
                        return true;
                    }
                }
            } else {
                let node = self.get_node(encoded_idx);
                let (mut bitmask, t_arr) = ray.intersect_bvh8(node, t_max);
                record(|s| s.nodes += 1);
                trace(Touch::Node(encoded_idx));

                // Highest slot pushed first, so slot 0 is popped first: children are tried in
                // slot order (the learned order once applied).
                while bitmask != 0 {
                    let lane = 31 - bitmask.leading_zeros() as usize;
                    bitmask &= !(1 << lane);
                    let encoded = node.children[lane];
                    if encoded != u32::MAX && t_arr[lane] < t_max {
                        record(|s| s.boxes_hit += 1);
                        stack.push(encoded, 0.0);
                    }
                }
            }
        }

        false
    }
}

impl BvhInner {
    /// Shadow rays, counting where occluders are: when a ray finds one, every node on its path
    /// counts the child it came through (see `apply_occluder_order`).
    fn any_hit_learning(&self, ray: &InternalRay) -> bool {
        record(|s| s.rays += 1);
        let t_max = ray.tmax;
        // (node or chunk range, depth, slot in its parent)
        let mut stack = vec![(0u32, 0usize, 0u8)];
        let mut path_node = [0u32; MAX_TRAVERSAL_DEPTH];
        let mut path_slot = [0u8; MAX_TRAVERSAL_DEPTH];
        while let Some((code, depth, slot)) = stack.pop() {
            path_slot[depth] = slot;
            if Bvh8Node::is_leaf(code) {
                for i in 0..Bvh8Node::leaf_count(code) {
                    let chunk_idx = Bvh8Node::leaf_start(code) + i;
                    let ds_mask = self.double_sided_masks[chunk_idx as usize];
                    if ray.any_triangle_simd8(self.get_chunk(chunk_idx), t_max, ds_mask) {
                        for k in 1..=depth {
                            let h = &self.occluder_hits[path_node[k - 1] as usize][path_slot[k] as usize];
                            h.store(h.load(Relaxed).saturating_add(1), Relaxed);
                        }
                        return true;
                    }
                }
            } else {
                path_node[depth] = code;
                let node = self.get_node(code);
                let (mut bitmask, t_arr) = ray.intersect_bvh8(node, t_max);
                while bitmask != 0 {
                    let lane = 31 - bitmask.leading_zeros() as usize;
                    bitmask &= !(1 << lane);
                    let child = node.children[lane];
                    if child != u32::MAX && t_arr[lane] < t_max {
                        stack.push((child, depth + 1, lane as u8));
                    }
                }
            }
        }
        false
    }
}

fn area(lo: [f32; 3], hi: [f32; 3]) -> f32 {
    let d = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
    2.0 * (d[0] * d[1] + d[1] * d[2] + d[2] * d[0])
}
