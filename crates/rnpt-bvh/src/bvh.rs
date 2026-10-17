use std::mem::MaybeUninit;

use crate::math::InternalRay;
use crate::stats::{record, TreeStats};
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
                    let ds_mask = self.double_sided_masks.get(chunk_idx as usize).copied().unwrap_or(0);

                    if let Some(simd_hit) = ray.closest_triangle_simd8(chunk, t_max, ds_mask) {
                        t_max = simd_hit.hit.t;
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
                    let ds_mask = self.double_sided_masks.get(chunk_idx as usize).copied().unwrap_or(0);
                    if ray.any_triangle_simd8(self.get_chunk(chunk_idx), t_max, ds_mask) {
                        return true;
                    }
                }
            } else {
                let node = self.get_node(encoded_idx);
                let (mut bitmask, t_arr) = ray.intersect_bvh8(node, t_max);
                record(|s| s.nodes += 1);

                while bitmask != 0 {
                    let lane = bitmask.trailing_zeros() as usize;
                    bitmask &= bitmask - 1;
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

fn area(lo: [f32; 3], hi: [f32; 3]) -> f32 {
    let d = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
    2.0 * (d[0] * d[1] + d[1] * d[2] + d[2] * d[0])
}
