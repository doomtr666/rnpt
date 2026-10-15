use std::mem::MaybeUninit;

use crate::math::InternalRay;
use crate::Hit;

const MAX_TRAVERSAL_DEPTH: usize = 64;

const CHILD_ORDER: [[u8; 8]; 8] = [
    [0, 1, 2, 4, 3, 5, 6, 7], // ray_oct 0: (+x,+y,+z)
    [1, 0, 3, 5, 2, 4, 7, 6], // ray_oct 1: (-x,+y,+z)
    [2, 0, 3, 6, 1, 4, 7, 5], // ray_oct 2: (+x,-y,+z)
    [3, 1, 2, 7, 0, 5, 6, 4], // ray_oct 3: (-x,-y,+z)
    [4, 0, 5, 6, 1, 2, 7, 3], // ray_oct 4: (+x,+y,-z)
    [5, 1, 4, 7, 0, 3, 6, 2], // ray_oct 5: (-x,+y,-z)
    [6, 2, 4, 7, 0, 3, 5, 1], // ray_oct 6: (+x,-y,-z)
    [7, 3, 5, 6, 1, 2, 4, 0], // ray_oct 7: (-x,-y,-z)
];

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

    pub fn closest_hit(&self, ray: &InternalRay) -> Option<Hit> {
        if self.nodes.is_empty() {
            return None;
        }

        let mut result: Option<Hit> = None;
        let mut t_max = ray.tmax;

        let oct = (ray.direction.x < 0.0) as usize
            | ((ray.direction.y < 0.0) as usize) << 1
            | ((ray.direction.z < 0.0) as usize) << 2;

        let mut stack = BvhStack::new();
        stack.push(0, 0.0);

        while !stack.is_empty() {
            let (encoded_idx, node_t) = stack.pop();

            if node_t >= t_max {
                continue;
            }

            if Bvh8Node::is_leaf(encoded_idx) {
                let start_chunk = Bvh8Node::leaf_start(encoded_idx);
                let chunk_count = Bvh8Node::leaf_count(encoded_idx);

                for i in 0..chunk_count {
                    let chunk_idx = start_chunk + i;
                    let chunk = self.get_chunk(chunk_idx);
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
                let (bitmask, t_arr) = ray.intersect_bvh8(node, t_max);

                for &slot in CHILD_ORDER[oct].iter().rev() {
                    let slot = slot as usize;
                    if bitmask & (1u32 << slot) != 0 {
                        let encoded = node.children[slot];
                        let t = t_arr[slot];
                        if encoded != u32::MAX && t < t_max {
                            stack.push(encoded, t);
                        }
                    }
                }
            }
        }

        result
    }

    pub fn any_hit(&self, ray: &InternalRay) -> bool {
        if self.nodes.is_empty() {
            return false;
        }

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
                    let ds_mask = self.double_sided_masks.get(chunk_idx as usize).copied().unwrap_or(0);
                    if ray.any_triangle_simd8(self.get_chunk(chunk_idx), t_max, ds_mask) {
                        return true;
                    }
                }
            } else {
                let node = self.get_node(encoded_idx);
                let (mut bitmask, t_arr) = ray.intersect_bvh8(node, t_max);

                while bitmask != 0 {
                    let lane = bitmask.trailing_zeros() as usize;
                    bitmask &= bitmask - 1;
                    let encoded = node.children[lane];
                    if encoded != u32::MAX && t_arr[lane] < t_max {
                        stack.push(encoded, 0.0);
                    }
                }
            }
        }

        false
    }
}
