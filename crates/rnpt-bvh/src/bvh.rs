use std::mem::MaybeUninit;

use crate::math::InternalRay;
use crate::Hit;

const MAX_TRAVERSAL_DEPTH: usize = 64;

/// Stack entries pointing to a leaf carry its chunk range instead of a node index.
pub const LEAF_FLAG: u32 = 0x8000_0000;

#[derive(Clone, Copy)]
pub struct FlatTriangle {
    pub v0: u32,
    pub v1: u32,
    pub v2: u32,
    pub orig_idx: u32,
    pub geom_id: u32,
    pub double_sided: bool,
}

/// 8 consecutive triangles laid out for SIMD, one lane per triangle: chunk `c`
/// holds triangles `8c..8c+8`. Padding lanes are degenerate and never hit.
#[repr(C, align(32))]
#[derive(Clone, Copy)]
pub struct FlatTriangles {
    pub v0_x: [f32; 8], pub v0_y: [f32; 8], pub v0_z: [f32; 8],
    pub e1_x: [f32; 8], pub e1_y: [f32; 8], pub e1_z: [f32; 8],
    pub e2_x: [f32; 8], pub e2_y: [f32; 8], pub e2_z: [f32; 8],
    /// Bit i set: lane i is double-sided.
    pub double_sided_mask: u8,
}

impl Default for FlatTriangles {
    fn default() -> Self {
        Self {
            v0_x: [0.0; 8], v0_y: [0.0; 8], v0_z: [0.0; 8],
            e1_x: [0.0; 8], e1_y: [0.0; 8], e1_z: [0.0; 8],
            e2_x: [0.0; 8], e2_y: [0.0; 8], e2_z: [0.0; 8],
            double_sided_mask: 0,
        }
    }
}

/// 8-wide node: the bounds of its (up to) 8 children in SoA, tested in one SIMD pass.
/// Unused slots have NaN bounds and never hit.
#[repr(C, align(32))]
#[derive(Clone, Copy)]
pub struct Bvh8Node {
    pub p_min_x: [f32; 8],
    pub p_min_y: [f32; 8],
    pub p_min_z: [f32; 8],
    pub p_max_x: [f32; 8],
    pub p_max_y: [f32; 8],
    pub p_max_z: [f32; 8],
    /// Child references, encoded once at build time: for a leaf,
    /// `first_chunk | (chunk_count << 24) | LEAF_FLAG`; for an internal node, its
    /// index. `u32::MAX` marks an empty slot.
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

pub struct BvhInner {
    pub nodes: Vec<Bvh8Node>,
    pub triangles: Vec<FlatTriangle>,
    pub soa_chunks: Vec<FlatTriangles>,
}

/// Traversal stack without bounds checks: the traversal never pushes more than
/// `MAX_TRAVERSAL_DEPTH` entries.
struct BvhStack {
    // Uninitialized on purpose: every slot is written by `push` before `pop`
    // reads it, so zeroing it was a dead 512-byte memset per ray.
    data: [MaybeUninit<(u32, f32)>; MAX_TRAVERSAL_DEPTH],
    ptr: usize,
}

impl BvhStack {
    #[inline(always)]
    fn new() -> Self {
        Self { data: [const { MaybeUninit::uninit() }; MAX_TRAVERSAL_DEPTH], ptr: 0 }
    }

    #[inline(always)]
    fn push(&mut self, node_idx: u32, t: f32) {
        unsafe { self.data.get_unchecked_mut(self.ptr).write((node_idx, t)) };
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

    // Indices are valid by construction (encoded at build time or derived from
    // in-range lanes), so the hot path skips bounds checks.
    #[inline(always)]
    fn get_chunk(&self, idx: usize) -> &FlatTriangles {
        unsafe { self.soa_chunks.get_unchecked(idx) }
    }

    #[inline(always)]
    fn get_triangle(&self, idx: usize) -> &FlatTriangle {
        unsafe { self.triangles.get_unchecked(idx) }
    }

    pub fn closest_hit(&self, ray: &InternalRay) -> Option<Hit> {
        if self.nodes.is_empty() {
            return None;
        }

        let mut result: Option<Hit> = None;
        let mut t_max = ray.tmax;

        // Each stack entry carries the distance at which the ray enters the node,
        // so nodes behind the closest hit found so far are skipped when popped.
        let mut stack = BvhStack::new();
        stack.push(0, 0.0);

        while !stack.is_empty() {
            let (encoded_idx, node_t) = stack.pop();

            if node_t >= t_max {
                continue;
            }

            if encoded_idx & LEAF_FLAG != 0 {
                let chunk_count = (encoded_idx >> 24) & 0x7F;
                let start_chunk = encoded_idx & 0x00FF_FFFF;

                for i in 0..chunk_count {
                    let chunk_idx = (start_chunk + i) as usize;
                    let chunk = self.get_chunk(chunk_idx);

                    if let Some(simd_hit) = ray.intersect_simd_8(chunk, t_max) {
                        t_max = simd_hit.hit.t;
                        let tri = self.get_triangle(chunk_idx * 8 + simd_hit.lane);
                        result = Some(Hit {
                            t: simd_hit.hit.t,
                            u: simd_hit.hit.u,
                            v: simd_hit.hit.v,
                            prim_id: tri.orig_idx,
                            geom_id: tri.geom_id,
                        });
                    }
                }
            } else {
                let node = self.get_node(encoded_idx);
                let (mut bitmask, t_arr) = ray.intersect_bvh8_dist(node, t_max);

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
                for &(encoded, t) in &hits[..hit_count] {
                    stack.push(encoded, t);
                }
            }
        }

        result
    }

    /// No dedicated occlusion query yet: any hit is a closest hit.
    pub fn any_hit(&self, ray: &InternalRay) -> bool {
        self.closest_hit(ray).is_some()
    }
}
