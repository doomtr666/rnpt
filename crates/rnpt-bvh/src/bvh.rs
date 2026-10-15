use nalgebra::Point3;

use crate::math::{InternalRay, AABB};
use crate::Hit;

const MAX_TRAVERSAL_DEPTH: usize = 64;

#[derive(Clone, Copy)]
pub struct FlatTriangle {
    pub v0: u32,
    pub v1: u32,
    pub v2: u32,
    pub orig_idx: u32,
    pub geom_id: u32,
    pub double_sided: bool,
}

#[derive(Clone, Copy)]
pub struct BvhNode {
    pub aabb: AABB,
    pub left_first: u32,
    pub tri_count: u32,
}

impl BvhNode {
    pub fn is_leaf(&self) -> bool {
        self.tri_count > 0
    }
}

pub struct BvhInner {
    pub nodes: Vec<BvhNode>,
    pub vertices: Vec<Point3<f32>>,
    pub triangles: Vec<FlatTriangle>,
}

impl BvhInner {
    pub fn closest_hit(&self, ray: &InternalRay) -> Option<Hit> {
        if self.nodes.is_empty() {
            return None;
        }

        let mut result: Option<Hit> = None;
        let mut t_max = ray.tmax;

        let mut stack = [0usize; MAX_TRAVERSAL_DEPTH];
        let mut stack_ptr = 0;
        stack[stack_ptr] = 0;
        stack_ptr += 1;

        while stack_ptr > 0 {
            stack_ptr -= 1;
            let node = &self.nodes[stack[stack_ptr]];

            if !node.aabb.intersect(ray, t_max) {
                continue;
            }

            if node.is_leaf() {
                let first = node.left_first as usize;
                let count = node.tri_count as usize;

                for i in 0..count {
                    let tri = &self.triangles[first + i];
                    let v0 = self.vertices[tri.v0 as usize];
                    let v1 = self.vertices[tri.v1 as usize];
                    let v2 = self.vertices[tri.v2 as usize];

                    let mut test_ray = ray.clone();
                    test_ray.tmax = t_max;

                    if let Some(hit) = test_ray.intersect_triangle(&v0, &v1, &v2, tri.double_sided) {
                        t_max = hit.t;
                        result = Some(Hit {
                            t: hit.t,
                            u: hit.u,
                            v: hit.v,
                            prim_id: tri.orig_idx,
                            geom_id: tri.geom_id,
                        });
                    }
                }
            } else if stack_ptr + 1 < MAX_TRAVERSAL_DEPTH {
                // Push both children, in no particular order.
                stack[stack_ptr] = node.left_first as usize;
                stack[stack_ptr + 1] = node.left_first as usize + 1;
                stack_ptr += 2;
            }
        }

        result
    }

    /// No dedicated occlusion query yet: any hit is a closest hit.
    pub fn any_hit(&self, ray: &InternalRay) -> bool {
        self.closest_hit(ray).is_some()
    }
}
