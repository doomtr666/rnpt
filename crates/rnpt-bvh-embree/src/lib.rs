pub use rnpt_bvh::{Hit, Ray, RayAccelerator};

/// Triangle-mesh geometry. Build with `Geometry::triangle_mesh`.
#[allow(dead_code)]
pub struct Geometry {
    verts: Vec<[f32; 3]>,
    tris:  Vec<[u32; 3]>,
}

impl Geometry {
    pub fn triangle_mesh(verts: &[[f32; 3]], tris: &[[u32; 3]]) -> Self {
        Self { verts: verts.to_vec(), tris: tris.to_vec() }
    }
}

/// Embree-backed BVH scene.
///
/// Enabled only when the `embree` feature is active.
/// Without the feature the struct exists but `commit` / queries are no-ops
/// (the crate still compiles so non-embree builds aren't broken).
pub struct Scene {
    #[cfg(feature = "embree")]
    inner: ffi::EmbreeInner,
    #[cfg(not(feature = "embree"))]
    _geometries: Vec<Geometry>,
}

impl Scene {
    pub fn new() -> Self {
        #[cfg(feature = "embree")]
        { Self { inner: ffi::EmbreeInner::new() } }
        #[cfg(not(feature = "embree"))]
        { Self { _geometries: Vec::new() } }
    }

    pub fn attach_geometry(&mut self, geom: Geometry) -> u32 {
        #[cfg(feature = "embree")]
        { self.inner.attach(geom) }
        #[cfg(not(feature = "embree"))]
        { self._geometries.push(geom); (self._geometries.len() - 1) as u32 }
    }

    pub fn commit(&mut self) {
        #[cfg(feature = "embree")]
        self.inner.commit();
    }
}

impl Default for Scene {
    fn default() -> Self { Self::new() }
}

impl RayAccelerator for Scene {
    fn closest_hit(&self, ray: &Ray) -> Option<Hit> {
        #[cfg(feature = "embree")]
        { self.inner.closest_hit(ray) }
        #[cfg(not(feature = "embree"))]
        { let _ = ray; None }
    }

    fn any_hit(&self, ray: &Ray) -> bool {
        #[cfg(feature = "embree")]
        { self.inner.any_hit(ray) }
        #[cfg(not(feature = "embree"))]
        { let _ = ray; false }
    }
}

// ── Embree FFI (only compiled when feature = "embree") ────────────────────────

#[cfg(feature = "embree")]
mod ffi {
    use super::{Geometry, Hit, Ray};
    use std::ptr;

    type RTCDevice   = *mut std::ffi::c_void;
    type RTCScene    = *mut std::ffi::c_void;
    type RTCGeometry = *mut std::ffi::c_void;

    const RTC_GEOMETRY_TYPE_TRIANGLE: u32 = 0;
    const RTC_BUFFER_TYPE_INDEX:      u32 = 0;
    const RTC_BUFFER_TYPE_VERTEX:     u32 = 1;
    const RTC_FORMAT_FLOAT3:          u32 = 0x9003;
    const RTC_FORMAT_UINT3:           u32 = 0x5003;

    #[repr(C, align(16))]
    #[derive(Clone, Copy, Default)]
    struct RTCRay {
        org_x:f32, org_y:f32, org_z:f32, tnear:f32,
        dir_x:f32, dir_y:f32, dir_z:f32, time:f32,
        tfar:f32,  mask:u32,  id:u32,    flags:u32,
    }

    #[repr(C, align(16))]
    #[derive(Clone, Copy)]
    struct RTCHit {
        ng_x:f32, ng_y:f32, ng_z:f32,
        u:f32, v:f32,
        prim_id:u32, geom_id:u32,
        inst_id:[u32;1], inst_prim_id:[u32;1],
    }
    impl Default for RTCHit {
        fn default() -> Self {
            Self { ng_x:0.,ng_y:0.,ng_z:0., u:0.,v:0.,
                prim_id:u32::MAX, geom_id:u32::MAX,
                inst_id:[u32::MAX], inst_prim_id:[u32::MAX] }
        }
    }

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct RTCRayHit { ray: RTCRay, hit: RTCHit }

    unsafe extern "C" {
        fn rtcNewDevice(cfg: *const std::ffi::c_char) -> RTCDevice;
        fn rtcReleaseDevice(d: RTCDevice);
        fn rtcNewScene(d: RTCDevice) -> RTCScene;
        fn rtcReleaseScene(s: RTCScene);
        fn rtcNewGeometry(d: RTCDevice, ty: u32) -> RTCGeometry;
        fn rtcReleaseGeometry(g: RTCGeometry);
        fn rtcSetNewGeometryBuffer(g:RTCGeometry, ty:u32, slot:u32, fmt:u32,
            stride:usize, count:usize) -> *mut std::ffi::c_void;
        fn rtcCommitGeometry(g: RTCGeometry);
        fn rtcAttachGeometry(s: RTCScene, g: RTCGeometry) -> u32;
        fn rtcCommitScene(s: RTCScene);
        fn rtcIntersect1(s:RTCScene, rh:*mut RTCRayHit, args:*const std::ffi::c_void);
        fn rtcOccluded1(s:RTCScene, r:*mut RTCRay, args:*const std::ffi::c_void);
    }

    pub struct EmbreeInner {
        device: RTCDevice,
        scene:  RTCScene,
    }

    impl EmbreeInner {
        pub fn new() -> Self {
            let device = unsafe { rtcNewDevice(ptr::null()) };
            assert!(!device.is_null(), "rtcNewDevice failed");
            let scene = unsafe { rtcNewScene(device) };
            assert!(!scene.is_null(), "rtcNewScene failed");
            Self { device, scene }
        }

        pub fn attach(&mut self, geom: Geometry) -> u32 {
            unsafe {
                let g = rtcNewGeometry(self.device, RTC_GEOMETRY_TYPE_TRIANGLE);
                let vb = rtcSetNewGeometryBuffer(
                    g, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, 12, geom.verts.len(),
                ) as *mut f32;
                ptr::copy_nonoverlapping(geom.verts.as_ptr() as *const f32, vb, geom.verts.len() * 3);
                let ib = rtcSetNewGeometryBuffer(
                    g, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, 12, geom.tris.len(),
                ) as *mut u32;
                ptr::copy_nonoverlapping(geom.tris.as_ptr() as *const u32, ib, geom.tris.len() * 3);
                rtcCommitGeometry(g);
                let id = rtcAttachGeometry(self.scene, g);
                rtcReleaseGeometry(g);
                id
            }
        }

        pub fn commit(&mut self) {
            unsafe { rtcCommitScene(self.scene); }
        }

        pub fn closest_hit(&self, ray: &Ray) -> Option<Hit> {
            let mut rh = RTCRayHit {
                ray: RTCRay {
                    org_x: ray.org[0], org_y: ray.org[1], org_z: ray.org[2], tnear: ray.tmin,
                    dir_x: ray.dir[0], dir_y: ray.dir[1], dir_z: ray.dir[2], time: 0.0,
                    tfar: ray.tmax, mask: u32::MAX, id: 0, flags: 0,
                },
                hit: RTCHit::default(),
            };
            unsafe { rtcIntersect1(self.scene, &mut rh, ptr::null()); }
            if rh.hit.geom_id == u32::MAX {
                None
            } else {
                Some(Hit {
                    t:       rh.ray.tfar,
                    u:       rh.hit.u,
                    v:       rh.hit.v,
                    prim_id: rh.hit.prim_id,
                    geom_id: rh.hit.geom_id,
                })
            }
        }

        pub fn any_hit(&self, ray: &Ray) -> bool {
            let mut r = RTCRay {
                org_x: ray.org[0], org_y: ray.org[1], org_z: ray.org[2], tnear: ray.tmin,
                dir_x: ray.dir[0], dir_y: ray.dir[1], dir_z: ray.dir[2], time: 0.0,
                tfar: ray.tmax, mask: u32::MAX, id: 0, flags: 0,
            };
            unsafe { rtcOccluded1(self.scene, &mut r, ptr::null()); }
            r.tfar < 0.0
        }
    }

    impl Drop for EmbreeInner {
        fn drop(&mut self) {
            unsafe {
                rtcReleaseScene(self.scene);
                rtcReleaseDevice(self.device);
            }
        }
    }
}
