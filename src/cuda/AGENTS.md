# Physics & CUDA Kernel Guidelines (RocketSim-CUDA)

> **Scope:** Applies to all files in `src/cuda/` and `include/rocketsim_cuda/physics/`.

---

## 1. Memory Architecture & SoA Layout

* **Mandatory Structure of Arrays (SoA):** `Array of Structures (AoS)` is strictly prohibited on GPU. All car and ball states must be contiguous arrays (`pos_x[N]`, `pos_y[N]`, etc.) ensuring 128-byte coalesced global memory transactions per warp.
* **Alignment:** All device array pointers must have 16-byte (`alignas(16)`) or 128-byte (cache line boundary) alignment.
* **Pointer Restriction:** Use `__restrict__` on all device pointers for compiler caching and register promotion.
* **Vector Loads:** Use `float4` loads exclusively when all 4 components are consumed simultaneously by the same thread.

---

## 2. Warp Divergence & Instructions

* **Eliminate Divergence:** Minimize data-dependent branches within warps. Favor branchless math: `__fminf`, `__fmaxf`, ternary `cond ? a : b`.
* **Block Uniformity:** Keep block size to multiples of 32 (default: 128 or 256 threads per block).
* **Const-Correctness:** All read-only configuration structs must be `const __restrict__` or passed by value if $\le 16$ bytes.

---

## 3. Suspension Model (`btRaycastVehicle`)

Faithfully replicate RocketSim's custom Bullet vehicle suspension (`btVehicleRL`):
1. **Raycast Query:** 4 suspension rays cast downwards from chassis-local suspension hardpoints.
2. **Contact Point & Normal:** Evaluated against ground/ramps/curved walls via arena SDF and ground plane.
3. **Spring & Damping Equation:**
   $$F_{\text{susp}} = (L_{\text{rest}} - L) \cdot k_{\text{stiff}} \cdot \text{clipped\_inv\_dot} - v_{\text{rel}} \cdot d$$
   where damping factor $d$ switches between compression ($0.83$) and relaxation ($0.88$).
4. **Extra Pushback:** Static object compression compensation matching `resolveSingleCollision`.
5. **Bilateral Friction Formulation:** Sideways and forward wheel impulse resolution identical to `resolveSingleBilateral` and `calcFrictionImpulses`.

---

## 4. Analytical Arena Signed Distance Fields (SDF)

* **Closed-Form $O(1)$ Queries:** Standard Soccar arena static collision queries are closed-form:
  * Rectangular bounds: $X \in [-4096, 4096]$, $Y \in [-5120, 5120]$, $Z \in [0, 2048]$.
  * Corner chamfers: $45^\circ$ vertical bevels with cylindrical fillets.
  * Floor-wall and ceiling-wall ramps: cylindrical arcs with radius $R_{\text{curv}} = 260\text{ UU}$.
  * Goal cavities: oriented bounding boxes with curved posts/crossbars (torus/cylinder segments).
* **Mesh Independence:** No triangle mesh files (`.cm`) in device kernels for standard Soccar.
