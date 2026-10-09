# Python / PyTorch / Nanobind Bindings Guidelines

> **Scope:** Applies to modules in `src/bindings/` and `python/`.

---

## 1. Zero-Copy VRAM Pipeline

* **On-Device Exclusivity:** `observations`, `rewards`, `terminated`, `truncated`, and `actions` tensors must reside exclusively in GPU memory (`torch::Tensor` / Nanobind `ndarray`).
* **No Host-Device Copies:** Zero `cudaMemcpy` (Host $\leftrightarrow$ Device) in simulation step loop.
* **Direct Export:** Use `torch::from_blob` pointing directly to device SoA arrays with appropriate strides.

---

## 2. Interface Standards

* **Static Contracts:** Fixed batch dimensions ($N$ arenas, up to 8 cars per arena, 1 ball, 34 boost pads).
* **Exposed Flags:** Expose boolean/uint8 state flags (`is_on_ground`, `has_jumped`, `is_supersonic`, `is_demoed`) directly via indexable tensors without serialization overhead.
