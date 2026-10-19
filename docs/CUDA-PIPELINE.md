# The CUDA pipeline, in full

How a captured frame becomes a censor bar without a single pixel of the frame
ever crossing PCIe. This is the document to read when the question is *how the
zero-copy claim is actually earned* — which component moves which bytes, what
runs where, and why each stage is shaped the way it is.

Sources for every factual claim are given as `file:line`. Numbers come from
`DESIGN.md` §2/§7 and this tree's own logs unless stated otherwise.

```
[1] xdg-desktop-portal ScreenCast (D-Bus, restore token)      portal.c
[2] PipeWire stream, DMA-BUF buffers                          capture.c
      |  an fd per plane, fourcc, stride, DRM modifier
      v
[3] EGLImage import -> GL texture, resident in VRAM            gpu.c
      |  never read back; aliased, not copied
      v
[4] Change gate: GPU reduction over the input tensor          preprocess.c:182
      |  4 bytes return across PCIe. Static frame -> skip everything
      v
[5] Preprocess compute-pass: letterbox + normalise +
    planarise straight into the inference tensor (SSBO)       preprocess.c:145
      |  VRAM -> VRAM; tensor never becomes host memory
      v
[6] GL->CUDA bridge: cudaGraphicsGLRegisterBuffer,
    MapResources -> device pointer                            preprocess.c:90-133
      |  the SAME buffer ORT sees; still no copy
      v
[7] ONNX Runtime, CUDA execution provider,
    input tensor spoken for over device memory                detect.c:810-827
      |  ~1 KB returns across PCIe: NMS-ready boxes + scalars
      v
[8] Decode / class policy / NMS / tracker / smoothing  (CPU)  detect.c, track.c
      |
      v
[9] Overlay render pass, sampling the frame already in VRAM   overlay.c
```

The per-frame PCIe budget is **~1 KB**: 4 bytes for the change gate, the box
list for the compositing decision, and nothing else. The frames themselves
(14.7 MB each at 2560×1440 RGBA) traverse this entire diagram without leaving
video memory.

---

## 1. Getting the frame: portal, PipeWire, DMA-BUF

`src/portal.c` talks to `org.freedesktop.portal.ScreenCast` over plain sd-bus:
source type Monitor, `persist` mode, and a `restore_token` round-tripped from
`.restore_token` so the consent dialog appears once per session rather than
every run (`portal.c:262-296`). The portal hands back a numbered node and a
writer fd.

`src/capture.c` then drives that fd as a PipeWire stream. The crux is buffer
negotiation: the stream takes the parameters the compositor's node offers and
consumes **DMA-BUF** buffers — file descriptors that alias buffers the
compositor already allocated in video memory. Each frame's metadata carries the
`fourcc`/stride/planes and, critically, the **DRM format modifier**, the
encoding of *how* the buffer is laid out in memory. On the reference machine
that modifier was `0x03000000004fe014` — a tiled, compressed NVIDIA layout, not
something a CPU can treat as rows of pixels.

The fd is a handle, not the data: it is handed to PipeWire via
`pw_context_connect_fd`, the buffers it names are mapped by the GPU, and
nothing inside them has ever been read by this process.

**Why?** The alternative — PipeWire's SHM path — delivers the frame into a
malloc'd ring buffer, and the design is over before it starts: you are then
committed to at least one full-frame copy *and* an upload per frame, because
from CPU memory the only road to inference goes over PCIe. Every good number
later in this document (the 0.31 ms preprocess, the ~1 KB budget, the 137
inferences/s at an idle desktop) is downstream of refusing SHM. The
cost is that DMA-BUF requires a compositor that provides it via the portal
(KDE/KWin, wlroots) — that is a deliberate dependency, not an accident.

**Why it's fast?** Because nothing is consumed. A DMA-BUF import does not move
14.7 MB; it hands your GPU driver a name for memory it already owns. The frame
is in VRAM before we ever see it, courtesy of the compositor, which produces a
frame at 180 Hz at a cost we do not pay.

---

## 2. Import: EGLImage as a texture alias

`src/gpu.c` holds the EGL context. `bc_gpu_import()` (`gpu.c:330-478`) turns
one frame's metadata into a texture:

1. Build an `EGL_LINUX_DMA_BUF_EXT` attribute list — plane fds, pitches,
   offsets (`gpu.c:388-430`).
2. Push the DRM modifier in as the two EGLAttrib halves it is
   (`gpu.c:424-431`). The code comments this exactly right at
   `gpu.c:420-423`: the modifier is not optional on NVIDIA — and
   `DESIGN.md` §7 phase 1 records the expensive lesson empirically. Ignoring
   the modifier does not produce a *slow* pipeline, it produces a *garbled*
   one, because there the modifier decoded to a tiled, compressed layout no
   CPU could ever have walked.
3. `eglCreateImage` → `glEGLImageTargetTexture2DOES`
   (`gpu.c:433-464`), which leaves a `GL_TEXTURE_2D` owned by a slot in an
   import table keyed on (fourcc, modifier, size) — the next frame from the
   same buffer set skips re-import entirely.

Note what this is not: not a copy that gets made once and refreshed. The
texture *is* the compositor's buffer, viewed through GL. There is no
"original" anywhere — that word would only have meant something if the data
had ever been in two places.

**Why?** Because EGL is the crossover point between everything upstream of
the tensor and everything downstream of it, and the only API on the system
where that crossover is one call. Vulkan was rejected deliberately
(`DESIGN.md` §3[2]): its external-memory path needs device-memory-type
negotiation, handle plumbing and manual barriers — more code for no
additional guarantee on a pipeline that has one queue and one frame texture.

**Why it's fast?** The import cost is driver bookkeeping, not bandwidth. It is
also *cached*: `layout_matches()` (`gpu.c:298-303`) means the steady-state cost
of getting a fresh 14.7 MB frame into GL is zero, because it was never
*out* of GL.

---

## 3. Building the tensor in place: the preprocess pass

`src/preprocess.c` owns the SSBO that ONNX Runtime will eventually read — one
structure of `3 × net_w × net_h` floats per tile, laid out **NCHW planar**
(channel-major) because that is what the YOLO exports expect
(`preprocess.h`; the write indexing is `t[n + i]` per channel at
`preprocess.c:167-171`).

`CS_SRC` (`preprocess.c:145-172`) is one GL compute shader that does three
things in a single dispatch:

- **Letterbox** — for each output texel `p`, the input coordinate is
  `(p + 0.5 - pad) / scale` (`preprocess.c:160`), sampling the import
  texture directly. Where the source does not reach the input (the letterbox
  bars), it writes the `114/255` grey Ultralytics models were trained to
  expect (`preprocess.c:163`).
- **Normalise** — `(c - mean) * invstd` per channel (`preprocess.c:166`), with
  the default being identity — the subtraction and multiply land in the same
  register as the 0..1 conversion of the texture fetch.
- **Planarise** — it is the *output address* that picks the channel slice:
  doing the transpose here is free, whereas doing it on the CPU after any
  readback would be a pass over ~184k floats per tile (`preprocess.c:137-139`,
  `167-171`).

One dispatch per tile: at a `3×2` grid plus full frame, `bc_preproc_run()`
(`preprocess.c:559-602`) issues seven dispatches with different `uScale`,
`uPad`, `uBase` — every tile into its slice of the same SSBO. The per-tile
scale is the recall lever: a 3×2 grid at 10% overlap views the screen at
~2× the pixel detail of a single 640×384 letterbox, and the full-frame tile
rides along so anything wider than one tile is not lost (measured: grid alone
detected 0/2146 frames, grid + full frame detected 90.5% — `NEXT-SESSION.md`).

**Why?** The tensor has to be *built* — resized, normalised, channel-major.
Almost every other tool in this niche does that on the CPU, between a capture
readback and an inference upload, paying for the frame's bandwidth twice. A
compute shader is the form that lets the work happen where the frame already
lives, and it is also the reason no `.cu` file ever had to exist: GL 4.6
compute covers the whole preprocessing stage, so `nvcc` stays out of the build
entirely (the Makefile links CUDA's runtime library for the GL↔CUDA bridge,
which is §5 — nothing else in the project compiles as CUDA), and the shaders
remain plain GLSL text the driver compiles at startup.

**Why it's fast?** One dispatch per tile over a 640×384 input is
microseconds — 0.31 ms measured mean for import + preprocess + readback *on the
old path that also read back* (`DESIGN.md` §2 table). It is fast because it is
one pass (no intermediate texture, no second kernel), because it reads a
texture that is a live alias of the compositor's buffer (no upload), and
because its output — the SSBO — is the very buffer the next stage consumes
(no staging copy between "resize" and "inference"). The letterbox maths being
in the shader rather than a library call is the same principle applied at line
level.

---

## 4. The change gate: four bytes, once per frame

`CS_DIFF_SRC` (`preprocess.c:182-199`) is a reduction over the tensor SSBO
against a `prev` buffer holding the last *inferred* frame: "how many elements
differ by more than eps". The result is one `uint` read back per frame
(`preprocess.c:637-639`), hence the subtitle of this section: the only thing
that crosses PCIe in this stage is **four bytes**.

The contract around it is more precise than the idea, and the precision is the
point:

- The reference frame is advanced by `bc_preproc_commit()`
  (`preprocess.c:716-749`) — a VRAM-to-VRAM `glCopyBufferSubData`, and it is
  called **before** the SSBO is handed to CUDA (`bc_main.c:690-693`), because
  once the buffer is mapped into CUDA's address space, a GL copy out of it is
  undefined.
- The commit-then-compare ordering answers the correct question: "changed
  since the last frame we actually *inferred*", not "since the last one we
  saw", so a gated frame never hides a change that no detector ever examined.
- If the gate *errors*, it returns 1.0 — "all of it changed"
  (`preprocess.c:686-689`) — which sends the caller to inference. The gate can
  only fail in the direction that keeps the screen covered.
- If a committed frame then fails to be inferred (the CUDA handoff fizzles),
  the reference is thrown away (`bc_main.c:697-704` → `bc_preproc_gate_reset`),
  for the reason in the comment at `bc_main.c:697-701`: advancing the
  reference past a frame no box was derived from would let a static screen
  pass as "already judged".

Measured on the idle desktop: **69.4% of frames gated**, inference rate
137.7 → 40.4 runs/s — the change gate is what turns "a 1080 Ti at 40 Hz
inference" into "a 1080 Ti at fractional idle load" (`DESIGN.md` §7 phase 4).
`BC_VERIFY_GATE=1` cross-checks the GPU reduction against a CPU loop each frame
(`preprocess.c:649-674`) — the test exists because the failure mode of a gate
that silently over-reports "unchanged" is exactly the one thing nothing else
would catch: stale boxes on live content.

**Why?** Two reasons, in order of how they appear in `preprocess.c:56-65`:
(a) because "how much did the frame change" is a reduction, and reductions
belong where the data already is; (b) because the *original* version of the
gate compared on the CPU, which meant dragging the whole tensor across PCIe
every frame to answer yes/no — measured at a 3×2 grid that was 668 readbacks of
~29 MB supporting only 13 inferences, and idle power to laugh at. The gate was
always the right idea; making it VRAM-resident is what made it a bargain
instead of a tax.

**Why it's fast?** On a static frame it is the difference between "run YOLO"
and "read one integer". 69% of desktop frames are static, and the gate turns
each of those into one shader dispatch and one scalar — the
GPU never touches the NN, the CPU never touches a pixel. It is also *exact*,
not heuristic: it compares the detector's own input, so an "unchanged" verdict
is provably an identical tensor, which means skipping inference is not an
approximation of running it (`DESIGN.md` §7 phase 4, change gate, final
paragraphs).

---

## 5. The GL↔CUDA bridge: where zero-copy actually happens

This is the stage the pipeline is named after. One buffer — the SSBO the
preprocess pass writes — becomes readable by ONNX Runtime's CUDA provider
without ever leaving VRAM:

```c
cudaGraphicsGLRegisterBuffer(&p->res, p->ssbo, RegisterFlagsReadOnly)  // :118-119
```

- Registered once per buffer; a grid change dumps and retakes the
  registration, because re-specifying storage under a live registration is not
  allowed and re-registering is cheaper than reasoning about stale handles
  (`cuda_unregister` / `alloc_ssbo`, `preprocess.c:84-104`).
- `cudaGraphicsRegisterFlagsReadOnly` is load-bearing, not decoration
  (`preprocess.c:111-117`): CUDA is promised the buffer is only ever *read*.
  A stray write on the inference side is then an error instead of the next
  frame's input being silently corrupted. Programs are built out of contracts
  this strict or they are not built out of contracts at all.
- Per submission, `bc_preproc_device_ptr()` (`preprocess.c:819-887`) maps the
  resource, takes the `void *` CUDA now addresses it by, and refuses anything
  short of the expected byte count — a short mapping would be an inference
  fault in someone else's memory rather than a wrong answer, and the readback
  path below catches the fallback.

Three ordering subtleties carry the whole stage, and all three are named in
situ:

1. **`glFinish()` before map** (`preprocess.c:840-848`). The dispatch is
   *queued*, not run. `cudaGraphicsMapResources` orders CUDA work against GL
   on the stream it is given, but ORT runs on its own stream, so the guarantee
   does not reach the inference. The cost is zero, because the call this
   replaces (`glGetBufferSubData`) blocked here too.
2. **The gate's reference is copied before the map** (§4), because GL may not
   touch a mapped buffer.
3. **The unmap is an ownership handoff, and the worker is the borrower.**
   `bc_detect_busy()` (`detect.c`) is the one-bit truth of "the tensor is
   being read by inference". `bc_main.c:2794-2812` will not rewrite the SSBO
   while it is, and `bc_preproc_gpu_release()` (`preprocess.c:889-898`) is
   called from the frame loop the moment the outcome is collected — the
   buffer returns to GL's name. The comment at `bc_main.c:2796-2812` states
   what this code's author put there deliberately: "Zero-copy removed the
   memcpy... so the exclusion has to be *real* rather than incidental."

The worker-side contract mirrors it (`detect.c:1458-1496`): a submission is
refused outright if the worker is busy (`busy || pending`); the caller keeps
the mapped pointer; a metadata snapshot (`work_conf`) rather than the live
value is what the run sees, so a slider moved mid-inference cannot change the
floor under a run already underway.

**Why?** Because the alternative exists on this very tree and its cost is in
the log line at `preprocess.c:780-782`: `BC_NO_ZEROCOPY=1` sends the tensor
"via host memory", and that path still works — it is what a non-CUDA machine
runs. The bridge exists because the *difference* between those two paths is
~29 MB of PCIe traffic per inference at 640m — up and back — plus the
synchronisation stall the readback imposes between preprocess and inference,
and because once the tensor is nothing but "memory that is already on the
device", the `OrtMemoryInfo` dispatch (§6) is four lines.

**Why it's fast?** Every frame's 29 MB of tensor never leaves the GPU. The
measured contrast is brutal when you toggle it: the host path moves each tile
twice (VRAM→host→VRAM across the same PCIe lane), pins a full frame's worth of
DRAM per submission, and inserts a `glGetBufferSubData` stall that serialises
the preprocess against the inference — the live costs are the difference
between 640m at ~16 ms per inference and 640m as "the accurate but unusable
model". It is one API boundary, not an abstraction: an EGLImage in GL is
already a CUDA-addressable range mapped through the driver's allocator rather
than copied by anyone.

---

## 6. ONNX Runtime: the input is told the truth about where it lives

`src/detect.c` builds the ORT session with a three-step best-effort ladder, all
three of which are covered by logs the operator can search for:

1. Append the CUDA execution provider (guarded by `BC_NO_CUDA`), logging
   either `detector: CUDA execution provider enabled (device 0)` or
   `CUDA EP unavailable (...)` with the ORT status string
   (`detect.c:1083-1095`).
2. **The session creation itself is the second test** — appending the EP only
   proves the provider library *loaded*, not that a CUDA context can be
   created (a driver mismatch reports itself as "out of memory" from
   `cudaSetDevice`). A failed creation on a GPU session is retried with fresh
   CPU options (`detect.c:1108-1133`), and the code documents the eminently
   reasonable bug the first version had: a GPU problem should never stop the
   censor from *starting*. Worse-than-slow is the one thing
   `DESIGN.md` §6 forbids.
3. When the session is genuinely on the CUDA EP, a second `OrtMemoryInfo`
   describing **device** memory is created (`detect.c:1145-1159`) —
   `CreateMemoryInfo("Cuda", OrtDeviceAllocator, ...)`. If this build of ORT
   does not know that allocator name, the failure is a WARN and zero-copy
   disappears, not the run.

The inference call itself (`run_tile`, `detect.c:809-886`) is short, and the
zero-copy change is smaller than it looks:

```c
OrtMemoryInfo *mem = on_device ? d->dev_mem : d->mem;
api->CreateTensorWithDataAsOrtValue(mem, tensor, ...);   // :827-833
api->Run(session, ..., &in, out);
```

ORT reads the `OrtMemoryInfo` and, knowing the graph is on a CUDA device and
the input is a device pointer under `OrtDeviceAllocator`, skips its own H2D
copy. That one decision is the whole "IoBinding" effect: the README's phrase
"IoBinding on a device pointer" is shorthand for what the code does, which is
to *describe the memory rather than bind it* — same mechanism from ORT's side,
one fewer API surface.

Note what is deliberately *not* done: the outputs are **not** bound to
device memory (the comment at `detect.c:822-826`): "the boxes have to be read
by the CPU", so binding them would move the copy without removing it.
NMS policy, tracker, thresholding all make a per-run box read mandatory, so
the design keeps outputs on the host path rather than pretending a second
device residency would pay for itself.

The one remaining consideration is scheduling: inference runs on a worker
thread (`detect.c` worker block, `run_once`), so the 179.4 fps present loop
never blocks on the model, and submissions are skipped, not queued, when the
worker is still busy — a dropped frame is free on this pipeline precisely
because the overlay is live and the tracker carries boxes between inferences;
there is no benefit in queueing for a "latest frame wins" model that has
already moved on.

**Why?** ORT is the pragmatic inference runtime for an ONNX-first pipeline:
every model the project runs is an Ultralytics-style ONNX export, and the CUDA
EP is the only inference backend that supports this generation of hardware
without an extra build artefact. The alternative, TensorRT, is analysed and
rejected in `DESIGN.md` §1.4 — version-capped at 8.6.1 on Pascal, which makes
it a frozen 2023 dependency for an INT8/dp4a win (`DESIGN.md` §1.2) that the
project never needed once the CUDA EP numbers above landed. FP32-on-ORT is
where this design sits; TensorRT and INT8 are recorded as the plan they remain,
not claims the code makes.

**Why it's fast?** Inference at FP32 on a 1080 Ti runs at a speed the display
pipeline never has to compete with, and `ORT_ENABLE_ALL` means the runtime
executes a pre-fused, allocator-planned graph rather than the naive one shipped
in the `.onnx` file. Combine that with never paying an H2D copy (§5) and the
entire cost of "inference" as seen from the frame loop is one `Run` call over
a device tensor — 15.7 ms for 640m, 6.4 ms for 320n, with outputs already back
as host tensors they were going to be read here anyway (`ROADMAP.md` [1] table).

---

## 7. What comes back: boxes, not bytes

There is no stage eight by design. The output ONNX Runtime produces on the
host (`detect.c:852-886`) is the raw detector grid, which `decode()` turns
into per-class boxes with logits, and by the time §8 sees them they have
already passed the class policy at `BC_CONF=0.35` (with the measured note at
`detect.c:27-35` that 0.20 was the observed noise floor of an ordinary
desktop), NMS at IoU 0.45, tile-seam rejoin, and the IoU tracker's
constant-velocity carry.

That is the whole reason the CPU is in the loop at all and the proof of the
architectural rule: from the CPU's seat, the frame is not 14.7 MB, it is
"a few dozen boxes". `DESIGN.md` §1.6 says the same thing as a warning: if
you find yourself writing AVX2 pixel code, the design has already been
violated upstream.

**Why?** Everything a censor needs to know is *where*, not *what* the pixels
were. The tracker + sticky/hysteresis stage exists not merely for smoothness
but because it is what decouples the *capture/current-overlay* bus from the
*inference* bus: boxes carry and predict at frame rate, inferences run at
model rate.

**Why it's fast?** A frame's 14.7 MB never had to answer any question the
CPU asked. CPU work at this point is tens-of-kilobytes scale, which is why
16 threads and AVX2 are described as *ample* rather than deployed
(`DESIGN.md` §1.6) — the CPU is not the accelerator here, it is the
bookkeeper the accelerator reports to.

---

## 8. The overlay renders from the same VRAM

`src/overlay.c` never uploads an image of the screen. The render pass binds
the *same import texture* from §2 (`bc_gpu_texture()` is handed to
`bc_overlay_set_source` in the frame loop, `bc_main.c:2676-2684`) as the
input of its sampling shader, and the effects are per-fragment reads of it:
a box-blur/Kawase, a mosaic, a fill, an RGB-shifted copy, a halftone —
all samples, not pixels.

**Why?** Covering a region by sampling the *already-resident* frame is the
only honest way to keep the §0 rule at both ends of the pipeline. Three stages
(the compositor's, the preprocess, the overlay) are reading the same 14.7 MB;
the alternative to sharing it is three uploads or three copies, and three
frame-latency penalties.

**Why it's fast?** Present is vsync-bound at 179.4 fps (`DESIGN.md` §7
phase 2) — the overlay costs one present per refresh, no more, because there
is no upload on the critical path to bound it. The overlay's own surface
changes are between-frames metadata, not pixel transfers.

---

## 9. The offline passes: same tools, different delivery

`bc-video` and `bc-image` link neither PipeWire nor Wayland
(`Makefile` offline objects; the README says as much). The frame source is
ffmpeg's decode→rawvideo stream into process memory (`video.c`), the GL
context is offscreen (`offline.c` offscreen g), and the pipeline is:

```c
bc_preproc_run(pp, g->src_tex, ...) && bc_preproc_readback(pp, tensor)
    && bc_detect_submit(det, tensor)                    // offline.c:737-739
```

— i.e. **the readback path, unconditionally.** The tensor comes down to host
RAM, and the CUDA EP's own H2D copy happens inside `Run`. The preprocess is
still the same compute pass, and the model still runs on the CUDA EP; what is
*not* carried over is the zero-copy delivery, because it is a live-mode-only
nicety — the ownership discipline it exists to enforce (§5, item 3) protects a
frame loop that the offline passes do not have.

**Why?** In the offline passes no one is watching in real time, so a 30 ms
tensor readback is not the product's failure mode — image-mode's cost is
"seconds per 100 images", and the same `bc_off_detect` contract
(`offline.h:117-127`) enforces the fail-closed ledger there that §10 below
applies to the live path. The bridge's discipline (map/unmap ownership,
busy-exclusion) exists to protect the frame loop's latency; shipping that
machinery to a process without a frame loop would be complexity paying for
nothing.

**Why it's fast?** The same reason plus a hedge: on a 1080 Ti the model is
still the bottleneck in offline mode (16 ms per image at 640m on the CUDA EP,
`ROADMAP.md`), so saving 1–2 ms of readback per image would barely move the
amortised total. The change gate is *also enabled here* — a still input image
still gets gated, one inference per distinct content, not per retry.

---

## 10. The degradation ladder: every way to lose the GPU and still run

Because `DESIGN.md` §6 ("all must fail closed") includes *slow down, do not
die*, the GPU path is wrapped so that every one of these is a log line, not a
crash:

| Losing | When | Falls back to | Log to find |
|---|---|---|---|
| `NO_CUDA=1` at build | Makefile probe | CPU inference; `BC_HAVE_CUDA` never defined | `make config` |
| toolkit absent | build | same | `CUDA toolkit: not found` |
| `BC_NO_CUDA` | runtime | CUDA EP not appended | `CUDA EP unavailable` or skip |
| EP appended but session fails | runtime | CPU session rebuilt clean | `CUDA session failed (...) -- retrying on CPU` |
| `dev_mem` refused by ORT build | runtime | input via host memory | `no CUDA memory descriptor` |
| `BC_NO_ZEROCOPY` | runtime | readback path, interop never taken | `tensor goes via host memory` |
| `cudaRegisterBuffer` fails | runtime | readback; never re-asked per frame | `CUDA cannot see the tensor buffer` |
| map/device-ptr/short buffer | runtime | readback of an unmapped, still-GL-owned SSBO | `could not map the tensor` / `mapped tensor is N bytes` |
| gate/GL error | runtime | `bc_preproc_change_frac` returns 1.0 | `change gate failed -- treating the frame as new` |
| `BC_FAULT_DETERR` | runtime | every run fails; the verdict deadline blanks | `det: fail from now` (`detect.c:1295-1301`) |
| model swap | runtime | screen blanked while the session rebuilds (~1 s) | `DESIGN.md` §6 table |

Every one of them preserves the invariant that matters: a run the detector
actually answered for, on whatever path it took. What the frame loop *never*
accepts is an empty result standing in for a failed run — a failed ORT call
publishes `BC_DET_FAILED`, not "no boxes" (`detect.c:769-794`), and that is
what triggers the fail-closed blank (`DESIGN.md` §6).

**Why?** Because a censor that will not start because a graphics library call
failed is strictly worse than a censor that runs slow — and `DESIGN.md` §6
says the same about every other failure in the system. The ladder is also what
makes the
CUDA path *safe to port to anything*: an AMD machine runs this entire
document with every CUDA stage on its fallback, at the price of tensors
through host memory, on the same code object.

**Why it's fast?** N/A by design — this section is not a performance claim.
It is the reason the performance sections can be bragged about
elsewhere: the fast path is not load-bearing.

---

## 11. Measured numbers on the whole pipeline

Reference machine: Ryzen 7 5700X, GTX 1080 Ti (sm_61), DP-3 at
2560×1440/180 Hz, CUDA 12.9, ORT 1.20.1 CUDA EP (`DESIGN.md` preamble).

| Stage | Measured | Bounded by |
|---|---|---|
| DMA-BUF import + preprocess + readback (320n) | 0.31 ms mean, 2.1 ms max | driver, not bandwidth |
| Same, 640m @ 640×384 + 3×2 grid | 0.85 ms mean, 4.7 ms max | tile count |
| Inference 320n | 4.8 ms **CPU** / ~6.4 ms CUDA | model |
| Inference 640m | 120.6 ms **CPU** / 15.7 ms CUDA | model/precision |
| Change gate | 69.4% idle frames gated, 137.7 → 40.4 inf/s | content |
| Present (overlay) | 179.4 fps, vsync | compositor refresh |
| Track churn / box jump | 0.0/s, 11.7 px (640m) | tracker settings |

Two precision notes in the code are worth carrying into any tuning
conversation (both live in `DESIGN.md` §1, both learned on real silicon):

- **fp16 does not exist here as a speed option.** On sm_61/GP102, FP16 is a
  1/64-rate compatibility path. The options are FP32 or an INT8/dp4a story
  that has not been built (§10 of `DESIGN.md` remains a plan); the numbers
  above are FP32.
- **CUDA 12.9 or death.** CUDA 13 removed this GPU's compute capability
  (`DESIGN.md` §1.3), so the toolkit is pinned and the README says as much.

---

## 12. One-sentence summary, per stage

- **Capture**: the frame arrives already in VRAM, as an fd nobody has read.
- **Import**: GL borrows that memory as a texture — the modifier is the
  difference between correct and garbage.
- **Preprocess**: resize/normalise/planarise is one GPU pass whose output is
  the inference buffer; the "original frame" never had to survive to any
  other address.
- **Gate**: a static frame answers "no change" in 4 bytes and skips the
  model, exactly and provably.
- **Bridge**: the SSBO is registered read-only with CUDA, mapped per
  submission, and given back as soon as the box count is known — ownership,
  not copying.
- **ORT**: the input tensor is described as device memory, so the runtime
  skips its own upload; outputs *stay* host-side, because the CPU must read
  them anyway.
- **Boxes out**: the CPU's whole world is ~1 KB of coordinates per frame.
- **Overlay**: covers what it finds by sampling the frame it already shares
  with the detector.
- **Offline**: same engine, tensors delivered by the readback path because
  nobody trades latency in that seat.
- **Fallbacks**: every CUDA failure downgrades some throughput and no
  correctness; the log tells you which line of the ladder the run took.
