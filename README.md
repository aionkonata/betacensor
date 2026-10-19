# betacensor
Real-time screen censorship for Linux/Wayland
**Curb your expectations: this only really works on my machine (CachyOS, KDE Plasma, Wayland). I haven't done much to get it working on other distros/window managers/desktop environments.**

## What it does

- **Live mode.** Captures a monitor, detects, and draws covers over the real
  screen through a separate always-on-top Wayland surface. Nothing about the
  underlying applications changes, and no application needs to cooperate. The
  surface is a fullscreen `xdg-toplevel` by default, which is what lets KWin
  hide it from the capture; `BC_LAYER=1` switches to `wlr-layer-shell`, which
  stacks over fullscreen windows but is visible to the detector's own capture.
- **Video mode.** Batches a file through the same pipeline and writes a
  censored copy, with the original audio copied across untouched.
- **Image mode.** Censors one image, a batch, or whole folders into an output
  folder, covering only the body parts you tick — Pury.fi's batch converter,
  done with the live censor's own detector and effects.
- **Settings panel.** A Qt GUI that edits `betacensor.conf`, which the running
  censor re-reads live. Model, effect, classes, thresholds and colours all
  change without a restart.
- **Panic / pause.** Signal-driven toggles: blank the screen entirely, or drop
  to the raw screen. Each command is its own undo.
- **Fixed regions.** Areas covered no matter what the detector says — typed in
  as coordinates, or traced on screen by hand with the mouse.


## How it works

> Pixels never leave VRAM. Nothing crosses PCIe per frame but box coordinates.

PipeWire hands out DMA-BUF handles to buffers the compositor already allocated
in video memory. Those are imported as GL textures, preprocessed into the
detector's input tensor in place, and sampled directly by the overlay shader.
The only thing read back per frame is the box list (~1 KB) and one scalar for
the change gate. Everything else in the design follows from holding to that
number.

```
portal ScreenCast → PipeWire DMA-BUF → EGLImage → GL texture (VRAM)
   → change gate (GPU reduce, 1 float back; static frame ⇒ skip inference)
   → preprocess compute shader (resize + letterbox + normalize, into the tensor)
   → ONNX Runtime, CUDA EP, IoBinding on a device pointer
   → NMS / class policy / IoU tracker / hysteresis / dilation  (CPU, boxes only)
   → overlay render pass, sampling the captured frame already in VRAM
```

Two things carry most of the performance:

- **The change gate.** Desktop content is static most of the time, and a static
  frame skips the entire inference path.
- **The tracker.** Detection runs at 10–30 Hz while the overlay is live, so
  boxes are carried and predicted between inferences instead of stuttering.
  This is what decouples detection rate from display rate.

Optionally the frame is split into a grid of tiles, each inferred separately
(plus a full-frame pass), which buys recall on small or distant subjects at the
cost of one inference per tile.

## Requirements

**Compositor.** Wayland with an `xdg-desktop-portal` ScreenCast backend, and
`wlr-layer-shell-unstable-v1` for the optional `BC_LAYER=1` overlay mode.
Developed and tested against **KWin**; see [the window
rule](#setup-the-kwin-window-rule) below, which is not optional.

**Build and runtime packages** (pkg-config names): `egl`, `gl`, `libdrm`,
`libpipewire-0.3`, `wayland-client`, `wayland-egl`, and `Qt6Widgets` for the
panels. Plus `wayland-scanner`, a C compiler, and `make`.

**sd-bus**, for the portal, from any of `libsystemd`, `basu` or `libelogind` —
the build takes whichever is installed, so a systemd-free distro works.

**ONNX Runtime**
Any directory matching `vendor/onnxruntime*` is a candidate and one carrying
the CUDA provider wins, so dropping in a different release needs no edit.
`./tools/setup.sh` downloads the right one for the machine it runs on —
matched to the **CUDA runtime version present**.

**CUDA** is optional. The build searches `/opt/cuda`, `/usr/local/cuda` and
`/usr`; finding one defines `BC_HAVE_CUDA` and hands the preprocessed tensor to
ONNX Runtime where it already lies. Without it every path falls back to a host
readback and still works — which is the whole of what an AMD or Intel machine
does. `NO_CUDA=1` forces it off. On NVIDIA, pin CUDA at **12.x**

**ffmpeg** (`ffmpeg` + `ffprobe` on `PATH`) is needed for video and image mode
only. WebP output needs it built with `libwebp`, AVIF with `libaom`.


## Build

```sh
./tools/setup.sh   # first time: fetch ONNX Runtime for this machine, then build
make               # everything (the Qt panels too, when Qt6 is present)
make config        # what the build detected: compiler, sd-bus, ORT, CUDA
make gui           # the three Qt panels only
make bc            # the live censor core only
make video         # the offline video pass only
make image         # the offline image pass only
make test          # unit tests
make dist          # a source tarball to hand to someone else
make clean
```
Missing binaries are built on demand by the `betacensor` launcher, so
`./build/betacensor live` works from a clean tree once `make launcher` has run.


## Setup: the KWin window rule

**Before live mode works, KWin/KDE needs a window rule on window class
`betacensor`** setting:

- **Layer = Overlay** (force)
- **Hide from screenshots and screen recordings** (force, yes)
- **"Match whole window class" — UNCHECKED**

Without it the capture stream contains the overlay's own censor bars, which
closes a feedback loop: the detector sees a censored frame, finds nothing,
drops the box, and the next frame is uncensored again. The visible symptom is
flicker.

---

## Running it

Everything goes through the `betacensor` launcher:

```
betacensor                    censor the screen, with the settings panel
betacensor live [out] [secs]  the same, naming an output and a duration
                              (default: DP-3, 24 hours)
betacensor gui                settings panel on its own
betacensor video              batch a video file, with its own panel
betacensor video IN OUT [m]   batch one file without the panel
betacensor image [IN...]      censor images or folders, with its own panel
betacensor image -o DIR IN... the same without the panel

betacensor panic              blank the screen entirely; run it again to undo
betacensor pause              show the real screen; again to undo
betacensor stop               stop the running censor
betacensor status             is one running, and with what settings
betacensor doctor             what this machine supports, and what that
                              means for the censor
```

With one monitor, no argument is needed. With several, it **refuses to guess**
and lists them: picking the wrong one would draw covers on a screen nobody is
watching and leave the watched one bare, which is the failure this program
exists to prevent. Name it — `betacensor live DP-2` — and pick the same monitor
in the portal dialog.

`panic` and `pause` are toggles sent as `SIGUSR1` / `SIGUSR2` to the running
core, so they work from a second terminal or a hotkey binding. There is
deliberately no `resume`: from outside the process there is no way to tell which
of the two is in force, and a command that sent both would switch on whichever
was off.

`BC_MODEL` overrides the configured model for one run. `run.sh` is a thin shim
kept so old shortcuts keep working.
