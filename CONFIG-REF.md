## Configuration reference

`betacensor.conf` lives beside the binaries, is written at first run, and is
re-read while running. `BC_CONFIG` (live), `BC_VIDEO_CONFIG` (video) and
`BC_IMAGE_CONFIG` (images) point elsewhere.

### `[global]`

| Key | Meaning |
|---|---|
| `enabled` | master switch; off shows the screen untouched |
| `draw` | puts the overlay in drawing mode; cleared by the core when done |
| `join` | merge overlapping covers into one region |
| `join_waste` | how much new area a join may add (0 = only exact overlaps) |
| `reverse` | mask the screen and reveal detections instead |
| `eye_bar` | cover only the eye band of each detected face |
| `eye_bar_pad` | how far the bar extends past the eyes themselves |
| `eye_bar_effect` | effect for the bar; `off` = it takes the face's effect |
| `eye_bar_face` | also cover the face under the bar |
| `model` | detector to load; swapped live when this changes |
| `landmark_model` | optional; locates eyes inside faces, for the eye bar |
| `seg_model` | optional; clips covers to the subject's outline, not a box |
| `seg_margin` | % of the short side the outline grows by before it clips |
| `effect` | the cover effect (see [Effects](#effects)) |
| `caption` | text for the caption and wall-of-text effects |
| `text_color` | glyph colour: `auto`, `rainbow`, a name, or `#RRGGBB` |
| `strength` | active effect's strength, % of box short side |
| `confidence` | detection floor |
| `dilate` | box growth %, covers motion since capture |
| `sticky_ms` | how long a box outlives its last detection |
| `smoothing` | inward edge ease; lower is steadier, slower to shrink |
| `section_cols` / `section_rows` | detection grid; >1 costs one inference per tile |
| `section_overlap` | fraction of a tile shared with its neighbour |
| `section_full_frame` | also infer the whole screen, for what a tile cannot hold |
| `censor_color` | bar colour: `auto` (sampled), a name, or `#RRGGBB` |
| `outline` / `outline_color` | px of outline drawn outward; 0 = off |
| `shadow` / `shadow_opacity` | px of drop shadow beyond the outline; 0 = off |
| `growth` / `growth_max` | % per second each cover grows, and its ceiling |
| `takeover_ms` | ms a detection must persist before the whole screen is covered |

### `[strength]`

One value per effect, because a block size, a blur radius and a dot pitch are
different quantities and want different numbers. Only the active effect's line
has any bearing.

### `[classes]` and `[classes.inactive]`

`NAME = true|false` per class of the loaded model. Classes belonging to other
models are parked in `[classes.inactive]` so a model swap never loses their
settings, and so the GUI knows which entries are real.

A class may also name the effect that covers it, and pin its own detection
floor with an `@score` suffix:

```ini
[classes]
FEMALE_BREAST_EXPOSED = fill@0.15    # this effect, and covered from 0.15 up
MALE_BREAST_EXPOSED   = true@0.60    # global effect, but only above 0.60
BELLY_EXPOSED         = false
```

The floor exists because one global `confidence` is a single compromise across
classes whose cost of being missed is not the same: the number that stops a
noisy class flickering is far too high for the class the program exists to
cover. A class without a suffix follows `confidence`, and keeps following it
when you move that slider — the suffix is an opt-out, not a snapshot. The
detector is then run at the lowest floor any censored class asks for, so
lowering one class cannot be silently undone inside the decoder.
