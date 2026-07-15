# BRIAN Route Sphere Terminal Dashboard

**Status:** implemented and validated
**Date:** 2026-07-16
**Branch:** `route-sphere-terminal-tui`

## Purpose

The Route Sphere dashboard is a detached terminal visualizer for live BRIAN
training. It combines optimizer telemetry with an animated view of one real
token route while keeping rendering work outside the training process.

The display is split into two regions:

- the left pane contains training progress, ETA, train/validation loss curves,
  learning-rate curve, throughput, CUDA memory, block-load balance, route
  entropy, and route depth;
- the right pane contains no words or numeric labels. It renders only a faint
  rotating sphere, one `*` per route block, faint all-to-all spatial edges, a
  brighter completed path, and a white moving transition.

IN and OUT are intentionally omitted from the sphere. A self-recurrent route
step is shown as a white pulse around the current `*`.

## Isolation and Performance Contract

The training process never starts, imports, or calls the Textual application.
When telemetry is enabled, rank 0 performs the following work after an
optimizer step has completed and CUDA has already been synchronized:

1. Stack one sample's already-computed `selected_actions`.
2. Transfer that small action vector to CPU once.
3. Append one compact JSONL event.
4. Snapshot normalized block positions only at the configured longer interval.

The UI tails this append-only file from another process and replays the route
at 24 FPS. There is no Python callback or GPU-to-CPU synchronization at each
route hop.

A 1,000-event local filesystem microbenchmark measured approximately `0.015
ms` per append and `413 bytes` per normal event. At 76,294 optimizer steps this
is about `31 MB`, before infrequent position snapshots. This is an I/O
microbenchmark, not a claim about full 5B training throughput.

Telemetry is disabled by default for existing configurations and is enabled
explicitly by:

```yaml
terminal_dashboard:
  enabled: true
  interval: 1
  position_interval: 100
  sample_index: 0
  output_dir: terminal_dashboard
```

## Route Geometry

The default screen layout uses a slowly rotating cube inscribed in the route
sphere for eight blocks. Its strict symmetry and orthographic projection keep
node spacing stable while avoiding PCA axis flips. The displayed block
identities and transitions are real; the display layout is not presented as a
lossless projection of learned 64-dimensional positions.

Pressing `p` switches to the latest learned-position PCA. Each refreshed PCA is
orthogonally aligned to the previous frame with Procrustes alignment to prevent
arbitrary sign and axis flips. The stable spherical layout remains the default.

The eight block nodes remain the vertices of an inscribed cube. Six unmarked
axis support vertices turn that structure into a 24-face triangular spherical
mesh; support vertices are display geometry, not model blocks. The dashboard
uses a neutral graphite background and white depth shading throughout. The
compact sphere omits a separate outer contour; its triangular mesh uses
low-contrast 2x4 Braille subpixels for finer geometry. Blocks use a larger
heavy-asterisk `✱` node glyph. A live route may still connect any two blocks:
completed route edges persist as brighter dots, while the active transition
uses a growing near-white dotted segment and a white leading point.

## Entry Points

```text
tools/route_sphere_tui/
src/brian_sphere_llm/train/live_telemetry.py
tests/test_live_terminal_dashboard.py
```

Run the standalone demo:

```bash
PYTHONPATH=src:. python tools/route_sphere_tui/run.py --demo
```

Attach to a run:

```bash
PYTHONPATH=src:. python tools/route_sphere_tui/run.py \
  --run-dir runs/bdre_rckv_r125_5b_ddp2_legacyval
```

Replay a completed stream:

```bash
PYTHONPATH=src:. python tools/route_sphere_tui/run.py \
  --replay runs/example/terminal_dashboard/events.jsonl --speed 4
```

`q` or `Esc` exits. `r` pauses or resumes visual rotation. `p` switches between
the stable sphere and learned-position projection.

## Validation

| Check | Result |
| --- | --- |
| Telemetry config and schema tests | passed |
| Partial JSONL and detached tail recovery | passed |
| Demo and recorded replay sources | passed |
| Sphere geometry and text-free render checks | passed |
| Textual headless render at 140x44 | passed |
| Compact terminal render at 80x24 | passed |
| Tiny BDRE real-forward train/eval telemetry | passed on B200 GPU 4 |
| Tiny BDRE DDP2 rank-0-only telemetry | passed on B200 GPUs 4-5 |
| Telemetry error logs | empty |

The captured live route is the final sequence token from the final microbatch
of a sampled optimizer step. The UI is intentionally delayed by one completed
forward; this is what prevents visualization from stalling route execution.
