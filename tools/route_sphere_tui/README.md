# BRIAN Route Sphere Terminal Dashboard

This directory contains a detached terminal UI for BRIAN training. The left
pane renders optimizer progress, loss and learning-rate curves, throughput,
memory, and route-health metrics. The interface uses a neutral graphite,
gray, and white palette. The right pane is deliberately text-free: it contains
only a compact rotating route sphere, eight `✱` block nodes on an inscribed
cube, a triangulated spherical wireframe, the current path trail, and a white
in-flight transition. Six unmarked support vertices round out the mesh without
being presented as model blocks. The sphere has no separate outer contour. Its
static triangular mesh uses low-contrast 2x4 Braille subpixels, while the live
route uses the same dot raster at higher brightness. Completed route segments
remain visibly brighter than the static mesh, and the active leading segment
approaches white.

## Demo

From the project root:

```bash
PYTHONPATH=src:. python tools/route_sphere_tui/run.py --demo
```

Use `q` or `Esc` to exit and `r` to pause/resume sphere rotation. Press `p` to
switch between the stable spherical display layout and the latest learned
position projection.

## Live Training

Enable telemetry in a train config:

```yaml
terminal_dashboard:
  enabled: true
  interval: 1
  position_interval: 100
  sample_index: 0
  output_dir: terminal_dashboard
```

Start training normally. Once its run directory exists, open a second terminal:

```bash
PYTHONPATH=src:. python tools/route_sphere_tui/run.py \
  --run-dir runs/bdre_rckv_r125_5b_ddp2_legacyval
```

The training process writes one compact rank-0 event after each configured
optimizer step. It never calls the UI and does not synchronize CUDA per route
hop. The UI tails the append-only event stream and replays the captured route
locally at 24 FPS.

## Replay

Recorded telemetry can be replayed without a model or GPU:

```bash
PYTHONPATH=src:. python tools/route_sphere_tui/run.py \
  --replay runs/example/terminal_dashboard/events.jsonl \
  --speed 4
```

The right-side layout is a stable spherical display layout. It represents real
block identities and real transitions but intentionally does not claim to be a
lossless projection of the learned 64-dimensional position geometry.
