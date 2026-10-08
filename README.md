# Fast Cutting of Cage Based Deformation with Walk on Spheres

Code for the SIGGRAPH 2026 paper by Hossam Saeed, Michael Lambiri, Teseo Schneider, Derek Nowrouzezahrai and
Paul G. Kry. [Project page](https://hoskillua.github.io/wos_cutting_siggraph26/) ·
[Paper](assets/paper.pdf) · [DOI](https://doi.org/10.1145/3799902.3811051)

A cage based elastic body is cut by a moving blade. Harmonic cage coordinates are computed with walk on
spheres, and each cut step updates only the walks the cut affects instead of re-solving them. The code
handles 3D polyhedral cages (cut by a blade) and 2D cages (cut by a ray).

## Install

Python 3.10+ and an NVIDIA GPU (CUDA) are recommended; `--device cpu` runs everything on the CPU, slowly.
Tested with warp-lang 1.18, numpy 2.2, scipy 1.15 and polyscope 2.6.

```bash
conda create -n woscut python=3.10 && conda activate woscut
pip install -r requirements.txt
```

## Run

```bash
python -m cagecut.app.main                          # starfish, data/star-config.json
python -m cagecut.app.main data/octo-config.json    # another case
python -m cagecut.app.main --cases data2d           # 2D cases (the Case selector lists this folder)
```

Options: `--walks N` (walks per point), `--device cpu|cuda:0`, `--ui full` (every parameter and debug layer).

In the window, **Run all** begins the cut and runs the simulation, **Run sim** only simulates, **Reset** returns to
the uncut body and keeps your blade and material edits. The **Case** section loads any case in the folder.
The **Blade** section (under Cut) edits the blade. *Finite blade* limits it to a segment, and *start inside* lets
it begin inside the material. Keys: `space` run / pause, `n` cut step, `b` begin cut, `r` reset
the simulation. `ctrl` + drag pulls the mesh.

## Repository structure

```
cagecut/
  scene.py        case (config) loading
  pipeline.py     Body: cut -> walk update -> mesh split -> simulation, one step at a time
  cut/            cage cutting: halfedge topology for a general blade (3D) and a knife (2D); CutDelta
  wos/            walk on spheres on the GPU (Warp): walk sets, per-walk caches, incremental update
  embed/          quadrature points, embedded mesh splitting, affine weight correction
  sim/            StVK elasticity with plastic rest strain, implicit step
  geometry/       cage snapshots, polygon helpers
  app/            polyscope viewer
data/  data2d/    the cases of the paper (3D / 2D)
```

Cutting and weights only talk through the `CutDelta` returned by `Cutter.advance`: the swept simplices, the
modified faces, and the vertices the cut created or moved. `Body` in `pipeline.py` ties the pieces together and
can be used from a script:

```python
from cagecut.pipeline import Body
from cagecut.scene import load_scene

body = Body(load_scene("data/star-config.json"))
body.run_cut(0)                  # begin and run cut 0, updating weights incrementally each step
for _ in range(100):
    body.step()                  # one simulation step
verts = body.mesh_positions()    # deformed embedded mesh
```

## Adding your own case

A case is a `.json` file next to its data; it shows up in the Case selector when it is in the loaded folder
(or pass `--cases your_folder`). Relative paths are looked up next to the json first, then in the repository
root.

```json
{
  "cage_file": "my-cage.obj",
  "mesh_file": "my-mesh.obj",
  "quad_points_file": "my-points.npy",
  "cut_spec": "my-cut.json",
  "pinned_faces": [3],
  "face_order": "file",
  "E": 32
}
```

* `cage_file`: closed cage `.obj` with planar faces (triangles, quads or polygons). It is scaled to the unit box
  (`"normalize": false` keeps your coordinates; the blade must then be given in them).
* `mesh_file`: the embedded surface mesh deformed by the cage (optional).
* `quad_points_file`: `.npy` of interior quadrature points, `N x 3` (optional, sampled if missing).
* `pinned_faces`: cage face indices to pin, in `.obj` order with `"face_order": "file"` (`pinned_verts` pins
  vertices instead).
* `E`, `nu`, `dt`, `g` (gravity along y), `rayleigh_alpha`, `rayleigh_beta`: material and time step.
* `cut_spec`: a blade file, a list of them (cuts run in order), or an inline object:

```json
{ "point1": [0.1, 0.5, 0.2], "point2": [0.9, 0.5, 0.2], "direction": [0, -1, 0],
  "cut_amount": 0.02, "bounded": false, "plunge": false }
```

  `point1`-`point2` is the blade line, `direction` the way it sweeps (the part perpendicular to the line is
  used), `cut_amount` its travel per cut step. `bounded` makes the blade the segment `point1`-`point2`
  instead of an infinite line, and `plunge` lets it begin inside the material. Without `direction`, `cut_angle`
  (degrees) rotates a default direction about the line. The Blade section of the viewer places a blade
  interactively, and *Save blade* writes this file.

  The blade line must not be parallel to a cage face it crosses. If you get "face N is parallel to the blade
  line" (typical for an axis aligned blade on an axis aligned box), tilt the blade slightly.

2D cases (see `data2d/`) use a `.npy` (or `.obj`) cage and a knife spec with `point` (or the cage edge
`line_to_cut` and `cut_line_alpha` along it), `angle`, `initial_cut_amount`, plus optional `scripted_cuts` that
start or grow cuts at given simulation steps.

## Citation

```bibtex
@inproceedings{saeed2026woscut,
  title     = {Fast Cutting of Cage Based Deformation with Walk on Spheres},
  author    = {Saeed, Hossam and Lambiri, Michael and Schneider, Teseo and Nowrouzezahrai, Derek and Kry, Paul G.},
  booktitle = {ACM SIGGRAPH 2026 Conference Papers},
  year      = {2026},
  doi       = {10.1145/3799902.3811051}
}
```

MIT license, see `LICENSE`.
