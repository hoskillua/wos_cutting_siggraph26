"""Scene configuration: one schema for 2D and 3D, accepting the legacy JSON configs.

All paths in a config are resolved relative to the config file first, then the repo root
(legacy configs use repo-root-relative paths like "data/star-cage.obj"). Nothing is read
from or written to the current working directory implicitly.
"""

import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

from cagecut import REPO_ROOT
from cagecut.io import legacy_face_order, read_obj, unit_box_transform


@dataclass
class WosParams:
	num_walks: int = 1024
	max_steps: int = 128
	eps: float = 1.0e-3  # termination shell of the outer walk
	eps_ngon: float = 1.0e-4  # termination shell of nested in-face walks (3D n-gons)
	max_steps_ngon: int = 64
	bounding_volume: str = "kdop"  # "kdop" or "aabb"
	distance: str = "bvh"  # "bvh" or "brute"
	seed: int = 1234  # global seed; per-point seeds are derived from it


@dataclass
class Material:
	E: float = 16.0
	nu: float = 0.3
	density: float | None = None  # None: legacy scaling (density = cage rest volume)
	rayleigh_alpha: float = 0.0
	rayleigh_beta: float = 0.0
	gravity: np.ndarray = None  # (dim,)
	dt: float = 0.03


@dataclass
class Scene:
	name: str
	dim: int
	config_path: str | None
	cage_verts: np.ndarray  # (V, dim) float64, rest positions
	cage_faces: list  # 3D: polygon faces (outward CCW). 2D: closed loops (CCW), each an int array
	mesh_verts: np.ndarray  # (M, dim) embedded geometry rest positions
	mesh_tris: np.ndarray | None  # (T, 3) for 3D surface meshes; None for 2D point sets
	quad_points: np.ndarray | None  # (Q, dim) or None (sample at load time)
	quad_points_path: str | None  # where "save quadrature points" writes
	cut_specs: list = field(default_factory=list)  # normalized cut specs, see blade_spec_* below
	schedule: list = field(default_factory=list)  # scripted cut events: dicts {step, action, amount, spec}
	pinned_verts: np.ndarray = None
	material: Material = None
	wos: WosParams = None
	floor_y: float | None = None
	env_verts: np.ndarray | None = None  # optional static environment mesh (display only)
	env_tris: np.ndarray | None = None


def resolve_path(path, config_dir):
	"""Config dir first, then repo root, then as given."""
	if path is None or path == "":
		return None
	if os.path.isabs(path):
		return path
	for base in (config_dir, REPO_ROOT):
		if base is not None:
			cand = os.path.normpath(os.path.join(base, path))
			if os.path.exists(cand):
				return cand
	return os.path.normpath(os.path.join(config_dir or REPO_ROOT, path))


# ---------------------------------------------------------------------------------------------
# Cut specs
#
# 3D blade:  {"kind": "blade", "p1": (3,), "p2": (3,), "direction": (3,) unit, "step_size": s}
#   The blade is the infinite line through p1, p2 swept along "direction" (perpendicular to the
#   line); sweep parameter t is the distance travelled. The cut plane contains line and direction.
# 2D slit:   {"kind": "slit", "edge": (a, b), "alpha": a, "direction": (2,) unit,
#             "initial": first cut length, "step_size": s}
#   Starts at (1-alpha) X_a + alpha X_b on cage edge (a, b) and grows straight along direction.
# ---------------------------------------------------------------------------------------------

def blade_direction_from_angle(p1, p2, angle_degrees):
	"""Legacy cut direction: reference vector orthogonal to the blade rotated about it by angle."""
	v = np.asarray(p2, dtype=np.float64) - np.asarray(p1, dtype=np.float64)
	if np.linalg.norm(v) < 1e-8:
		v = np.array([0.0, 0.0, 1.0])
	v = v / np.linalg.norm(v)
	up = np.array([0.0, 0.0, 1.0])
	if abs(np.dot(up, v)) > 0.999:
		up = np.array([0.0, 1.0, 0.0])
	r = up - np.dot(up, v) * v
	r = r / np.linalg.norm(r)
	th = math.radians(angle_degrees)
	n = r * math.cos(th) + np.cross(v, r) * math.sin(th) + v * np.dot(v, r) * (1.0 - math.cos(th))
	return n / np.linalg.norm(n)


def blade_spec_from_legacy(data):
	p1 = np.array(data.get("point1", [0.0, 0.0, 0.0]), dtype=np.float64)
	p2 = np.array(data.get("point2", [1.0, 1.0, 1.0]), dtype=np.float64)
	if "direction" in data:
		d = np.array(data["direction"], dtype=np.float64)
	else:
		d = blade_direction_from_angle(p1, p2, data.get("cut_angle", 45.0))
	# keep only the component perpendicular to the blade so t is a true distance
	l = (p2 - p1) / np.linalg.norm(p2 - p1)
	d = d - np.dot(d, l) * l
	d = d / np.linalg.norm(d)
	# bounded: the blade is the segment point1-point2 (else an infinite line); plunge: material the blade
	# already meets at its start position is entered there (else cut as if the blade came from outside)
	return {"kind": "blade", "p1": p1, "p2": p2, "direction": d,
		"step_size": float(data.get("cut_amount", data.get("step_size", 0.01))),
		"angle": float(data.get("cut_angle", 0.0)),
		"bounded": bool(data.get("bounded", False)), "plunge": bool(data.get("plunge", False))}


def normals_all(verts, faces):
	out = []
	for f in faces:
		p = verts[f]
		q = np.roll(p, -1, axis=0)
		nf = np.array([np.sum((p[:, 1] - q[:, 1]) * (p[:, 2] + q[:, 2])),
			np.sum((p[:, 2] - q[:, 2]) * (p[:, 0] + q[:, 0])), np.sum((p[:, 0] - q[:, 0]) * (p[:, 1] + q[:, 1]))])
		out.append(nf / np.linalg.norm(nf))
	return out


def blade_spec_from_face_anchor(data, verts, file_faces):
	"""Oldest legacy format (blob): anchor on edge `cut_face_edge` of `cut_face` (legacy face order)
	at `cut_face_edge_alpha`, cut plane normal `cut_plane_normal`. The blade sweeps from just outside
	the anchor into the cage, along the inward bisector of the two faces at the edge (projected on
	the cut plane). file_faces: faces in file order and file orientation."""
	order = legacy_face_order(file_faces)
	face = file_faces[order[int(data["cut_face"])]]
	e = int(data.get("cut_face_edge", 0))
	a, b = int(face[e % len(face)]), int(face[(e + 1) % len(face)])
	al = float(data.get("cut_face_edge_alpha", 0.5))
	pa = (1.0 - al) * verts[a] + al * verts[b]
	n = np.asarray(data.get("cut_plane_normal", [1.0, 0.0, 0.0]), dtype=np.float64)
	n = n / np.linalg.norm(n)
	normals = [nf for f, nf in zip(file_faces, normals_all(verts, file_faces)) if a in f and b in f]
	# legacy patch: a plane normal parallel to a face at the edge gave zero tip directions and the
	# old code moved the tips along -n_f2 / -n_f1, i.e. cut the plane perpendicular to the edge
	if any(np.linalg.norm(np.cross(nf, n)) < 1e-6 for nf in normals):
		n = (verts[b] - verts[a]) / np.linalg.norm(verts[b] - verts[a])
	# orientation of the file faces is unknown: point the bisector towards the cage centre
	d = np.sum(normals, axis=0) if normals else np.zeros(3)
	d = d - np.dot(d, n) * n
	if np.linalg.norm(d) < 1e-9:
		d = np.mean(verts, axis=0) - pa
		d = d - np.dot(d, n) * n
	d = d / np.linalg.norm(d)
	if np.dot(np.mean(verts, axis=0) - pa, d) < 0.0:
		d = -d
	# the sweep direction is rotated in the cut plane by the spec angle (this also tilts the blade
	# line). The angle closest to it that keeps the line clear (> 2 deg) of being parallel to any
	# face the plane crosses is used: the topology cannot host a tip in such faces.
	pa_off = np.dot(n, pa)
	crossed = []
	for f, nf in zip(file_faces, normals_all(verts, file_faces)):
		s = verts[f] @ n - pa_off
		if s.min() < 1e-6 and s.max() > -1e-6:  # crossing or touching (begin shifts the blade off vertices)
			crossed.append(nf)
	crossed = np.array(crossed).reshape(-1, 3)
	th0 = float(data.get("cut_direction_angle_degrees", 0.0))
	d0 = d
	for k in range(721):
		th = math.radians(th0 + 0.5 * ((k + 1) // 2) * (1 if k % 2 else -1))
		d = math.cos(th) * d0 + math.sin(th) * np.cross(n, d0)
		l = np.cross(d, n)
		if crossed.shape[0] == 0 or np.min(np.abs(crossed @ l)) > math.sin(math.radians(2.0)):
			break
	p1 = pa - 0.05 * d
	return {"kind": "blade", "p1": p1, "p2": p1 + l, "direction": d,
		"step_size": float(data.get("cut_amount_increment", data.get("cut_amount", 0.01))),
		"angle": float(data.get("cut_direction_angle_degrees", 0.0))}


def slit_direction_legacy(xa, xb, angle):
	"""Legacy 2D cut direction: edge left normal rotated by angle (radians) towards the edge."""
	e = np.asarray(xb, dtype=np.float64) - np.asarray(xa, dtype=np.float64)
	e = e / np.linalg.norm(e)
	n = np.array([-e[1], e[0]])
	d = n * math.cos(angle) + e * math.sin(angle)
	return d / np.linalg.norm(d)


def slit_spec(verts, loop_edges, line_to_cut, alpha, initial, angle, step_size):
	a, b = loop_edges[line_to_cut]
	return {"kind": "slit", "edge": (int(a), int(b)), "alpha": float(alpha),
		"direction": slit_direction_legacy(verts[a], verts[b], angle),
		"initial": float(initial), "step_size": float(step_size), "angle": float(angle)}


# ---------------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------------

def _signed_volume(verts, faces):
	vol = 0.0
	for f in faces:
		p0 = verts[f[0]]
		for i in range(1, len(f) - 1):
			vol += np.dot(p0, np.cross(verts[f[i]], verts[f[i + 1]])) / 6.0
	return vol


def _signed_area(verts, loop):
	p = verts[loop]
	q = np.roll(p, -1, axis=0)
	return 0.5 * np.sum(p[:, 0] * q[:, 1] - q[:, 0] * p[:, 1])


def _load_json(path):
	with open(path, "r") as f:
		return json.load(f)


def _wos_params(config, dim):
	p = WosParams()
	if dim == 2:
		p.eps = 1.0e-4
	for k, v in config.get("wos", {}).items():
		setattr(p, k, v)
	return p


def _material(config, dim):
	g = float(config.get("g", -9.8 if dim == 3 else -1.0))
	grav = np.zeros(dim)
	grav[1] = g
	if "gravity" in config:
		grav = np.array(config["gravity"], dtype=np.float64)
	return Material(E=float(config.get("E", 16.0)), nu=float(config.get("nu", 0.3)),
		density=config.get("density", None),
		rayleigh_alpha=float(config.get("rayleigh_alpha", 0.0)),
		rayleigh_beta=float(config.get("rayleigh_beta", 0.0)),
		gravity=grav, dt=float(config.get("dt", 0.03 if dim == 3 else 0.01)))


def load_scene(config_path):
	config_path = os.path.abspath(config_path)
	config = _load_json(config_path)
	config_dir = os.path.dirname(config_path)
	dim = int(config.get("dim", 0))
	cage_file = resolve_path(config.get("cage_file"), config_dir)
	if dim == 0:
		# legacy configs: 2D ones use .npy cages or carry 2D-only keys
		dim = 2 if (cage_file.endswith(".npy") or "pinned_verts" in config or "points_file" in config) else 3
	if dim == 3:
		return _load_scene_3d(config, config_path, config_dir)
	return _load_scene_2d(config, config_path, config_dir)


def _load_scene_3d(config, config_path, config_dir):
	cage_file = resolve_path(config["cage_file"], config_dir)
	verts, faces = read_obj(cage_file)
	normalize = config.get("normalize", True)
	xf = unit_box_transform(verts) if normalize else (lambda p: np.asarray(p, dtype=np.float64))
	verts = xf(verts)
	file_faces = faces
	if _signed_volume(verts, faces) < 0.0:
		faces = [f[::-1].copy() for f in faces]

	mesh_file = resolve_path(config.get("mesh_file"), config_dir)
	mesh_verts = np.zeros((0, 3))
	mesh_tris = np.zeros((0, 3), dtype=np.int32)
	if mesh_file is not None and os.path.exists(mesh_file):
		mv, mf = read_obj(mesh_file)
		mesh_verts = xf(mv)
		tris = []
		for f in mf:
			for i in range(1, len(f) - 1):
				tris.append([f[0], f[i], f[i + 1]])
		mesh_tris = np.array(tris, dtype=np.int32).reshape(-1, 3)

	env_verts = env_tris = None
	env_file = resolve_path(config.get("scene_file"), config_dir)
	if env_file is not None and os.path.exists(env_file):
		ev, ef = read_obj(env_file)
		env_verts = xf(ev)
		env_tris = np.array([[f[0], f[i], f[i + 1]] for f in ef for i in range(1, len(f) - 1)], dtype=np.int32)

	qp_path = resolve_path(config.get("quad_points_file"), config_dir)
	quad_points = None
	if qp_path is not None and os.path.exists(qp_path):
		quad_points = np.load(qp_path).astype(np.float64).reshape(-1, 3)

	# pinned faces: legacy configs index faces in quads-tris-polygons order
	pinned_faces = np.array(config.get("pinned_faces", []), dtype=np.int64)
	if config.get("face_order", "legacy") == "legacy" and pinned_faces.size > 0:
		pinned_faces = legacy_face_order(faces)[pinned_faces]
	pinned = set()
	for fi in pinned_faces:
		pinned.update(int(v) for v in faces[fi])
	pinned.update(int(v) for v in config.get("pinned_verts", []))

	cut_specs = []
	spec = config.get("cut_spec", None)
	spec_list = spec if isinstance(spec, list) else ([spec] if spec else [])
	for s in spec_list:
		data = _load_json(resolve_path(s, config_dir)) if isinstance(s, str) else s
		if "cut_face" in data and "point1" not in data:
			cut_specs.append(blade_spec_from_face_anchor(data, verts, file_faces))
		else:
			cut_specs.append(blade_spec_from_legacy(data))

	return Scene(name=os.path.splitext(os.path.basename(config_path))[0], dim=3, config_path=config_path,
		cage_verts=verts, cage_faces=faces, mesh_verts=mesh_verts, mesh_tris=mesh_tris,
		quad_points=quad_points, quad_points_path=qp_path, cut_specs=cut_specs, schedule=[],
		pinned_verts=np.array(sorted(pinned), dtype=np.int64), material=_material(config, 3),
		wos=_wos_params(config, 3), floor_y=config.get("floor_y", None),
		env_verts=env_verts, env_tris=env_tris)


def _load_points_2d(path):
	"""Legacy 2D point files are (2, N) with flipped y; (N, 2) files are used as is."""
	a = np.load(path).astype(np.float64)
	if a.ndim == 2 and a.shape[0] == 2 and a.shape[1] != 2:
		a = a.T.copy()
		a[:, 1] = 1.0 - a[:, 1]
	return a.reshape(-1, 2)


def _load_scene_2d(config, config_path, config_dir):
	cage_file = resolve_path(config["cage_file"], config_dir)
	if cage_file.endswith(".obj"):
		v3, faces = read_obj(cage_file)
		verts = v3[:, :2].copy()
		loops = [faces[0].astype(np.int32)]
	else:
		c = np.load(cage_file).T.astype(np.float64)
		verts = c[:-1, :].copy()  # legacy files repeat the first vertex at the end
		verts[:, 1] = 1.0 - verts[:, 1]
		loops = [np.arange(verts.shape[0], dtype=np.int32)]
	# legacy cut edges are (i, i+1) in the file loop order, record them before reorienting
	legacy_edges = [(int(l[i]), int(l[(i + 1) % len(l)])) for l in loops for i in range(len(l))]
	loops = [l if _signed_area(verts, l) > 0.0 else l[::-1].copy() for l in loops]

	pts_path = resolve_path(config.get("points_file"), config_dir)
	mesh_verts = _load_points_2d(pts_path) if pts_path is not None and os.path.exists(pts_path) else np.array([[0.5, 0.5]])

	qp_path = resolve_path(config.get("quad_points_file"), config_dir)
	quad_points = None
	if qp_path is not None and os.path.exists(qp_path):
		quad_points = np.load(qp_path).astype(np.float64).reshape(-1, 2)

	step_size = float(config.get("cut_step_size", 0.01))
	cut_specs = []
	schedule = []
	spec_path = resolve_path(config.get("cut_spec"), config_dir)
	if spec_path is not None and os.path.exists(spec_path):
		d = _load_json(spec_path)
		if "point" in d:
			# knife starting inside the material at a point, direction given or as an angle (radians) from +x
			dirn = np.array(d["direction"], dtype=np.float64) if "direction" in d else np.array(
				[math.cos(d.get("angle", 0.0)), math.sin(d.get("angle", 0.0))])
			cut_specs.append({"kind": "slit", "point": np.array(d["point"], dtype=np.float64),
				"direction": dirn / np.linalg.norm(dirn), "initial": float(d.get("initial_cut_amount", 0.04)),
				"step_size": step_size, "angle": float(d.get("angle", 0.0))})
		else:
			cut_specs.append(slit_spec(verts, legacy_edges, d.get("line_to_cut", 0), d.get("cut_line_alpha", 0.5),
				d.get("initial_cut_amount", 0.04), d.get("angle", 0.0), step_size))
	for ev in sorted(config.get("scripted_cuts", []), key=lambda e: e["step"]):
		if ev["mode"] == "first":
			cut_specs.append(slit_spec(verts, legacy_edges, ev["line_to_cut"], ev["alpha"], ev["amount"],
				ev["angle"], step_size))
			schedule.append({"step": int(ev["step"]), "action": "start", "spec": len(cut_specs) - 1,
				"amount": float(ev["amount"])})
		elif ev["mode"] == "grow":
			period = int(ev.get("period", 1))
			for r in range(int(ev.get("repeat", 1))):
				schedule.append({"step": int(ev["step"]) + r * period, "action": "grow",
					"amount": float(ev.get("growth_amount", 0.001))})

	return Scene(name=os.path.splitext(os.path.basename(config_path))[0], dim=2, config_path=config_path,
		cage_verts=verts, cage_faces=loops, mesh_verts=mesh_verts, mesh_tris=None,
		quad_points=quad_points, quad_points_path=qp_path, cut_specs=cut_specs, schedule=schedule,
		pinned_verts=np.array(config.get("pinned_verts", []), dtype=np.int64), material=_material(config, 2),
		wos=_wos_params(config, 2), floor_y=config.get("floor_y", None))
