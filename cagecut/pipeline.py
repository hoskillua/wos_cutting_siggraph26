"""One pipeline for the app and for scripting: scene -> cage cut -> walk update -> simulation.

A Body owns the cut topology, the device cage, two walk sets (quadrature points with gradients
for the simulation, embedded mesh points for rendering), the embedded mesh splitter and the
elastic simulation. One cut step:

	delta = cutter.advance(amount)
	new GpuCage for delta.new
	quad.update / mesh.update                  (incremental walk reuse)
	quadrature points the cut brought within the margin are pushed off it and re-walked
	3D: embedded mesh split along the cut region (all cut faces of the blade), new mesh points appended
	sim.add_vertices (state from delta.new_vertex_sources), sim.set_weights
	keep the new GpuCage; on completion begin the next spec (unless a 2D schedule drives cuts)

A blade (3D) cuts all material it sweeps: several cut regions / front segments may be active at once;
they are born, split, merge and close (delta.events). It is completed when the whole blade is done.
A 2D knife keeps travelling along its ray and starts new slits where it re-enters material; it is
completed when no material is left ahead. Neither needs special casing here: advance until completed.

Nothing here reads or writes files except the explicit save_* methods.
"""

import dataclasses
import json
import time
from dataclasses import dataclass, field

import numpy as np
import warp as wp

from cagecut.cut.cutter import Cutter, CutError
from cagecut.embed.correction import AffineCorrector
from cagecut.embed.mesh_split import MeshSplitter, cut_face_pairs
from cagecut.embed.quadrature import cage_volume, inside, sample_quadrature
from cagecut.geometry.snapshot import piece_labels
from cagecut.scene import Scene
from cagecut.sim.elastic import ElasticSim
from cagecut.wos import GpuCage, SolveStats, UpdateStats, WalkSet

__all__ = ["Body", "CutStepInfo", "StepInfo", "CutError", "spec_to_legacy_json"]


@dataclass
class CutStepInfo:
	spec_index: int = -1
	amount: float = 0.0
	changed: bool = False  # the cage changed (False while the blade travels towards the cage)
	completed: bool = False
	t: float = 0.0  # blade sweep time (3D) / slit length (2D) after the step
	delta: object = None
	quad: UpdateStats = field(default_factory=UpdateStats)
	mesh: UpdateStats = field(default_factory=UpdateStats)
	n_relocated: int = 0  # quadrature points pushed off the new boundary and re-walked
	n_deactivated: int = 0  # relocated points that could not be placed inside (ignored by the sim)
	n_split_points: int = 0  # new embedded mesh points
	n_new_cage_verts: int = 0
	t_topology_ms: float = 0.0
	t_gpu_cage_ms: float = 0.0
	t_quad_ms: float = 0.0
	t_mesh_ms: float = 0.0
	t_relocate_ms: float = 0.0
	t_split_ms: float = 0.0  # mesh split + walks of the new mesh points
	t_sim_ms: float = 0.0  # state transfer + set_weights
	t_total_ms: float = 0.0
	next_spec: int = -1  # spec begun automatically after completion (-1: none)
	error: str = ""  # message if beginning the next spec failed
	events: list = field(default_factory=list)  # topology events of the step ('birth', 'split', 'tip', ...)
	n_regions: int = 0  # live cut regions (3D) / slits (2D) of the current blade after the step


@dataclass
class StepInfo:
	step: int = 0
	sim: dict = field(default_factory=dict)
	cuts: list = field(default_factory=list)  # CutStepInfo of every cut advance in this step
	t_cut_ms: float = 0.0
	t_total_ms: float = 0.0


def _sync(device):
	wp.synchronize_device(device)


def _loops_to_segments(loops):
	return [[int(l[i]), int(l[(i + 1) % len(l)])] for l in loops for i in range(len(l))]


def spec_to_legacy_json(spec, scene=None):
	"""A cut spec as a dict in the legacy JSON format the scene loader reads back.

	3D: point1/point2/direction/cut_amount/cut_angle. 2D: line_to_cut (index into the cage loop in
	file order, needs the scene), cut_line_alpha, initial_cut_amount, angle."""
	if spec["kind"] == "blade":
		return {"point1": [float(x) for x in spec["p1"]], "point2": [float(x) for x in spec["p2"]],
			"direction": [float(x) for x in spec["direction"]], "cut_amount": float(spec["step_size"]),
			"cut_angle": float(spec.get("angle", 0.0)), "bounded": bool(spec.get("bounded", False)),
			"plunge": bool(spec.get("plunge", False))}
	if "point" in spec:
		return {"point": [float(x) for x in spec["point"]], "direction": [float(x) for x in spec["direction"]],
			"initial_cut_amount": float(spec.get("initial", 0.04)), "angle": float(spec.get("angle", 0.0))}
	a, b = (int(x) for x in spec["edge"])
	line = -1
	if scene is not None:
		loop = [int(x) for x in scene.cage_faces[0]]
		for orient in (loop, loop[::-1]):  # the loader may have reversed the file loop
			n = len(orient)
			for i in range(n):
				if (orient[i], orient[(i + 1) % n]) == (a, b):
					line = i
			if line >= 0:
				break
	return {"line_to_cut": line, "cut_line_alpha": float(spec["alpha"]),
		"initial_cut_amount": float(spec.get("initial", spec["step_size"])), "angle": float(spec.get("angle", 0.0))}


def _loop_area(P, n):
	return 0.5 * float(np.dot(np.sum(np.cross(P, np.roll(P, -1, axis=0)), axis=0), n))


def _in_polygon(q, poly, tol):
	"""Point q in the closed 2D polygon (crossing number; points within tol of its boundary count as inside)."""
	a, b = poly, np.roll(poly, -1, axis=0)
	d = b - a
	t = np.clip(np.einsum("ij,ij->i", q - a, d) / np.maximum(np.einsum("ij,ij->i", d, d), 1e-300), 0.0, 1.0)
	if np.min(np.linalg.norm(q - (a + t[:, None] * d), axis=1)) < tol:
		return True
	cond = (a[:, 1] > q[1]) != (b[:, 1] > q[1])
	xi = a[:, 0] + (q[1] - a[:, 1]) * d[:, 0] / np.where(cond, d[:, 1], 1.0)
	return bool(np.sum(cond & (q[0] < xi)) % 2 == 1)


class Body:
	"""Cage-based elastic body that can be cut progressively.

	Options (keyword arguments):
		wos: WosParams overriding scene.wos (copied); num_walks: shortcut for wos.num_walks
		quad_spacing: Poisson disk spacing when the scene has no quadrature points (None: about
			quad_count points); quad_margin: minimum distance of quadrature points from the boundary
		mesh_points: (M, dim) override of the embedded points (drops the mesh triangles)
		start_at_contact: 3D blades start at their first contact (legacy) instead of travelling
		auto_cut: step() advances the active cut by its step size (default: 3D, and 2D scenes
			without a schedule); run_schedule: 2D scripted cuts; auto_next: begin the next spec
			after a completion (when no schedule drives the cuts)
		split_mesh / split_offset: embedded mesh splitting (3D)
		psd_projection: sim Hessian projection; compute: run the initial solve now
	"""

	def __init__(self, scene: Scene, device=None, wos=None, num_walks=None, quad_spacing=None, quad_count=800,
		quad_margin=0.01, mesh_points=None, start_at_contact=True, auto_cut=None, run_schedule=True, auto_next=True,
		split_mesh=True, split_offset=1.0e-4, psd_projection=True, quad_seed=0, compute=True):
		self.scene = scene
		self.dim = scene.dim
		self.device = wp.get_device(device)
		params = dataclasses.replace(wos if wos is not None else scene.wos)
		if num_walks is not None:
			params.num_walks = int(num_walks)
		self.params = params
		self.quad_margin = float(quad_margin)
		self.start_at_contact = bool(start_at_contact)
		self.auto_cut = (self.dim == 3 or not scene.schedule) if auto_cut is None else bool(auto_cut)
		self.run_schedule = bool(run_schedule)
		self.auto_next = bool(auto_next)
		self.split_mesh = bool(split_mesh)
		self.split_offset = float(split_offset)
		self.on_delta = None  # optional callable(body, delta, old_gpu, new_gpu) run before the walk updates
		self.psd_projection = bool(psd_projection)
		self.specs = [dict(s) for s in scene.cut_specs]

		# embedded points / mesh (rest space), quadrature points: fixed for the life of the body
		if mesh_points is not None:
			self._mesh_verts0 = np.asarray(mesh_points, dtype=np.float64).reshape(-1, self.dim)
			self._mesh_tris0 = None
		else:
			self._mesh_verts0 = np.asarray(scene.mesh_verts, dtype=np.float64).reshape(-1, self.dim)
			self._mesh_tris0 = scene.mesh_tris if (scene.mesh_tris is not None and len(scene.mesh_tris) > 0) else None
		cutter = Cutter.from_scene(scene)
		self.n_quad_outside = 0
		if scene.quad_points is not None and len(scene.quad_points) > 0:
			q = np.asarray(scene.quad_points, dtype=np.float64).reshape(-1, self.dim)
			# legacy files contain a few points outside the cage: useless for the sim, and their walks
			# wander outside (long, every cut step rewalks them), so they are dropped
			ins = inside(cutter.snapshot, q)
			self.n_quad_outside = int(np.sum(~ins))
			self._quad_points0 = q[ins]
			self.quad_sampled = False
		else:
			vol = cage_volume(cutter.snapshot)
			sp = quad_spacing if quad_spacing is not None else (0.7 * vol / max(quad_count, 1)) ** (1.0 / self.dim)
			self._quad_points0 = sample_quadrature(cutter.snapshot, sp, self.quad_margin, seed=quad_seed)
			self.quad_sampled = True
		self._build(cutter)
		if compute:
			self.compute_weights()

	# -----------------------------------------------------------------------------------------
	# construction / reset

	def _build(self, cutter=None):
		scene = self.scene
		self.cutter = cutter if cutter is not None else Cutter.from_scene(scene)
		snap = self.cutter.snapshot
		self.gpu_cage = GpuCage.from_snapshot(snap, self.params, self.device)
		self.quad = WalkSet(self.dim, self._quad_points0, self.params, gradients=True, device=self.device, seed_base=0)
		self.quad_active = inside(snap, self._quad_points0) if len(self._quad_points0) else np.zeros(0, dtype=bool)
		self._quad_dist = np.full(len(self._quad_points0), np.inf)
		self.mesh = WalkSet(self.dim, self._mesh_verts0, self.params, gradients=False, device=self.device, seed_base=1) \
			if len(self._mesh_verts0) else None
		self.mesh_tris = None if self._mesh_tris0 is None else np.array(self._mesh_tris0, dtype=np.int64)
		self.splitter = MeshSplitter(self._mesh_verts0, self._mesh_tris0) \
			if (self.dim == 3 and self._mesh_tris0 is not None and self.split_mesh) else None
		Q = max(len(self._quad_points0), 1)
		mat = dataclasses.replace(scene.material)
		self.sim = ElasticSim(self.dim, snap.verts, cage_volume(snap) / Q, mat, scene.pinned_verts,
			floor_y=scene.floor_y, psd_projection=self.psd_projection, device=self.device)
		self.spec_index = -1
		self.completed_specs = []
		self.cut_face_ids = set()  # every face (3D) / segment (2D) created as a cut face so far
		self.cut_pairs = []  # (plus, minus) cut face pairs of the current blade (delta.cut_faces)
		self.step_count = 0
		self.solved = False
		self.last_cut = None
		self.last_full = {}
		# monotonic across resets: bump when the cage or mesh connectivity / any weight changes (UI refresh)
		self.topology_version = getattr(self, "topology_version", -1) + 1
		self.weights_version = getattr(self, "weights_version", -1) + 1
		self._W_mesh = None
		self._T_quad = None
		self._corrector = None

	def reset(self):
		"""Back to the uncut cage at rest (same quadrature and mesh points, same seeds), weights solved."""
		self._build()
		self.compute_weights()

	def reset_sim(self):
		"""Rest positions and zero velocity on the current (possibly cut) cage."""
		self.sim.rest = self.cutter.snapshot.verts.copy()
		self.sim.reset()

	def rebuild_walks(self, **params):
		"""Change WoS parameters (num_walks, eps, max_steps, ...) and re-solve on the current cage.
		Current points are kept (incl. relocated / split ones); seeds are fresh."""
		for k, v in params.items():
			setattr(self.params, k, v)
		self.gpu_cage = GpuCage.from_snapshot(self.cutter.snapshot, self.params, self.device)
		self.quad = WalkSet(self.dim, self.quad.points, self.params, gradients=True, device=self.device, seed_base=0)
		if self.mesh is not None:
			self.mesh = WalkSet(self.dim, self.mesh.points, self.params, gradients=False, device=self.device, seed_base=1)
		return self.compute_weights()

	# -----------------------------------------------------------------------------------------
	# weights

	def compute_weights(self):
		"""Full solve of both walk sets on the current cage, then refresh the sim."""
		stats = self._solve_all()
		self._quad_dist = self._closest_dist(self.gpu_cage, self.quad.points)
		self.solved = True
		return stats

	def full_recompute(self) -> SolveStats:
		"""Baseline: re-solve every walk on the current cage with the same seeds."""
		return self._solve_all()

	def _solve_all(self):
		sq = self.quad.solve(self.gpu_cage)
		sm = self.mesh.solve(self.gpu_cage) if self.mesh is not None else SolveStats()
		self.last_full = {"quad": sq, "mesh": sm}
		self._weights_changed()
		self._refresh_sim()
		n = sq.n_points + sm.n_points
		tf = (sq.terminated_fraction * sq.n_points + sm.terminated_fraction * sm.n_points) / max(n, 1)
		return SolveStats(n, sq.n_walks + sm.n_walks, tf, sq.t_total_ms + sm.t_total_ms)

	def _weights_changed(self):
		self.weights_version += 1
		self._W_mesh = None
		self._T_quad = None

	def _refresh_sim(self):
		snap = self.cutter.snapshot
		T = self.quad.weights()
		G = self.quad.gradients()
		self._T_quad = T
		self.sim.set_weights(T, G, self.quad_active, snap.verts, snap.vert_alive)

	def _closest_dist(self, cage, points):
		if len(points) == 0:
			return np.zeros(0)
		return cage.closest(points)[0].astype(np.float64)

	# -----------------------------------------------------------------------------------------
	# cutting

	@property
	def cut_active(self):
		return self.cutter.active

	@property
	def spec(self):
		return self.specs[self.spec_index] if 0 <= self.spec_index < len(self.specs) else None

	def set_spec(self, index, spec):
		"""Replace (or append, index == len) a cut spec, e.g. after editing it in the UI."""
		if index == len(self.specs):
			self.specs.append(dict(spec))
		else:
			self.specs[index] = dict(spec)

	def begin_cut(self, spec_index: int):
		"""Set the blade / slit of spec `spec_index` (no geometry change). Raises CutError."""
		spec = self.specs[spec_index]
		if self.dim == 3:
			self.cutter.begin(spec, start_at_contact=self.start_at_contact)
		else:
			self.cutter.begin(spec)
		self.spec_index = int(spec_index)
		self.cut_pairs = []
		if self.splitter is not None:
			self.splitter.reset_spec()

	def default_amount(self):
		spec = self.spec
		if spec is None:
			return 0.0
		if self.dim == 2 and not self.cutter.started:
			return float(spec.get("initial", spec["step_size"]))
		return float(spec["step_size"])

	def advance_cut(self, amount=None) -> CutStepInfo:
		"""One cut step and every update it implies. amount None: the spec's step size (2D: the
		initial slit length for the first advance)."""
		info = CutStepInfo(spec_index=self.spec_index)
		if not self.cutter.active:
			return info
		amount = self.default_amount() if amount is None else float(amount)
		info.amount = amount
		dev = self.device
		_sync(dev)
		t0 = time.perf_counter()
		delta = self.cutter.advance(amount)
		t1 = time.perf_counter()
		info.t_topology_ms = (t1 - t0) * 1e3
		info.t = float(delta.t)
		info.delta = delta
		info.events = list(getattr(delta, "events", None) or [])
		if delta.new is delta.old:
			info.t_total_ms = info.t_topology_ms
			self.last_cut = info
			return info
		info.changed = True
		info.completed = bool(delta.completed)
		self.cut_pairs = cut_face_pairs(delta.cut_faces)
		info.n_regions = len(self.cut_pairs)
		self.cut_face_ids.update(int(f) for pair in self.cut_pairs for f in pair if f >= 0)
		info.n_new_cage_verts = int(len(delta.new_vertices))

		old_gpu = self.gpu_cage
		new_gpu = GpuCage.from_snapshot(delta.new, self.params, dev)
		_sync(dev)
		t2 = time.perf_counter()
		info.t_gpu_cage_ms = (t2 - t1) * 1e3
		hook_ms = 0.0
		if self.on_delta is not None:
			# the walks can be inspected against the delta before they are updated (not timed)
			self.on_delta(self, delta, old_gpu, new_gpu)
			_sync(dev)
			th = time.perf_counter()
			hook_ms = (th - t2) * 1e3
			t2 = th

		info.quad = self.quad.update(old_gpu, new_gpu, delta)
		t3 = time.perf_counter()
		info.t_quad_ms = (t3 - t2) * 1e3
		if self.mesh is not None:
			info.mesh = self.mesh.update(old_gpu, new_gpu, delta)
		t4 = time.perf_counter()
		info.t_mesh_ms = (t4 - t3) * 1e3

		info.n_relocated, info.n_deactivated = self._relocate_quadrature(new_gpu)
		_sync(dev)
		t5 = time.perf_counter()
		info.t_relocate_ms = (t5 - t4) * 1e3

		if self.splitter is not None and self.mesh is not None:
			res = self.splitter.split(delta, self.split_offset)
			if res.changed:
				if len(res.new_points):
					ids = self.mesh.append_points(new_gpu, res.new_points)
					assert np.array_equal(ids, res.new_point_ids), "mesh walk set and splitter disagree on ids"
				self.mesh_tris = self.splitter.tris
				info.n_split_points = int(len(res.new_points))
				self.topology_version += 1
		_sync(dev)
		t6 = time.perf_counter()
		info.t_split_ms = (t6 - t5) * 1e3

		self.gpu_cage = new_gpu
		self._transfer_sim_state(delta, old_gpu)
		self._weights_changed()
		self._refresh_sim()
		self._corrector = None
		if delta.topology_changed:
			self.topology_version += 1
		t7 = time.perf_counter()
		info.t_sim_ms = (t7 - t6) * 1e3
		info.t_total_ms = (t7 - t0) * 1e3 - hook_ms

		if delta.completed:
			self.cut_pairs = []
			self.completed_specs.append(self.spec_index)
			nxt = self.spec_index + 1
			if self.auto_next and not (self.dim == 2 and self.scene.schedule and self.run_schedule) \
				and nxt < len(self.specs):
				try:
					self.begin_cut(nxt)
					info.next_spec = nxt
				except CutError as e:
					info.error = f"spec {nxt}: {e}"
		self.last_cut = info
		return info

	def run_cut(self, spec_index=None, amount=None, max_steps=100000, callback=None):
		"""Begin (optional) and advance until the current cut completes. Returns the CutStepInfos."""
		if spec_index is not None:
			self.begin_cut(spec_index)
		infos = []
		idx = self.spec_index
		for _ in range(max_steps):
			if not self.cutter.active or self.spec_index != idx:
				break
			info = self.advance_cut(amount)
			infos.append(info)
			if callback is not None:
				callback(info)
			if info.completed:
				break
		return infos

	def _relocate_quadrature(self, cage):
		"""Points the cut brought within the margin of the boundary are pushed away from their closest
		boundary point (the inward normal of the closest face when the projection is inside it) and get
		fresh walks. Points already close to the uncut boundary are left alone."""
		P = self.quad.num_points
		if P == 0:
			return 0, 0
		pts = self.quad.points
		d_all, f_all, cp_all = cage.closest(pts)
		d_new = d_all.astype(np.float64)
		d_old = self._quad_dist if self._quad_dist.shape[0] == P else np.full(P, np.inf)
		m = self.quad_margin
		sel = np.nonzero((d_new < m) & (d_new < d_old * (1.0 - 1e-4) - 1e-7))[0]
		if sel.size == 0:
			self._quad_dist = d_new
			return 0, 0
		snap = cage.snapshot
		p = pts[sel].copy()
		d, f, cp = d_new[sel], f_all[sel], cp_all[sel]
		target = 1.05 * m
		for _ in range(4):
			far = d >= m
			if np.all(far):
				break
			for j in np.nonzero(~far)[0]:
				dirv = p[j] - cp[j]
				nrm = np.linalg.norm(dirv)
				if nrm > 1e-9:
					dirv = dirv / nrm
				else:
					dirv = -snap.face_normal[f[j]]
				p[j] = p[j] + (target - d[j]) * dirv
			d, f, cp = cage.closest(p)
			d = d.astype(np.float64)
		ok = inside(snap, p) & (d >= 0.5 * m)
		self.quad.reset_points(cage, sel, p)
		self.quad_active[sel] = self.quad_active[sel] & ok
		d_new[sel] = d
		self._quad_dist = d_new
		return int(sel.size), int(np.sum(~ok))

	@staticmethod
	def _pinned_region(snap, pinned, tol=1e-7):
		"""Closure of the pinned part of the cage as a point test: the faces (3D polygons by their outer loop,
		2D segments) whose vertices are all pinned."""
		pieces = []
		for f in range(snap.num_faces):
			if not snap.face_alive[f]:
				continue
			ids = [int(i) for i in snap.face(f)]
			if not ids or not all(i in pinned for i in ids):
				continue
			if snap.dim == 2:
				pieces.append((snap.verts[ids[0]], snap.verts[ids[1]], None, None))
				continue
			loops = snap.face_loops(f)
			outer = max(loops, key=lambda lp: abs(_loop_area(snap.verts[lp], snap.face_normal[f])))
			P = snap.verts[outer]
			e = P[1] - P[0]
			u = e / np.linalg.norm(e)
			n = np.asarray(snap.face_normal[f], dtype=np.float64)
			pieces.append((P, n, u, np.cross(n, u)))

		def contains(x):
			x = np.asarray(x, dtype=np.float64)
			for a, b, u, w in pieces:
				if u is None and w is None and snap.dim == 2:
					d = b - a
					t = np.clip(np.dot(x - a, d) / max(np.dot(d, d), 1e-300), 0.0, 1.0)
					if np.linalg.norm(x - (a + t * d)) < tol:
						return True
					continue
				P, n = a, b
				if abs(np.dot(n, x - P[0])) > tol:
					continue
				q = np.array([(x - P[0]) @ u, (x - P[0]) @ w])
				poly = np.stack([(P - P[0]) @ u, (P - P[0]) @ w], axis=1)
				if _in_polygon(q, poly, tol):
					return True
			return False
		return contains

	def _transfer_sim_state(self, delta, old_gpu):
		"""Grow the sim by the new cage vertices and place new / moved vertices in deformed space.

		New vertices take the weighted sum of their sources. Vertices whose rest position is not that
		sum (cut tips, which slide along their face) and moved vertices get an affine map fitted to the
		deformed positions of nearby cage vertices (rings of shared faces)."""
		sim = self.sim
		snap = delta.new
		V0 = sim.num_verts
		Vn = snap.num_verts
		X = snap.verts
		x = np.vstack([sim.x, np.zeros((Vn - V0, self.dim))])
		v = np.vstack([sim.v, np.zeros((Vn - V0, self.dim))])
		refit = set(int(i) for i in delta.moved_vertices)
		pinned = set(int(i) for i in sim.pinned)
		new_pins = []
		# a vertex the cut puts inside the pinned part of the cage (a cut tip inside a pinned face, a slit
		# vertex on its boundary) is pinned itself, at its rest position; a tip that leaves the pinned part
		# is freed again
		in_pin = self._pinned_region(delta.old, pinned) if pinned else (lambda x: False)
		cand = set(range(V0, Vn)) | {i for i in refit if i < V0}
		held = set()
		for vid in sorted(cand):
			if in_pin(X[vid]):
				held.add(vid)
				x[vid] = X[vid]
				v[vid] = 0.0
				refit.discard(vid)
				if vid not in pinned:
					new_pins.append(vid)
		freed = [i for i in refit if i < V0 and i in pinned and i not in held]
		for vid in range(V0, Vn):
			if vid in held:
				continue
			src = delta.new_vertex_sources.get(vid)
			if src is None:
				refit.add(vid)
				continue
			ids = np.asarray(src[0], dtype=np.int64)
			w = np.asarray(src[1], dtype=np.float64)
			x[vid] = w @ x[ids]
			v[vid] = w @ v[ids]
			if np.linalg.norm(w @ X[ids] - X[vid]) > 1e-9:
				refit.add(vid)
			elif all(int(i) in pinned for i in ids):
				# a slit vertex on an edge between pinned vertices stays pinned (a cut through a
				# pinned face must not free part of it); cut tips (refit above) are never pinned
				new_pins.append(vid)
		if refit:
			self._affine_refit(snap, x, v, sorted(refit), V0, old_gpu)
		if Vn > V0:
			sim.add_vertices(X[V0:Vn], x[V0:Vn], v[V0:Vn])
		sim.x[:V0] = x[:V0]
		sim.v[:V0] = v[:V0]
		if new_pins or freed:
			keep = np.array([i for i in sim.pinned if int(i) not in set(freed)], dtype=np.int64)
			sim.set_pinned(np.concatenate([keep, np.array(new_pins, dtype=np.int64)]))

	def _affine_refit(self, snap, x, v, ids, V0, old_gpu):
		rest_old = np.vstack([self.sim.rest, snap.verts[V0:]])  # rest positions the deformed x belong to
		X = snap.verts
		inc = {}
		for f in range(snap.num_faces):
			if not snap.face_alive[f]:
				continue
			for j in snap.face(f):
				inc.setdefault(int(j), []).append(f)
		skip = set(ids)
		lost = []  # no full-rank neighbourhood (a crack pocket inside the material): use harmonic coordinates
		for vid in ids:
			ring = {vid}
			nb = set()
			for _ in range(3):
				grow = set()
				for a in ring:
					for f in inc.get(a, ()):
						grow.update(int(j) for j in snap.face(f))
				ring |= grow
				nb = [j for j in ring if j not in skip and snap.vert_alive[j]]
				if len(nb) >= self.dim + 1:
					R = rest_old[nb] - rest_old[nb].mean(axis=0)
					if np.linalg.matrix_rank(R, tol=1e-9) >= self.dim:
						break
			if len(nb) < self.dim + 1 or np.linalg.matrix_rank(rest_old[nb] - rest_old[nb].mean(axis=0), tol=1e-9) < self.dim:
				lost.append(vid)
				continue
			r0 = rest_old[nb].mean(axis=0)
			R = rest_old[nb] - r0
			Y = np.hstack([x[nb], v[nb]])
			y0 = Y.mean(axis=0)
			A = np.linalg.lstsq(R, Y - y0, rcond=1e-9)[0]
			y = y0 + (X[vid] - r0) @ A
			x[vid] = y[:self.dim]
			v[vid] = y[self.dim:]
		if lost:
			# deformed position of the rest point = its harmonic coordinates in the cage before the step
			ws = WalkSet(self.dim, X[lost], self.params, gradients=False, device=self.device, seed_base=7)
			ws.solve(old_gpu)
			W = ws.weights()[:, :V0].astype(np.float64)
			W /= np.maximum(W.sum(axis=1, keepdims=True), 1e-12)
			x[lost] = W @ x[:V0]
			v[lost] = W @ v[:V0]

	def blade_front(self):
		"""Current cut front as cage vertex ids, for display: (tips (k,), front segments (m, 2)).

		3D: the front segments of the active blade (blade edges between two tips, several at once when the
		blade cuts several parts or around notches) and their tips. 2D: the tip of the slit being cut."""
		if not self.cutter.active:
			return np.zeros(0, dtype=np.int64), np.zeros((0, 2), dtype=np.int64)
		return self.cutter.front()

	# -----------------------------------------------------------------------------------------
	# time stepping

	def step(self, ext_force=None) -> StepInfo:
		"""Scheduled (2D) or automatic cut advance, then one simulation step."""
		info = StepInfo(step=self.step_count)
		t0 = time.perf_counter()
		ran = False
		if self.dim == 2 and self.scene.schedule and self.run_schedule:
			for ev in self.scene.schedule:
				if ev["step"] != self.step_count:
					continue
				ran = True
				if ev["action"] == "start":
					try:
						self.begin_cut(int(ev["spec"]))
					except CutError as e:
						info.cuts.append(CutStepInfo(spec_index=int(ev["spec"]), error=str(e)))
						continue
					info.cuts.append(self.advance_cut(ev["amount"]))
				elif ev["action"] == "grow" and self.cutter.active:
					info.cuts.append(self.advance_cut(ev["amount"]))
		if not ran and self.auto_cut and self.cutter.active:
			info.cuts.append(self.advance_cut(None))
		t1 = time.perf_counter()
		if ext_force is not None:
			ext_force = np.asarray(ext_force, dtype=np.float64).reshape(-1, self.dim)
			if ext_force.shape[0] < self.sim.num_verts:
				ext_force = np.vstack([ext_force, np.zeros((self.sim.num_verts - ext_force.shape[0], self.dim))])
		info.sim = self.sim.step(ext_force)
		self.step_count += 1
		info.t_cut_ms = (t1 - t0) * 1e3
		info.t_total_ms = (time.perf_counter() - t0) * 1e3
		return info

	# -----------------------------------------------------------------------------------------
	# outputs

	def corrected_mesh_weights(self):
		"""(M, V) affine-corrected weights of the embedded points (evaluation time only, cached)."""
		if self.mesh is None:
			return np.zeros((0, self.sim.num_verts))
		if self._W_mesh is None:
			snap = self.cutter.snapshot
			if self._corrector is None:
				self._corrector = AffineCorrector(snap.verts, piece_labels(snap))
			W = self.mesh.weights()[:, :snap.num_verts]
			self._W_mesh = self._corrector(W, self.mesh.points)
		return self._W_mesh

	def mesh_positions(self) -> np.ndarray:
		"""Deformed embedded points (corrected weights applied to the deformed cage)."""
		return self.corrected_mesh_weights() @ self.sim.x

	def mesh_rest_positions(self):
		return np.zeros((0, self.dim)) if self.mesh is None else self.mesh.points

	def quad_weights(self):
		if self._T_quad is None:
			self._T_quad = self.quad.weights()
		return self._T_quad[:, :self.sim.num_verts]

	def quad_positions(self):
		"""Deformed quadrature points (uncorrected weights; debug display)."""
		return self.quad_weights() @ self.sim.x

	def cage_positions(self):
		return self.sim.x

	def reconstruction_error(self, corrected=False):
		"""Per mesh point |sum_j T_j X_j - x| on the rest cage (linear precision error)."""
		if self.mesh is None:
			return np.zeros(0)
		X = self.cutter.snapshot.verts
		W = self.corrected_mesh_weights() if corrected else self.mesh.weights()[:, :X.shape[0]]
		return np.linalg.norm(W @ X - self.mesh.points, axis=1)

	def drag_force(self, point_id, target, stiffness, which="mesh"):
		"""Spring force pulling embedded point `point_id` (deformed) towards `target`, distributed to
		the cage through the point's weights: f_j = w_j * k (target - x_p). Returns (V, dim)."""
		W = self.corrected_mesh_weights() if which == "mesh" else self.quad_weights()
		w = W[int(point_id)]
		xp = w @ self.sim.x
		f = float(stiffness) * (np.asarray(target, dtype=np.float64).reshape(self.dim) - xp)
		return w[:, None] * f[None, :]

	def walk_cache_bytes(self):
		"""Device memory of the per-walk caches (face, landing point, gradient factor, bounding volume)."""
		total = 0
		for ws in (self.quad, self.mesh):
			if ws is None:
				continue
			per = 4 + 4 * self.dim + (4 * self.dim if ws.grads else 0) + 4 * 2 * ws.NA
			total += per * ws.K * ws.num_points
		return total

	# -----------------------------------------------------------------------------------------
	# explicit saving

	def save_quadrature(self, path=None):
		"""Save the current quadrature points (rest space) as .npy; path defaults to the scene's."""
		path = path if path is not None else self.scene.quad_points_path
		if path is None:
			raise ValueError("no path for the quadrature points")
		np.save(path, self.quad.points)
		return path

	def save_spec(self, index, path):
		with open(path, "w") as f:
			json.dump(spec_to_legacy_json(self.specs[index], self.scene), f, indent=2)
		return path
