"""AppState: the only mutable UI state. Commands forward to the pipeline Body; no algorithm here."""

import os
from collections import deque
from dataclasses import dataclass

import numpy as np

from cagecut.app.cases import list_cases
from cagecut.cut.cutter import CutError
from cagecut.pipeline import Body
from cagecut.scene import blade_direction_from_angle, slit_direction_legacy


@dataclass
class DragState:
	active: bool = False
	point: int = -1  # embedded point id (mesh vertex in 3D, mesh point in 2D)
	which: str = "mesh"
	depth: float = 0.0  # distance from the camera along the pick ray
	cam_pos: np.ndarray = None
	target: np.ndarray = None
	stiffness: float = 20.0


@dataclass
class ViewFlags:
	"""What is shown. Kept in sync both ways with polyscope's structure list (see viz.py)."""
	mesh: bool = True
	cage: str = "off"  # "off" | "wireframe" | "faces" | "both" (2D: off / on)
	cage_opacity: float = 0.25
	cut_faces: bool = False
	quad_points: bool = False
	pinned: bool = True  # pinned cage vertices
	ground: bool = False  # ground plane (3D)
	blade: bool = True  # blade preview (line, sweep direction, cut plane)
	front: bool = True  # cut front segments and tips of the active blade
	environment: bool = True


@dataclass
class DebugFlags:
	"""Analysis layers (off by default)."""
	rewalk_heat: bool = False
	onface_heat: bool = False
	swept: bool = False
	weight_field: bool = False
	weight_vertex: int = 0
	timing: bool = True


class AppState:
	def __init__(self, scene, body_kwargs=None, device=None):
		self.scene = scene
		self.body_kwargs = dict(body_kwargs or {})
		self.device = device
		self.body = Body(scene, device=device, **self.body_kwargs)
		self.running = False
		self.run_mode = "all"  # what space / the last Run button starts: "all" (cut + sim) or "sim"
		self._done_sig = None  # blade whose cut just completed (Run all does not repeat it)
		self.message = ""
		self.view = ViewFlags()
		self.debug = DebugFlags()
		self.drag = DragState()
		self.history = deque(maxlen=120)  # recent StepInfo / CutStepInfo timings
		self.events = deque(maxlen=200)  # cut event log: (spec index, cut step, event names)
		self.cut_steps = 0  # cut steps that changed the cage since the last reset
		self.last_step = None
		self.last_cut = None
		self.frame = 0
		self.pending_case = None  # config path to load once the frame is done (see app.main)
		self.case_dir = os.path.dirname(os.path.abspath(scene.config_path)) if scene.config_path else os.getcwd()
		self.cases = []
		self.case_index = 0
		self.refresh_cases()
		self._init_ui_values()

	def _init_ui_values(self):
		b = self.body
		p = b.params
		self.num_walks = int(p.num_walks)
		self.eps = float(p.eps)
		self.max_steps = int(p.max_steps)
		m = b.sim
		self.E, self.nu = float(m.E), float(m.nu)
		self.rayleigh_alpha, self.rayleigh_beta = float(m.rayleigh_alpha), float(m.rayleigh_beta)
		self.dt = float(m.dt)
		self.gravity = np.array(m.gravity, dtype=np.float64)
		self.gravity0 = self.gravity.copy()  # the scene's gravity, for the on / off toggle
		self.gravity_on = bool(np.any(self.gravity != 0.0))
		self.floor_on = m.floor_y is not None
		self.floor_y = float(m.floor_y) if m.floor_y is not None else float(np.min(self.scene.cage_verts[:, 1]) - 0.05)
		self.auto_cut = bool(b.auto_cut)
		cfg_dir = os.path.dirname(self.scene.config_path) if self.scene.config_path else os.getcwd()
		self.spec_path = os.path.join(cfg_dir, f"{self.scene.name}-cut-0.json")
		self.quad_path = self.scene.quad_points_path or os.path.join(cfg_dir, f"{self.scene.name}-quad-points.npy")
		self.select_spec(0)

	# -----------------------------------------------------------------------------------------
	# cases (scene configs in a folder)

	def refresh_cases(self, folder=None):
		"""List the loadable configs of the case folder; the loaded scene's config is selected when it is there."""
		if folder is not None:
			self.case_dir = folder
		self.cases = list_cases(self.case_dir)
		cur = os.path.abspath(self.scene.config_path) if self.scene.config_path else None
		paths = [c.path for c in self.cases]
		self.case_index = paths.index(cur) if cur in paths else min(self.case_index, max(len(paths) - 1, 0))

	def cmd_load_case(self, index):
		"""Ask for the case to be loaded after this frame (the app swaps the whole state)."""
		if 0 <= index < len(self.cases):
			self.pending_case = self.cases[index].path

	# -----------------------------------------------------------------------------------------
	# cut spec editing (UI values <-> spec dicts)

	def _default_edit(self):
		if self.scene.dim == 3:
			c = np.mean(self.scene.cage_verts, axis=0)
			return {"kind": "blade", "p1": c + np.array([0.0, 0.5, -0.5]), "p2": c + np.array([0.0, 0.5, 0.5]),
				"angle": 0.0, "step_size": 0.01}
		return {"kind": "slit", "edge": (int(self.scene.cage_faces[0][0]), int(self.scene.cage_faces[0][1])),
			"alpha": 0.5, "angle": 0.0, "initial": 0.05, "step_size": 0.01}

	def _spec_to_edit(self, spec):
		e = {k: (np.array(v, dtype=np.float64) if isinstance(v, np.ndarray) else v) for k, v in spec.items()}
		e.setdefault("angle", 0.0)
		return e

	def edit_to_spec(self):
		"""Spec dict from the edited values (direction recomputed from the angle like the loader)."""
		e = self.edit
		if e["kind"] == "blade":
			p1, p2 = np.asarray(e["p1"], dtype=np.float64), np.asarray(e["p2"], dtype=np.float64)
			if "direction" in e and e.get("_keep_dir", False):
				d = np.asarray(e["direction"], dtype=np.float64)
			else:
				d = blade_direction_from_angle(p1, p2, float(e["angle"]))
			l = (p2 - p1) / max(np.linalg.norm(p2 - p1), 1e-12)
			d = d - np.dot(d, l) * l
			d = d / max(np.linalg.norm(d), 1e-12)
			return {"kind": "blade", "p1": p1, "p2": p2, "direction": d, "step_size": float(e["step_size"]),
				"angle": float(e["angle"]), "bounded": bool(e.get("bounded", False)), "plunge": bool(e.get("plunge", False))}
		if "point" in e:
			if "direction" in e and e.get("_keep_dir", False):
				d = np.asarray(e["direction"], dtype=np.float64)
			else:
				d = np.array([np.cos(float(e["angle"])), np.sin(float(e["angle"]))])
			return {"kind": "slit", "point": np.asarray(e["point"], dtype=np.float64), "direction": d / np.linalg.norm(d),
				"initial": float(e.get("initial", 0.05)), "step_size": float(e["step_size"]), "angle": float(e["angle"])}
		a, b = e["edge"]
		X = self.scene.cage_verts
		return {"kind": "slit", "edge": (int(a), int(b)), "alpha": float(e["alpha"]),
			"direction": slit_direction_legacy(X[a], X[b], float(e["angle"])), "initial": float(e["initial"]),
			"step_size": float(e["step_size"]), "angle": float(e["angle"])}

	def select_spec(self, index):
		self.spec_index = int(index)
		b = self.body
		self.edit = self._spec_to_edit(b.specs[index]) if index < len(b.specs) else self._default_edit()
		self.edit["_keep_dir"] = index < len(b.specs)  # keep the loaded direction until the angle is edited
		cfg_dir = os.path.dirname(self.spec_path)
		self.spec_path = os.path.join(cfg_dir, f"{self.scene.name}-cut-{self.spec_index}.json")

	# -----------------------------------------------------------------------------------------
	# commands

	def cmd_flip_blade(self):
		"""Sweep the blade / slit the opposite way."""
		e = self.edit
		if "direction" in e and e.get("_keep_dir", False):
			e["direction"] = -np.asarray(e["direction"], dtype=np.float64)
		elif self.scene.dim == 3:
			e["angle"] = (float(e["angle"]) + 180.0) % 360.0
		else:
			e["angle"] = float(e["angle"]) + np.pi if float(e["angle"]) <= 0.0 else float(e["angle"]) - np.pi

	def _edit_sig(self):
		sp = self.edit_to_spec()
		return repr({k: (np.round(v, 9).tolist() if isinstance(v, np.ndarray) else v) for k, v in sp.items()
			if k != "step_size"})

	def cmd_run(self, cut=True):
		"""Run all (cut + simulation) or the simulation alone. Run all begins the edited blade's cut unless it
		is already running or just completed."""
		b = self.body
		if cut and not b.cut_active and self._done_sig != self._edit_sig():
			self.cmd_begin_cut()
			if not b.cut_active:
				return  # the message says why; stay paused
		self.auto_cut = bool(cut)
		self.run_mode = "all" if cut else "sim"
		self.running = True

	def cmd_toggle_run(self):
		if self.running:
			self.running = False
		else:
			self.cmd_run(self.run_mode == "all")

	def cmd_begin_cut(self):
		b = self.body
		b.set_spec(self.spec_index, self.edit_to_spec())
		try:
			b.begin_cut(self.spec_index)
			self.message = f"cut {self.spec_index} begun"
		except CutError as e:
			self.message = f"cannot begin cut: {e}"

	def _sync_step_size(self):
		"""The step size stays editable while cutting: apply it to the active spec."""
		b = self.body
		if b.cut_active and b.spec_index == self.spec_index and b.spec is not None:
			b.spec["step_size"] = float(self.edit["step_size"])

	def cmd_cut_step(self):
		if not self.body.cut_active:
			self.message = "no active cut (begin one first)"
			return None
		self._sync_step_size()
		info = self.body.advance_cut()
		self._record_cut(info)
		return info

	def _record_cut(self, info):
		if info is None:
			return
		self.last_cut = info
		if info.changed:
			self.cut_steps += 1
		if info.events:
			self.events.append((int(info.spec_index), self.cut_steps, list(info.events)))
		if info.error:
			self.message = info.error
		elif info.completed:
			self._done_sig = self._edit_sig()
			self.message = f"cut {info.spec_index} completed" + \
				(f", cut {info.next_spec} begun" if info.next_spec >= 0 else "")
		if info.next_spec >= 0:
			self.select_spec(info.next_spec)

	def cmd_step(self):
		b = self.body
		b.auto_cut = self.auto_cut
		self._sync_step_size()
		f = None
		if self.drag.active and self.drag.target is not None:
			f = b.drag_force(self.drag.point, self.drag.target, self.drag.stiffness, self.drag.which)
		info = b.step(ext_force=f)
		self.last_step = info
		for c in info.cuts:
			self._record_cut(c)
		self.history.append(info)
		return info

	def cmd_reset(self):
		"""Back to the uncut cage at rest. The blade and material settings stay as last edited (not the
		config's); only the simulation and cut state are rebuilt."""
		self.body.set_spec(self.spec_index, self.edit_to_spec())
		self.body.reset()
		self.cmd_apply_material()
		self.running = False
		self._done_sig = None
		self.history.clear()
		self.events.clear()
		self.cut_steps = 0
		self.last_step = self.last_cut = None
		self.message = "reset to the uncut cage (blade and material settings kept)"

	def cmd_reset_sim(self):
		self.body.reset_sim()
		self.message = "simulation reset to rest"

	def cmd_recompute(self):
		st = self.body.full_recompute()
		self.message = f"full recompute: {st.n_walks} walks in {st.t_total_ms:.1f} ms"

	def cmd_apply_wos(self):
		st = self.body.rebuild_walks(num_walks=int(self.num_walks), eps=float(self.eps), max_steps=int(self.max_steps))
		self.message = f"re-solved with {self.num_walks} walks ({st.t_total_ms:.1f} ms)"

	def cmd_apply_material(self):
		s = self.body.sim
		s.set_material(self.E, self.nu, self.rayleigh_alpha, self.rayleigh_beta)
		s.set_dt(self.dt)
		s.gravity[:] = self.gravity
		s.floor_y = self.floor_y if self.floor_on else None

	def cmd_save_quadrature(self):
		try:
			path = self.body.save_quadrature(self.quad_path)
			self.message = f"saved quadrature points to {path}"
		except (OSError, ValueError) as e:
			self.message = f"save failed: {e}"

	def cmd_save_spec(self):
		b = self.body
		b.set_spec(self.spec_index, self.edit_to_spec())
		try:
			path = b.save_spec(self.spec_index, self.spec_path)
			self.message = f"saved cut spec to {path}"
		except OSError as e:
			self.message = f"save failed: {e}"

	# drag (ctrl + click): the UI provides the picked point and the target, the body the force
	def start_drag(self, point, which, depth, cam_pos):
		self.drag.active = True
		self.drag.point = int(point)
		self.drag.which = which
		self.drag.depth = float(depth)
		self.drag.cam_pos = np.asarray(cam_pos, dtype=np.float64)
		self.drag.target = None

	def event_log(self, n=8):
		"""The last n cut event entries as text lines (newest last)."""
		return [f"cut {s} step {k}: {', '.join(ev)}" for s, k, ev in list(self.events)[-n:]]

	def stop_drag(self):
		self.drag.active = False
		self.drag.target = None
