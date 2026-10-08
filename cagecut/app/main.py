"""cagecut interactive app: python -m cagecut.app.main [config.json] [--ui full] [--walks N] [--backend openGL_mock]

The default panel has Run / Reset, Case, Cut, View and Material sections (--ui full adds every parameter
and debug layer). Ctrl + left click on the embedded mesh picks a point and
drags it with a spring (force applied to the cage through the point's weights).
"""

import argparse
import os
import sys
import types

if __package__ in (None, ""):
	# run as a file (python cagecut/app/main.py): make the package importable
	sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import warp as wp

from cagecut import REPO_ROOT
from cagecut.app import panels, panels_full
from cagecut.app.state import AppState
from cagecut.app.viz import Viz
from cagecut.scene import load_scene

DEFAULT_CONFIG = os.path.join(REPO_ROOT, "data", "star-config.json")


def build(config_path=DEFAULT_CONFIG, walks=None, device=None, backend=None, body_kwargs=None):
	"""Initialize polyscope, load the scene, build the single AppState and the structures."""
	ps.set_program_name("cagecut")
	ps.set_verbosity(0)
	ps.set_use_prefs_file(False)
	if backend:
		ps.init(backend)
	else:
		ps.init()
	scene = load_scene(config_path)
	kw = dict(body_kwargs or {})
	if walks is not None:
		kw["num_walks"] = int(walks)
	state = AppState(scene, kw, device=device)
	viz = Viz(state)
	return state, viz


# ---------------------------------------------------------------------------------------------
# mouse: ctrl + click picks an embedded point, dragging moves the spring target

def _pick_point(st, res):
	"""Embedded point id of a pick result (None if the pick did not hit the embedded geometry)."""
	b = st.body
	if not res.is_hit:
		return None, None
	if res.structure_name == "Mesh" and b.mesh is not None:
		data = res.structure_data
		et = data.get("element_type", "vertex")
		idx = int(data.get("index", res.local_index))
		if et == "vertex" or b.mesh_tris is None or st.scene.dim == 2:
			return idx, "mesh"
		if et == "face" and "bary_coords" in data:
			return int(b.mesh_tris[idx][int(np.argmax(data["bary_coords"]))]), "mesh"
		pts = b.mesh_positions()
		return int(np.argmin(np.sum((pts - res.position[:st.scene.dim]) ** 2, axis=1))), "mesh"
	if res.structure_name == "Quadrature points":
		return int(res.structure_data.get("index", res.local_index)), "quad"
	return None, None


def _target(st, mouse):
	ray = np.asarray(ps.screen_coords_to_world_ray(mouse), dtype=np.float64)
	ray = ray / max(np.linalg.norm(ray), 1e-12)
	cam = st.drag.cam_pos
	if st.scene.dim == 2:
		t = -cam[2] / ray[2] if abs(ray[2]) > 1e-12 else st.drag.depth
		return (cam + t * ray)[:2]
	return cam + st.drag.depth * ray


def handle_mouse(st):
	io = psim.GetIO()
	if st.drag.active:
		if not psim.IsMouseDown(0):
			st.stop_drag()
			return
		psim.SetNextFrameWantCaptureMouse(True)  # keep the camera still while dragging
		st.drag.target = _target(st, psim.GetMousePos())
		return
	if io.KeyCtrl and psim.IsMouseClicked(0) and not io.WantCaptureMouse:
		mouse = psim.GetMousePos()
		res = ps.pick(screen_coords=(mouse[0], mouse[1]))
		pid, which = _pick_point(st, res)
		if pid is None:
			return
		cam = np.asarray(ps.get_view_camera_parameters().get_position(), dtype=np.float64)
		st.start_drag(pid, which, np.linalg.norm(np.asarray(res.position) - cam), cam)
		st.drag.target = _target(st, mouse)
		psim.SetNextFrameWantCaptureMouse(True)


def handle_keys(st):
	"""space: run / pause (cut + sim, or sim only as last chosen), n: cut step, b: begin cut, r: reset sim
	(not while typing in a field)."""
	if psim.GetIO().WantTextInput:
		return
	if psim.IsKeyPressed(psim.ImGuiKey_Space, False):
		st.cmd_toggle_run()
	elif psim.IsKeyPressed(psim.ImGuiKey_N, False):
		st.cmd_cut_step()
	elif psim.IsKeyPressed(psim.ImGuiKey_B, False) and not st.body.cut_active:
		st.cmd_begin_cut()
	elif psim.IsKeyPressed(psim.ImGuiKey_R, False):
		st.cmd_reset_sim()


def swap_case(ctx, path):
	"""Load the scene config `path` into a fresh AppState / Viz (ctx.st, ctx.viz). On any failure the current
	case stays untouched and says why. The View flags carry over; everything else (body, GPU caches, drag,
	history, edits) belongs to the old case and is dropped."""
	old = ctx.st
	name = os.path.basename(path)
	try:
		scene = load_scene(path)
		new = AppState(scene, old.body_kwargs, device=old.device)
	except Exception as e:  # a bad config must not take the app down
		old.message = f"cannot load {name}: {type(e).__name__}: {e}"
		return False
	new.view = old.view
	new.refresh_cases(old.case_dir)
	new.message = f"loaded {name}"
	ps.remove_all_structures()
	ctx.st = new
	ctx.viz = Viz(new)
	ps.reset_camera_to_home_view()
	return True


def make_callback(st, viz, ui="simple"):
	"""The polyscope user callback. `callback.ctx` holds the live state and structures (they are replaced
	when another case is loaded, so look them up there)."""
	ui_mod = panels_full if ui == "full" else panels
	ctx = types.SimpleNamespace(st=st, viz=viz)

	def callback():
		st = ctx.st
		st.frame += 1
		handle_keys(st)
		handle_mouse(st)
		if st.running:
			st.cmd_step()
		ui_mod.draw(st)
		ctx.viz.sync()
		if st.pending_case:  # after the frame: the swap rebuilds every structure
			path, st.pending_case = st.pending_case, None
			st.running = False
			swap_case(ctx, path)
	callback.ctx = ctx
	return callback


def main(argv=None):
	ap = argparse.ArgumentParser(description="cagecut: cutting cage based deformation with walk on spheres")
	ap.add_argument("config", nargs="?", default=DEFAULT_CONFIG)
	ap.add_argument("--walks", type=int, default=None, help="walks per point (default: scene value)")
	ap.add_argument("--device", default=None)
	ap.add_argument("--backend", default=None, help="polyscope backend, e.g. openGL_mock for headless runs")
	ap.add_argument("--frames", type=int, default=0, help="headless: run this many frames and exit")
	ap.add_argument("--run", action="store_true", help="start with the simulation running")
	ap.add_argument("--cases", default=None, help="folder whose scene configs the Case selector lists "
		"(default: the folder of the config)")
	ap.add_argument("--ui", choices=("simple", "full"), default="simple",
		help="simple: cut / view / material essentials (default); full: every parameter and debug layer")
	args = ap.parse_args(argv)
	wp.config.log_level = wp.LOG_WARNING  # no per-module load messages
	st, viz = build(args.config, walks=args.walks, device=args.device, backend=args.backend)
	if args.cases:
		st.refresh_cases(os.path.abspath(args.cases))
	st.running = args.run
	cb = make_callback(st, viz, args.ui)
	ps.set_user_callback(cb)
	if args.frames > 0:
		ps.show(forFrames=args.frames)
		print(cb.ctx.st.message)
		return cb.ctx.st
	ps.show()
	return cb.ctx.st


if __name__ == "__main__":
	main()
