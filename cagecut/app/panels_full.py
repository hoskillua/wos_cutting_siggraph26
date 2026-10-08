"""Full imgui panels (python -m cagecut.app.main --ui full): every WoS / material parameter, view toggle and
debug layer. The default slim panel is panels.py. Read the AppState, write UI values, issue commands."""

import numpy as np
import polyscope.imgui as psim

from cagecut.app.panels import case_section
from cagecut.app.viz import CAGE_MODES

try:
	_OPEN = int(psim.ImGuiCond_FirstUseEver)
except TypeError:
	_OPEN = psim.ImGuiCond_FirstUseEver

SHORTCUTS = "space: run / pause   n: cut step   b: begin cut   r: reset sim   ctrl+drag: pull the mesh"


def _header(label, default_open=True):
	psim.SetNextItemOpen(default_open, _OPEN)
	return psim.CollapsingHeader(label)


def _button(label, enabled=True):
	"""A button that is greyed out (and never fires) when not enabled."""
	if enabled:
		return psim.Button(label)
	psim.BeginDisabled(True)
	psim.Button(label)
	psim.EndDisabled()
	return False


def status_line(st):
	b = st.body
	cut = "cutting" if b.cut_active else ("cut done" if b.cutter.completed else "no cut")
	sim = st.last_step.sim if st.last_step is not None and st.last_step.sim else None
	ms = f", sim {sim['t_total_ms']:.0f} ms" if sim else ""
	psim.Text(f"{'RUNNING' if st.running else 'paused'}  |  {cut} (spec {b.spec_index}, t = {b.cutter.t:.3f}){ms}")
	psim.TextDisabled(SHORTCUTS)
	psim.Separator()


def scene_panel(st):
	if not _header("Scene", default_open=False):
		return
	b = st.body
	snap = b.cutter.snapshot
	psim.Text(f"{st.scene.name} ({st.scene.dim}D)  cage: {int(snap.vert_alive.sum())} verts, "
		f"{int(snap.face_alive.sum())} faces")
	nm = 0 if b.mesh is None else b.mesh.num_points
	psim.Text(f"quadrature points: {b.quad.num_points} ({int(np.sum(~b.quad_active))} inactive)  mesh points: {nm}")
	psim.Text(f"walk cache: {b.walk_cache_bytes() / 2 ** 20:.1f} MiB")
	if psim.Button("Reset (uncut)"):
		st.cmd_reset()
	psim.SameLine()
	if psim.Button("Reset sim"):
		st.cmd_reset_sim()
	psim.SameLine()
	if psim.Button("Recompute weights"):
		st.cmd_recompute()
	psim.PushItemWidth(120)
	_, st.num_walks = psim.InputInt("walks", st.num_walks, 64, 256)
	st.num_walks = int(max(1, st.num_walks))
	_, st.eps = psim.InputFloat("eps", st.eps, 0.0, 0.0, "%.5f")
	st.eps = float(max(st.eps, 1e-6))
	_, st.max_steps = psim.InputInt("max steps", st.max_steps, 8, 32)
	st.max_steps = int(max(1, st.max_steps))
	psim.PopItemWidth()
	if psim.Button("Apply WoS parameters"):
		st.cmd_apply_wos()
	_, st.quad_path = psim.InputText("quad file", st.quad_path)
	if psim.Button("Save quadrature points"):
		st.cmd_save_quadrature()


def view_panel(st):
	"""Visibility of every structure (also toggled from polyscope's structure list)."""
	if not _header("View"):
		return
	v = st.view
	_, v.mesh = psim.Checkbox("mesh", v.mesh)
	psim.SameLine()
	_, v.cut_faces = psim.Checkbox("cut faces", v.cut_faces)
	psim.SameLine()
	_, v.pinned = psim.Checkbox("pinned", v.pinned)
	modes = CAGE_MODES if st.scene.dim == 3 else ("off", "on")
	cur = v.cage if st.scene.dim == 3 else ("off" if v.cage == "off" else "on")
	psim.PushItemWidth(140)
	c, i = psim.Combo("cage", modes.index(cur), list(modes))
	if c:
		v.cage = modes[i] if st.scene.dim == 3 else ("off" if i == 0 else "wireframe")
	if st.scene.dim == 3 and v.cage in ("faces", "both"):
		psim.SameLine()
		_, v.cage_opacity = psim.SliderFloat("opacity", float(v.cage_opacity), 0.0, 1.0)
	psim.PopItemWidth()
	_, v.blade = psim.Checkbox("blade preview", v.blade)
	psim.SameLine()
	_, v.front = psim.Checkbox("cut front / tips", v.front)
	psim.SameLine()
	_, v.quad_points = psim.Checkbox("quadrature points", v.quad_points)
	if st.scene.env_verts is not None:
		_, v.environment = psim.Checkbox("environment", v.environment)
	if st.scene.dim == 3:
		_, v.ground = psim.Checkbox("ground", v.ground)


def cut_panel(st):
	if not _header("Cut"):
		return
	b = st.body
	n = len(b.specs)
	items = [f"spec {i}" for i in range(n)] + ["new spec"]
	changed, idx = psim.Combo("spec", min(st.spec_index, n), items)
	if changed:
		st.select_spec(idx)
	e = st.edit
	editable = not b.cut_active  # the active blade's spec is fixed until it completes or is reset
	if not editable:
		psim.BeginDisabled(True)
	ca = False
	if e["kind"] == "blade":
		c1, p1 = psim.InputFloat3("p1", [float(x) for x in e["p1"]])
		c2, p2 = psim.InputFloat3("p2", [float(x) for x in e["p2"]])
		if c1:
			e["p1"] = np.array(p1)
		if c2:
			e["p2"] = np.array(p2)
		ca, e["angle"] = psim.SliderFloat("angle (deg)", float(e["angle"]), 0.0, 360.0)
		_, e["bounded"] = psim.Checkbox("bounded (blade is the segment p1-p2)", bool(e.get("bounded", False)))
		_, e["plunge"] = psim.Checkbox("plunge (start inside the material)", bool(e.get("plunge", False)))
	elif "point" in e:
		c1, q = psim.InputFloat2("start point", [float(x) for x in e["point"]])
		if c1:
			e["point"] = np.array(q)
		ca, e["angle"] = psim.SliderFloat("angle (rad, from +x)", float(e["angle"]), -3.1416, 3.1416)
	else:
		psim.Text(f"edge {tuple(int(v) for v in e['edge'])}")
		_, e["alpha"] = psim.SliderFloat("alpha", float(e["alpha"]), 0.01, 0.99)
		ca, e["angle"] = psim.SliderFloat("angle (rad)", float(e["angle"]), -1.5, 1.5)
		_, e["initial"] = psim.InputFloat("initial length", float(e["initial"]), 0.0, 0.0, "%.4f")
	if ca:
		e["_keep_dir"] = False
	if not editable:
		psim.EndDisabled()
	_, e["step_size"] = psim.InputFloat("step size", float(e["step_size"]), 0.0, 0.0, "%.4f")
	e["step_size"] = float(max(e["step_size"], 1e-6))

	if _button("Begin cut", not b.cut_active):
		st.cmd_begin_cut()
	psim.SameLine()
	if _button("Cut step", b.cut_active):
		st.cmd_cut_step()
	psim.SameLine()
	_, st.auto_cut = psim.Checkbox("auto-advance while simulating", st.auto_cut)
	kind = "cut regions" if st.scene.dim == 3 else "slits"
	psim.Text(f"completed specs {b.completed_specs}, {kind}: {len(b.cut_pairs)}")
	log = st.event_log()
	if log:
		psim.Text("events:")
		for line in log:
			psim.BulletText(line)
	_, st.spec_path = psim.InputText("spec file", st.spec_path)
	if psim.Button("Save spec JSON"):
		st.cmd_save_spec()


def sim_panel(st):
	if not _header("Simulation"):
		return
	label = "Pause" if st.running else "Run"
	if psim.Button(label):
		st.running = not st.running
	psim.SameLine()
	if _button("Step", not st.running):
		st.cmd_step()
	psim.SameLine()
	if psim.Button("Reset sim"):
		st.cmd_reset_sim()
	psim.PushItemWidth(120)
	ch = False
	c, st.dt = psim.InputFloat("dt", st.dt, 0.0, 0.0, "%.4f")
	ch |= c
	c, st.E = psim.InputFloat("E", st.E, 1.0, 10.0, "%.2f")
	ch |= c
	c, st.nu = psim.SliderFloat("nu", st.nu, 0.0, 0.49)
	ch |= c
	c, st.rayleigh_alpha = psim.InputFloat("Rayleigh alpha", st.rayleigh_alpha, 0.0, 0.0, "%.4f")
	ch |= c
	c, st.rayleigh_beta = psim.InputFloat("Rayleigh beta", st.rayleigh_beta, 0.0, 0.0, "%.5f")
	ch |= c
	psim.PopItemWidth()
	c, g = psim.InputFloat3("gravity", [float(x) for x in np.resize(st.gravity, 3)])
	if c:
		st.gravity = np.array(g[:st.scene.dim])
		ch = True
	c, st.floor_on = psim.Checkbox("floor", st.floor_on)
	ch |= c
	if st.floor_on:
		psim.SameLine()
		psim.PushItemWidth(100)
		c, st.floor_y = psim.InputFloat("floor y", st.floor_y, 0.0, 0.0, "%.3f")
		psim.PopItemWidth()
		ch |= c
	if ch:
		st.dt = max(st.dt, 1e-5)
		st.cmd_apply_material()
	_, st.drag.stiffness = psim.InputFloat("drag stiffness (ctrl+drag)", st.drag.stiffness, 1.0, 10.0, "%.1f")


def debug_panel(st):
	if not _header("Debug", default_open=False):
		return
	d = st.debug
	_, d.rewalk_heat = psim.Checkbox("rewalk heat map (last update)", d.rewalk_heat)
	_, d.onface_heat = psim.Checkbox("on-face heat map (last update)", d.onface_heat)
	_, d.swept = psim.Checkbox("swept geometry (last step)", d.swept)
	_, d.weight_field = psim.Checkbox("weight field", d.weight_field)
	if d.weight_field:
		psim.SameLine()
		psim.PushItemWidth(100)
		_, d.weight_vertex = psim.InputInt("cage vertex", d.weight_vertex)
		psim.PopItemWidth()
		d.weight_vertex = int(np.clip(d.weight_vertex, 0, st.body.sim.num_verts - 1))
	_, d.timing = psim.Checkbox("timing", d.timing)
	if d.timing:
		timing_text(st)


def timing_text(st):
	c = st.last_cut
	if c is not None and c.changed:
		psim.Text(f"cut step: total {c.t_total_ms:.1f} ms (topology {c.t_topology_ms:.1f}, cage {c.t_gpu_cage_ms:.1f}, "
			f"quad {c.t_quad_ms:.1f}, mesh {c.t_mesh_ms:.1f}, relocate {c.t_relocate_ms:.1f}, split {c.t_split_ms:.1f}, "
			f"sim {c.t_sim_ms:.1f})")
		for name, u in (("quad", c.quad), ("mesh", c.mesh)):
			if u.n_walks:
				psim.Text(f"  {name}: rewalk {u.n_rewalk} ({100.0 * u.n_rewalk / u.n_walks:.2f}%), on-face {u.n_onface} "
					f"({100.0 * u.n_onface / u.n_walks:.2f}%); check {u.t_check_ms:.2f} / rewalk {u.t_rewalk_ms:.2f} / "
					f"on-face {u.t_onface_ms:.2f} ms")
		psim.Text(f"  relocated quadrature points {c.n_relocated}, new mesh points {c.n_split_points}")
	s = st.last_step
	if s is not None and s.sim:
		m = s.sim
		psim.Text(f"sim step: {m['t_total_ms']:.1f} ms (forces {m['t_forces_ms']:.1f}, assemble {m['t_assemble_ms']:.1f}, "
			f"solve {m['t_solve_ms']:.1f}), {m['num_dofs']} dofs")


def draw(st):
	status_line(st)
	case_section(st)
	view_panel(st)
	cut_panel(st)
	sim_panel(st)
	scene_panel(st)
	debug_panel(st)
	if st.message:
		psim.Separator()
		psim.TextWrapped(st.message)
