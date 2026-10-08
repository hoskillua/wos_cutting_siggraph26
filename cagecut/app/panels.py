"""Default (slim) imgui panel: only what is needed to cut and play. The full panel with every parameter
and debug layer is panels_full.py (python -m cagecut.app.main --ui full). No algorithm logic here."""

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

try:
	_OPEN = int(psim.ImGuiCond_FirstUseEver)
except TypeError:
	_OPEN = psim.ImGuiCond_FirstUseEver
_LOG = int(psim.ImGuiSliderFlags_Logarithmic)

SHORTCUTS = "space run / pause   n cut step   b begin cut   r reset sim   ctrl+drag pull"


def _header(label, default_open=True):
	psim.SetNextItemOpen(default_open, _OPEN)
	return psim.CollapsingHeader(label)


def _tip(text):
	"""Tooltip of the previous item."""
	if psim.IsItemHovered():
		psim.SetTooltip(text)


def _button(label, enabled=True):
	"""A button that is greyed out (and never fires) when not enabled."""
	if enabled:
		return psim.Button(label)
	psim.BeginDisabled(True)
	psim.Button(label)
	psim.EndDisabled()
	return False


def case_section(st):
	"""Case selector: the scene configs found in the case folder; Load swaps the whole scene."""
	if not _header("Case"):
		return
	labels = [c.label for c in st.cases]
	if labels:
		psim.PushItemWidth(190)
		changed, idx = psim.Combo("##case", min(st.case_index, len(labels) - 1), labels)
		psim.PopItemWidth()
		if changed:
			st.case_index = idx
		psim.SameLine()
		if psim.Button("Load"):
			st.cmd_load_case(st.case_index)
		_tip("Load the selected case (the current cut, blade edits and simulation are dropped)")
	else:
		psim.TextDisabled("no scene configs in this folder")
	psim.SetNextItemOpen(False, _OPEN)
	if psim.TreeNode("Folder"):
		psim.PushItemWidth(260)
		c, folder = psim.InputText("##folder", st.case_dir)
		psim.PopItemWidth()
		if c:
			st.case_dir = folder  # typed, listed on Enter / Refresh
		psim.SameLine()
		if psim.Button("Refresh"):
			st.refresh_cases(st.case_dir)
		psim.TreePop()


def top(st):
	b = st.body
	if st.running:
		if psim.Button("Pause", (150, 0)):
			st.running = False
		_tip("Pause (space)")
	else:
		if psim.Button("Run all", (72, 0)):
			st.cmd_run(True)
		_tip("Begin the cut and run the simulation (space repeats the last choice)")
		psim.SameLine()
		if psim.Button("Run sim", (72, 0)):
			st.cmd_run(False)
		_tip("Run the simulation only; the blade does not move")
	psim.SameLine()
	if psim.Button("Reset"):
		st.cmd_reset()
	_tip("Back to the uncut body at rest. Keeps your blade and material settings")
	if b.cut_active:
		cut = f"cutting (blade {b.cutter.t:.3f})" if st.scene.dim == 3 else f"cutting (slit {b.cutter.t:.3f})"
	else:
		cut = "cut done" if b.cutter.completed else "not cut"
	mode = ("all" if st.auto_cut else "sim only") if st.running else "paused"
	psim.Text(f"{mode}, {cut}")
	psim.TextDisabled(SHORTCUTS)


def cut_section(st):
	if not _header("Cut"):
		return
	b = st.body
	if len(b.specs) > 1:
		psim.PushItemWidth(120)
		changed, idx = psim.Combo("blade", min(st.spec_index, len(b.specs) - 1), [f"blade {i}" for i in range(len(b.specs))])
		psim.PopItemWidth()
		if changed:
			st.select_spec(idx)
	if _button("Begin cut", not b.cut_active):
		st.cmd_begin_cut()
	_tip("Place the blade and start cutting (b)")
	psim.SameLine()
	if _button("Step", b.cut_active):
		st.cmd_cut_step()
	_tip("Advance the blade by one cut step (n)")
	e = st.edit
	psim.PushItemWidth(180)
	_, e["step_size"] = psim.SliderFloat("cut speed", float(e["step_size"]), 1e-4, 0.1, "%.4f", _LOG)
	psim.PopItemWidth()
	_tip("Blade travel per cut step (smaller: smoother cut, more steps)")
	e["step_size"] = float(max(e["step_size"], 1e-6))
	_blade_editor(st)


def _blade_editor(st):
	"""Blade placement, collapsed by default; locked while its cut is running."""
	e = st.edit
	if not _header("Blade", default_open=False):
		return
	locked = st.body.cut_active
	if locked:
		psim.TextDisabled("(locked while cutting)")
		psim.BeginDisabled(True)
	ca = False
	if e["kind"] == "blade":
		c1, p1 = psim.InputFloat3("p1", [float(x) for x in e["p1"]])
		c2, p2 = psim.InputFloat3("p2", [float(x) for x in e["p2"]])
		if c1:
			e["p1"] = np.array(p1)
		if c2:
			e["p2"] = np.array(p2)
		ca, e["angle"] = psim.SliderFloat("angle", float(e["angle"]), 0.0, 360.0, "%.1f deg")
		_, e["bounded"] = psim.Checkbox("finite blade", bool(e.get("bounded", False)))
		psim.SameLine()
		_, e["plunge"] = psim.Checkbox("start inside", bool(e.get("plunge", False)))
	elif "point" in e:
		c1, q = psim.InputFloat2("start point", [float(x) for x in e["point"]])
		if c1:
			e["point"] = np.array(q)
		ca, e["angle"] = psim.SliderFloat("angle", float(e["angle"]), -3.1416, 3.1416, "%.2f rad")
	else:
		_, e["alpha"] = psim.SliderFloat("position on edge", float(e["alpha"]), 0.01, 0.99)
		ca, e["angle"] = psim.SliderFloat("angle", float(e["angle"]), -1.5, 1.5, "%.2f rad")
	if ca:
		e["_keep_dir"] = False
	if e["kind"] == "blade" or "point" in e:
		if psim.Button("Flip direction"):
			st.cmd_flip_blade()
		_tip("Sweep the blade the other way")
	if locked:
		psim.EndDisabled()
	if psim.Button("Save blade"):
		st.cmd_save_spec()
	_tip("Write this blade to its cut spec file")


def view_section(st):
	if not _header("View"):
		return
	v = st.view
	_, v.mesh = psim.Checkbox("mesh", v.mesh)
	psim.SameLine()
	_, v.cut_faces = psim.Checkbox("cut faces", v.cut_faces)
	psim.SameLine()
	_, v.blade = psim.Checkbox("blade", v.blade)
	_tip("Blade, cut plane and where it meets the body")
	_, v.pinned = psim.Checkbox("pinned faces", v.pinned)
	if st.scene.dim == 3:
		psim.SameLine()
		_, v.ground = psim.Checkbox("ground", v.ground)
	if st.scene.dim == 3:
		modes = ("off", "wireframe", "faces")
		cur = "faces" if v.cage == "both" else v.cage
		psim.PushItemWidth(120)
		c, i = psim.Combo("cage", modes.index(cur), list(modes))
		psim.PopItemWidth()
		if c:
			v.cage = modes[i]
		if v.cage in ("faces", "both"):
			psim.PushItemWidth(120)
			_, v.cage_opacity = psim.SliderFloat("cage opacity", float(v.cage_opacity), 0.05, 1.0, "%.2f")
			psim.PopItemWidth()
	else:
		on = v.cage != "off"
		c, on = psim.Checkbox("cage", on)
		if c:
			v.cage = "wireframe" if on else "off"
	if psim.Button("Reset camera"):
		ps.reset_camera_to_home_view()


def material_section(st):
	if not _header("Material"):
		return
	psim.PushItemWidth(180)
	c, st.E = psim.SliderFloat("stiffness", float(st.E), 0.1, 1000.0, "%.1f", _LOG)
	psim.PopItemWidth()
	g, st.gravity_on = psim.Checkbox("gravity", st.gravity_on)
	if g:
		st.gravity = st.gravity0.copy() if st.gravity_on else np.zeros_like(st.gravity0)
	if c or g:
		st.cmd_apply_material()


def draw(st):
	top(st)
	case_section(st)
	cut_section(st)
	view_section(st)
	material_section(st)
	if st.message:
		psim.Separator()
		psim.TextWrapped(st.message)
