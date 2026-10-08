"""Polyscope structures for the app. Registered once, updated in place; re-registered only when the
cage or mesh connectivity changes (Body.topology_version).

Visibility is two-way: the View flags of the AppState decide what is shown, and toggling a structure in
polyscope's own structure list writes back into those flags (instead of being undone next frame).
Hidden structures are not updated."""

import numpy as np
import polyscope as ps

from cagecut.app.preview import affine_fit, apply_affine, blade_outline, outline_nodes
from cagecut.geometry.polygon import FACE_NGON, triangulate_polygon_with_holes
from cagecut.geometry.snapshot import piece_labels


def pad3(x):
	x = np.asarray(x, dtype=np.float64)
	if x.ndim == 2 and x.shape[1] == 2:
		return np.concatenate([x, np.zeros((x.shape[0], 1))], axis=1)
	return x


NAMES = ("Mesh", "Cage", "Cage edges", "Cut faces", "Quadrature points", "Pinned", "Swept", "Blade", "Cut plane",
	"Cut outline", "Cut front", "Cut tips", "Drag", "Weight vertex", "Environment")

# appearance the user may have changed in polyscope's structure panel; kept when a structure is re-registered
STYLE_KEYS = ("color", "edge_color", "edge_width", "material", "smooth_shade")
PINNED_COLOR = (0.8, 0.8, 0.82)

CAGE_MODES = ("off", "wireframe", "faces", "both")


def face_polygons(snap, faces):
	"""Polygons (vertex id lists) rendering the given 3D faces: triangles and convex quads as they are,
	n-gons (non-convex, slits, holes) triangulated in their face frame."""
	out = []
	for f in faces:
		ids = snap.face(f)
		if snap.face_kind[f] != FACE_NGON:
			out.append([int(x) for x in ids])
			continue
		s, n = int(snap.face_start[f]), int(snap.face_size[f])
		nxt = np.asarray(snap.face_vnext[s:s + n], dtype=np.int64) - s
		# loops (slot order) as 2D frame coordinates
		seen = np.zeros(n, dtype=bool)
		loops, order = [], []
		for j0 in range(n):
			j, loop = j0, []
			while not seen[j]:
				seen[j] = True
				loop.append(j)
				j = int(nxt[j])
			if loop:
				loops.append(snap.face_coords2d[s + np.asarray(loop)])
				order += loop
		for t in triangulate_polygon_with_holes(loops):
			out.append([int(ids[order[k]]) for k in t])
	return out


def face_edges(snap, faces):
	"""Undirected edges (vertex id pairs) of the given faces (every loop)."""
	edges = set()
	for f in faces:
		s, n = int(snap.face_start[f]), int(snap.face_size[f])
		for j in range(s, s + n):
			a, b = int(snap.face_verts[j]), int(snap.face_verts[snap.face_vnext[j]])
			if a != b:
				edges.add((min(a, b), max(a, b)))
	return np.array(sorted(edges), dtype=np.int64).reshape(-1, 2)


class Viz:
	def __init__(self, state):
		self.state = state
		self.dim = state.scene.dim
		self._topo = None
		self._wver = None
		self._dbg = None
		self._swept_of = None
		self._front_of = None
		self._applied = {}  # structure name -> enabled state last applied by the app
		self._opacity = None
		self._net_ids = {}  # curve network name -> cage vertex ids of its nodes
		self._style = {}  # structure name -> appearance carried over a re-registration
		self._outline_of = None
		self._outline = (np.zeros((0, 2, 3)), np.zeros((0, 2)))
		self._pin_polys = None
		self._blade_done = False  # the edited blade's cut is finished: its preview is hidden
		self._labels = (None, None)  # (topology version, piece label per cage vertex)
		self._register_static()
		self.sync(force=True)

	# -----------------------------------------------------------------------------------------
	# visibility

	def _vis_table(self):
		"""Structure name -> (wanted visibility, setter writing a polyscope toggle back into the View flags)."""
		v = self.state.view

		def flag(attr):
			return (lambda: bool(getattr(v, attr)), lambda x: setattr(v, attr, bool(x)))
		t = {
			"Mesh": flag("mesh"),
			"Cut faces": flag("cut_faces"),
			"Quadrature points": flag("quad_points"),
			"Pinned": flag("pinned"),
			"Blade": (lambda: bool(v.blade) and not self._blade_done, lambda x: setattr(v, "blade", bool(x))),
			"Cut plane": (lambda: bool(v.blade) and not self._blade_done, lambda x: setattr(v, "blade", bool(x))),
			"Cut outline": flag("blade"),
			"Cut front": flag("front"),
			"Cut tips": flag("front"),
			"Environment": flag("environment"),
		}
		if self.dim == 3:
			t["Cage"] = (lambda: v.cage in ("faces", "both"), lambda x: self._set_cage(faces=x))
			t["Cage edges"] = (lambda: v.cage in ("wireframe", "both"), lambda x: self._set_cage(wire=x))
		else:
			t["Cage"] = (lambda: v.cage != "off", lambda x: setattr(v, "cage", "wireframe" if x else "off"))
		return t

	def _set_cage(self, faces=None, wire=None):
		v = self.state.view
		f = v.cage in ("faces", "both") if faces is None else bool(faces)
		w = v.cage in ("wireframe", "both") if wire is None else bool(wire)
		v.cage = {(False, False): "off", (False, True): "wireframe", (True, False): "faces", (True, True): "both"}[(f, w)]

	def visible(self, name):
		get = self._vis_table().get(name)
		return bool(get[0]()) if get is not None else True

	def _registered(self, name, s):
		"""Apply the wanted visibility to a freshly registered structure."""
		for key, val in self._style.get(name, {}).items():
			try:
				getattr(s, "set_" + key)(val)
			except (AttributeError, TypeError, ValueError):
				pass
		on = self.visible(name)
		s.set_enabled(on)
		self._applied[name] = on
		return s

	def _network(self, name, x, edges, **kw):
		"""Curve network on the nodes the edges use only (a curve network draws every node it is given, so
		unused / dead cage vertices would show as stray dots). Positions are updated with _net_nodes."""
		ids, local = np.unique(np.asarray(edges, dtype=np.int64), return_inverse=True)
		self._net_ids[name] = ids
		return self._registered(name, ps.register_curve_network(name, x[ids], local.reshape(-1, 2), **kw))

	def _net_nodes(self, name, x):
		return x[self._net_ids[name]]

	def _capture_styles(self):
		"""Remember the appearance of the live structures, so a re-registration (the cage changed) keeps it."""
		for name in NAMES:
			if not self._has(name):
				continue
			s = self._get(name)
			d = {}
			for key in STYLE_KEYS:
				try:
					d[key] = getattr(s, "get_" + key)()
				except (AttributeError, TypeError):
					pass
			self._style[name] = d

	def _remove(self, name):
		for rm in (ps.remove_surface_mesh, ps.remove_point_cloud, ps.remove_curve_network):
			try:
				rm(name, error_if_absent=False)
			except TypeError:
				pass

	def _sync_ground(self):
		if self.dim != 3:
			return
		want = bool(self.state.view.ground)
		if want != self._ground:
			ps.set_ground_plane_mode("tile" if want else "none")
			self._ground = want

	def _sync_visibility(self):
		for name, (get, set_) in self._vis_table().items():
			if not self._has(name):
				self._applied.pop(name, None)
				continue
			s = self._get(name)
			cur = bool(s.is_enabled())
			last = self._applied.get(name)
			if last is not None and cur != last:
				set_(cur)  # toggled in polyscope's structure list
			want = bool(get())
			if cur != want:
				s.set_enabled(want)
			self._applied[name] = want
		v = self.state.view
		if self.dim == 3 and ps.has_surface_mesh("Cage") and self._opacity != v.cage_opacity:
			ps.get_surface_mesh("Cage").set_transparency(float(v.cage_opacity))
			self._opacity = v.cage_opacity

	# -----------------------------------------------------------------------------------------

	def _register_static(self):
		s = self.state.scene
		self._ground = None
		if self.dim == 2:
			ps.set_navigation_style("planar")
			ps.set_up_dir("y_up")
			ps.set_ground_plane_mode("none")
		else:
			ps.set_navigation_style("turntable")
			ps.set_up_dir("y_up")
		if s.env_verts is not None and s.env_tris is not None:
			self._registered("Environment", ps.register_surface_mesh("Environment", s.env_verts, s.env_tris,
				color=(0.6, 0.6, 0.6)))
		# blade preview: line, sweep arrow (curve network) and the cut plane quad
		nodes = np.zeros((5, 3))
		edges = np.array([[0, 1], [2, 3], [3, 4]])
		self._registered("Blade", ps.register_curve_network("Blade", nodes, edges, radius=0.002, color=(0.9, 0.2, 0.1)))
		if self.dim == 3:
			self._registered("Cut plane", ps.register_surface_mesh("Cut plane", np.zeros((4, 3)), np.array([[0, 1, 2, 3]]),
				color=(0.95, 0.6, 0.2), transparency=0.3))
		dr = ps.register_curve_network("Drag", np.zeros((2, 3)), np.array([[0, 1]]), radius=0.002, color=(0.1, 0.3, 0.9))
		dr.set_enabled(False)

	def _register_topology(self):
		st = self.state
		b = st.body
		snap = b.cutter.snapshot
		self._capture_styles()
		x = pad3(b.cage_positions())
		alive = [f for f in range(snap.num_faces) if snap.face_alive[f]]
		cut_alive = [f for f in sorted(b.cut_face_ids) if snap.face_alive[f]]
		if self.dim == 3:
			self._registered("Cage", ps.register_surface_mesh("Cage", x, face_polygons(snap, alive), color=(0.3, 0.5, 0.9),
				transparency=float(st.view.cage_opacity), edge_width=0.0))
			self._opacity = st.view.cage_opacity
			self._network("Cage edges", x, face_edges(snap, alive), radius=0.0012, color=(0.15, 0.15, 0.25))
			# cut faces may have several loops (holes) and merge: triangulated per face
			cut = face_polygons(snap, cut_alive)
			if cut:
				self._registered("Cut faces", ps.register_surface_mesh("Cut faces", x, cut, color=(0.95, 0.3, 0.2),
					edge_width=1.0))
			elif ps.has_surface_mesh("Cut faces"):
				ps.remove_surface_mesh("Cut faces")
			pts = pad3(b.mesh_positions())
			if b.mesh_tris is not None and len(b.mesh_tris):
				self._registered("Mesh", ps.register_surface_mesh("Mesh", pts, b.mesh_tris, color=(0.95, 0.75, 0.4),
					smooth_shade=True))
			elif len(pts):
				self._registered("Mesh", ps.register_point_cloud("Mesh", pts, radius=0.003, color=(0.95, 0.75, 0.4)))
		else:
			segs = np.array([snap.face(f) for f in alive], dtype=np.int64).reshape(-1, 2)
			self._network("Cage", x, segs, radius=0.002, color=(0.2, 0.3, 0.8))
			cut = np.array([snap.face(f) for f in cut_alive], dtype=np.int64).reshape(-1, 2)
			if len(cut):
				self._network("Cut faces", x, cut, radius=0.003, color=(0.95, 0.3, 0.2))
			elif ps.has_curve_network("Cut faces"):
				ps.remove_curve_network("Cut faces")
			pts = pad3(b.mesh_positions())
			if len(pts):
				self._registered("Mesh", ps.register_point_cloud("Mesh", pts, radius=0.004, color=(0.95, 0.6, 0.2)))
		self._registered("Quadrature points", ps.register_point_cloud("Quadrature points", pad3(b.quad_positions()),
			radius=0.003, color=(0.2, 0.8, 0.3)))
		self._register_pinned(snap, alive, x)
		self._topo = b.topology_version
		self._wver = None
		self._dbg = None
		self._front_of = None

	# -----------------------------------------------------------------------------------------

	def _register_pinned(self, snap, alive, x):
		"""Pinned faces as a filled light grey surface (3D) / thick segments (2D): the faces whose vertices
		are all pinned."""
		pin = set(int(i) for i in self.state.body.sim.pinned)
		faces = [f for f in alive if all(int(i) in pin for i in snap.face(f))] if pin else []
		self._pin_polys = None
		if not faces:
			self._remove("Pinned")
			return
		if self.dim == 3:
			polys = face_polygons(snap, faces)
			ids = np.concatenate([np.asarray(p, dtype=np.int64) for p in polys])
			sizes = np.array([len(p) for p in polys])
			starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
			self._pin_polys = (ids, starts, [list(range(a, a + k)) for a, k in zip(starts, sizes)])
			self._registered("Pinned", ps.register_surface_mesh("Pinned", self._pin_positions(x), self._pin_polys[2],
				color=PINNED_COLOR, smooth_shade=False))
		else:
			segs = np.array([snap.face(f) for f in faces], dtype=np.int64).reshape(-1, 2)
			self._network("Pinned", x, segs, radius=0.005, color=PINNED_COLOR)

	def _pin_positions(self, x):
		"""Vertices of the pinned polygons, lifted a hair along the face normal so they are not hidden by the
		cage faces beneath them."""
		ids, starts, _ = self._pin_polys
		P = x[ids]
		a, b, c = P[starts], P[starts + 1], P[starts + 2]
		nrm = np.cross(b - a, c - a)
		nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-30)
		sizes = np.diff(np.concatenate([starts, [len(ids)]]))
		ext = float(np.max(np.ptp(self.state.scene.cage_verts, axis=0)))
		return P + 0.002 * ext * np.repeat(nrm, sizes, axis=0)

	def _sync_pinned(self, x):
		if not self._has("Pinned") or not self._get("Pinned").is_enabled():
			return
		if self.dim == 3 and self._pin_polys is not None:
			ps.get_surface_mesh("Pinned").update_vertex_positions(self._pin_positions(x))
		elif self.dim == 2:
			ps.get_curve_network("Pinned").update_node_positions(self._net_nodes("Pinned", x))

	def sync(self, force=False):
		"""Bring every structure up to date with the state (cheap when nothing changed)."""
		st = self.state
		b = st.body
		if force or self._topo != b.topology_version or self._cage_count() != b.sim.num_verts:
			self._register_topology()
		self._sync_visibility()
		self._sync_ground()
		x = pad3(b.cage_positions())
		for name in ("Cage", "Cage edges", "Cut faces"):
			if self._has(name) and self._get(name).is_enabled():
				s = self._get(name)
				if hasattr(s, "update_vertex_positions"):
					s.update_vertex_positions(x)
				else:
					s.update_node_positions(self._net_nodes(name, x))
		self._sync_pinned(x)
		if b.mesh is not None and self._has("Mesh") and self._get("Mesh").is_enabled():
			pts = pad3(b.mesh_positions())
			m = self._get("Mesh")
			if hasattr(m, "update_vertex_positions"):
				m.update_vertex_positions(pts)
			else:
				m.update_point_positions(pts)
		if self._has("Quadrature points") and self._get("Quadrature points").is_enabled():
			ps.get_point_cloud("Quadrature points").update_point_positions(pad3(b.quad_positions()))
		self._sync_debug()
		self._sync_swept()
		self._sync_front(x)
		self._sync_preview()
		self._sync_drag()

	def _sync_front(self, x):
		"""Front segments (3D) and tips of the active blade on the deformed cage."""
		st = self.state
		b = st.body
		on = bool(st.view.front) and b.cut_active
		key = (self._topo, id(None if st.last_cut is None else st.last_cut.delta), b.spec_index, on)
		if key != self._front_of:
			self._front_of = key
			self._front = b.blade_front() if on else (np.zeros(0, np.int64), np.zeros((0, 2), np.int64))
			for name in ("Cut front", "Cut tips"):
				if self._has(name):
					(ps.remove_curve_network if name == "Cut front" else ps.remove_point_cloud)(name)
			tips, segs = self._front
			if len(segs):
				# only the front's own nodes: a curve network draws every node it is given
				self._front_ids, local = np.unique(segs, return_inverse=True)
				self._registered("Cut front", ps.register_curve_network("Cut front", x[self._front_ids],
					local.reshape(-1, 2), radius=0.003, color=(1.0, 0.85, 0.1)))
			if len(tips):
				self._registered("Cut tips", ps.register_point_cloud("Cut tips", x[tips], radius=0.006, color=(1.0, 0.5, 0.0)))
			return
		tips, segs = self._front
		if len(segs) and ps.has_curve_network("Cut front"):
			ps.get_curve_network("Cut front").update_node_positions(x[self._front_ids])
		if len(tips) and ps.has_point_cloud("Cut tips"):
			ps.get_point_cloud("Cut tips").update_point_positions(x[tips])

	def _cage_count(self):
		if self.dim == 3:
			return ps.get_surface_mesh("Cage").n_vertices() if ps.has_surface_mesh("Cage") else -1
		return -2 if not ps.has_curve_network("Cage") else self.state.body.sim.num_verts

	def _has(self, name):
		return ps.has_surface_mesh(name) or ps.has_point_cloud(name) or ps.has_curve_network(name)

	def _get(self, name):
		if ps.has_surface_mesh(name):
			return ps.get_surface_mesh(name)
		if ps.has_point_cloud(name):
			return ps.get_point_cloud(name)
		return ps.get_curve_network(name)

	def _set_enabled(self, name, on):
		if self._has(name):
			s = self._get(name)
			if s.is_enabled() != bool(on):
				s.set_enabled(bool(on))

	def _sync_debug(self):
		"""Scalar quantities: rewalk / on-face heat maps and the weight field (only when they change)."""
		st = self.state
		b = st.body
		d = st.debug
		key = (b.weights_version, self._topo, d.rewalk_heat, d.onface_heat, d.weight_field, d.weight_vertex)
		if key == self._dbg:
			return
		self._dbg = key
		q = ps.get_point_cloud("Quadrature points")
		marks_q = b.quad.last_marks
		for name, on, arr in (("rewalks", d.rewalk_heat, marks_q["rewalk"]), ("on-face updates", d.onface_heat, marks_q["onface"])):
			if on:
				q.add_scalar_quantity(name, np.asarray(arr, dtype=np.float64), enabled=True, cmap="reds")
			else:
				q.remove_quantity(name)
		if b.mesh is None or not self._has("Mesh"):
			return
		m = self._get("Mesh")
		marks = b.mesh.last_marks
		for name, on, arr in (("rewalks", d.rewalk_heat, marks["rewalk"]), ("on-face updates", d.onface_heat, marks["onface"])):
			if on:
				m.add_scalar_quantity(name, np.asarray(arr, dtype=np.float64), enabled=not d.weight_field, cmap="reds")
			else:
				m.remove_quantity(name)
		V = b.sim.num_verts
		j = int(np.clip(d.weight_vertex, 0, V - 1))
		if d.weight_field:
			w = b.corrected_mesh_weights()[:, j]
			m.add_scalar_quantity("weight field", w, enabled=True, cmap="coolwarm", vminmax=(-0.2, 1.0))
			wv = ps.register_point_cloud("Weight vertex", pad3(b.cage_positions()[j:j + 1]), radius=0.008,
				color=(1.0, 0.9, 0.1))
			wv.set_enabled(True)
		else:
			m.remove_quantity("weight field")
			if ps.has_point_cloud("Weight vertex"):
				ps.remove_point_cloud("Weight vertex")

	def _sync_swept(self):
		st = self.state
		info = st.last_cut
		delta = None if info is None else info.delta
		key = (id(delta), st.debug.swept)
		if key == self._swept_of:
			return
		self._swept_of = key
		for rm in (ps.remove_surface_mesh, ps.remove_curve_network):
			try:
				rm("Swept", error_if_absent=False)
			except TypeError:
				pass
		if not st.debug.swept or delta is None or delta.new_cut.shape[0] == 0:
			return
		sw = delta.new_cut
		if self.dim == 3:
			v = sw.reshape(-1, 3)
			ps.register_surface_mesh("Swept", v, np.arange(v.shape[0]).reshape(-1, 3), color=(0.9, 0.1, 0.6))
		else:
			v = pad3(sw.reshape(-1, 2))
			ps.register_curve_network("Swept", v, np.arange(v.shape[0]).reshape(-1, 2), radius=0.004, color=(0.9, 0.1, 0.6))

	def _display_map(self):
		"""Affine map from the rest cage (where the cut is computed) onto the displayed, deformed cage: the
		blade preview follows the body instead of staying where it was at rest. Fitted to the pieces that
		hold the pinned vertices (else all of them), so a piece the cut dropped does not drag the preview
		along with it."""
		b = self.state.body
		snap = b.cutter.snapshot
		alive = np.asarray(snap.vert_alive, dtype=bool)
		x = np.asarray(b.cage_positions(), dtype=np.float64)
		rest = snap.verts
		if x.shape != rest.shape or alive.sum() < self.dim + 1:
			return None
		use = alive
		pin = np.asarray(b.sim.pinned, dtype=np.int64)
		pin = pin[(pin < len(alive)) & alive[pin]] if len(pin) else pin
		if len(pin):
			if self._labels[0] != self._topo or len(self._labels[1]) != len(alive):
				self._labels = (self._topo, np.asarray(piece_labels(snap)))
			held = alive & np.isin(self._labels[1], np.unique(self._labels[1][pin]))
			if held.sum() >= self.dim + 1:
				use = held
		if np.array_equal(rest[use], x[use]):
			return None
		return affine_fit(rest[use], x[use])

	def _sync_preview(self):
		"""Blade line, sweep direction, cut plane and the outline where the plane meets the cage, for the
		edited spec (the live blade while cutting). All drawn on the displayed (deformed) body."""
		st = self.state
		if not st.view.blade:
			return
		spec = st.edit_to_spec()
		b = st.body
		self._blade_done = spec["kind"] == "blade" and self._cut_done(spec)
		ext = float(np.max(np.ptp(st.scene.cage_verts, axis=0)))
		M = self._display_map()
		to_world = (lambda P: pad3(P)) if M is None else (lambda P: pad3(apply_affine(M, P)))
		live = b.cut_active and b.spec_index == st.spec_index
		if spec["kind"] == "blade":
			p1, p2, d = spec["p1"], spec["p2"], spec["direction"]
			l = (p2 - p1) / np.linalg.norm(p2 - p1)
			t = b.cutter.t if live else 0.0
			if spec.get("bounded", False):
				a, e2 = p1, p2  # a finite blade: its segment
			else:
				a, e2 = p1 - ext * l, p1 + ext * l
			m = 0.5 * (a + e2)
			nodes = np.array([a + t * d, e2 + t * d, m + t * d, m + (t + 0.3 * ext) * d, m + (t + 0.3 * ext) * d])
			ps.get_curve_network("Blade").update_node_positions(to_world(nodes))
			quad = np.array([a, e2, e2 + 1.5 * ext * d, a + 1.5 * ext * d])
			ps.get_surface_mesh("Cut plane").update_vertex_positions(to_world(quad))
			self._sync_outline(spec, self._cut_done(spec))
		else:
			X = st.scene.cage_verts
			if "point" in spec:
				q = np.asarray(spec["point"], dtype=np.float64)
			else:
				a, e = spec["edge"]
				q = (1.0 - spec["alpha"]) * X[a] + spec["alpha"] * X[e]
			d = spec["direction"]
			t = b.cutter.t if live else spec["initial"]
			nodes = np.array([q, q + t * d, q, q + 0.3 * ext * d, q + 0.3 * ext * d])
			ps.get_curve_network("Blade").update_node_positions(to_world(nodes))

	def _cut_done(self, spec):
		"""The edited blade is the one whose cut just completed (nothing left to preview)."""
		b = self.state.body
		done = b.spec
		if not b.cutter.completed or done is None or done.get("kind") != "blade":
			return False
		same = all(np.allclose(spec[k], done[k], atol=1e-9) for k in ("p1", "p2", "direction"))
		return same and bool(spec.get("bounded", False)) == bool(done.get("bounded", False))

	def _sync_outline(self, spec, hidden):
		"""Where the blade plane meets the cage (the part a finite blade covers): computed on the rest cage
		when the blade or the cage changes, placed on the deformed cage every frame. Hidden while cutting
		(the cut front shows the intersection then)."""
		st = self.state
		b = st.body
		key = (None if hidden else tuple(np.round(np.concatenate([spec["p1"], spec["p2"], spec["direction"]]), 9)) + (
			bool(spec.get("bounded", False)), self._topo, b.cutter.snapshot.num_verts))
		if key != self._outline_of:
			self._outline_of = key
			self._remove("Cut outline")
			self._outline = (np.zeros((0, 2, 3)), np.zeros((0, 2)))
			if key is not None:
				try:
					self._outline = blade_outline(b.cutter.snapshot, spec)
				except (FloatingPointError, ValueError):
					pass
			ends, tau = self._outline
			if len(ends):
				m = len(ends)
				x = np.asarray(b.cage_positions(), dtype=np.float64)
				self._registered("Cut outline", ps.register_curve_network("Cut outline", outline_nodes(ends, tau, x),
					np.stack([2 * np.arange(m), 2 * np.arange(m) + 1], axis=1), radius=0.003, color=(0.95, 0.5, 0.1)))
			return
		ends, tau = self._outline
		if len(ends) and ps.has_curve_network("Cut outline") and ps.get_curve_network("Cut outline").is_enabled():
			x = np.asarray(b.cage_positions(), dtype=np.float64)
			ps.get_curve_network("Cut outline").update_node_positions(outline_nodes(ends, tau, x))

	def _sync_drag(self):
		dr = self.state.drag
		on = dr.active and dr.target is not None
		self._set_enabled("Drag", on)
		if on:
			b = self.state.body
			W = b.corrected_mesh_weights() if dr.which == "mesh" else b.quad_weights()
			p = W[dr.point] @ b.sim.x
			ps.get_curve_network("Drag").update_node_positions(pad3(np.array([p, dr.target])))
