"""3D cut topology: halfedge mesh of planar polygon faces cut by a swept infinite blade line.

The blade cuts all material it
sweeps over: the cross-section R (cut plane ∩ cage interior) may have several components, be
non-convex and have holes, so any number of tip pairs (front segments) and cut regions can be active.

Events are the crossing edges (cage edges whose endpoints lie on opposite logical sides), processed
in increasing (time, edge id) order. Each cage face hosts at most one tip at a time (the chords of a
planar face are disjoint intervals of one line), so a tip "heads to" a crossing edge iff it lives in
one of the edge's two faces:
	no tip   -> BIRTH (new cut region) or SPLIT (the crossing lies inside a front segment)
	one tip  -> TIP (the tip moves into the next face)
	two tips -> CLOSE (same front segment), MERGE (two regions join), HOLE (same region, other front)

Halfedges are append-only arrays; `a -> b` denotes a halfedge with tail a, head b,
tail(h) = head(prev(h)). Faces are CCW around their outward normal; a face may have several
boundary loops (outer CCW first, holes CW), listed by one representative halfedge per loop. Face and
vertex ids are persistent (append-only). Merged-away cut faces stay as dead faces.

Tips are never integrated: their positions are recomputed in closed form from the blade time and the
plane of the face hosting them. Vertex "sides" are logical labels (+1/-1, 0 for tips): computed from
geometry at `begin` for existing vertices and assigned on creation for the coincident slit copies, so
crossing edges are found combinatorially and never by testing points that lie on the cut plane.
"""

import numpy as np

from cagecut.cut.blade import Blade3D, CutError
from cagecut.cut.delta import CutDelta
from cagecut.geometry.polygon import (FACE_NGON, classify_face, frame_from_normal, newell_normal, point_in_polygon_2d,
	signed_area_2d, to_plane_2d)
from cagecut.geometry.snapshot import CUT_ROOT, CageSnapshot, build_snapshot, piece_labels

_INT = np.int32
EVENTS = ("birth", "split", "tip", "close", "merge", "hole", "plunge", "end_birth", "tip_end", "end_tip", "end_close")


class _Step:
	"""Bookkeeping of one advance (several events may happen)."""

	def __init__(self, num_faces, num_verts, t_old):
		self.F_old = num_faces
		self.V_old = num_verts
		self.t_old = t_old
		self.anc = {}  # new face id -> step-start ancestor (-1 for faces created from nothing)
		self.modified = set()  # step-start face ids that changed
		self.sources = {}  # new vertex id -> (ids, weights)
		self.orphaned = []
		self.merged = {}  # step-start dead face -> step-start face it was merged into
		self.events = []
		self.tris = []  # swept triangles
		self.anchor = {}  # tip -> (time, point): exact tip position at its last event of this step


class _Region:
	"""A connected cut region of the current blade: its + and - cut faces, and the label of the cage
	component (at `begin`) it lies in."""
	__slots__ = ("faces", "alive", "label")

	def __init__(self, plus, minus, label):
		self.faces = {1: plus, -1: minus}
		self.alive = True
		self.label = label


class Cutter3D:
	def __init__(self, verts, faces):
		verts = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
		faces = [np.asarray(f, dtype=np.int64) for f in faces]
		self._pos = [verts[i].copy() for i in range(verts.shape[0])]
		self._valive = [True] * verts.shape[0]
		self._side = [0] * verts.shape[0]

		self.he_head, self.he_next, self.he_prev, self.he_twin, self.he_face = [], [], [], [], []
		# line-defining (tail, head) vertex ids of split edge pieces (None: the halfedge's own endpoints)
		self.he_ref = []
		self.face_reps = []  # per face: one halfedge per boundary loop, outer loop first ([] when dead)
		self.face_alive = []
		self.face_root = []  # split-invariant ancestor id (see snapshot.py)
		edge = {}
		for f, fv in enumerate(faces):
			n = len(fv)
			if n < 3:
				raise ValueError(f"face {f} has fewer than 3 vertices")
			if len(set(int(x) for x in fv)) != n:
				raise ValueError(f"face {f} repeats a vertex")
			base = len(self.he_head)
			for i in range(n):
				tail, head = int(fv[i]), int(fv[(i + 1) % n])
				h = self._new_he(head, f)
				if (tail, head) in edge:
					raise ValueError(f"non-manifold or inconsistently oriented edge ({tail}, {head})")
				edge[(tail, head)] = h
			for i in range(n):
				self._link(base + i, base + (i + 1) % n)
			# the representative's head is fv[0], so loop order == input order
			self.face_reps.append([base + n - 1])
			self.face_alive.append(True)
			self.face_root.append(f)
		for (tail, head), h in edge.items():
			t = edge.get((head, tail))
			if t is None:
				raise ValueError(f"cage is not closed: edge ({tail}, {head}) has no twin")
			self.he_twin[h] = t
		used = np.zeros(len(self._pos), dtype=bool)
		for fv in faces:
			used[fv] = True
		self._valive = [bool(u) for u in used]

		snap = build_snapshot(3, verts, faces, vert_alive=np.array(self._valive))
		self.face_n = [snap.face_normal[f].copy() for f in range(len(faces))]
		self.face_o = [float(snap.face_offset[f]) for f in range(len(faces))]
		# planarity tolerance per face: input faces are only planar up to their own deviation
		self.face_tol = []
		for f, fv in enumerate(faces):
			dev = np.max(np.abs(verts[fv] @ self.face_n[f] - self.face_o[f]))
			self.face_tol.append(max(1e-9, 1.01 * dev + 1e-14))
		# per face export cache: (loops, edge refs, local vnext); dead faces keep the export of their death
		self._export = []
		for fv in faces:
			fl = np.asarray(fv, dtype=_INT)
			self._export.append(([fl], np.stack([fl, np.roll(fl, -1)], axis=1).astype(_INT), _local_next([fl])))
		self._face_geom = [(snap.face_u[f], snap.face_v[f],
			snap.face_coords2d[snap.face_start[f]:snap.face_start[f] + snap.face_size[f]], int(snap.face_kind[f]))
			for f in range(len(faces))]
		self._dead = set()
		self.snapshot = snap
		self.version = 0

		self.blade = None
		self.blade_count = 0  # blades begun so far (cut face roots are per blade and side)
		self.spec = None
		self.active = False
		self.contacted = False
		self.completed = False
		self.t = 0.0
		self.t_contact = np.inf
		self._reset_blade_state()

	def _reset_blade_state(self):
		self.tips = []  # live tip vertex ids
		self.tip_out = {}  # tip -> halfedge T -> x_b in its face
		self.tip_in = {}  # tip -> halfedge x_a -> T in its face
		self.regions = []
		self._region_of = {}  # live cut face of the current blade -> region index
		self._queue = []  # remaining crossings (time, edge) of the current blade
		self._qi = 0
		self._flabel = []  # per face: cage component label at `begin` (inherited by split children)
		self._bt_anc = []  # per face: face at `begin` it descends from (-1 for cut faces of this blade)
		# (blade end k, material piece label) -> live end tip vertex (bounded blades). Per piece: where pieces
		# separated by an earlier cut touch, the end leaves one and enters the other at the same point
		self.end_tip = {}
		self.end_of = {}  # end tip vertex -> blade end k
		self.loops_ignored = 0
		self.event_counts = {e: 0 for e in EVENTS}

	# -----------------------------------------------------------------------------------------
	# primitive helpers

	def _new_he(self, head, face):
		h = len(self.he_head)
		self.he_head.append(head)
		self.he_next.append(-1)
		self.he_prev.append(-1)
		self.he_twin.append(-1)
		self.he_face.append(face)
		self.he_ref.append(None)
		return h

	def _kill_he(self, h):
		self.he_face[h] = -1
		self.he_next[h] = self.he_prev[h] = self.he_twin[h] = -1
		self.he_head[h] = -1

	def _link(self, a, b):
		self.he_next[a] = b
		self.he_prev[b] = a

	def _set_twin(self, a, b):
		self.he_twin[a] = b
		self.he_twin[b] = a

	def _tail(self, h):
		return self.he_head[self.he_prev[h]]

	def _ref(self, h):
		"""Vertex ids defining the supporting line of halfedge h (inherited by the pieces of a split)."""
		r = self.he_ref[h]
		return r if r is not None else (self._tail(h), self.he_head[h])

	def _loop(self, r):
		"""Halfedges of the loop through r, the j-th going from vertex j to j+1 of _loop_verts(r)."""
		out = []
		h = r
		while True:
			h = self.he_next[h]
			out.append(h)
			if h == r:
				return out
			if len(out) > len(self.he_head):
				raise AssertionError(f"loop through halfedge {r} does not close")

	def _loop_verts(self, r):
		hes = self._loop(r)
		return [self.he_head[r]] + [self.he_head[h] for h in hes[:-1]]

	def face_loops(self, f):
		"""Vertex id lists of the loops of face f (outer first)."""
		return [self._loop_verts(r) for r in self.face_reps[f]]

	def face_loop(self, f):
		"""Vertex ids of the outer loop of face f."""
		return self._loop_verts(self.face_reps[f][0])

	def face_halfedges(self, f):
		"""Halfedges of face f, the j-th one going from slot j to its loop successor slot."""
		return [h for r in self.face_reps[f] for h in self._loop(r)]

	def _face_export(self, f):
		loops = [np.array(self._loop_verts(r), dtype=_INT) for r in self.face_reps[f]]
		refs = np.array([self._ref(h) for h in self.face_halfedges(f)], dtype=_INT).reshape(-1, 2)
		return loops, refs, _local_next(loops)

	def _new_vert(self, pos, side, step, sources):
		"""sources None: a crack vertex inside a face or the volume (no exact linear source; the
		pipeline places it in deformed space by fitting)."""
		v = len(self._pos)
		self._pos.append(np.array(pos, dtype=np.float64))
		self._valive.append(True)
		self._side.append(side)
		if sources is not None:
			step.sources[v] = sources
		return v

	def _anc(self, f, step):
		return f if f < step.F_old else step.anc[f]

	def _touch(self, f, step):
		a = self._anc(f, step)
		if a >= 0:
			step.modified.add(a)

	def _cut_root(self, side):
		"""face_root of the cut faces of the current blade on `side`: one root per (blade, side), so the
		closest-face tie order is unchanged by merges (same blade and side) and by later splits (children
		inherit the root). A shared CUT_ROOT would order cut faces of different blades by id, which changes
		when a later blade splits one of them."""
		return CUT_ROOT + 2 * (self.blade_count - 1) + (0 if side > 0 else 1)

	def _new_face(self, parent, normal, offset, tol, step, root=None):
		f = len(self.face_reps)
		self.face_reps.append([])
		self.face_alive.append(True)
		self.face_root.append(self.face_root[parent] if parent >= 0 else root)
		self.face_n.append(np.array(normal, dtype=np.float64))
		self.face_o.append(float(offset))
		self.face_tol.append(tol)
		self._export.append(None)
		self._face_geom.append(None)
		self._flabel.append(self._flabel[parent] if parent >= 0 else -1)
		self._bt_anc.append(self._bt_anc[parent] if parent >= 0 and parent < len(self._bt_anc) else -1)
		step.anc[f] = self._anc(parent, step) if parent >= 0 else -1
		return f

	def _relabel(self, r, f):
		for h in self._loop(r):
			self.he_face[h] = f

	def _cut_plane(self, side):
		# C_+ bounds the + side: outward normal -n_c
		n = -self.blade.n if side > 0 else self.blade.n.copy()
		return n, float(np.dot(n, self.blade.p1))

	@property
	def verts(self):
		return np.array(self._pos)

	# -----------------------------------------------------------------------------------------
	# blade setup

	def _edges(self):
		"""Live undirected edges as their smaller halfedge id."""
		return [h for h in range(len(self.he_head)) if self.he_face[h] >= 0 and h < self.he_twin[h]]

	def _crossing(self, h):
		"""Crossing of undirected edge h with the cut plane, computed canonically (by vertex id).

		Returns (time, point, (u, v, a)) with point = X_u + a (X_v - X_u), u < v.
		"""
		x, y = self._tail(h), self.he_head[h]
		u, v = (x, y) if x < y else (y, x)
		p, a = self.blade.crossing(self._pos[u], self._pos[v])
		return float(self.blade.sweep_time(p)), p, (u, v, float(a))

	def _is_crossing(self, h):
		su, sv = self._side[self._tail(h)], self._side[self.he_head[h]]
		return su != 0 and sv != 0 and su != sv

	def begin(self, spec, start_at_contact=False):
		"""Set the blade. No geometry change. Raises CutError for specs the topology cannot handle.

		start_at_contact: put the blade at the first contact time (legacy behaviour: the first
		advance then starts cutting immediately instead of travelling to the cage first).

		The blade cuts every component of the cross-section it sweeps over from its start time on.
		If the blade line already meets the cross-section at t = 0, the sweep starts at the first
		crossing of that material (the swept region is the same as coming from outside); parts of the
		cross-section lying entirely behind the start are left alone.
		"""
		if self.active and self.contacted:
			raise CutError("previous cut is still in progress; finish it before beginning another")
		self.active = False  # stays inactive if the spec is rejected
		blade = Blade3D(spec, verts=np.array(self._pos), alive=np.array(self._valive))
		sides = blade.side(np.array(self._pos))
		self.blade = blade
		self.blade_count += 1
		self.spec = dict(spec)
		self._side = [int(s) if a else 0 for s, a in zip(sides, self._valive)]
		self.t = 0.0
		self.contacted = False
		self.completed = False
		self._reset_blade_state()

		cross = {}
		for h in self._edges():
			if self._is_crossing(h):
				cross[h] = self._crossing(h)[0]
		if not cross:
			raise CutError("blade plane does not intersect the cage")
		loops = self.cross_section_loops(cross)
		chords = self._face_chords(cross)
		lab = piece_labels(self.snapshot)
		flabel = [int(lab[self.snapshot.face_verts[self.snapshot.face_start[f]]]) for f in range(len(self.face_reps))]
		self._flabel = flabel
		self._bt_anc = list(range(len(self.face_reps)))
		lo, hi = blade.s_range
		plunge = []
		if not blade.bounded and not blade.plunge:
			# start time: the earliest crossing of the material the blade line meets at or after t = 0,
			# extended until no included loop is partly behind it (nested loops are always included
			# together with the loop around them)
			rng = [(min(cross[h] for h in lp), max(cross[h] for h in lp)) for lp in loops]
			t_s = 0.0
			while True:
				inc = [k for k, (a, b) in enumerate(rng) if b >= t_s]
				if not inc:
					raise CutError("blade never touches the cage: every intersected part lies behind it")
				t_new = min(t_s, min(rng[k][0] for k in inc))
				if t_new == t_s:
					break
				t_s = t_new
			queue = sorted((cross[h], (1, h), ("x", h)) for k in inc for h in loops[k])
			self.loops_ignored = len(loops) - len(inc)
		else:
			# events: crossings inside the strip and the strip sides crossing face chords
			# queue order (time, (priority, id)): at equal times (a blade end crossing a zero-thickness crack:
			# leaving through one face and entering through the coincident one) exits come before entries
			events = [(cross[h], (1, h), ("x", h)) for h in cross if blade.in_strip(blade.along(self._crossing(h)[1]))]
			if blade.bounded:
				for f, pairs in chords.items():
					for (ta, ha), (tb, hb) in pairs:
						sa, sb = blade.along(self._crossing(ha)[1]), blade.along(self._crossing(hb)[1])
						for k in (0, 1):
							if (sa - blade.s_range[k]) * (sb - blade.s_range[k]) < 0.0:
								te = blade.end_time(k, self.face_n[f], self.face_o[f])
								exit_ = float(np.dot(blade.d, self.face_n[f])) > 0.0
								events.append((te, (0 if exit_ else 2, 2 * f + k), ("side", k, f)))
			segs = self._line_segments(0.0, chords)
			if blade.plunge:
				t_s = 0.0
				plunge = segs
			elif segs:
				t_s = min(e[0] for e in events)  # cut the strip's material as if coming from outside
			else:
				t_s = 0.0
			queue = sorted(e for e in events if e[0] >= t_s)
			if plunge:
				queue.insert(0, (t_s, (-1, 0), ("plunge", plunge)))
			self.loops_ignored = 0
			if not queue:
				raise CutError("blade never touches the cage: every intersected part lies behind it")
		for item in queue:
			if item[2][0] != "x":
				continue
			for g in (item[2][1], self.he_twin[item[2][1]]):
				f = self.he_face[g]
				if abs(np.dot(self.face_n[f], blade.l)) < 1e-12:
					raise CutError(f"face {f} is parallel to the blade line; it cannot host a cut tip")
		# pieces separated by earlier cuts may touch along coincident cut faces: fronts are only ever
		# looked up within the material piece of the crossing (a surface component, together with the
		# shells of the cavities it encloses)
		self._check_generic([(it[0], it[2][1]) for it in queue if it[2][0] == "x"], flabel)
		self._queue = queue
		self._qi = 0
		self.t_contact = queue[0][0]
		if t_s < 0.0 or start_at_contact or plunge:
			self.t = self.t_contact
		self.active = True

	def _face_chords(self, cross):
		"""Per face: its chords as pairs ((t_a, h_a), (t_b, h_b)) of crossing edges, t_a < t_b."""
		per_face = {}
		for h, t in cross.items():
			for g in (h, self.he_twin[h]):
				per_face.setdefault(self.he_face[g], []).append((t, h))
		out = {}
		for f, lst in per_face.items():
			lst.sort()
			out[f] = [(lst[k], lst[k + 1]) for k in range(0, len(lst) - 1, 2)]
		return out

	def _line_segments(self, t, chords):
		"""Segments of blade line L(t) ∩ material ∩ strip, ordered along l. Each segment is
		(label, left, right) with an end ('face', f, x) on a cage face or ('end', k, x) at a blade end."""
		b = self.blade
		pts = []
		for f, pairs in chords.items():
			for (ta, _), (tb, _) in pairs:
				if ta < t < tb:
					x = b.tip(self.face_n[f], self.face_o[f], t)
					pts.append((float(b.along(x)), f, x))
		# where pieces separated by an earlier cut touch, an exit and an entry coincide: exit first
		pts.sort(key=lambda p: (p[0], 0 if np.dot(b.l, self.face_n[p[1]]) > 0.0 else 1))
		if len(pts) % 2:
			raise CutError("blade line meets the cage in an odd number of points (not in generic position)")
		lo, hi = b.s_range
		out = []
		for i in range(0, len(pts), 2):
			(sa, fa, xa), (sb, fb, xb) = pts[i], pts[i + 1]
			if np.dot(b.l, self.face_n[fa]) >= 0.0 or np.dot(b.l, self.face_n[fb]) <= 0.0:
				raise CutError("blade line enters and leaves the material inconsistently (non-planar faces?)")
			if sb <= lo or sa >= hi:
				continue
			left = ("face", fa, xa) if sa > lo else ("end", 0, b.end(0, t))
			right = ("face", fb, xb) if sb < hi else ("end", 1, b.end(1, t))
			out.append((self._flabel[fa], left, right))
		return out

	def _check_generic(self, queue, flabel, tol=1e-12):
		"""Distinct crossing edges of one cage component must cross the plane at distinct points.

		Coincident crossings happen only where the cage has a zero-thickness crack (coincident vertex
		copies inside one component, e.g. a cut stopped half way); the order of the tips along the
		crack is then undefined.
		"""
		pts = {}
		for t, h in queue:
			pts.setdefault(flabel[self.he_face[h]], []).append((t, self._crossing(h)[1], h))
		scale = max(1.0, float(np.max(np.abs(np.array(self._pos)))))
		for lst in pts.values():
			for i in range(len(lst)):
				j = i + 1
				while j < len(lst) and lst[j][0] - lst[i][0] <= tol * scale:
					if np.linalg.norm(lst[j][1] - lst[i][1]) <= tol * scale:
						raise CutError("the cut plane crosses a zero-thickness crack of the cage (coincident crossings "
							f"of edges {lst[i][2]} and {lst[j][2]}): not in generic position")
					j += 1

	def cross_section_loops(self, cross):
		"""Loops of the cross-section of the cage with the cut plane.

		cross: {undirected edge (smaller halfedge id): sweep time} of the crossing edges. The
		cross-section vertices are the crossing edges; two of them are adjacent when they bound the
		same chord of a face (crossings of a face paired in sweep-time order along its chord).
		Returns a list of loops (lists of edge ids in loop order).
		"""
		per_face = {}
		for h, t in cross.items():
			for g in (h, self.he_twin[h]):
				per_face.setdefault(self.he_face[g], []).append((t, h))
		adj = {h: [] for h in cross}
		for f, lst in per_face.items():
			if len(lst) % 2:
				raise CutError(f"face {f} has an odd number of crossing edges (non-planar face?)")
		for f, pairs in self._face_chords(cross).items():
			for (_, a), (_, b) in pairs:
				adj[a].append(b)
				adj[b].append(a)
		for h, nb in adj.items():
			if len(nb) != 2:
				raise CutError("degenerate cross-section (edge not shared by two crossing chords)")
		seen = set()
		loops = []
		for h0 in sorted(adj):
			if h0 in seen:
				continue
			loop = [h0]
			seen.add(h0)
			prev, cur = h0, adj[h0][0]
			while cur != h0:
				loop.append(cur)
				seen.add(cur)
				a, b = adj[cur]
				prev, cur = cur, (b if a == prev else a)
			loops.append(loop)
		return loops

	# -----------------------------------------------------------------------------------------
	# tips, fronts and regions

	def _partner(self, T):
		"""The other tip of T's front segment (head of T's blade halfedge)."""
		return self.he_head[self.he_next[self.he_twin[self.tip_out[T]]]]

	def _tip_region(self, T):
		return self.regions[self._region_of[self.he_face[self.he_twin[self.tip_out[T]]]]]

	def _tip_face(self, T):
		return self.he_face[self.tip_out[T]]

	def _hosts(self):
		host = {}
		for T in self.tips:
			if T in self.end_of:
				continue
			f = self._tip_face(T)
			if f in host:
				raise AssertionError(f"face {f} hosts two tips ({host[f]}, {T})")
			host[f] = T
		return host

	def fronts(self):
		"""Front segments as (U, W) tip pairs, U < W."""
		out = []
		for T in self.tips:
			P = self._partner(T)
			if T < P:
				out.append((T, P))
		return out

	def _tip_pos(self, T, t, step):
		a = step.anchor.get(T)
		if a is not None and a[0] == t:
			return a[1]
		if T in self.end_of:
			return self.blade.end(self.end_of[T], t)
		f = self._tip_face(T)
		return self.blade.tip(self.face_n[f], self.face_o[f], t)

	def _emit(self, step, ta, tb, ends):
		"""Swept trapezoids of all front segments over [ta, tb] (tips move linearly inside it).

		ends: tip -> exact position at tb (the crossing point of the event ending the interval).
		"""
		if not tb > ta:
			return
		for U, W in self.fronts():
			a0, b0 = self._tip_pos(U, ta, step), self._tip_pos(W, ta, step)
			a1 = ends[U] if U in ends else self._tip_pos(U, tb, step)
			b1 = ends[W] if W in ends else self._tip_pos(W, tb, step)
			for tri in ((a0, b0, b1), (a0, b1, a1)):
				if np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0])) > 0.0:
					step.tris.append(np.array(tri, dtype=np.float64))

	def _front_at(self, p, t, step, label):
		"""Front segment of cage component `label` whose interior contains p at time t (the one p is
		deepest inside), or None."""
		l, p1 = self.blade.l, self.blade.p1
		sp = float(np.dot(p - p1, l))
		best = None
		for U, W in self.fronts():
			if self._tip_region(U).label != label:
				continue
			su = float(np.dot(self._tip_pos(U, t, step) - p1, l))
			sw = float(np.dot(self._tip_pos(W, t, step) - p1, l))
			m = min(sp - min(su, sw), max(su, sw) - sp)
			if m > 0.0 and (best is None or m > best[0]):
				best = (m, (U, W))
		return None if best is None else best[1]

	def _wedge(self, h):
		"""Orientation of the two chords leaving crossing edge h (no tip in its faces): positive when the
		wedge between them is material (BIRTH: the side(a) cut face T1 -> q_a -> T2 is CCW around its
		outward normal), negative when it is a notch or hole of a front segment (SPLIT)."""
		f1, f2 = self.he_face[h], self.he_face[self.he_twin[h]]
		w1 = self.blade.chord_direction(self.face_n[f1])
		w2 = self.blade.chord_direction(self.face_n[f2])
		n, _ = self._cut_plane(self._side[self._tail(h)])
		return float(np.dot(np.cross(w2, w1), n))

	# -----------------------------------------------------------------------------------------
	# topology operations

	def _set_single_rep(self, f, h):
		if len(self.face_reps[f]) == 1:
			self.face_reps[f] = [h]

	def _contact(self, h, te, p, src, step):
		"""Cage face part of BIRTH / SPLIT: f1: a -> q_a -> T1 -> q_b -> b, f2: b -> q_b -> T2 -> q_a -> a."""
		hp = self.he_twin[h]
		av, bv = self._tail(h), self.he_head[h]
		f1, f2 = self.he_face[h], self.he_face[hp]
		sa, sb = self._side[av], self._side[bv]
		qa = self._new_vert(p, sa, step, src)
		qb = self._new_vert(p, sb, step, src)
		T1 = self._new_vert(p, 0, step, src)
		T2 = self._new_vert(p, 0, step, src)
		self._touch(f1, step)
		self._touch(f2, step)
		ref_ab, ref_ba = self._ref(h), self._ref(hp)
		hn = self.he_next[h]
		self.he_head[h] = qa
		h1 = self._new_he(T1, f1)
		h2 = self._new_he(qb, f1)
		h3 = self._new_he(bv, f1)
		self._link(h, h1)
		self._link(h1, h2)
		self._link(h2, h3)
		self._link(h3, hn)
		gn = self.he_next[hp]
		self.he_head[hp] = qb
		g1 = self._new_he(T2, f2)
		g2 = self._new_he(qa, f2)
		g3 = self._new_he(av, f2)
		self._link(hp, g1)
		self._link(g1, g2)
		self._link(g2, g3)
		self._link(g3, gn)
		self._set_single_rep(f1, h)
		self._set_single_rep(f2, hp)
		self.he_ref[h] = self.he_ref[h3] = ref_ab
		self.he_ref[hp] = self.he_ref[g3] = ref_ba
		self._set_twin(h, g3)
		self._set_twin(h3, hp)
		self.tips += [T1, T2]
		self.tip_out[T1], self.tip_in[T1] = h2, h1
		self.tip_out[T2], self.tip_in[T2] = g2, g1
		step.anchor[T1] = step.anchor[T2] = (te, np.array(p))
		return sa, sb, qa, qb, T1, T2, h1, h2, g1, g2

	def _birth(self, h, te, p, src, step):
		label = self._flabel[self.he_face[h]]
		sa, sb, qa, qb, T1, T2, h1, h2, g1, g2 = self._contact(h, te, p, src, step)
		na, oa = self._cut_plane(sa)
		nb, ob = self._cut_plane(sb)
		Ca = self._new_face(-1, na, oa, 1e-9, step, root=self._cut_root(sa))
		Cb = self._new_face(-1, nb, ob, 1e-9, step, root=self._cut_root(sb))
		# C_side(a) = T1 -> q_a -> T2 -> T1
		c1 = self._new_he(qa, Ca)
		c2 = self._new_he(T2, Ca)
		c3 = self._new_he(T1, Ca)
		self._link(c1, c2)
		self._link(c2, c3)
		self._link(c3, c1)
		# C_side(b) = q_b -> T1 -> T2 -> q_b
		e1 = self._new_he(T1, Cb)
		e2 = self._new_he(T2, Cb)
		e3 = self._new_he(qb, Cb)
		self._link(e1, e2)
		self._link(e2, e3)
		self._link(e3, e1)
		self.face_reps[Ca] = [c1]
		self.face_reps[Cb] = [e1]
		self._set_twin(h1, c1)
		self._set_twin(h2, e1)
		self._set_twin(g1, e3)
		self._set_twin(g2, c2)
		self._set_twin(c3, e2)
		k = len(self.regions)
		self._flabel[Ca] = self._flabel[Cb] = label
		self.regions.append(_Region(Ca if sa > 0 else Cb, Cb if sa > 0 else Ca, label))
		self._region_of[Ca] = self._region_of[Cb] = k
		step.events.append("birth")

	def _split(self, h, front, te, p, src, step):
		R = self._tip_region(front[0])
		bo = self.he_next[self.he_twin[self.tip_out[front[0]]]]  # blade halfedge out of front[0]
		sa, sb, qa, qb, T1, T2, h1, h2, g1, g2 = self._contact(h, te, p, src, step)
		CA, CB = R.faces[sa], R.faces[sb]
		hA = bo if self.he_face[bo] == CA else self.he_twin[bo]
		hB = self.he_twin[hA]
		if self.he_face[hA] != CA or self.he_face[hB] != CB:
			raise AssertionError("front blade halfedges are not in the region's cut faces")
		U, W = self._tail(hA), self.he_head[hA]
		self._touch(CA, step)
		self._touch(CB, step)
		nA, nB = self.he_next[hA], self.he_next[hB]
		# C_A: U -> W becomes U -> T1 -> q_a -> T2 -> W
		self.he_head[hA] = T1
		c1 = self._new_he(qa, CA)
		c2 = self._new_he(T2, CA)
		c3 = self._new_he(W, CA)
		self._link(hA, c1)
		self._link(c1, c2)
		self._link(c2, c3)
		self._link(c3, nA)
		# C_B: W -> U becomes W -> T2 -> q_b -> T1 -> U
		self.he_head[hB] = T2
		e1 = self._new_he(qb, CB)
		e2 = self._new_he(T1, CB)
		e3 = self._new_he(U, CB)
		self._link(hB, e1)
		self._link(e1, e2)
		self._link(e2, e3)
		self._link(e3, nB)
		self._set_twin(hA, e3)
		self._set_twin(c3, hB)
		self._set_twin(c1, h1)
		self._set_twin(c2, g2)
		self._set_twin(e1, g1)
		self._set_twin(e2, h2)
		step.events.append("split")

	def _tip_event(self, T, hcd, te, r, src, step):
		f = self.he_face[hcd]
		hout, hin = self.tip_out[T], self.tip_in[T]
		if self.he_face[hout] != f:
			raise AssertionError(f"tip {T} is not in face {f}")
		c, d = self._tail(hcd), self.he_head[hcd]
		s, s2 = self._side[c], self._side[d]
		xb, xa = self.he_head[hout], self._tail(hin)
		if self._side[xb] not in (0, s) or self._side[xa] not in (0, s2):
			raise CutError(f"tip {T}: crossing edge ({c}, {d}) of face {f} is not between the slit sides "
				"(non-planar face?)")
		hdc = self.he_twin[hcd]
		g = self.he_face[hdc]
		k1, m1 = self.he_twin[hout], self.he_twin[hin]
		Cs, Cs2 = self.he_face[k1], self.he_face[m1]
		R = self._tip_region(T)
		if R.faces[s] != Cs or R.faces[s2] != Cs2:
			raise AssertionError("tip halfedges are not adjacent to the region's cut faces")
		rc = self._new_vert(r, s, step, src)
		rd = self._new_vert(r, s2, step, src)
		for x in (f, g, Cs, Cs2):
			self._touch(x, step)
		lt, le = self._loop_of(f, hout), self._loop_of(f, hcd)
		ref_cd, ref_dc = self._ref(hcd), self._ref(hdc)
		dn = self.he_next[hcd]
		# F_s = x_b ... c -> r_c -> x_b (keeps f)
		self.he_head[hcd] = rc
		self._link(hcd, hout)
		# F_s' = r_d -> d ... x_a -> r_d
		n1 = self._new_he(d, f)
		self.he_head[hin] = rd
		self._link(hin, n1)
		self._link(n1, dn)
		# C_s: x_b -> r_c -> T
		kn = self.he_next[k1]
		self.he_head[k1] = rc
		k2 = self._new_he(T, Cs)
		self._link(k1, k2)
		self._link(k2, kn)
		# C_s': T -> r_d -> x_a
		mp = self.he_prev[m1]
		m0 = self._new_he(rd, Cs2)
		self._link(mp, m0)
		self._link(m0, m1)
		# g: d -> r_d -> T -> r_c -> c
		gn = self.he_next[hdc]
		self.he_head[hdc] = rd
		g1 = self._new_he(T, g)
		g2 = self._new_he(rc, g)
		g3 = self._new_he(c, g)
		self._link(hdc, g1)
		self._link(g1, g2)
		self._link(g2, g3)
		self._link(g3, gn)
		self._set_single_rep(g, hdc)
		self.he_ref[hcd] = self.he_ref[n1] = ref_cd
		self.he_ref[hdc] = self.he_ref[g3] = ref_dc
		self._set_twin(hcd, g3)
		self._set_twin(n1, hdc)
		self._set_twin(k2, g2)
		self._set_twin(m0, g1)
		self.tip_out[T] = g2
		self.tip_in[T] = g1
		self._refresh_face(f, [hout, n1], lt, le, step)
		step.anchor[T] = (te, np.array(r))
		# the tip now follows the closed form in g; its sources follow its last crossing
		if T >= step.V_old:
			step.sources[T] = src
		step.events.append("tip")

	def _converge(self, T1, T2, hcd, te, r, src, step):
		"""CLOSE / MERGE / HOLE: tips T1 (face of hcd) and T2 (face of its twin) meet at r."""
		f1 = self.he_face[hcd]
		hdc = self.he_twin[hcd]
		f2 = self.he_face[hdc]
		c, d = self._tail(hcd), self.he_head[hcd]
		s, s2 = self._side[c], self._side[d]
		hout1, hin1 = self.tip_out[T1], self.tip_in[T1]
		hout2, hin2 = self.tip_out[T2], self.tip_in[T2]
		if self.he_face[hout1] != f1 or self.he_face[hout2] != f2:
			raise AssertionError("converging tips are not in the faces of the crossing edge")
		if (self._side[self.he_head[hout1]] not in (0, s) or self._side[self._tail(hin1)] not in (0, s2)
				or self._side[self.he_head[hout2]] not in (0, s2) or self._side[self._tail(hin2)] not in (0, s)):
			raise CutError("converging tips: crossing edge is not between the slit sides (non-planar face?)")
		k1, k3 = self.he_twin[hout1], self.he_twin[hin2]  # x_b -> T1, T2 -> y_2 (side s)
		m1, m3 = self.he_twin[hout2], self.he_twin[hin1]  # y_1 -> T2, T1 -> x_a (side s')
		bs, bs2 = self.he_next[k1], self.he_prev[m3]  # T1 -> Ta (side s), Ta -> T1 (side s')
		bt, bt2 = self.he_prev[k3], self.he_next[m1]  # Tb -> T2 (side s), T2 -> Tb (side s')
		Ta, Tb = self.he_head[bs], self._tail(bt)
		if not (self._tail(bs2) == Ta and self.he_head[bt2] == Tb and self.he_twin[bs] == bs2
				and self.he_twin[bt] == bt2):
			raise AssertionError("blade edges are not where the converging tips expect them")
		R1, R2 = self._tip_region(T1), self._tip_region(T2)
		A1, A2 = self.he_face[k1], self.he_face[m3]  # region of T1: side s, side s'
		B1, B2 = self.he_face[k3], self.he_face[m1]  # region of T2
		if R1.faces[s] != A1 or R1.faces[s2] != A2 or R2.faces[s] != B1 or R2.faces[s2] != B2:
			raise AssertionError("converging tips are not adjacent to their region's cut faces")
		if Ta == T2:
			if not (Tb == T1 and bs == bt):
				raise AssertionError("inconsistent front segment")
			kind = "close"
		else:
			kind = "merge" if R1 is not R2 else "hole"
		if kind == "merge":
			# the higher id face per side dies; freeze its export at death
			dead = (max(A1, B1), max(A2, B2))
			for x in dead:
				self._export[x] = self._face_export(x)
		rs = self._new_vert(r, s, step, src)
		rs2 = self._new_vert(r, s2, step, src)
		for x in (f1, f2, A1, A2, B1, B2):
			self._touch(x, step)
		l1 = (self._loop_of(f1, hout1), self._loop_of(f1, hcd))
		l2 = (self._loop_of(f2, hout2), self._loop_of(f2, hdc))
		# cage faces (as in CLOSE): f1 -> x_b ... c -> r_s -> x_b (keeps f1) + r_s' -> d ... x_a -> r_s',
		# f2 -> y_1 ... d -> r_s' -> y_1 (keeps f2) + r_s -> c ... y_2 -> r_s
		ref_cd, ref_dc = self._ref(hcd), self._ref(hdc)
		dn1, dn2 = self.he_next[hcd], self.he_next[hdc]
		self.he_head[hcd] = rs
		self._link(hcd, hout1)
		n1 = self._new_he(d, f1)
		self.he_head[hin1] = rs2
		self._link(hin1, n1)
		self._link(n1, dn1)
		self.he_head[hdc] = rs2
		self._link(hdc, hout2)
		n2 = self._new_he(c, f2)
		self.he_head[hin2] = rs
		self._link(hin2, n2)
		self._link(n2, dn2)
		self.he_ref[hcd] = self.he_ref[n1] = ref_cd
		self.he_ref[hdc] = self.he_ref[n2] = ref_dc
		self._set_twin(hcd, n2)
		self._set_twin(n1, hdc)
		# cut faces: x_b -> r_s -> y_2 and y_1 -> r_s' -> x_a
		nbs, nbt2 = self.he_next[bs], self.he_next[bt2]
		self.he_head[k1] = rs
		self._link(k1, k3)
		self.he_head[m1] = rs2
		self._link(m1, m3)
		if kind == "close":
			self._kill_he(bs)
			self._kill_he(bs2)
			self.face_reps[A1][0] = k1
			self.face_reps[A2][0] = m1
		else:
			# one blade edge Tb -> Ta (side s) / Ta -> Tb (side s')
			self.he_head[bt] = Ta
			self._link(bt, nbs)
			self.he_head[bs2] = Tb
			self._link(bs2, nbt2)
			self._kill_he(bs)
			self._kill_he(bt2)
			self._set_twin(bt, bs2)
			if kind == "merge":
				self._merge_faces(A1, B1, bt, step)
				self._merge_faces(A2, B2, bs2, step)
				ka, kb = sorted((self._region_of[A1], self._region_of[B1]))
				keep, gone = self.regions[ka], self.regions[kb]
				gone.alive = False
				keep.faces = {s: min(A1, B1), s2: min(A2, B2)}
				for x in dead:
					del self._region_of[x]
				for x in keep.faces.values():
					self._region_of[x] = ka
			else:
				# the loop pinches: the part through r_s / r_s' (no blade edge) is a hole
				for C, outer, hole in ((A1, bt, k1), (A2, bs2, m1)):
					tips = set(self.tips)
					for h in self._loop(hole):
						if self.he_head[h] in tips and self._tail(h) in tips:
							raise AssertionError("HOLE: the pinched-off loop contains a front segment")
					self.face_reps[C] = [outer] + self.face_reps[C][1:] + [hole]
		self._refresh_face(f1, [hout1, n1], *l1, step)
		self._refresh_face(f2, [hout2, n2], *l2, step)
		for T in (T1, T2):
			self._valive[T] = False
			self._pos[T] = np.array(r)
			step.orphaned.append(T)
			self.tips.remove(T)
			del self.tip_out[T]
			del self.tip_in[T]
		step.events.append(kind)

	# -----------------------------------------------------------------------------------------
	# plunge and blade ends

	def _new_region(self, label, side_x, step):
		"""Two new cut faces (C_x, C_y) of a new region; C_x bounds side side_x. Returns (C_x, C_y)."""
		nx, ox = self._cut_plane(side_x)
		ny, oy = self._cut_plane(-side_x)
		Cx = self._new_face(-1, nx, ox, 1e-9, step, root=self._cut_root(side_x))
		Cy = self._new_face(-1, ny, oy, 1e-9, step, root=self._cut_root(-side_x))
		k = len(self.regions)
		self._flabel[Cx] = self._flabel[Cy] = label
		self.regions.append(_Region(Cx if side_x > 0 else Cy, Cy if side_x > 0 else Cx, label))
		self._region_of[Cx] = self._region_of[Cy] = k
		return Cx, Cy

	def _dangling_slit(self, g, S, T, step):
		"""Slit in the interior of cage face g from crack vertex S to tip T: a zero-area hole loop
		S -> T -> S of g. Returns (hin: S -> T, hout: T -> S); the caller sets their twins."""
		hin = self._new_he(T, g)
		hout = self._new_he(S, g)
		self._link(hin, hout)
		self._link(hout, hin)
		self.face_reps[g].append(hout)
		self._touch(g, step)
		self.tip_out[T], self.tip_in[T] = hout, hin
		return hin, hout

	def _slit_side(self, g, w):
		"""Side of the material of face g left of a slit edge running along w (around g's normal)."""
		return 1 if np.dot(self.blade.n, np.cross(self.face_n[g], w)) > 0.0 else -1

	def _face_at(self, f0, x):
		"""Live face descending from begin-time face f0 whose region contains x (on its plane)."""
		u, v = frame_from_normal(self.face_n[f0])
		p = to_plane_2d(np.asarray(x)[None], u, v)[0]
		hit = []
		for g in range(len(self.face_reps)):
			if not self.face_alive[g] or self._bt_anc[g] != f0:
				continue
			inside = False
			for lp in self.face_loops(g):
				if point_in_polygon_2d(p, to_plane_2d(np.array([self._pos[i] for i in lp]), u, v)):
					inside = not inside
			if inside:
				hit.append(g)
		if len(hit) != 1:
			raise CutError(f"blade end / plunge point is not inside exactly one face (generic position): {hit}")
		return hit[0]

	def _plunge(self, te, segs, t_cur, step):
		"""Zero-area cut regions for every segment of L(t_s) ∩ material ∩ strip: back edge B_L - B_R
		(fixed crack vertices), front F_L - F_R (tips). Ends on a cage face become dangling slits of that
		face, ends at a blade end become end tips with a crack track B -> F along d.
		C_- = B_L -> B_R -> F_R -> F_L, C_+ = B_R -> B_L -> F_L -> F_R."""
		for label, left, right in segs:
			xl, xr = np.array(left[2]), np.array(right[2])
			BL = self._new_vert(xl, 0, step, None)
			BR = self._new_vert(xr, 0, step, None)
			FL = self._new_vert(xl, 0, step, None)
			FR = self._new_vert(xr, 0, step, None)
			Cm, Cp = self._new_region(label, -1, step)
			# C_-: m4 BL->BR, m1 BR->FR, m2 FR->FL, m3 FL->BL
			m1, m2, m3, m4 = self._new_he(FR, Cm), self._new_he(FL, Cm), self._new_he(BL, Cm), self._new_he(BR, Cm)
			for a, b in ((m4, m1), (m1, m2), (m2, m3), (m3, m4)):
				self._link(a, b)
			# C_+: p4 BR->BL, p1 BL->FL, p2 FL->FR, p3 FR->BR
			p1, p2, p3, p4 = self._new_he(FL, Cp), self._new_he(FR, Cp), self._new_he(BR, Cp), self._new_he(BL, Cp)
			for a, b in ((p4, p1), (p1, p2), (p2, p3), (p3, p4)):
				self._link(a, b)
			self.face_reps[Cm] = [m3]
			self.face_reps[Cp] = [p3]
			self._set_twin(m4, p4)  # back edge
			self._set_twin(m2, p2)  # blade edge
			# per end: (kind, index, point, front vertex, the C_- / C_+ side halfedges, tip_out, tip_in if an end tip)
			ends = (
				(right, xr, FR, BR, m1, p3, p3, m1),  # m1 BR->FR (next: blade FR->FL), p3 FR->BR (prev: blade FL->FR)
				(left, xl, FL, BL, m3, p1, m3, p1),  # p1 BL->FL (next: blade FL->FR), m3 FL->BL (prev: blade FR->FL)
			)
			for (kind, idx, _), x, F, B, h_m, h_p, out, inn in ends:
				if kind == "end":
					self._set_twin(h_m, h_p)
					self.tip_out[F], self.tip_in[F] = out, inn
					self.end_tip[(idx, label)] = F
					self.end_of[F] = idx
				else:
					g = self._face_at(idx, x)
					hin, hout = self._dangling_slit(g, B, F, step)
					# twin(tip_out) is followed by the blade halfedge out of F, twin(tip_in) preceded by the one in
					nxt_blade = h_m if F == FR else h_p
					prv_blade = h_p if F == FR else h_m
					self._set_twin(hout, nxt_blade)
					self._set_twin(hin, prv_blade)
				self.tips.append(F)
				step.anchor[F] = (te, x)
			step.tris.append(np.array([xl, xr, xr]))  # the back edge becomes boundary
		step.events.append("plunge")

	def _side_event(self, te, k, f0, t_cur, step):
		"""Blade end k crosses the chord of (a descendant of) begin-time face f0 at x."""
		b = self.blade
		x = b.end(k, te)
		n = self.face_n[f0]
		w = b.chord_direction(n)
		sig = 1.0 if k == 0 else -1.0  # inward direction of the strip at this side (along l)
		end_before = float(np.dot(b.d, n)) > 0.0  # the strip corner was inside material before
		chord_before = float(np.dot(w, b.l)) * sig < 0.0  # the chord ran inside the strip before
		host = self._hosts()
		key = (k, self._flabel[f0])
		E = self.end_tip.get(key)
		T = None
		if chord_before:
			# a non-convex face may host tips on several chords: take the one arriving at x
			cands = sorted((float(np.linalg.norm(self._tip_pos(host[g], te, step) - x)), host[g])
				for g in host if self._bt_anc[g] == f0)
			if not cands or cands[0][0] > 1e-7 * max(1.0, float(np.max(np.abs(x)))):
				raise AssertionError(f"side event: no tip arriving at the chord of face {f0}")
			T = cands[0][1]
		if end_before and E is None:
			raise AssertionError("side event: blade end inside material without an end tip")
		ends = {}
		if T is not None:
			ends[T] = x
		if end_before:
			ends[E] = x
		self._emit(step, t_cur, te, ends)
		if end_before and chord_before:
			self._end_close(E, T, key, x, step)
		elif end_before:
			self._end_to_tip(E, key, self._face_at(f0, x), x, te, step)
		elif chord_before:
			self._tip_to_end(T, k, x, te, step)
		else:
			self._end_birth(k, self._face_at(f0, x), x, w, te, step)

	def _end_birth(self, k, g, x, w, te, step):
		"""Blade end k enters material at x on face g: new region with crack vertex S, end tip E and
		face tip T (dangling slit S -> T in g). C_x = S -> T -> E, C_y = S -> E -> T."""
		S = self._new_vert(x, 0, step, None)
		E = self._new_vert(x, 0, step, None)
		T = self._new_vert(x, 0, step, None)
		# C_y holds the twin of the slit edge S -> T of g: it bounds g's side left of S -> T
		sy = self._slit_side(g, w)
		Cx, Cy = self._new_region(self._flabel[g], -sy, step)
		x1, x2, x3 = self._new_he(T, Cx), self._new_he(E, Cx), self._new_he(S, Cx)  # S->T, T->E, E->S
		y1, y2, y3 = self._new_he(E, Cy), self._new_he(T, Cy), self._new_he(S, Cy)  # S->E, E->T, T->S
		for a, b in ((x1, x2), (x2, x3), (x3, x1), (y1, y2), (y2, y3), (y3, y1)):
			self._link(a, b)
		self.face_reps[Cx] = [x3]
		self.face_reps[Cy] = [y3]
		hin, hout = self._dangling_slit(g, S, T, step)
		self._set_twin(hout, x1)
		self._set_twin(hin, y3)
		self._set_twin(x2, y2)
		self._set_twin(x3, y1)
		self.tip_out[E], self.tip_in[E] = x3, y1
		self.tips += [T, E]
		self.end_tip[(k, self._flabel[g])] = E
		self.end_of[E] = k
		step.anchor[T] = step.anchor[E] = (te, x)
		step.events.append("end_birth")

	def _tip_to_end(self, T, k, x, te, step):
		"""Face tip T reaches the strip side: it stops at x, a new end tip E continues the front."""
		g = self._tip_face(T)
		kk = self.he_twin[self.tip_out[T]]  # x_b -> T (C_s)
		bo = self.he_next[kk]  # T -> P (blade, C_s)
		mm = self.he_twin[self.tip_in[T]]  # T -> x_a (C_s')
		bi = self.he_prev[mm]  # P -> T (blade, C_s')
		Cs, Cs2 = self.he_face[kk], self.he_face[mm]
		E = self._new_vert(x, 0, step, None)
		e1 = self._new_he(E, Cs)  # T -> E
		self._link(kk, e1)
		self._link(e1, bo)
		self.he_head[bi] = E
		e2 = self._new_he(T, Cs2)  # E -> T
		self._link(bi, e2)
		self._link(e2, mm)
		self._set_twin(e1, e2)
		for f in (g, Cs, Cs2):
			self._touch(f, step)
		self._pos[T] = np.array(x)
		step.sources.pop(T, None)  # stopped inside a face: no longer on the edge it last crossed
		self.tips.remove(T)
		del self.tip_out[T], self.tip_in[T]
		self.tip_out[E], self.tip_in[E] = e2, e1
		self.tips.append(E)
		self.end_tip[(k, self._tip_region(E).label)] = E
		self.end_of[E] = k
		step.anchor[E] = (te, x)
		step.events.append("tip_end")

	def _end_to_tip(self, E, key, g, x, te, step):
		"""End tip E reaches cage face g at x: E stops there and a face tip T continues the front with a
		dangling slit E -> T in g."""
		a1 = self.tip_in[E]  # S -> E (C_a)
		b = self.he_next[a1]  # E -> P (blade, C_a)
		a2 = self.tip_out[E]  # E -> S (C_b)
		b2 = self.he_prev[a2]  # P -> E (blade, C_b)
		Ca, Cb = self.he_face[a1], self.he_face[a2]
		T = self._new_vert(x, 0, step, None)
		c1 = self._new_he(T, Ca)  # E -> T
		self._link(a1, c1)
		self._link(c1, b)
		self.he_head[b2] = T
		c2 = self._new_he(E, Cb)  # T -> E
		self._link(b2, c2)
		self._link(c2, a2)
		self._pos[E] = np.array(x)
		self.tips.remove(E)
		del self.tip_out[E], self.tip_in[E], self.end_of[E]
		del self.end_tip[key]
		hin, hout = self._dangling_slit(g, E, T, step)
		self._set_twin(hout, c1)
		self._set_twin(hin, c2)
		for f in (Ca, Cb):
			self._touch(f, step)
		self.tips.append(T)
		step.anchor[T] = (te, x)
		step.events.append("end_tip")

	def _end_close(self, E, T, key, x, step):
		"""End tip E and face tip T (one front segment) meet at x: T stays as a crack vertex, E dies."""
		a1 = self.tip_in[E]  # S -> E (C_a)
		b = self.he_next[a1]  # E -> T (blade, C_a)
		a2 = self.tip_out[E]  # E -> S (C_b)
		b2 = self.he_prev[a2]  # T -> E (blade, C_b)
		if self.he_head[b] != T or self._tail(b2) != T or self.he_twin[b] != b2:
			raise AssertionError("end close: end tip and face tip are not one front segment")
		g = self._tip_face(T)
		Ca, Cb = self.he_face[a1], self.he_face[a2]
		nb = self.he_next[b]  # T -> x_a
		pb = self.he_prev[b2]  # x_b -> T
		self.he_head[a1] = T
		self._link(a1, nb)
		self._link(pb, a2)
		for r, keep in ((Ca, a1), (Cb, a2)):
			reps = self.face_reps[r]
			if reps[0] in (b, b2):
				reps[0] = keep
		self._kill_he(b)
		self._kill_he(b2)
		for f in (g, Ca, Cb):
			self._touch(f, step)
		self._pos[T] = np.array(x)
		step.sources.pop(T, None)
		self._pos[E] = np.array(x)
		self._valive[E] = False
		step.orphaned.append(E)
		self.tips.remove(T)
		self.tips.remove(E)
		del self.tip_out[T], self.tip_in[T], self.tip_out[E], self.tip_in[E], self.end_of[E]
		del self.end_tip[key]
		step.events.append("end_close")

	def _merge_faces(self, A, B, rep, step):
		"""Cut faces A, B (same side) became one loop through rep: the lower id survives."""
		keep, gone = min(A, B), max(A, B)
		holes = self.face_reps[A][1:] + self.face_reps[B][1:]
		self.face_reps[keep] = [rep] + holes
		self.face_reps[gone] = []
		self.face_alive[gone] = False
		self._dead.add(gone)
		for r in self.face_reps[keep]:
			self._relabel(r, keep)
		if gone < step.F_old:
			step.merged[gone] = keep

	def _loop_of(self, f, h):
		"""Index (in face_reps[f]) of the loop of face f through halfedge h."""
		reps = self.face_reps[f]
		if len(reps) > 1:
			for i, r in enumerate(reps):
				if h == r or h in self._loop(r):
					return i
			raise AssertionError(f"halfedge {h} is on no loop of face {f}")
		return 0

	def _refresh_face(self, f, hints, la, lb, step):
		"""Recompute the loop structure of cage face f after a slit from loop la (tip side) to loop lb
		(crossing edge side) was closed, hints = halfedges (tip side, edge side) of the relinked loops.

		Classified topologically, never by the sign of a (possibly sliver sized) area: la != lb joins two
		loops into one (outer if either was the outer loop, else a hole); la == lb splits that loop in two,
		two outer loops if it was the outer loop, else a hole and a new piece of material (the larger signed
		area). Other holes go to the outer loop containing them. The outer loop through hints[0] keeps id f
		(else the first outer loop), other outer loops become new faces (split children inheriting f's plane).
		"""
		reps = self.face_reps[f]
		rest = [r for i, r in enumerate(reps) if i not in (la, lb)]
		if la != lb:
			if 0 in (la, lb):
				self.face_reps[f] = [hints[0]] + rest
			else:
				self.face_reps[f] = [reps[0]] + [r for r in rest if r != reps[0]] + [hints[0]]
			return
		ha, hb = hints
		if self.he_face[hb] < 0 or hb in self._loop(ha):
			raise AssertionError(f"face {f}: a closed slit did not split its loop")
		X = self._pos
		u, v = frame_from_normal(self.face_n[f])

		def poly(r):
			return to_plane_2d(np.array([X[x] for x in self._loop_verts(r)]), u, v)
		if la == 0:
			loops = [ha, hb] + rest
			outer = [0, 1]
		else:
			piece = 0 if signed_area_2d(poly(ha)) >= signed_area_2d(poly(hb)) else 1
			loops = [ha, hb, reps[0]] + [r for r in rest if r != reps[0]]
			outer = [piece, 2]
		holes = [i for i in range(len(loops)) if i not in outer]
		polys = [poly(r) for r in loops]
		areas = [signed_area_2d(p) for p in polys]
		keep = outer[0]
		owner = {}
		for j in holes:
			best = None
			for i in outer:
				votes = sum(point_in_polygon_2d(p, polys[i]) for p in polys[j])
				key = (votes, -areas[i])
				if best is None or key > best[0]:
					best = (key, i)
			owner[j] = best[1]
		loops = [(r, None) for r in loops]
		kids = []
		for i in [keep] + [i for i in outer if i != keep]:
			g = f if i == keep else self._new_face(f, self.face_n[f], self.face_o[f], self.face_tol[f], step)
			reps = [loops[i][0]] + [loops[j][0] for j in holes if owner[j] == i]
			self.face_reps[g] = reps
			kids.append(g)
			if g != f:
				for r in reps:
					self._relabel(r, g)
		self._separated_slits(kids, step)

	def _separated_slits(self, kids, step):
		"""Slits of a face (pairs of coincident edges: the chord just closed, and earlier chords that only
		joined two loops) whose sides now lie in different pieces become real boundary. A walk that landed on
		the parent next to such a slit would be relocated on-face to a piece whose boundary passes through
		its landing point (ambiguous within rounding, unlike a fresh walk), so every separated slit edge goes
		into `swept` as a degenerate triangle: walks near it are rewalked (exact reuse)."""
		X = self._pos
		owner = {}
		for g in kids:
			for h in self.face_halfedges(g):
				a, b = X[self._tail(h)], X[self.he_head[h]]
				key = tuple(sorted((tuple(a.tolist()), tuple(b.tolist()))))
				owner.setdefault(key, set()).add(g)
		for key, gs in owner.items():
			if len(gs) > 1:
				a, b = (np.array(x, dtype=np.float64) for x in key)
				step.tris.append(np.array([a, b, b]))

	# -----------------------------------------------------------------------------------------
	# public API

	def _cut_faces(self):
		return [(R.faces[1], R.faces[-1]) for R in self.regions if R.alive]

	def _empty_delta(self):
		s = self.snapshot
		e = np.zeros(0, dtype=_INT)
		return CutDelta(dim=3, old=s, new=s, swept=np.zeros((0, 3, 3)), modified_faces=e,
			face_ancestor=np.arange(s.num_faces, dtype=_INT), new_vertices=e, orphaned_vertices=e.copy(),
			moved_vertices=e.copy(), new_vertex_sources={}, topology_changed=False, completed=False,
			plane_point=None if self.blade is None else self.blade.p1.copy(),
			plane_normal=None if self.blade is None else self.blade.n.copy(),
			cut_faces=self._cut_faces(), t=self.t)

	def _event(self, item, t_cur, step):
		te, _, payload = item
		kind = payload[0]
		if kind == "x":
			self._crossing_event(te, payload[1], t_cur, step)
		elif kind == "side":
			self._side_event(te, payload[1], payload[2], t_cur, step)
		else:
			self._plunge(te, payload[1], t_cur, step)
		self.event_counts[step.events[-1]] += 1
		self.contacted = True

	def _crossing_event(self, te, h, t_cur, step):
		hp = self.he_twin[h]
		f1, f2 = self.he_face[h], self.he_face[hp]
		if f1 == f2:
			raise CutError(f"crossing edge has face {f1} on both sides")
		host = self._hosts()
		T1, T2 = host.get(f1), host.get(f2)
		_, p, (u, v, a) = self._crossing(h)
		src = ([u, v], [1.0 - a, a])
		ends = {T: p for T in (T1, T2) if T is not None}
		self._emit(step, t_cur, te, ends)
		if T1 is None and T2 is None:
			# decided locally by the wedge of the two chords; the front lookup decides only at cusps
			w = self._wedge(h)
			front = None if w > 1e-9 else self._front_at(p, te, step, self._flabel[f1])
			if front is None and w < -1e-9:
				raise CutError("a crossing opens a notch or hole of the cross-section outside every front segment "
					"(degenerate geometry)")
			if front is None:
				self._birth(h, te, p, src, step)
			else:
				self._split(h, front, te, p, src, step)
		elif T2 is None:
			self._tip_event(T1, h, te, p, src, step)
		elif T1 is None:
			self._tip_event(T2, hp, te, p, src, step)
		else:
			self._converge(T1, T2, h, te, p, src, step)

	def advance(self, amount):
		if not self.active or amount <= 0.0:
			return self._empty_delta()
		old = self.snapshot
		t_target = self.t + float(amount)
		step = _Step(len(self.face_reps), len(self._pos), self.t)
		t_cur = self.t
		while self._qi < len(self._queue) and self._queue[self._qi][0] <= t_target:
			item = self._queue[self._qi]
			self._qi += 1
			self._event(item, t_cur, step)
			t_cur = max(t_cur, item[0])
		if self._qi == len(self._queue):
			if self.tips:
				raise AssertionError("every crossing was processed but front segments remain")
			self.completed = True
			self.active = False
			self.t = t_cur
		else:
			self._emit(step, t_cur, t_target, {})
			self.t = t_target
		if not step.events and not self.tips:
			return self._empty_delta()  # travelling between parts of the cross-section
		for T in self.tips:
			if T in self.end_of:
				x = self.blade.end(self.end_of[T], self.t)
				faces = (self.he_face[self.tip_out[T]], self.he_face[self.tip_in[T]])
			else:
				f = self._tip_face(T)
				x = self.blade.tip(self.face_n[f], self.face_o[f], self.t)
				faces = (f, self.he_face[self.he_twin[self.tip_out[T]]], self.he_face[self.he_twin[self.tip_in[T]]])
			if np.array_equal(x, self._pos[T]):
				continue  # did not move (step within rounding of the last position): its faces are unchanged
			self._pos[T] = x
			for g in faces:
				self._touch(g, step)
		if not step.events and not step.modified:
			return self._empty_delta()  # the step was within rounding of the last blade position

		new = self._rebuild_snapshot(step)
		anc = np.arange(new.num_faces, dtype=_INT)
		for f, a in step.anc.items():
			anc[f] = a
		merged = np.full(step.F_old, -1, dtype=_INT)
		for f, g in step.merged.items():
			while g in step.merged:
				g = step.merged[g]
			merged[f] = g
		# tips, and tips that stopped this step (blade ends), are the only existing vertices that move
		moved = [v for v in range(step.V_old) if self._valive[v] and not np.array_equal(old.verts[v], self._pos[v])]
		swept = np.array(step.tris, dtype=np.float64).reshape(-1, 3, 3)
		return CutDelta(dim=3, old=old, new=new, swept=swept,
			modified_faces=np.array(sorted(step.modified), dtype=_INT), face_ancestor=anc,
			new_vertices=np.arange(step.V_old, len(self._pos), dtype=_INT),
			orphaned_vertices=np.array(step.orphaned, dtype=_INT), moved_vertices=np.array(moved, dtype=_INT),
			new_vertex_sources=dict(step.sources), face_merged_into=merged, topology_changed=bool(step.events),
			completed=self.completed, events=list(step.events), plane_point=self.blade.p1.copy(),
			plane_normal=self.blade.n.copy(), cut_faces=self._cut_faces(), t=self.t)

	def _rebuild_snapshot(self, step):
		"""New snapshot, recomputing only faces touched in this step (and dead faces' geometry).

		Bit-identical to build_snapshot(3, verts, face loops, face_planes=...) (verified in check()):
		the per-face frame, planar coordinates and kind are computed by the same functions.
		"""
		dirty = set(step.modified) | set(step.anc.keys())
		X = np.array(self._pos)
		for f in range(len(self.face_reps)):
			if f in self._dead:
				self._face_geom[f] = None  # frozen loops, but their vertices may still move
			elif self._export[f] is None or f in dirty:
				self._export[f] = self._face_export(f)
				self._face_geom[f] = None
		for f in range(len(self.face_reps)):
			if self._face_geom[f] is None:
				loops = self._export[f][0]
				u, v = frame_from_normal(self.face_n[f])
				c = to_plane_2d(X[np.concatenate(loops)], u, v)
				self._face_geom[f] = (u, v, c, classify_face(c) if len(loops) == 1 else FACE_NGON)
		self.version += 1
		self.snapshot = self._assemble(X)
		return self.snapshot

	def _assemble(self, X):
		F = len(self.face_reps)
		ex = self._export
		sizes = np.array([e[2].shape[0] for e in ex], dtype=np.int32)
		start = np.zeros(F, dtype=np.int32)
		if F > 1:
			start[1:] = np.cumsum(sizes)[:-1]
		vnext = (np.concatenate([e[2] for e in ex]) + np.repeat(start, sizes)).astype(np.int32)
		geom = self._face_geom
		return CageSnapshot(3, X, np.array(self._valive, dtype=bool),
			np.concatenate([np.concatenate(e[0]) for e in ex]).astype(np.int32), start, sizes,
			np.array(self.face_alive, dtype=bool), np.array([g[3] for g in geom], dtype=np.int32),
			np.array(self.face_n), np.array(self.face_o), np.array([g[0] for g in geom]),
			np.array([g[1] for g in geom]), np.concatenate([g[2] for g in geom]), self.version,
			np.array(self.face_root, dtype=np.int32), np.concatenate([e[1] for e in ex]).astype(np.int32),
			vnext, np.array([len(e[0]) for e in ex], dtype=np.int32))

	# -----------------------------------------------------------------------------------------
	# invariants

	def check(self):
		H = len(self.he_head)
		head, nxt, prv, twin, hf = self.he_head, self.he_next, self.he_prev, self.he_twin, self.he_face
		live = [h for h in range(H) if hf[h] >= 0]
		out_count = {}
		out_any = {}
		directed = {}
		slit2 = set(h for h in range(H) if hf[h] >= 0 and nxt[nxt[h]] == h)  # two-edge (dangling slit) loops
		for h in live:
			assert 0 <= nxt[h] < H and 0 <= prv[h] < H, f"halfedge {h}: dangling next/prev"
			assert prv[nxt[h]] == h and nxt[prv[h]] == h, f"halfedge {h}: next/prev not inverse"
			assert hf[nxt[h]] == hf[h], f"halfedge {h}: next in another face"
			t = twin[h]
			assert 0 <= t < H and hf[t] >= 0, f"halfedge {h}: no live twin (mesh not closed)"
			assert twin[t] == h, f"halfedge {h}: twin not an involution"
			assert head[h] == self._tail(t), f"halfedge {h}: head != tail(twin)"
			assert self._valive[head[h]], f"halfedge {h}: points to dead vertex {head[h]}"
			e = (self._tail(h), head[h])
			assert e[0] != e[1], f"halfedge {h}: degenerate edge"
			if e in directed:
				# the only multi-edges are the two sides of a dangling slit: once in the cage face, once in
				# a cut face
				g = directed[e]
				assert g is not None and ((g in slit2) != (h in slit2)), f"duplicate directed edge {e}"
				directed[e] = None
			else:
				directed[e] = h
			out_count[e[0]] = out_count.get(e[0], 0) + 1
			out_any[e[0]] = h
		F = len(self.face_reps)
		visited = np.zeros(H, dtype=bool)
		for f in range(F):
			if not self.face_alive[f]:
				assert self.face_reps[f] == [] and f in self._dead, f"dead face {f} still has loops"
				continue
			assert len(self.face_reps[f]) >= 1, f"face {f}: no loop"
			for r in self.face_reps[f]:
				assert hf[r] == f, f"face {f}: loop representative in face {hf[r]}"
				n = 0
				h = r
				while True:
					assert not visited[h], f"halfedge {h} visited twice"
					visited[h] = True
					assert hf[h] == f
					h = nxt[h]
					n += 1
					assert n <= H, f"face {f}: loop does not close"
					if h == r:
						break
				# a dangling slit is a two-edge hole loop of a cage face
				assert n >= 3 or (n == 2 and r != self.face_reps[f][0]), \
					f"face {f}: loop with fewer than 3 edges"
		for h in live:
			assert visited[h], f"halfedge {h} not reachable from its face"
		# vertex manifoldness: outgoing halfedges form a single fan
		for v, cnt in out_count.items():
			h0 = out_any[v]
			h = h0
			k = 0
			while True:
				h = nxt[twin[h]]
				k += 1
				assert k <= cnt, f"vertex {v}: fan longer than outgoing count"
				if h == h0:
					break
			assert k == cnt, f"vertex {v}: non-manifold (fan {k} of {cnt} outgoing halfedges)"
		used = set(out_count.keys())
		for v in range(len(self._pos)):
			assert self._valive[v] == (v in used), f"vertex {v}: alive flag {self._valive[v]} vs use"
		# snapshot agrees with the loops, planes hold, orientation matches the planes
		s = self.snapshot
		assert s.num_faces == F and s.num_verts == len(self._pos)
		X = np.array(self._pos)
		assert np.array_equal(s.verts, X), "snapshot vertex positions are stale"
		faces = []
		for f in range(F):
			ls = s.face_loops(f)
			faces.append(ls if len(ls) > 1 else ls[0])
		ref = build_snapshot(3, X, faces, face_planes=(np.array(self.face_n), np.array(self.face_o)),
			vert_alive=np.array(self._valive), face_alive=np.array(self.face_alive), face_root=np.array(self.face_root),
			face_edge_ref=np.concatenate([self._face_export(f)[1] if self.face_alive[f] else self._export[f][1]
				for f in range(F)]))
		for name in ("vert_alive", "face_verts", "face_start", "face_size", "face_alive", "face_kind", "face_root",
				"face_edge_ref", "face_vnext", "face_nloops",
				"face_normal", "face_offset", "face_u", "face_v", "face_coords2d"):
			assert np.array_equal(getattr(s, name), getattr(ref, name)), f"snapshot {name} differs from build_snapshot"
		for f in range(F):
			if not self.face_alive[f]:
				continue
			loops = self.face_loops(f)
			sl = [[int(x) for x in lp] for lp in s.face_loops(f)]
			assert sl == loops, f"face {f}: snapshot loops differ from the halfedge loops"
			allv = [x for lp in loops for x in lp]
			assert len(set(allv)) == len(allv), f"face {f}: repeated vertex"
			n, o = self.face_n[f], self.face_o[f]
			assert np.array_equal(s.face_normal[f], n) and s.face_offset[f] == o, f"face {f}: plane changed"
			dev = np.max(np.abs(X[allv] @ n - o))
			assert dev < self.face_tol[f], f"face {f}: not planar ({dev:.3e} >= {self.face_tol[f]:.1e})"
			nw = sum(newell_normal(X[lp]) for lp in loops)
			if np.linalg.norm(nw) > 1e-10:
				assert np.dot(nw, n) > 0.0, f"face {f}: orientation disagrees with its plane normal"
			if len(loops) > 1:
				u, v = frame_from_normal(n)
				polys = [to_plane_2d(X[lp], u, v) for lp in loops]
				# rounding of the area sum (sliver loops a 1e-7 generic shift away from a vertex)
				eps = 1e-14 * max(1.0, float(np.max(np.abs(np.concatenate(polys))))) ** 2 * len(allv)
				assert signed_area_2d(polys[0]) > -eps, f"face {f}: outer loop not CCW"
				for P in polys[1:]:
					assert signed_area_2d(P) < eps, f"face {f}: hole loop not CW"
					assert _inside(P, polys[0]), f"face {f}: hole loop not inside the outer loop"
		if self.blade is not None:
			self._check_cut_state(X)

	def _check_cut_state(self, X):
		hf = self.he_face
		live_faces = set()
		for k, R in enumerate(self.regions):
			if not R.alive:
				continue
			for sd, C in R.faces.items():
				assert self.face_alive[C], f"region {k}: cut face {C} is dead"
				n, o = self._cut_plane(sd)
				assert np.array_equal(self.face_n[C], n) and self.face_o[C] == o, f"cut face {C}: wrong plane"
				assert self.face_root[C] == self._cut_root(sd), f"cut face {C}: root is not its blade side's root"
				assert self._region_of.get(C) == k, f"cut face {C}: region bookkeeping"
				live_faces.add(C)
		assert set(self._region_of.keys()) == live_faces, "region map lists dead or unknown cut faces"
		tips = set(self.tips)
		assert len(tips) == len(self.tips)
		assert set(self.tip_out.keys()) == tips and set(self.tip_in.keys()) == tips
		if not self.active:
			assert not tips, "inactive cutter with live tips"
		hosts = set()
		l, d, p1 = self.blade.l, self.blade.d, self.blade.p1
		assert set(self.end_of.keys()) <= tips
		assert all(self.end_tip.get((k, self._tip_region(E).label)) == E for E, k in self.end_of.items())
		assert len(self.end_tip) == len(self.end_of)
		lo, hi_s = self.blade.s_range
		for T in tips:
			ho, hi = self.tip_out[T], self.tip_in[T]
			assert self._valive[T] and self._side[T] == 0, f"tip {T}: dead or sided"
			assert self._tail(ho) == T and self.he_head[hi] == T, f"tip {T}: tip halfedges"
			assert abs(self.blade.signed_dist(X[T])) < 1e-9, f"tip {T}: off the cut plane"
			assert abs(np.dot(X[T] - p1, d) - self.t) < 1e-9 * max(1.0, abs(self.t)), f"tip {T}: off the blade line"
			R = self._tip_region(T)
			assert R.alive
			if T in self.end_of:
				# end tip: its track (tip_out / tip_in, twins) runs between the region's two cut faces
				k = self.end_of[T]
				assert self.he_twin[ho] == hi, f"end tip {T}: track halfedges are not twins"
				assert {hf[ho], hf[hi]} == set(R.faces.values()), f"end tip {T}: track not between its cut faces"
				assert np.allclose(X[T], self.blade.end(k, self.t), atol=1e-9), f"end tip {T}: not at blade end {k}"
			else:
				f = hf[ho]
				assert hf[hi] == f and f not in self._region_of, f"tip {T}: not inside one cage face"
				assert f not in hosts, f"face {f} hosts two tips"
				hosts.add(f)
				assert abs(np.dot(self.face_n[f], X[T]) - self.face_o[f]) < 1e-9, f"tip {T}: off its face plane"
				sb = float(np.dot(X[T] - p1, l))
				assert lo - 1e-9 <= sb <= hi_s + 1e-9, f"tip {T}: outside the blade's strip"
				xb, xa = self.he_head[ho], self._tail(hi)
				# slit sides: by the side of the slit vertex, or (crack vertex at the base of a dangling slit)
				# by the cut face across the slit
				cb, ca = hf[self.he_twin[ho]], hf[self.he_twin[hi]]
				assert cb in R.faces.values() and ca in R.faces.values() and cb != ca, f"tip {T}: slit not on its cut faces"
				if self._side[xb] != 0:
					assert cb == R.faces[self._side[xb]], f"tip {T}: out halfedge not on its cut face"
				if self._side[xa] != 0:
					assert ca == R.faces[self._side[xa]], f"tip {T}: in halfedge not on its cut face"
			bo = self.he_next[self.he_twin[ho]]
			P = self.he_head[bo]
			assert P in tips and P != T, f"tip {T}: blade edge does not join two tips"
			assert self.he_twin[bo] == self.he_prev[self.he_twin[hi]], f"tip {T}: blade edge twins"
			assert self._partner(P) == T, f"tip {T}: partner mismatch"
			assert self._tip_region(P) is R
			# blade halfedges run -l in C_- and +l in C_+ (the region lies behind its front)
			sgn = 1.0 if R.faces[1] == hf[bo] else -1.0
			assert sgn * (np.dot(X[P] - X[T], l)) >= -1e-9, f"front ({T}, {P}): wrong orientation"
		# tip-to-tip halfedges are exactly the blade edges; hole loops of cut faces carry no tips
		nblade = 0
		for h in range(len(self.he_head)):
			if hf[h] >= 0 and self.he_head[h] in tips and self._tail(h) in tips:
				assert hf[h] in self._region_of, f"halfedge {h} joins two tips outside a cut face"
				nblade += 1
		assert nblade == len(tips), "blade edge count differs from the number of tips"
		for C in live_faces:
			for r in self.face_reps[C][1:]:
				for v in self._loop_verts(r):
					assert v not in tips, f"cut face {C}: tip {v} on a hole loop"
		if self.active:
			# right after begin(start_at_contact=True) the first crossing is exactly at t
			nxt = self._queue[self._qi][0] if self._qi < len(self._queue) else -np.inf
			assert nxt > self.t or (self._qi == 0 and nxt == self.t), "pending event behind the blade"


def _local_next(loops):
	"""Slot successor (relative to the face start) of every slot of a face given by its loops."""
	nxt = []
	k = 0
	for lp in loops:
		n = len(lp)
		nxt.extend(range(k + 1, k + n))
		nxt.append(k)
		k += n
	return np.array(nxt, dtype=np.int32)


def _inside(P, Q, tol=1e-9):
	"""Every vertex of P not on Q's boundary lies inside Q (even-odd), and at least one is off it."""
	a, b = Q, np.roll(Q, -1, axis=0)
	e = b - a
	ee = np.maximum(np.sum(e * e, axis=1), 1e-300)
	off = 0
	for p in P:
		t = np.clip(((p - a) * e).sum(axis=1) / ee, 0.0, 1.0)
		if np.min(np.linalg.norm(p - (a + t[:, None] * e), axis=1)) <= tol:
			continue
		off += 1
		if not point_in_polygon_2d(p, Q):
			return False
	return off > 0
