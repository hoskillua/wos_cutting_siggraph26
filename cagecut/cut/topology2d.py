"""2D cut topology: a knife travelling along a ray, cutting straight slits into a cage of CCW loops.

Segments are the faces of the 2D snapshot and have
persistent ids (append-only, never deleted). A segment keeps the line (normal, offset) of the
segment it was split from, so split pieces lie on the parent line bit-for-bit.

The knife position is lam (distance from the start point q along d); the tip of the slit being cut is
computed in closed form T = q + lam d (never integrated). The first slit starts on the spec's start
segment; after the knife exits the material it keeps travelling and starts a new slit wherever the
ray enters material again. Crossings are found combinatorially from logical vertex sides (assigned at
`begin`, and on creation for the coincident slit copies): a segment from the + to the - side is an
entry, from - to + an exit; processed crossings are split into same-side pieces and disappear.
"""

import numpy as np

from cagecut.cut.blade import CutError, Slit2D
from cagecut.cut.delta import CutDelta
from cagecut.cut.topology3d import _inside
from cagecut.geometry.polygon import points_in_loops_2d, signed_area_2d
from cagecut.geometry.snapshot import build_snapshot

_INT = np.int32


def _cross(a, b):
	return a[0] * b[1] - a[1] * b[0]


class _Step:
	def __init__(self, num_faces, num_verts, lam):
		self.F_old = num_faces
		self.V_old = num_verts
		self.lam_old = lam
		self.anc = {}
		self.modified = set()
		self.sources = {}
		self.orphaned = []
		self.events = []
		self.swept = []


class Cutter2D:
	def __init__(self, verts, loops):
		verts = np.asarray(verts, dtype=np.float64).reshape(-1, 2)
		self._pos = [verts[i].copy() for i in range(verts.shape[0])]
		self._valive = [False] * verts.shape[0]
		self.seg_a, self.seg_b, self.seg_next, self.seg_prev = [], [], [], []
		self.seg_alive = []
		self.seg_root = []  # split-invariant ancestor id (see snapshot.py)
		for loop in loops:
			loop = [int(v) for v in loop]
			n = len(loop)
			if n < 3:
				raise ValueError("a loop needs at least 3 vertices")
			if len(set(loop)) != n:
				raise ValueError("a loop repeats a vertex")
			base = len(self.seg_a)
			for i in range(n):
				self.seg_a.append(loop[i])
				self.seg_b.append(loop[(i + 1) % n])
				self.seg_next.append(base + (i + 1) % n)
				self.seg_prev.append(base + (i - 1) % n)
				self.seg_alive.append(True)
				self.seg_root.append(len(self.seg_root))
				self._valive[loop[i]] = True
		snap = build_snapshot(2, verts, [[a, b] for a, b in zip(self.seg_a, self.seg_b)],
			vert_alive=np.array(self._valive))
		self.seg_n = [snap.face_normal[s].copy() for s in range(len(self.seg_a))]
		self.seg_o = [float(snap.face_offset[s]) for s in range(len(self.seg_a))]
		self.snapshot = snap
		self.version = 0
		self._side = [0] * len(self._pos)

		self.slit = None
		self.spec = None
		self.active = False
		self.started = False
		self.completed = False
		self.length = 0.0  # knife position lam (kept under its old name)
		self.t = 0.0
		self.tip = -1
		self.seg_in = -1  # x_a -> T of the slit being cut (bounds the + side)
		self.seg_out = -1  # T -> x_b (bounds the - side)
		self.slits = []  # (seg_in, seg_out) of every slit of the current knife
		self.slit_starts = []  # knife position where each slit started
		self._start_seg = -1
		self._alpha = 0.0  # position of q on the start segment, measured from its tail
		self._start_inside = False  # the knife starts at a point inside the material (spec "point")

	# -----------------------------------------------------------------------------------------

	def _new_vert(self, pos, side, step, sources):
		"""sources None: a crack vertex inside the material (knife started inside)."""
		v = len(self._pos)
		self._pos.append(np.array(pos, dtype=np.float64))
		self._valive.append(True)
		self._side.append(side)
		if sources is not None:
			step.sources[v] = sources
		return v

	def _new_seg(self, a, b, normal, offset, parent, step):
		s = len(self.seg_a)
		self.seg_a.append(a)
		self.seg_b.append(b)
		self.seg_next.append(-1)
		self.seg_prev.append(-1)
		self.seg_alive.append(True)
		self.seg_root.append(self.seg_root[parent] if parent >= 0 else s)
		self.seg_n.append(np.array(normal, dtype=np.float64))
		self.seg_o.append(float(offset))
		step.anc[s] = self._anc(parent, step) if parent >= 0 else -1
		return s

	def _link(self, s, t):
		self.seg_next[s] = t
		self.seg_prev[t] = s

	def _anc(self, s, step):
		return s if s < step.F_old else step.anc[s]

	def _touch(self, s, step):
		a = self._anc(s, step)
		if a >= 0:
			step.modified.add(a)

	@property
	def verts(self):
		return np.array(self._pos)

	@property
	def cutting(self):
		"""A slit is being cut (the knife is inside material)."""
		return self.tip >= 0

	def loops(self):
		"""Current loops as lists of segment ids."""
		seen = set()
		out = []
		for s0 in range(len(self.seg_a)):
			if not self.seg_alive[s0] or s0 in seen:
				continue
			loop = []
			s = s0
			while s not in seen:
				seen.add(s)
				loop.append(s)
				s = self.seg_next[s]
			out.append(loop)
		return out

	# -----------------------------------------------------------------------------------------

	def begin(self, spec):
		if self.active and self.started:
			raise CutError("previous slit is still growing; finish it before beginning another")
		self.active = False  # stays inactive if the spec is rejected
		if "point" in spec:
			self._begin_inside(spec)
			return
		a, b = (int(x) for x in spec["edge"])
		alpha = float(spec["alpha"])
		seg = -1
		for s in range(len(self.seg_a)):
			if not self.seg_alive[s]:
				continue
			if (self.seg_a[s], self.seg_b[s]) == (a, b):
				seg, al = s, alpha
				break
			if (self.seg_a[s], self.seg_b[s]) == (b, a):
				seg, al = s, 1.0 - alpha
				break
		if seg < 0:
			raise CutError(f"slit start edge ({a}, {b}) is not a segment of the cage")
		if not 0.0 < al < 1.0:
			raise CutError("slit start alpha must be strictly inside the edge")
		xa, xb = self._pos[self.seg_a[seg]], self._pos[self.seg_b[seg]]
		q = (1.0 - al) * xa + al * xb
		d = np.asarray(spec["direction"], dtype=np.float64)
		e = xb - xa
		if _cross(e, d) <= 1e-12 * np.linalg.norm(e) * np.linalg.norm(d):
			raise CutError("slit direction does not point into the cage")
		self.spec = dict(spec)
		self.slit = Slit2D(q, d)
		sides = self.slit.side(np.array(self._pos))
		self._side = [int(sd) if al_ else 0 for sd, al_ in zip(sides, self._valive)]
		if self._side[self.seg_a[seg]] != 1 or self._side[self.seg_b[seg]] != -1:
			raise CutError("slit start edge endpoints are not on opposite sides of the slit line")
		self._start_seg = seg
		self._alpha = al
		self._start_inside = False
		self.seg_in = self.seg_out = -1
		self.tip = -1
		self.slits = []
		self.slit_starts = []
		if self._next_crossing(exit_=True, after=0.0, exclude=seg) is None:
			raise CutError("slit ray never leaves the cage")
		self.started = False
		self.completed = False
		self.length = 0.0
		self.t = 0.0
		self.active = True

	def _begin_inside(self, spec):
		"""Knife starting at a point inside the material: the first slit is a crack S -> T -> S (a
		zero-area loop) that joins the boundary where the knife first leaves the material."""
		q = np.asarray(spec["point"], dtype=np.float64)
		live = [s for s in range(len(self.seg_a)) if self.seg_alive[s]]
		a = np.array([self._pos[self.seg_a[s]] for s in live])
		b = np.array([self._pos[self.seg_b[s]] for s in live])
		if not points_in_loops_2d(q[None], a, b)[0]:
			raise CutError("knife start point is not inside the material")
		self.spec = dict(spec)
		self.slit = Slit2D(q, spec["direction"])
		sides = self.slit.side(np.array(self._pos))
		self._side = [int(sd) if al_ else 0 for sd, al_ in zip(sides, self._valive)]
		self._start_seg = -1
		self._alpha = 0.0
		self._start_inside = True
		self.seg_in = self.seg_out = -1
		self.tip = -1
		self.slits = []
		self.slit_starts = []
		if self._next_crossing(exit_=True, after=0.0) is None:
			raise CutError("slit ray never leaves the cage")
		self.started = False
		self.completed = False
		self.length = 0.0
		self.t = 0.0
		self.active = True

	def _start_crack(self, step):
		"""First slit of a knife starting inside: crack vertex S and tip T at q, loop S -> T -> S."""
		q = self.slit.q
		S = self._new_vert(q, 0, step, None)
		T = self._new_vert(q, 0, step, None)
		n = self.slit.n
		o = float(np.dot(n, q))
		s_in = self._new_seg(S, T, -n, -o, -1, step)
		s_out = self._new_seg(T, S, n, o, -1, step)
		self._link(s_in, s_out)
		self._link(s_out, s_in)
		self.tip, self.seg_in, self.seg_out = T, s_in, s_out
		self.slits.append((s_in, s_out))
		self.slit_starts.append(0.0)
		self.started = True
		step.events.append("start")

	def _next_crossing(self, exit_, after, exclude=-1):
		"""Nearest crossing of the knife ray beyond `after`: (lam, seg, point, mu) or None.

		exit_: segments crossed from inside to outside (tail on the - side, head on the + side), else
		entries (tail +, head -). Where the ray crosses an earlier, completed slit it meets two coincident
		segments at the same point, the exit of one piece and the entry of the other; the knife state
		(inside / outside material) decides which one is next, so their rounding order does not matter.
		"""
		q, d, n = self.slit.q, self.slit.d, self.slit.n
		want = (-1, 1) if exit_ else (1, -1)
		best = None
		for s in range(len(self.seg_a)):
			if not self.seg_alive[s] or s == exclude or s in (self.seg_in, self.seg_out):
				continue
			c, e = self.seg_a[s], self.seg_b[s]
			if (self._side[c], self._side[e]) != want:
				continue
			xc, xe = self._pos[c], self._pos[e]
			# canonical arithmetic (endpoints in lexicographic position order): the coincident segments of
			# an earlier slit give a bit-identical point, so an exit and the re-entry there coincide exactly
			flip = tuple(xe) < tuple(xc)
			u, w = (xe, xc) if flip else (xc, xe)
			a = np.dot(n, q - u) / np.dot(n, w - u)
			r = u + a * (w - u)
			mu = 1.0 - a if flip else a
			lam = float(np.dot(r - q, d))
			if lam > after and (best is None or lam < best[0]):
				best = (lam, s, r, float(mu))
		return best

	def _start(self, s0, al, q, lam, step):
		"""Start a slit at point q of segment s0 = a -> b (a on the + side): a -> q_a, q_a -> T, T -> q_b,
		q_b -> b."""
		a, b = self.seg_a[s0], self.seg_b[s0]
		src = ([a, b], [1.0 - al, al])
		sa, sb = self._side[a], self._side[b]
		if sa != 1 or sb != -1:
			raise CutError("slit start segment endpoints are not on opposite sides of the slit line")
		qa = self._new_vert(q, sa, step, src)
		qb = self._new_vert(q, sb, step, src)
		T = self._new_vert(q, 0, step, src)
		self._touch(s0, step)
		nxt = self.seg_next[s0]
		n = self.slit.n
		o = float(np.dot(n, self.slit.q))  # every slit lies on the knife line
		self.seg_b[s0] = qa
		s_in = self._new_seg(qa, T, -n, -o, -1, step)
		s_out = self._new_seg(T, qb, n, o, -1, step)
		s_qb = self._new_seg(qb, b, self.seg_n[s0], self.seg_o[s0], s0, step)
		self._link(s0, s_in)
		self._link(s_in, s_out)
		self._link(s_out, s_qb)
		self._link(s_qb, nxt)
		self.tip, self.seg_in, self.seg_out = T, s_in, s_out
		self.slits.append((s_in, s_out))
		self.slit_starts.append(lam)
		self.started = True
		step.events.append("start")

	def _exit(self, hit, step):
		lam, sc, r, mu = hit
		c, e = self.seg_a[sc], self.seg_b[sc]
		T = self.tip
		s_in, s_out = self.seg_in, self.seg_out
		xa, xb = self.seg_a[s_in], self.seg_b[s_out]
		# the base of a crack (knife started inside) is a side-0 vertex
		if self._side[xb] not in (0, self._side[c]) or self._side[xa] not in (0, self._side[e]):
			raise CutError("slit exit segment is not between the slit sides")
		# same loop: x_b ... c -> r_c -> x_b and r_d -> d ... x_a -> r_d (the loop splits in two);
		# another loop (a hole): the same relinking joins the two loops
		src = ([c, e], [1.0 - mu, mu])
		rc = self._new_vert(r, self._side[c], step, src)
		rd = self._new_vert(r, self._side[e], step, src)
		for x in (sc, s_in, s_out):
			self._touch(x, step)
		nxt = self.seg_next[sc]
		s_new = self._new_seg(rd, e, self.seg_n[sc], self.seg_o[sc], sc, step)
		self.seg_b[sc] = rc
		self.seg_a[s_out] = rc
		self._link(sc, s_out)
		self.seg_b[s_in] = rd
		self._link(s_in, s_new)
		self._link(s_new, nxt)
		self._valive[T] = False
		self._pos[T] = np.array(r)
		step.orphaned.append(T)
		self.tip = self.seg_in = self.seg_out = -1
		step.events.append("exit")

	# -----------------------------------------------------------------------------------------

	def _cut_faces(self):
		return list(self.slits)

	def _empty_delta(self):
		s = self.snapshot
		e = np.zeros(0, dtype=_INT)
		return CutDelta(dim=2, old=s, new=s, new_cut=np.zeros((0, 2, 2)), modified_faces=e,
			face_ancestor=np.arange(s.num_faces, dtype=_INT), new_vertices=e, orphaned_vertices=e.copy(),
			moved_vertices=e.copy(), new_vertex_sources={}, topology_changed=False, completed=False,
			plane_point=None if self.slit is None else self.slit.q.copy(),
			plane_normal=None if self.slit is None else self.slit.n.copy(),
			cut_faces=self._cut_faces(), t=self.length)

	def advance(self, amount):
		if not self.active or amount <= 0.0:
			return self._empty_delta()
		old = self.snapshot
		step = _Step(len(self.seg_a), len(self._pos), self.length)
		lam = self.length
		l1 = lam + float(amount)
		p_old = None
		if not self.started:
			if self._start_inside:
				self._start_crack(step)
			else:
				self._start(self._start_seg, self._alpha, self.slit.q, 0.0, step)
			p_old = self.slit.q.copy()
		elif self.cutting:
			p_old = self._pos[self.tip].copy()
		while True:
			if self.cutting:
				# measured from the slit start (not the tip) so a tip within rounding of the boundary
				# cannot tunnel through it
				hit = self._next_crossing(True, self.slit_starts[-1])
				if hit is None:
					raise AssertionError("slit ray never leaves the cage")
				if hit[0] > l1:
					lam = l1
					p_new = self.slit.point(l1)
					self._pos[self.tip] = p_new
					self._touch(self.seg_in, step)
					self._touch(self.seg_out, step)
					self._sweep(step, p_old, p_new)
					break
				self._exit(hit, step)
				self._sweep(step, p_old, hit[2])
				lam = max(lam, hit[0])
			else:
				ent = self._next_crossing(False, 0.0)
				if ent is None:
					self.completed = True
					self.active = False
					break
				if ent[0] > l1:
					lam = l1
					break
				lam = max(lam, ent[0])
				self._start(ent[1], ent[3], ent[2], lam, step)
				p_old = np.array(ent[2])
		self.length = self.t = lam
		if not step.events and not self.cutting:
			return self._empty_delta()  # travelling outside the material
		new = self._rebuild_snapshot()
		anc = np.arange(new.num_faces, dtype=_INT)
		for s, a in step.anc.items():
			anc[s] = a
		moved = []
		T = self.tip
		if 0 <= T < step.V_old and not np.array_equal(old.verts[T], self._pos[T]):
			moved.append(T)
		swept = np.array(step.swept, dtype=np.float64).reshape(-1, 2, 2)
		return CutDelta(dim=2, old=old, new=new, new_cut=swept,
			modified_faces=np.array(sorted(step.modified), dtype=_INT), face_ancestor=anc,
			new_vertices=np.arange(step.V_old, len(self._pos), dtype=_INT),
			orphaned_vertices=np.array(step.orphaned, dtype=_INT), moved_vertices=np.array(moved, dtype=_INT),
			new_vertex_sources=dict(step.sources), topology_changed=bool(step.events), completed=self.completed,
			events=list(step.events), plane_point=self.slit.q.copy(), plane_normal=self.slit.n.copy(),
			cut_faces=self._cut_faces(), t=self.length)

	@staticmethod
	def _sweep(step, a, b):
		if not np.array_equal(a, b):
			step.swept.append([np.array(a), np.array(b)])

	def _rebuild_snapshot(self):
		self.version += 1
		faces = [[a, b] for a, b in zip(self.seg_a, self.seg_b)]
		self.snapshot = build_snapshot(2, np.array(self._pos), faces,
			face_planes=(np.array(self.seg_n), np.array(self.seg_o)), vert_alive=np.array(self._valive),
			face_alive=np.array(self.seg_alive), version=self.version, face_root=np.array(self.seg_root))
		return self.snapshot

	# -----------------------------------------------------------------------------------------

	def loop_vertices(self):
		return [[self.seg_a[s] for s in loop] for loop in self.loops()]

	def check(self):
		S = len(self.seg_a)
		X = np.array(self._pos)
		used = set()
		for s in range(S):
			if not self.seg_alive[s]:
				continue
			nx, pv = self.seg_next[s], self.seg_prev[s]
			assert 0 <= nx < S and 0 <= pv < S, f"segment {s}: dangling links"
			assert self.seg_prev[nx] == s and self.seg_next[pv] == s, f"segment {s}: next/prev not inverse"
			assert self.seg_alive[nx] and self.seg_alive[pv]
			assert self.seg_b[s] == self.seg_a[nx], f"segment {s}: head != tail(next)"
			a, b = self.seg_a[s], self.seg_b[s]
			assert a != b, f"segment {s}: degenerate"
			assert self._valive[a] and self._valive[b], f"segment {s}: dead endpoint"
			used.add(a)
			# line inherited from the parent holds, orientation matches
			n, o = self.seg_n[s], self.seg_o[s]
			assert abs(np.dot(n, X[a]) - o) < 1e-9 and abs(np.dot(n, X[b]) - o) < 1e-9, f"segment {s}: off its line"
			e = X[b] - X[a]
			if np.linalg.norm(e) > 1e-12:
				assert np.dot(n, [e[1], -e[0]]) > 0.0, f"segment {s}: normal flipped"
		for v in range(len(self._pos)):
			assert self._valive[v] == (v in used), f"vertex {v}: alive flag vs use"
		outgoing = {}
		for s in range(S):
			if self.seg_alive[s]:
				assert self.seg_a[s] not in outgoing, f"vertex {self.seg_a[s]} has two outgoing segments"
				outgoing[self.seg_a[s]] = s
		# loops are CCW, or CW holes inside a CCW loop
		polys, areas = [], []
		for loop in self.loops():
			if len(loop) == 2:
				# the crack of a knife that started inside the material, until it reaches the boundary
				assert self.cutting and self.tip in (self.seg_a[loop[0]], self.seg_a[loop[1]]), "two-segment loop"
				continue
			assert len(loop) >= 3, "loop with fewer than 3 segments"
			p = X[[self.seg_a[s] for s in loop]]
			polys.append(p)
			areas.append(signed_area_2d(p))
			assert areas[-1] != 0.0, "degenerate loop"
		for P, a in zip(polys, areas):
			if a < 0.0:
				assert any(_inside(P, Q) for Q, b in zip(polys, areas) if b > 0.0), "CW loop is not a hole"
		# no proper crossings between segments (slit sides overlap, touching is allowed)
		segs = [s for s in range(S) if self.seg_alive[s]]
		P0 = X[[self.seg_a[s] for s in segs]]
		P1 = X[[self.seg_b[s] for s in segs]]
		for i in range(len(segs)):
			a0, a1 = P0[i], P1[i]
			da = a1 - a0
			db = P1 - P0
			o1 = da[0] * (P0[:, 1] - a0[1]) - da[1] * (P0[:, 0] - a0[0])
			o2 = da[0] * (P1[:, 1] - a0[1]) - da[1] * (P1[:, 0] - a0[0])
			o3 = db[:, 0] * (a0[1] - P0[:, 1]) - db[:, 1] * (a0[0] - P0[:, 0])
			o4 = db[:, 0] * (a1[1] - P0[:, 1]) - db[:, 1] * (a1[0] - P0[:, 0])
			tol = 1e-14
			bad = (o1 * o2 < -tol) & (o3 * o4 < -tol)
			bad[i] = False
			assert not np.any(bad), f"segment {segs[i]} crosses segment {segs[int(np.argmax(bad))]}"
		s = self.snapshot
		assert np.array_equal(s.verts, X), "snapshot vertex positions are stale"
		for f in range(S):
			assert list(s.face(f)) == [self.seg_a[f], self.seg_b[f]], f"segment {f}: snapshot differs"
		assert np.array_equal(s.face_root, np.array(self.seg_root)), "snapshot face_root differs"
		if self.slit is not None:
			assert len(self.slits) == len(self.slit_starts)
			assert all(self.slit_starts[i] <= self.slit_starts[i + 1] for i in range(len(self.slit_starts) - 1))
			for si, so in self.slits:
				assert self.seg_alive[si] and self.seg_alive[so]
				assert np.array_equal(self.seg_n[si], -self.slit.n) and np.array_equal(self.seg_n[so], self.slit.n)
			if self.cutting:
				T = self.tip
				assert self.active and (self.seg_in, self.seg_out) == self.slits[-1]
				assert self.seg_b[self.seg_in] == T and self.seg_a[self.seg_out] == T
				assert np.allclose(X[T], self.slit.point(self.length), atol=1e-12)
			if not self.active:
				assert not self.cutting
