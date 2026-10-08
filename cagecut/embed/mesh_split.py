"""Splitting an embedded 3D surface mesh along the growing cut.

A triangle is split when it straddles the cut plane and its intersection segment with the plane
lies inside the current cut region: the union of the plus cut faces of all cut regions of the blade
(delta.cut_faces), each possibly with several loops (holes), tested even-odd. Each crossing edge gets
two new vertices at the crossing point, offset by +-offset along the plane normal (one per side). They
are shared with neighbouring triangles through an edge map that lives for the whole blade, so the two
sides of the crack stay watertight among split triangles. Each triangle is split at most once per blade
(cut spec); the pieces keep the original orientation. Unsplit neighbours keep the original edge (the
crack front has T-junctions until they are split too).
"""

from dataclasses import dataclass, field

import numpy as np

from cagecut.geometry.polygon import frame_from_normal, points_in_loops_2d, to_plane_2d


@dataclass
class SplitResult:
	new_points: np.ndarray  # (n, 3) rest positions of new mesh vertices (ids = new_point_ids)
	new_point_sources: list = field(default_factory=list)  # [(old vertex ids, weights)] per new point
	changed: bool = False
	new_point_ids: np.ndarray = None  # (n,) mesh vertex ids
	new_point_side: np.ndarray = None  # (n,) +1 / -1: side of the cut plane
	split_triangles: np.ndarray = None  # ids of triangles replaced in place this call


def cut_face_pairs(cut_faces):
	"""delta.cut_faces as a list of (plus, minus) int pairs. Accepts the list form and the older single
	(plus, minus) tuple ((-1, -1) before contact)."""
	if cut_faces is None:
		return []
	cf = list(cut_faces)
	if len(cf) == 2 and all(np.ndim(x) == 0 for x in cf):
		cf = [tuple(cf)]
	out = []
	for pair in cf:
		a, b = (int(x) for x in pair)
		if a >= 0 or b >= 0:
			out.append((a, b))
	return out


def region_edges(snapshot, faces, u, v):
	"""2D edges (a, b) in the frame (u, v) of every loop of the given faces (slot j -> face_vnext[j])."""
	a, b = [], []
	for f in faces:
		s, n = int(snapshot.face_start[f]), int(snapshot.face_size[f])
		slots = np.arange(s, s + n)
		pts = snapshot.face_points(f)
		nxt = np.asarray(snapshot.face_vnext, dtype=np.int64)[slots] - s
		a.append(to_plane_2d(pts, u, v))
		b.append(to_plane_2d(pts[nxt], u, v))
	if not a:
		return np.zeros((0, 2)), np.zeros((0, 2))
	return np.concatenate(a), np.concatenate(b)


class MeshSplitter:
	def __init__(self, verts, tris):
		self.verts = np.array(verts, dtype=np.float64).reshape(-1, 3)
		self.tris = np.array(tris, dtype=np.int64).reshape(-1, 3)
		self._key = None
		self._edges = {}  # (min, max) -> (v_plus, v_minus)
		self._done = np.zeros(self.tris.shape[0], dtype=bool)

	def _empty(self):
		return SplitResult(np.zeros((0, 3)), [], False, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64),
			np.zeros(0, dtype=np.int64))

	def reset_spec(self):
		"""A new blade begins: fresh edge map, every triangle may be split once again."""
		self._key = None
		self._edges = {}
		self._done = np.zeros(self.tris.shape[0], dtype=bool)

	def split(self, delta, offset=1e-4) -> SplitResult:
		if delta.dim != 3 or delta.plane_normal is None:
			return self._empty()
		snap = delta.new
		plus = [p for p, m in cut_face_pairs(delta.cut_faces) if 0 <= p < snap.num_faces and snap.face_alive[p]]
		if not plus:
			return self._empty()
		n = np.asarray(delta.plane_normal, dtype=np.float64)
		n = n / np.linalg.norm(n)
		p0 = np.asarray(delta.plane_point, dtype=np.float64)
		key = (tuple(np.round(n, 12)), round(float(n @ p0), 12))
		if key != self._key:
			# another blade plane: a new cut spec
			self.reset_spec()
			self._key = key
		u, v = frame_from_normal(n)
		ea, eb = region_edges(snap, plus, u, v)

		s = (self.verts - p0) @ n
		side = s >= 0.0
		ts = side[self.tris]
		cnt = ts.sum(axis=1)
		cand = np.nonzero((cnt == 1) | (cnt == 2))[0]
		cand = cand[~self._done[cand]]
		if cand.size == 0:
			return self._empty()
		# rotate each candidate so its lone vertex comes first (orientation preserved)
		lone_val = (cnt[cand] == 1)
		lone = np.argmax(ts[cand] == lone_val[:, None], axis=1)
		r = np.arange(cand.size)
		tc = self.tris[cand]
		a = tc[r, lone]
		b = tc[r, (lone + 1) % 3]
		c = tc[r, (lone + 2) % 3]
		t_ab = s[a] / (s[a] - s[b])
		t_ca = s[c] / (s[c] - s[a])
		X = self.verts
		r_ab = X[a] + t_ab[:, None] * (X[b] - X[a])
		r_ca = X[c] + t_ca[:, None] * (X[a] - X[c])
		ok = points_in_loops_2d(to_plane_2d(r_ab, u, v), ea, eb)
		ok &= points_in_loops_2d(to_plane_2d(r_ca, u, v), ea, eb)
		ok &= points_in_loops_2d(to_plane_2d(0.5 * (r_ab + r_ca), u, v), ea, eb)
		sel = np.nonzero(ok)[0]
		if sel.size == 0:
			return self._empty()

		new_pts, sources, sides = [], [], []
		nv0 = self.verts.shape[0]

		def edge_verts(i, j, t, pos):
			# crossing of edge i -> j at parameter t (point pos); returns (v_plus, v_minus)
			k = (min(i, j), max(i, j))
			if k not in self._edges:
				base = nv0 + len(new_pts)
				for sgn in (1.0, -1.0):
					new_pts.append(pos + sgn * offset * n)
					sources.append((np.array([i, j], dtype=np.int64), np.array([1.0 - t, t])))
					sides.append(int(sgn))
				self._edges[k] = (base, base + 1)
			return self._edges[k]

		new_tris = []
		for m in sel:
			ia, ib, ic = int(a[m]), int(b[m]), int(c[m])
			pab = edge_verts(ia, ib, float(t_ab[m]), r_ab[m])
			pca = edge_verts(ic, ia, float(t_ca[m]), r_ca[m])
			sa = 0 if side[ia] else 1  # index into (plus, minus)
			so = 1 - sa
			self.tris[cand[m]] = (ia, pab[sa], pca[sa])
			new_tris.append((pab[so], ib, ic))
			new_tris.append((pab[so], ic, pca[so]))
		self._done[cand[sel]] = True
		if new_pts:
			self.verts = np.vstack([self.verts, np.asarray(new_pts)])
		self.tris = np.vstack([self.tris, np.asarray(new_tris, dtype=np.int64)])
		self._done = np.concatenate([self._done, np.ones(len(new_tris), dtype=bool)])
		ids = np.arange(nv0, self.verts.shape[0], dtype=np.int64)
		return SplitResult(self.verts[nv0:].copy(), sources, True, ids, np.array(sides, dtype=np.int64),
			cand[sel].astype(np.int64))
