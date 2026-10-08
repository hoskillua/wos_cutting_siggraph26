"""Cage volume, inside test (winding numbers), boundary distance and Poisson disk quadrature (numpy).

3D cages use the snapshot's planar model (face_points); fan triangles are only used for signed
quantities (volume, solid angle), where they are exact for any planar polygon, also with several loops
(holes). Zero-thickness slits (two coincident faces with opposite orientation) cancel in the winding
number.
"""

import math

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

_CHUNK = 1 << 19  # max points * primitives per vectorized block


class _Boundary:
	def __init__(self, snapshot):
		self.dim = snapshot.dim
		alive = np.asarray(snapshot.face_alive, dtype=bool)
		faces = np.nonzero(alive)[0]
		if self.dim == 2:
			seg = snapshot.verts[snapshot.face_verts].reshape(-1, 2, 2)[alive]
			self.a, self.b = seg[:, 0], seg[:, 1]
			self.lo = np.minimum(self.a, self.b)
			self.hi = np.maximum(self.a, self.b)
			return
		# every slot of every alive face, vectorized (faces are stored contiguously in order); the edge of
		# slot j ends at slot face_vnext[j] (multi-loop faces: outer loops and holes)
		size = snapshot.face_size.astype(np.int64)
		start = snapshot.face_start.astype(np.int64)
		f_of = snapshot.face_of_slot()
		keep = alive[f_of]
		P = snapshot.planar_points()
		nxt = np.asarray(snapshot.face_vnext, dtype=np.int64).reshape(-1)
		local = np.repeat(np.cumsum(alive) - 1, size)  # index of the face among alive faces
		self.a = P[keep]
		self.b = P[nxt[keep]]
		self.a2 = snapshot.face_coords2d[keep]
		self.b2 = snapshot.face_coords2d[nxt[keep]]
		self.edge_face = local[keep]
		self.lo = np.minimum.reduceat(P, start, axis=0)[alive] if P.shape[0] else np.zeros((0, 3))
		self.hi = np.maximum.reduceat(P, start, axis=0)[alive] if P.shape[0] else np.zeros((0, 3))
		# signed fan triangles (apex, a, b) over every edge of every loop, apex = the face's first slot:
		# exact signed area / solid angle of a planar face with holes (hole loops are reversed). Edges
		# touching the apex are degenerate and skipped; for one loop these are the triangles (s0, sj, sj+1).
		slot = np.arange(P.shape[0])
		s0 = start[f_of]
		fan = keep & (slot != s0) & (nxt != s0)
		self.tris = np.stack([P[s0[fan]], P[slot[fan]], P[nxt[fan]]], axis=1).reshape(-1, 3, 3)
		self.n = snapshot.face_normal[faces]
		self.o = snapshot.face_offset[faces]
		self.u = snapshot.face_u[faces]
		self.v = snapshot.face_v[faces]

	@property
	def num_faces(self):
		return self.lo.shape[0]

	# -- signed measure ------------------------------------------------------------------------

	def volume(self):
		if self.dim == 2:
			return 0.5 * float(np.sum(self.a[:, 0] * self.b[:, 1] - self.b[:, 0] * self.a[:, 1]))
		t = self.tris
		return float(np.sum(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])))) / 6.0

	# -- winding number ------------------------------------------------------------------------

	def winding(self, points):
		pts = np.asarray(points, dtype=np.float64).reshape(-1, self.dim)
		out = np.zeros(pts.shape[0])
		m = max(1, (self.a.shape[0] if self.dim == 2 else self.tris.shape[0]))
		step = max(1, _CHUNK // m)
		for s in range(0, pts.shape[0], step):
			p = pts[s:s + step]
			if self.dim == 2:
				a = self.a[None] - p[:, None]
				b = self.b[None] - p[:, None]
				cr = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
				dt = np.sum(a * b, axis=-1)
				out[s:s + step] = np.sum(np.arctan2(cr, dt), axis=1) / (2.0 * math.pi)
			else:
				a = self.tris[None, :, 0] - p[:, None]
				b = self.tris[None, :, 1] - p[:, None]
				c = self.tris[None, :, 2] - p[:, None]
				la, lb, lc = (np.sqrt(np.sum(x * x, axis=-1)) for x in (a, b, c))
				det = np.sum(a * np.cross(b, c), axis=-1)
				den = la * lb * lc + np.sum(a * b, -1) * lc + np.sum(a * c, -1) * lb + np.sum(b * c, -1) * la
				out[s:s + step] = np.sum(2.0 * np.arctan2(det, den), axis=1) / (4.0 * math.pi)
		return out

	# -- local primitive selection -------------------------------------------------------------

	def near_faces(self, center, radius):
		"""Faces whose bounding box is within radius of center (sorted ids)."""
		d = np.maximum(np.maximum(self.lo - center, center - self.hi), 0.0)
		return np.nonzero(np.sum(d * d, axis=1) <= radius * radius)[0]

	def _edges_of(self, faces):
		"""Edge ids of the given faces (3D), grouped by face, and the group starts."""
		if faces is None:
			e = np.arange(self.a.shape[0])
			return e, np.searchsorted(self.edge_face, np.arange(self.num_faces))
		e = np.nonzero(np.isin(self.edge_face, faces))[0]
		return e, np.searchsorted(self.edge_face[e], faces)

	def _in_faces(self, p, faces, e, starts):
		"""(n, len(faces)) bool: projection of p inside each planar face polygon (even-odd)."""
		pu = p @ self.u[faces].T
		pv = p @ self.v[faces].T
		loc = np.searchsorted(faces, self.edge_face[e])  # local face index of each edge
		x = pu[:, loc]
		y = pv[:, loc]
		x0, y0 = self.a2[e, 0], self.a2[e, 1]
		x1, y1 = self.b2[e, 0], self.b2[e, 1]
		straddle = (y0[None] > y) != (y1[None] > y)
		dy = np.where(y1 == y0, 1.0, y1 - y0)
		xc = x0[None] + (y - y0[None]) * (x1 - x0)[None] / dy[None]
		cross = (straddle & (xc > x)).astype(np.int64)
		return (np.add.reduceat(cross, starts, axis=1) & 1).astype(bool)

	# -- unsigned distance ---------------------------------------------------------------------

	def distance(self, points, faces=None):
		"""Exact distance to the boundary (or to the given subset of faces; inf if empty)."""
		pts = np.asarray(points, dtype=np.float64).reshape(-1, self.dim)
		out = np.full(pts.shape[0], np.inf)
		if self.dim == 2:
			e = np.arange(self.a.shape[0]) if faces is None else np.asarray(faces)
			fl = None
		else:
			fl = np.arange(self.num_faces) if faces is None else np.asarray(faces)
			e, starts = self._edges_of(None if faces is None else fl)
		if e.size == 0:
			return out
		a, b = self.a[e], self.b[e]
		ab = b - a
		ab2 = np.maximum(np.sum(ab * ab, axis=1), 1e-300)
		step = max(1, _CHUNK // e.size)
		for s in range(0, pts.shape[0], step):
			p = pts[s:s + step]
			ap = p[:, None] - a[None]
			t = np.clip(np.sum(ap * ab[None], axis=-1) / ab2[None], 0.0, 1.0)
			d = ap - t[..., None] * ab[None]
			dist = np.sqrt(np.min(np.sum(d * d, axis=-1), axis=1))
			if self.dim == 3:
				plane = np.abs(p @ self.n[fl].T - self.o[fl][None])
				plane = np.where(self._in_faces(p, fl, e, starts), plane, np.inf)
				dist = np.minimum(dist, plane.min(axis=1))
			out[s:s + step] = dist
		return out

	# -- segment crossings ---------------------------------------------------------------------

	def crossing_parity(self, s, c, faces):
		"""Parity of the number of boundary crossings of segments s -> c[k] with the given faces.
		Assumes generic position (segments do not pass through edges or vertices)."""
		c = np.asarray(c, dtype=np.float64).reshape(-1, self.dim)
		if len(faces) == 0:
			return np.zeros(c.shape[0], dtype=bool)
		if self.dim == 2:
			a, b = self.a[faces], self.b[faces]

			def orient(p, q, r):
				return (q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0])
			ss = np.broadcast_to(s, c.shape)
			o1 = orient(ss[:, None], c[:, None], a[None])
			o2 = orient(ss[:, None], c[:, None], b[None])
			o3 = orient(a[None], b[None], ss[:, None])
			o4 = orient(a[None], b[None], c[:, None])
			hit = ((o1 > 0) != (o2 > 0)) & ((o3 > 0) != (o4 > 0))
			return (np.sum(hit, axis=1) & 1).astype(bool)
		faces = np.asarray(faces)
		e, starts = self._edges_of(faces)
		n, o = self.n[faces], self.o[faces]
		sa = s @ n.T - o  # (F,)
		sc = c @ n.T - o[None]  # (k, F)
		straddle = (sa[None] > 0) != (sc > 0)
		den = np.where(sa[None] - sc == 0.0, 1.0, sa[None] - sc)
		t = sa[None] / den  # (k, F)
		hit = np.zeros(straddle.shape, dtype=bool)
		for k in range(c.shape[0]):
			fs = np.nonzero(straddle[k])[0]
			if fs.size == 0:
				continue
			x = s[None] + t[k, fs, None] * (c[k] - s)[None]  # intersection with each face plane
			ins = self._in_faces(x, faces, e, starts)  # (len(fs), F)
			hit[k, fs] = ins[np.arange(fs.size), fs]
		return (np.sum(hit, axis=1) & 1).astype(bool)


def cage_volume(snapshot):
	"""Enclosed volume (area in 2D) of the cage, summed over components."""
	return _Boundary(snapshot).volume()


def winding_number(snapshot, points):
	return _Boundary(snapshot).winding(points)


def inside(snapshot, points):
	"""Boolean inside test by winding number (> 0.5)."""
	return _Boundary(snapshot).winding(points) > 0.5


def boundary_distance(snapshot, points):
	"""Exact unsigned distance to the cage boundary (planar model)."""
	return _Boundary(snapshot).distance(points)


def _classify_cells(bd, lo, hi, h, margin):
	"""Background grid: 0 = cell entirely outside, 1 = entirely inside with every point at distance
	>= margin from the boundary, 2 = undecided (touched by a face box grown by margin). Undecided
	cells separate the others, so each connected component of decided cells has one status,
	evaluated by the winding number at one representative."""
	shape = np.maximum(np.ceil((hi - lo) / h).astype(np.int64), 1)
	und = np.zeros(tuple(shape), dtype=bool)
	a = np.clip(np.floor((bd.lo - margin - lo) / h).astype(np.int64), 0, shape - 1)
	b = np.clip(np.floor((bd.hi + margin - lo) / h).astype(np.int64), 0, shape - 1)
	for f in range(a.shape[0]):
		und[tuple(slice(a[f, i], b[f, i] + 1) for i in range(bd.dim))] = True
	labels, nlab = ndimage.label(~und)
	cls = np.full(tuple(shape), 2, dtype=np.int8)
	if nlab > 0:
		flat = labels.reshape(-1)
		_, first = np.unique(flat, return_index=True)
		first = first[1:]  # skip label 0 (undecided cells)
		centers = lo + (np.stack(np.unravel_index(first, shape), axis=-1) + 0.5) * h
		status = np.concatenate([[2], np.where(bd.winding(centers) > 0.5, 1, 0)]).astype(np.int8)
		cls = status[labels]
	return cls, shape


def sample_quadrature(snapshot, spacing, margin, seed=0, k=30):
	"""Bridson Poisson disk samples with pairwise distance >= spacing, inside the cage and at
	distance >= margin from its boundary. Disconnected regions are seeded by dart throwing until
	a batch of random darts finds no free spot. Deterministic for a given seed.

	A background cell classification decides most candidates without geometry; the rest are tested
	locally around their parent sample s (inside iff the segment s -> c crosses the boundary an
	even number of times; distance against faces near s, exact below a cap)."""
	bd = _Boundary(snapshot)
	d = bd.dim
	r = float(spacing)
	margin = float(margin)
	cap = 2.0 * r + margin  # stored sample distances are exact below cap, lower bounds above
	rng = np.random.default_rng(seed)
	verts = snapshot.verts[snapshot.vert_alive]
	lo = verts.min(axis=0) + margin
	hi = verts.max(axis=0) - margin
	if np.any(hi <= lo):
		return np.zeros((0, d))
	ext = float(np.max(hi - lo))
	hcell = max(r, ext / (256 if d == 2 else 96))
	clo = lo - 0.5 * hcell
	cls, cshape = _classify_cells(bd, clo, hi + 0.5 * hcell, hcell, margin)
	open_cells = np.nonzero(cls.reshape(-1) > 0)[0]
	if open_cells.size == 0:
		return np.zeros((0, d))

	def cell_class(c):
		ci = np.clip(np.floor((c - clo) / hcell).astype(np.int64), 0, cshape - 1)
		return cls[tuple(ci[:, i] for i in range(d))]

	cell = r / math.sqrt(d)
	shape = np.maximum(np.ceil((hi - lo) / cell).astype(np.int64), 1)
	grid = -np.ones(tuple(shape), dtype=np.int64)
	reach = int(math.ceil(math.sqrt(d)))
	offs = np.stack(np.meshgrid(*[np.arange(-reach, reach + 1)] * d, indexing="ij"), axis=-1).reshape(-1, d)
	pts = np.zeros((1024, d))
	dist = np.zeros(1024)  # (capped) boundary distance of each sample
	count = 0
	active = []

	def far_from_samples(c):
		ci = np.clip(((c - lo) / cell).astype(np.int64), 0, shape - 1)
		nb = ci[:, None, :] + offs[None]
		inb = np.all((nb >= 0) & (nb < shape), axis=-1)
		idx = grid[tuple(np.clip(nb, 0, shape - 1)[..., i] for i in range(d))]
		idx = np.where(inb, idx, -1)
		q = pts[np.maximum(idx, 0)]
		close = (idx >= 0) & (np.sum((q - c[:, None]) ** 2, axis=-1) < r * r)
		return ~np.any(close, axis=1)

	def add(p):
		nonlocal pts, dist, count
		if count == pts.shape[0]:
			pts = np.vstack([pts, np.zeros_like(pts)])
			dist = np.concatenate([dist, np.zeros_like(dist)])
		pts[count] = p
		dist[count] = min(bd.distance(p[None], bd.near_faces(p, cap))[0], cap)
		active.append(count)
		grid[tuple(np.clip(((p - lo) / cell).astype(np.int64), 0, shape - 1))] = count
		count += 1

	while True:
		# dart throwing seed for a (new) region: uniform over the non-outside cells
		seeded = False
		for _ in range(4):
			cid = open_cells[rng.integers(open_cells.size, size=2048)]
			darts = clo + (np.stack(np.unravel_index(cid, cshape), axis=-1) + rng.uniform(size=(2048, d))) * hcell
			darts = darts[np.all((darts >= lo) & (darts <= hi), axis=1)]
			if count > 0 and darts.shape[0] > 0:
				darts = darts[cKDTree(pts[:count]).query(darts)[0] >= r]
			cc_all = cell_class(darts)
			sure = np.nonzero(cc_all == 1)[0]
			if sure.size > 0:
				add(darts[sure[0]])
				seeded = True
				break
			darts = darts[cc_all == 2]
			for s0 in range(0, darts.shape[0], 128):
				cc = darts[s0:s0 + 128]
				ok = bd.distance(cc) >= margin
				if np.any(ok):
					ok[ok] = bd.winding(cc[ok]) > 0.5
				if np.any(ok):
					add(cc[int(np.argmax(ok))])
					seeded = True
					break
			if seeded:
				break
		if not seeded:
			break
		while active:
			i = active[rng.integers(len(active))]
			s = pts[i].copy()
			dirs = rng.standard_normal((k, d))
			dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
			rad = r * (1.0 + rng.uniform(size=k) * (2.0 ** d - 1.0)) ** (1.0 / d)
			c = s + dirs * rad[:, None]
			keep = np.all((c >= lo) & (c <= hi), axis=1)
			c, rad = c[keep], rad[keep]
			if c.shape[0] > 0:
				keep = far_from_samples(c) & (cell_class(c) > 0)
				c, rad = c[keep], rad[keep]
			j = -1
			if c.shape[0] > 0:
				cc = cell_class(c)
				if cc[0] == 1 or rad[0] <= dist[i] - margin:
					# decided by the cell grid, or inside the empty ball around s (1-Lipschitz)
					j = 0
				else:
					near = bd.near_faces(s, 2.0 * r + cap)
					ok = cc == 1
					und = np.nonzero(~ok)[0]
					okd = bd.distance(c[und], near) >= margin
					if np.any(okd):
						okd[okd] = ~bd.crossing_parity(s, c[und[okd]], near)
					ok[und] = okd
					j = int(np.argmax(ok)) if np.any(ok) else -1
			if j >= 0:
				add(c[j])
			else:
				active.remove(i)
	return pts[:count].copy()
