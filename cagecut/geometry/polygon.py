"""Planar polygon helpers shared by topology and snapshots (numpy, CPU)."""

import numpy as np

FACE_TRI = 0
FACE_QUAD = 1  # convex quad, mean value coordinates
FACE_NGON = 2  # anything else, nested 2D walk
FACE_SEGMENT = 3  # 2D cage segment


def newell_normal(points):
	"""Unnormalized Newell normal of a (possibly non-planar) polygon, (n,3) -> (3,)."""
	p = np.asarray(points, dtype=np.float64)
	q = np.roll(p, -1, axis=0)
	return np.array([
		np.sum((p[:, 1] - q[:, 1]) * (p[:, 2] + q[:, 2])),
		np.sum((p[:, 2] - q[:, 2]) * (p[:, 0] + q[:, 0])),
		np.sum((p[:, 0] - q[:, 0]) * (p[:, 1] + q[:, 1])),
	])


def frame_from_normal(n):
	"""Deterministic orthonormal tangent frame (u, v) with u x v = n (Duff et al. 2017).

	Children of a split face inherit the parent normal, so they get bit-identical frames.
	"""
	n = np.asarray(n, dtype=np.float64)
	s = 1.0 if n[2] >= 0.0 else -1.0
	a = -1.0 / (s + n[2])
	b = n[0] * n[1] * a
	u = np.array([1.0 + s * n[0] * n[0] * a, s * b, -s * n[0]])
	v = np.array([b, s + n[1] * n[1] * a, -n[1]])
	return u, v


def to_plane_2d(points, u, v):
	p = np.asarray(points, dtype=np.float64)
	return np.stack([p @ u, p @ v], axis=-1)


def signed_area_2d(p2):
	q = np.roll(p2, -1, axis=0)
	return 0.5 * np.sum(p2[:, 0] * q[:, 1] - q[:, 0] * p2[:, 1])


def is_convex_2d(p2, tol=1e-12):
	n = p2.shape[0]
	sign = 0.0
	for i in range(n):
		a, b, c = p2[i], p2[(i + 1) % n], p2[(i + 2) % n]
		cr = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
		if abs(cr) <= tol:
			continue
		if sign == 0.0:
			sign = np.sign(cr)
		elif np.sign(cr) != sign:
			return False
	return True


def point_in_polygon_2d(p, poly):
	"""Even-odd crossing test; coincident slit edges cancel, so slit polygons work."""
	x, y = p
	inside = False
	n = poly.shape[0]
	for i in range(n):
		x0, y0 = poly[i]
		x1, y1 = poly[(i + 1) % n]
		if (y0 > y) != (y1 > y):
			xc = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
			if xc > x:
				inside = not inside
	return inside


def classify_face(p2):
	n = p2.shape[0]
	if n == 3:
		return FACE_TRI
	if n == 4 and is_convex_2d(p2):
		return FACE_QUAD
	return FACE_NGON


def points_in_loops_2d(pts, a, b):
	"""Vectorized even-odd test of points (n, 2) against a set of edges a[k] -> b[k] (any number of loops,
	coincident slit edges cancel). Returns (n,) bool."""
	pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
	a = np.asarray(a, dtype=np.float64).reshape(-1, 2)
	b = np.asarray(b, dtype=np.float64).reshape(-1, 2)
	if a.shape[0] == 0 or pts.shape[0] == 0:
		return np.zeros(pts.shape[0], dtype=bool)
	x, y = pts[:, 0:1], pts[:, 1:2]
	x0, y0, x1, y1 = a[None, :, 0], a[None, :, 1], b[None, :, 0], b[None, :, 1]
	straddle = (y0 > y) != (y1 > y)
	dy = np.where(y1 == y0, 1.0, y1 - y0)
	xc = x0 + (y - y0) * (x1 - x0) / dy
	return (np.sum(straddle & (xc > x), axis=1) & 1).astype(bool)


def _cross2(o, a, b):
	return (a[..., 0] - o[..., 0]) * (b[..., 1] - o[..., 1]) - (a[..., 1] - o[..., 1]) * (b[..., 0] - o[..., 0])


def _segments_cross(p, q, a, b):
	"""Proper crossings of segment p-q with segments a[k]-b[k] (touching endpoints do not count)."""
	o1 = _cross2(p, q, a)
	o2 = _cross2(p, q, b)
	o3 = _cross2(a, b, p[None])
	o4 = _cross2(a, b, q[None])
	return (((o1 > 0) & (o2 < 0)) | ((o1 < 0) & (o2 > 0))) & (((o3 > 0) & (o4 < 0)) | ((o3 < 0) & (o4 > 0)))


def _in_wedge(P, prv, v, nxt, m):
	"""Direction v -> m lies inside the interior wedge (CCW polygon) at v between v -> nxt and v -> prv."""
	two_pi = 2.0 * np.pi
	ang = lambda w: np.arctan2(w[1], w[0])
	a0 = ang(P[nxt] - P[v])
	span = (ang(P[prv] - P[v]) - a0) % two_pi
	if span == 0.0:
		span = two_pi  # needle: the whole circle
	return (ang(m - P[v]) - a0) % two_pi <= span


def _bridge(P, poly, hole, blockers):
	"""Splice a CW hole into a CCW polygon (index lists into P) through a mutually visible pair of vertices
	(hole's rightmost vertex m, nearest visible polygon vertex v): poly .. v, m, hole .., m, v .. poly."""
	h = np.asarray(hole)
	im = int(np.argmax(P[h, 0]))
	m = P[h[im]]
	ea = P[np.concatenate([np.asarray(e[0]) for e in blockers])]
	eb = P[np.concatenate([np.asarray(e[1]) for e in blockers])]
	pv = np.asarray(poly)
	order = np.argsort(np.sum((P[pv] - m) ** 2, axis=1), kind="stable")
	n = len(poly)
	pick = int(order[0])
	for k in order:
		v = P[pv[k]]
		# edges sharing an endpoint position with the bridge do not block it
		keep = ~(np.all(ea == v, axis=1) | np.all(eb == v, axis=1) | np.all(ea == m, axis=1) | np.all(eb == m, axis=1))
		if np.any(_segments_cross(m, v, ea[keep], eb[keep])):
			continue
		if not _in_wedge(P, pv[(k - 1) % n], pv[k], pv[(k + 1) % n], m):
			continue
		pick = int(k)
		break
	hl = list(h[im:]) + list(h[:im + 1])
	return list(pv[:pick + 1]) + hl + list(pv[pick:])


def _ear_clip(P, idx, tol):
	"""Ear clipping of a weakly simple CCW polygon (index list into P). Zero-area vertices (collinear
	points, slit needles, bridge duplicates) are removed without a triangle."""
	idx = list(idx)
	tris = []
	k = 0
	misses = 0
	while len(idx) > 3:
		n = len(idx)
		k %= n
		ia, ib, ic = idx[k - 1], idx[k], idx[(k + 1) % n]
		a, b, c = P[ia], P[ib], P[ic]
		cr = _cross2(a, b, c)
		if abs(cr) <= tol:
			idx.pop(k)
			misses = 0
			continue
		ear = cr > 0.0
		if ear:
			q = P[np.asarray(idx)]
			mask = ~(np.all(q == a, axis=1) | np.all(q == b, axis=1) | np.all(q == c, axis=1))
			q = q[mask]
			if q.shape[0]:
				ins = (_cross2(a, b, q) >= 0.0) & (_cross2(b, c, q) >= 0.0) & (_cross2(c, a, q) >= 0.0)
				ear = not np.any(ins)
		if ear or misses > n:
			if not ear:
				# numerical trouble (not weakly simple): clip the most convex vertex
				qa = P[np.roll(idx, 1)]
				qb = P[np.asarray(idx)]
				qc = P[np.roll(idx, -1)]
				k = int(np.argmax(_cross2(qa, qb, qc)))
				ia, ib, ic = idx[k - 1], idx[k], idx[(k + 1) % n]
				if _cross2(P[ia], P[ib], P[ic]) <= tol:
					break
			tris.append((ia, ib, ic))
			idx.pop(k)
			misses = 0
			continue
		k += 1
		misses += 1
	if len(idx) == 3 and _cross2(P[idx[0]], P[idx[1]], P[idx[2]]) > tol:
		tris.append(tuple(idx))
	return tris


def triangulate_polygon_with_holes(loops2d):
	"""Triangulate a planar region bounded by several loops (for rendering multi-loop faces).

	loops2d: list of (n_i, 2) arrays (any orientation). Loop nesting is decided by containment (even-odd:
	depth 0 = outer boundary, 1 = hole, 2 = island in a hole, ...). Each hole is bridged into its outer
	loop, then the result is ear clipped. Returns (k, 3) int64 triangles indexing the concatenated loop
	vertices, CCW in the 2D frame. Zero-area parts (slits, collinear points) produce no triangles.
	"""
	loops = [np.asarray(l, dtype=np.float64).reshape(-1, 2) for l in loops2d]
	offs = np.concatenate([[0], np.cumsum([l.shape[0] for l in loops])]).astype(np.int64)
	if offs[-1] == 0:
		return np.zeros((0, 3), dtype=np.int64)
	P = np.concatenate(loops)
	ext = float(np.max(np.ptp(P, axis=0))) if P.shape[0] > 1 else 0.0
	tol = 1e-14 * max(ext, 1e-300) ** 2
	ids = [np.arange(offs[i], offs[i + 1]) for i in range(len(loops))]
	use = [i for i in range(len(loops)) if loops[i].shape[0] >= 3 and abs(signed_area_2d(loops[i])) > tol]
	# nesting depth: loops containing (most of) the vertices of loop i
	depth = {}
	parent = {}
	for i in use:
		cont = []
		for j in use:
			if j == i:
				continue
			a, b = loops[j], np.roll(loops[j], -1, axis=0)
			if np.mean(points_in_loops_2d(loops[i], a, b)) > 0.5:
				cont.append(j)
		depth[i] = len(cont)
		parent[i] = cont
	tris = []
	for o in use:
		if depth[o] % 2:
			continue
		poly = list(ids[o] if signed_area_2d(loops[o]) > 0 else ids[o][::-1])
		holes = [h for h in use if depth[h] == depth[o] + 1 and o in parent[h]]
		holes = [list(ids[h] if signed_area_2d(loops[h]) < 0 else ids[h][::-1]) for h in holes]
		holes.sort(key=lambda h: -np.max(P[h, 0]))
		for k, h in enumerate(holes):
			rest = holes[k:]
			blockers = [(poly, np.roll(poly, -1))] + [(r, np.roll(r, -1)) for r in rest]
			poly = _bridge(P, poly, h, blockers)
		tris += _ear_clip(P, poly, tol)
	return np.asarray(tris, dtype=np.int64).reshape(-1, 3)
