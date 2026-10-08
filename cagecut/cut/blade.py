"""Blade kinematics shared by the cut topology code (numpy, float64).

3D blade: infinite line through p1 with unit direction l, swept along the unit direction d
(perpendicular to l). Cut plane normal n = normalize(l x d) through p1. The line at time t is
L(t) = {p1 + t d + s l}; a point x on the cut plane is reached at t(x) = (x - p1) . d.

2D slit: ray from q along unit direction d; normal n = (-d_y, d_x), side "+" is n.(x - q) >= 0.
"""

import numpy as np


class CutError(RuntimeError):
	"""A cut spec that violates the assumptions of the topology code (generic position, no face parallel
	to the blade line among the faces it cuts)."""


class Blade3D:
	def __init__(self, spec, verts=None, alive=None, clearance=1e-9, shift=1e-7, max_shifts=1000):
		p1 = np.asarray(spec["p1"], dtype=np.float64)
		p2 = np.asarray(spec["p2"], dtype=np.float64)
		l = p2 - p1
		if np.linalg.norm(l) < 1e-12:
			raise CutError("blade p1 and p2 coincide")
		l = l / np.linalg.norm(l)
		d = np.asarray(spec["direction"], dtype=np.float64)
		d = d - np.dot(d, l) * l
		if np.linalg.norm(d) < 1e-12:
			raise CutError("blade sweep direction is parallel to the blade line")
		d = d / np.linalg.norm(d)
		n = np.cross(l, d)
		n = n / np.linalg.norm(n)
		self.l, self.d, self.n = l, d, n
		self.p1 = p1.copy()
		self.shifts = 0
		# bounded: the blade is the segment p1-p2 (s in [0, |p2 - p1|] along l), its ends sweep the sides
		# of a strip; plunge: material the blade line already meets at t = 0 is entered there (zero-area
		# cut regions) instead of being cut as if the blade had come from outside
		self.bounded = bool(spec.get("bounded", False))
		self.plunge = bool(spec.get("plunge", False))
		self.s_range = (0.0, float(np.dot(p2 - p1, l))) if self.bounded else (-np.inf, np.inf)
		# generic position: no live cage vertex may lie on the cut plane. If one does, shift the
		# blade (p1, p2) along the plane normal by `shift` until all vertices are clear.
		if verts is not None:
			v = np.asarray(verts, dtype=np.float64)
			if alive is not None:
				v = v[np.asarray(alive, dtype=bool)]
			while v.shape[0] > 0 and np.min(np.abs((v - self.p1) @ n)) < clearance:
				if self.shifts >= max_shifts:
					raise CutError("could not move the blade into generic position")
				self.p1 = self.p1 + shift * n
				self.shifts += 1
		self.offset = float(np.dot(n, self.p1))

	def signed_dist(self, x):
		return (np.asarray(x, dtype=np.float64) - self.p1) @ self.n

	def side(self, x):
		"""+1 / -1 per point (sigma)."""
		return np.where(self.signed_dist(x) >= 0.0, 1, -1).astype(np.int8)

	def sweep_time(self, x):
		return (np.asarray(x, dtype=np.float64) - self.p1) @ self.d

	def along(self, x):
		"""Coordinate s of x along the blade line (0 at p1)."""
		return (np.asarray(x, dtype=np.float64) - self.p1) @ self.l

	def in_strip(self, s):
		return self.s_range[0] < s < self.s_range[1]

	def end(self, k, t):
		"""Position of blade end k (0: p1 side, 1: p2 side) at time t (bounded blades)."""
		return self.p1 + self.s_range[k] * self.l + t * self.d

	def end_time(self, k, normal, offset):
		"""Time at which blade end k crosses the plane n.x = o."""
		nd = float(np.dot(normal, self.d))
		if abs(nd) < 1e-12:
			raise CutError("a blade end slides along a face plane (face parallel to the sweep direction)")
		return (offset - float(np.dot(normal, self.p1 + self.s_range[k] * self.l))) / nd

	def crossing(self, xu, xv):
		"""Intersection of segment xu -> xv with the cut plane: (point, a) with point = xu + a (xv - xu)."""
		du = np.dot(self.n, self.p1 - xu)
		den = np.dot(self.n, xv - xu)
		a = du / den
		return xu + a * (xv - xu), a

	def tip(self, normal, offset, t):
		"""Closed-form tip: intersection of the blade line L(t) with the plane n.x = o."""
		nl = float(np.dot(normal, self.l))
		if abs(nl) < 1e-12:
			raise CutError("face plane is parallel to the blade line; it cannot host a cut tip")
		base = self.p1 + t * self.d
		s = (offset - np.dot(normal, base)) / nl
		return base + s * self.l

	def chord_direction(self, normal):
		"""Unit direction of the chord (cut plane ∩ face plane) along which t increases."""
		w = np.cross(self.n, normal)
		wd = np.dot(w, self.d)
		if abs(wd) < 1e-15:
			raise CutError("face plane is parallel to the blade line; it cannot host a cut tip")
		w = w if wd > 0.0 else -w
		return w / np.linalg.norm(w)

	def frame(self):
		"""In-plane 2D frame (l, d) of the cut plane: x -> ((x-p1).l, (x-p1).d)."""
		return self.l, self.d


class Slit2D:
	def __init__(self, q, d):
		self.q = np.asarray(q, dtype=np.float64)
		d = np.asarray(d, dtype=np.float64)
		self.d = d / np.linalg.norm(d)
		self.n = np.array([-self.d[1], self.d[0]])

	def signed_dist(self, x):
		return (np.asarray(x, dtype=np.float64) - self.q) @ self.n

	def side(self, x):
		return np.where(self.signed_dist(x) >= 0.0, 1, -1).astype(np.int8)

	def point(self, length):
		return self.q + length * self.d


def triangulate_polygon_2d(p2, eps=1e-14):
	"""Ear clipping of a simple polygon (either orientation). Returns index triples.

	Collinear / duplicate vertices are tolerated (zero-area ears are clipped, then dropped).
	"""
	n = p2.shape[0]
	if n < 3:
		return []
	area = 0.5 * np.sum(p2[:, 0] * np.roll(p2[:, 1], -1) - np.roll(p2[:, 0], -1) * p2[:, 1])
	sgn = 1.0 if area >= 0.0 else -1.0
	idx = list(range(n))
	tris = []

	def cross(o, a, b):
		return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

	guard = 0
	while len(idx) > 3 and guard < 10 * n * n:
		guard += 1
		m = len(idx)
		clipped = False
		# prefer degenerate ears first so collinear runs disappear cleanly
		for degenerate_pass in (True, False):
			for k in range(m):
				i0, i1, i2 = idx[(k - 1) % m], idx[k], idx[(k + 1) % m]
				a, b, c = p2[i0], p2[i1], p2[i2]
				cr = sgn * cross(a, b, c)
				scale = max(1e-300, np.sum((b - a) ** 2) + np.sum((c - b) ** 2))
				if degenerate_pass:
					if abs(cr) > eps * scale:
						continue
				else:
					if cr <= eps * scale:
						continue
					inside = False
					for j in idx:
						if j in (i0, i1, i2):
							continue
						p = p2[j]
						if (sgn * cross(a, b, p) >= 0.0 and sgn * cross(b, c, p) >= 0.0
								and sgn * cross(c, a, p) >= 0.0):
							# points coinciding with the ear corners do not block it
							if (np.array_equal(p, a) or np.array_equal(p, b) or np.array_equal(p, c)):
								continue
							inside = True
							break
					if inside:
						continue
				tris.append((i0, i1, i2))
				del idx[k]
				clipped = True
				break
			if clipped:
				break
		if not clipped:
			# numerically stuck (nearly degenerate polygon): fan the rest
			for k in range(1, len(idx) - 1):
				tris.append((idx[0], idx[k], idx[k + 1]))
			return tris
	if len(idx) == 3:
		tris.append(tuple(idx))
	return tris
