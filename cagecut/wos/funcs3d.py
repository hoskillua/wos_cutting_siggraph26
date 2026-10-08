"""Warp functions for 3D walk on spheres: planar polygon distances, boundary values, k-DOPs.

All geometry uses the snapshot's planar model: face f is the polygon with 2D frame coordinates
`coords` in the plane (normal, offset) with tangent frame (fu, fv).
"""

import math

import numpy as np
import warp as wp

from cagecut.geometry.polygon import FACE_QUAD, FACE_TRI

# tie window for equidistant faces (float32 geometry, so a bit wider than float64 rounding)
TIE_REL = 1.0e-6
# nested n-gon walks use their own rng stream
NGON_SEED_XOR = 0x2545F491

_S3 = 1.0 / math.sqrt(3.0)
_S2 = 1.0 / math.sqrt(2.0)

# k-DOP axes: 3 coordinate axes, 4 cube diagonals, 6 face diagonals (normalized). Must match kdop13_proj.
KDOP13_AXES = np.array([
	[1, 0, 0], [0, 1, 0], [0, 0, 1],
	[_S3, _S3, _S3], [_S3, _S3, -_S3], [_S3, -_S3, _S3], [-_S3, _S3, _S3],
	[_S2, _S2, 0], [_S2, -_S2, 0], [_S2, 0, _S2], [_S2, 0, -_S2], [0, _S2, _S2], [0, _S2, -_S2],
], dtype=np.float64)
AABB3_AXES = np.eye(3)

vec13 = wp.types.vector(length=13, dtype=wp.float32)

S3 = wp.constant(wp.float32(_S3))
S2 = wp.constant(wp.float32(_S2))
TIE = wp.constant(wp.float32(TIE_REL))
NGON_XOR = wp.constant(wp.int32(NGON_SEED_XOR))
KIND_TRI = wp.constant(wp.int32(FACE_TRI))
KIND_QUAD = wp.constant(wp.int32(FACE_QUAD))


@wp.struct
class Cage3:
	face_start: wp.array(dtype=wp.int32)
	face_size: wp.array(dtype=wp.int32)
	face_kind: wp.array(dtype=wp.int32)
	face_alive: wp.array(dtype=wp.int32)
	face_root: wp.array(dtype=wp.int32)
	face_verts: wp.array(dtype=wp.int32)
	vnext: wp.array(dtype=wp.int32)  # per slot: slot of the next vertex along its loop (multi-loop faces)
	coords: wp.array(dtype=wp.vec2)
	edge_line: wp.array(dtype=wp.vec3)  # per face edge: supporting line (unit normal, offset) in the face frame
	normal: wp.array(dtype=wp.vec3)
	offset: wp.array(dtype=wp.float32)
	fu: wp.array(dtype=wp.vec3)
	fv: wp.array(dtype=wp.vec3)
	bvh: wp.uint64
	num_faces: wp.int32
	diam: wp.float32
	use_bvh: wp.int32


@wp.func
def kdop13_proj(p: wp.vec3):
	return vec13(p[0], p[1], p[2],
		(p[0] + p[1] + p[2]) * S3, (p[0] + p[1] - p[2]) * S3, (p[0] - p[1] + p[2]) * S3, (-p[0] + p[1] + p[2]) * S3,
		(p[0] + p[1]) * S2, (p[0] - p[1]) * S2, (p[0] + p[2]) * S2, (p[0] - p[2]) * S2, (p[1] + p[2]) * S2,
		(p[1] - p[2]) * S2)


@wp.func
def aabb3_proj(p: wp.vec3):
	return p


@wp.func
def sample_dir3(rng: wp.uint32):
	"""Uniform direction; returns the advanced rng state too (func args are copies)."""
	d = wp.sample_unit_sphere_surface(rng)
	return d, rng


@wp.func
def tie_tol(d: float):
	return TIE * wp.max(1.0, d)


@wp.func
def seg_closest2(q: wp.vec2, a: wp.vec2, b: wp.vec2):
	"""Closest point parameter t in [0,1] on segment a-b and squared distance."""
	e = b - a
	ee = wp.dot(e, e)
	t = float(0.0)
	if ee > 0.0:
		t = wp.clamp(wp.dot(q - a, e) / ee, 0.0, 1.0)
	c = a + t * e
	return t, wp.length_sq(q - c), c


@wp.func
def face_scan3(cage: Cage3, f: int, q: wp.vec2):
	"""In-plane part of the face distance for frame coords q: (inside, squared distance to the boundary,
	closest edge slot, its parameter). Edges (j, vnext[j]) of all loops: the even-odd test treats points
	over a hole as outside."""
	start = cage.face_start[f]
	size = cage.face_size[f]
	inside = bool(False)
	best2 = float(1.0e30)
	bj = int(0)
	bt = float(0.0)
	for j in range(size):
		a = cage.coords[start + j]
		b = cage.coords[cage.vnext[start + j]]
		# even-odd crossing test (coincident slit edges cancel)
		if (a[1] > q[1]) != (b[1] > q[1]):
			xi = a[0] + (q[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
			if xi > q[0]:
				inside = not inside
		# distance to the edge; when q projects strictly inside it, from the edge's stored supporting
		# line, which the pieces of a split edge inherit: distances near a split edge are bit-identical
		# before and after the split (exact walk reuse)
		e = b - a
		ee = wp.dot(e, e)
		t = float(0.0)
		if ee > 0.0:
			t = wp.dot(q - a, e) / ee
		dd = float(0.0)
		if t > 0.0 and t < 1.0:
			ln = cage.edge_line[start + j]
			sl = ln[0] * q[0] + ln[1] * q[1] - ln[2]
			dd = sl * sl
		elif t >= 1.0:
			dd = wp.length_sq(q - b)
		else:
			dd = wp.length_sq(q - a)
		if dd < best2:
			best2 = dd
			bj = j
			bt = t
	return inside, best2, bj, bt


@wp.func
def face_dist_only3(cage: Cage3, f: int, p: wp.vec3):
	"""Exact distance from p to planar face f and the signed plane distance (no closest point)."""
	s = wp.dot(cage.normal[f], p) - cage.offset[f]
	q = wp.vec2(wp.dot(cage.fu[f], p), wp.dot(cage.fv[f], p))
	inside, best2, bj, bt = face_scan3(cage, f, q)
	d = float(0.0)
	if inside:
		d = wp.abs(s)
	else:
		d = wp.sqrt(s * s + best2)
	return d, s


@wp.func
def face_dist3(cage: Cage3, f: int, p: wp.vec3):
	"""Exact distance from p to planar face f. Returns (d, closest point, signed plane distance)."""
	n = cage.normal[f]
	o = cage.offset[f]
	u = cage.fu[f]
	v = cage.fv[f]
	s = wp.dot(n, p) - o
	q = wp.vec2(wp.dot(u, p), wp.dot(v, p))
	inside, best2, bj, bt = face_scan3(cage, f, q)
	d = float(0.0)
	cb = q
	if inside:
		d = wp.abs(s)
	else:
		d = wp.sqrt(s * s + best2)
		start = cage.face_start[f]
		if bt > 0.0 and bt < 1.0:
			ln = cage.edge_line[start + bj]
			sl = ln[0] * q[0] + ln[1] * q[1] - ln[2]
			cb = q - sl * wp.vec2(ln[0], ln[1])
		elif bt >= 1.0:
			cb = cage.coords[cage.vnext[start + bj]]
		else:
			cb = cage.coords[start + bj]
	xc = cb[0] * u + cb[1] * v + o * n
	return d, xc, s


@wp.func
def scan3(cage: Cage3, f: int, p: wp.vec3, d1: float, f1: int, d2: float):
	"""Track the smallest (distance, id) and the second smallest distance (order independent)."""
	s0 = wp.dot(cage.normal[f], p) - cage.offset[f]
	if wp.abs(s0) > d2 * 1.000001:
		return d1, f1, d2  # d >= |s0| cannot change the two smallest
	d, s = face_dist_only3(cage, f, p)
	if d < d1 or (d == d1 and f < f1):
		d2 = d1
		d1 = d
		f1 = f
	elif d < d2:
		d2 = d
	return d1, f1, d2


@wp.func
def tie_better(f: int, inner: int, root: int, bf: int, bi: int, broot: int):
	"""Tie order: inner side first, then the lower split-invariant root id, then the lower id."""
	if bf < 0 or inner > bi:
		return True
	if inner < bi:
		return False
	if root != broot:
		return root < broot
	return f < bf


@wp.func
def scan_tie3(cage: Cage3, f: int, p: wp.vec3, dlim: float, bf: int, bi: int):
	"""Among faces with d <= dlim prefer p on the inner side (n.p - o < 0), then the lower root id
	(split invariant, see snapshot.py), then the lower id."""
	s0 = wp.dot(cage.normal[f], p) - cage.offset[f]
	if wp.abs(s0) > dlim * 1.000001:
		return bf, bi
	d, s = face_dist_only3(cage, f, p)
	if d <= dlim:
		inner = int(0)
		if s < 0.0:
			inner = 1
		broot = int(-1)
		if bf >= 0:
			broot = cage.face_root[bf]
		if tie_better(f, inner, cage.face_root[f], bf, bi, broot):
			bf = f
			bi = inner
	return bf, bi


@wp.func
def closest_face3(cage: Cage3, p: wp.vec3, bound: float):
	"""Exact closest face. `bound` is a guess of the distance: any value works (doubling fallback,
	brute force beyond the cage diameter), a tight upper bound makes it fast.
	Faces within the tie window of the minimum are resolved by the inner-side rule, then the lower
	root id, then the lower id.
	Returns (d, face, closest point on that face); face == -1 only if the cage has no alive face."""
	d1 = float(1.0e30)
	d2 = float(1.0e30)
	f1 = int(-1)
	brute = cage.use_bvh == 0
	b = wp.max(bound, 1.0e-6)
	hb = float(0.0)
	while not brute:
		hb = b * (1.0 + 4.0 * TIE) + 2.0 * TIE
		query = wp.bvh_query_aabb(cage.bvh, p - wp.vec3(hb), p + wp.vec3(hb))
		f = int(0)
		while wp.bvh_query_next(query, f):
			if cage.face_alive[f] != 0:
				d1, f1, d2 = scan3(cage, f, p, d1, f1, d2)
		if f1 >= 0 and d1 <= b:
			break
		if b > cage.diam:
			brute = True
		b = 2.0 * b
	if brute:
		for f in range(cage.num_faces):
			if cage.face_alive[f] != 0:
				d1, f1, d2 = scan3(cage, f, p, d1, f1, d2)
	dlim = d1 + tie_tol(d1)
	if f1 >= 0 and d2 <= dlim:
		# tie: second pass over the same candidates
		bf = int(-1)
		bi = int(0)
		if brute:
			for f in range(cage.num_faces):
				if cage.face_alive[f] != 0:
					bf, bi = scan_tie3(cage, f, p, dlim, bf, bi)
		else:
			query = wp.bvh_query_aabb(cage.bvh, p - wp.vec3(hb), p + wp.vec3(hb))
			f = int(0)
			while wp.bvh_query_next(query, f):
				if cage.face_alive[f] != 0:
					bf, bi = scan_tie3(cage, f, p, dlim, bf, bi)
		f1 = bf
	x1 = wp.vec3(0.0)
	if f1 >= 0:
		dx, x1, sx = face_dist3(cage, f1, p)
	return d1, f1, x1


# ---------------------------------------------------------------------------------------------
# boundary values: up to 4 (vertex, weight) pairs; unused slots have id -1 and weight 0
# ---------------------------------------------------------------------------------------------

@wp.func
def cross2(a: wp.vec2, b: wp.vec2):
	return a[0] * b[1] - a[1] * b[0]


@wp.func
def tri_bary(q: wp.vec2, a: wp.vec2, b: wp.vec2, c: wp.vec2):
	area = cross2(b - a, c - a)
	wb = float(0.0)
	wc = float(0.0)
	if area != 0.0:
		wb = cross2(q - a, c - a) / area
		wc = cross2(b - a, q - a) / area
	return wp.vec3(1.0 - wb - wc, wb, wc)


@wp.func
def quad_mvc(q: wp.vec2, p0: wp.vec2, p1: wp.vec2, p2: wp.vec2, p3: wp.vec2):
	"""Mean value coordinates of a convex quad (Floater), robust on edges and vertices."""
	s0 = p0 - q
	s1 = p1 - q
	s2 = p2 - q
	s3 = p3 - q
	r = wp.vec4(wp.length(s0), wp.length(s1), wp.length(s2), wp.length(s3))
	scale = wp.max(wp.max(r[0], r[1]), wp.max(r[2], r[3]))
	tiny = 1.0e-7 * scale
	out = wp.vec4(0.0)
	# at a vertex
	for i in range(4):
		if r[i] <= tiny:
			out[i] = 1.0
			return out
	A = wp.vec4(cross2(s0, s1), cross2(s1, s2), cross2(s2, s3), cross2(s3, s0))
	D = wp.vec4(wp.dot(s0, s1), wp.dot(s1, s2), wp.dot(s2, s3), wp.dot(s3, s0))
	# on an edge: linear interpolation
	for i in range(4):
		j = (i + 1) % 4
		if wp.abs(A[i]) <= 1.0e-7 * r[i] * r[j] and D[i] < 0.0:
			t = r[i] / (r[i] + r[j])
			out[i] = 1.0 - t
			out[j] = t
			return out
	# tan(alpha_i / 2) = A / (r r' + D) = (r r' - D) / A, picking the form without cancellation;
	# near an edge it blows up and the coordinates tend to the edge lerp
	tn = wp.vec4(0.0)
	for i in range(4):
		j = (i + 1) % 4
		rr = r[i] * r[j]
		if D[i] >= 0.0:
			tn[i] = A[i] / (rr + D[i])
		else:
			tn[i] = (rr - D[i]) / A[i]
	w = wp.vec4(0.0)
	tot = float(0.0)
	for i in range(4):
		im = (i + 3) % 4
		w[i] = (tn[im] + tn[i]) / r[i]
		tot += w[i]
	return w / tot


@wp.func
def ngon_closest_edge(cage: Cage3, start: int, size: int, x: wp.vec2):
	"""Closest edge (of all loops) of a face polygon in 2D; ties prefer x on the edge's inner (left) side
	(outer loops CCW, holes CW: the left side is always the face interior)."""
	best_j = int(-1)
	best_d = float(1.0e30)
	best_t = float(0.0)
	best_in = int(0)
	for j in range(size):
		a = cage.coords[start + j]
		b = cage.coords[cage.vnext[start + j]]
		t, dd, c = seg_closest2(x, a, b)
		d = wp.sqrt(dd)
		inner = int(0)
		if cross2(b - a, x - a) > 0.0:
			inner = 1
		take = bool(False)
		if best_j < 0:
			take = True
		elif wp.abs(d - best_d) <= tie_tol(wp.min(d, best_d)):
			if inner > best_in:
				take = True
		elif d < best_d:
			take = True
		if take:
			best_j = j
			best_d = d
			best_t = t
			best_in = inner
	return best_d, best_j, best_t


@wp.func
def boundary_value3(cage: Cage3, f: int, xc: wp.vec3, seed: int, k: int, eps_ngon: float, max_steps_ngon: int):
	start = cage.face_start[f]
	size = cage.face_size[f]
	kind = cage.face_kind[f]
	q = wp.vec2(wp.dot(cage.fu[f], xc), wp.dot(cage.fv[f], xc))
	ids = wp.vec4i(-1, -1, -1, -1)
	w = wp.vec4(0.0)
	if kind == KIND_TRI:
		b = tri_bary(q, cage.coords[start], cage.coords[start + 1], cage.coords[start + 2])
		for j in range(3):
			ids[j] = cage.face_verts[start + j]
			w[j] = b[j]
	elif kind == KIND_QUAD:
		w = quad_mvc(q, cage.coords[start], cage.coords[start + 1], cage.coords[start + 2], cage.coords[start + 3])
		for j in range(4):
			ids[j] = cage.face_verts[start + j]
	else:
		# nested 2D walk on spheres in the face plane; every edge (outer, hole, slit) absorbs
		rng = wp.rand_init(seed ^ NGON_XOR, k)
		x = q
		d, j, t = ngon_closest_edge(cage, start, size, x)
		for step in range(max_steps_ngon):
			if d < eps_ngon:
				break
			th = 2.0 * wp.pi * wp.randf(rng)
			x = x + d * wp.vec2(wp.cos(th), wp.sin(th))
			d, j, t = ngon_closest_edge(cage, start, size, x)
		ids[0] = cage.face_verts[start + j]
		ids[1] = cage.face_verts[cage.vnext[start + j]]
		w[0] = 1.0 - t
		w[1] = t
	return ids, w
