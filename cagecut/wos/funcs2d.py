"""Warp functions for 2D walk on spheres: segment distances, boundary values, 8-DOPs."""

import math

import numpy as np
import warp as wp

from cagecut.wos.funcs3d import TIE, seg_closest2, tie_better, tie_tol

_S2 = 1.0 / math.sqrt(2.0)

# 8-DOP axes: x, y and the two diagonals. Must match kdop8_proj.
KDOP8_AXES = np.array([[1, 0], [0, 1], [_S2, _S2], [_S2, -_S2]], dtype=np.float64)
AABB2_AXES = np.eye(2)

S2 = wp.constant(wp.float32(_S2))


@wp.struct
class Cage2:
	face_start: wp.array(dtype=wp.int32)
	face_size: wp.array(dtype=wp.int32)
	face_kind: wp.array(dtype=wp.int32)
	face_alive: wp.array(dtype=wp.int32)
	face_root: wp.array(dtype=wp.int32)
	face_verts: wp.array(dtype=wp.int32)
	verts: wp.array(dtype=wp.vec2)
	normal: wp.array(dtype=wp.vec2)
	offset: wp.array(dtype=wp.float32)
	bvh: wp.uint64
	num_faces: wp.int32
	diam: wp.float32
	use_bvh: wp.int32


@wp.func
def kdop8_proj(p: wp.vec2):
	return wp.vec4(p[0], p[1], (p[0] + p[1]) * S2, (p[0] - p[1]) * S2)


@wp.func
def aabb2_proj(p: wp.vec2):
	return p


@wp.func
def sample_dir2(rng: wp.uint32):
	"""Uniform direction; returns the advanced rng state too (func args are copies)."""
	th = 2.0 * wp.pi * wp.randf(rng)
	return wp.vec2(wp.cos(th), wp.sin(th)), rng


@wp.func
def face_dist2(cage: Cage2, f: int, p: wp.vec2):
	"""Exact distance from p to segment f. Returns (d, closest point, signed line distance).

	When p projects inside the segment the distance comes from the stored line (normal, offset),
	so it does not depend on the endpoint positions: a split or a growing slit segment inherits
	its line and gives bit-identical distances (exact walk reuse, as the 3D planar model)."""
	s0 = cage.face_start[f]
	a = cage.verts[cage.face_verts[s0]]
	b = cage.verts[cage.face_verts[s0 + 1]]
	n = cage.normal[f]
	s = wp.dot(n, p) - cage.offset[f]
	e = b - a
	ee = wp.dot(e, e)
	t = float(0.0)
	if ee > 0.0:
		t = wp.dot(p - a, e) / ee
	d = float(0.0)
	c = a
	if t > 0.0 and t < 1.0:
		d = wp.abs(s)
		c = p - s * n
	else:
		if t >= 1.0:
			c = b
		d = wp.length(p - c)
	return d, c, s


@wp.func
def scan2(cage: Cage2, f: int, p: wp.vec2, d1: float, f1: int, x1: wp.vec2, d2: float):
	"""Track the smallest (distance, id) and the second smallest distance (order independent)."""
	d, xc, s = face_dist2(cage, f, p)
	if d < d1 or (d == d1 and f < f1):
		d2 = d1
		d1 = d
		f1 = f
		x1 = xc
	elif d < d2:
		d2 = d
	return d1, f1, x1, d2


@wp.func
def scan_tie2(cage: Cage2, f: int, p: wp.vec2, dlim: float, bf: int, bi: int, bx: wp.vec2):
	"""Among segments with d <= dlim prefer p on the inner side (n.p - o < 0), then the lower root id,
	then the lower id."""
	d, xc, s = face_dist2(cage, f, p)
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
			bx = xc
	return bf, bi, bx


@wp.func
def _box_lo(p: wp.vec2, h: float):
	return wp.vec3(p[0] - h, p[1] - h, 0.0)


@wp.func
def _box_hi(p: wp.vec2, h: float):
	return wp.vec3(p[0] + h, p[1] + h, 0.0)


@wp.func
def closest_face2(cage: Cage2, p: wp.vec2, bound: float):
	"""Exact closest segment, same contract as closest_face3."""
	d1 = float(1.0e30)
	d2 = float(1.0e30)
	f1 = int(-1)
	x1 = wp.vec2(0.0)
	brute = cage.use_bvh == 0
	b = wp.max(bound, 1.0e-6)
	hb = float(0.0)
	while not brute:
		hb = b * (1.0 + 4.0 * TIE) + 2.0 * TIE
		query = wp.bvh_query_aabb(cage.bvh, _box_lo(p, hb), _box_hi(p, hb))
		f = int(0)
		while wp.bvh_query_next(query, f):
			if cage.face_alive[f] != 0:
				d1, f1, x1, d2 = scan2(cage, f, p, d1, f1, x1, d2)
		if f1 >= 0 and d1 <= b:
			break
		if b > cage.diam:
			brute = True
		b = 2.0 * b
	if brute:
		for f in range(cage.num_faces):
			if cage.face_alive[f] != 0:
				d1, f1, x1, d2 = scan2(cage, f, p, d1, f1, x1, d2)
	dlim = d1 + tie_tol(d1)
	if f1 >= 0 and d2 <= dlim:
		bf = int(-1)
		bi = int(0)
		bx = x1
		if brute:
			for f in range(cage.num_faces):
				if cage.face_alive[f] != 0:
					bf, bi, bx = scan_tie2(cage, f, p, dlim, bf, bi, bx)
		else:
			query = wp.bvh_query_aabb(cage.bvh, _box_lo(p, hb), _box_hi(p, hb))
			f = int(0)
			while wp.bvh_query_next(query, f):
				if cage.face_alive[f] != 0:
					bf, bi, bx = scan_tie2(cage, f, p, dlim, bf, bi, bx)
		f1 = bf
		x1 = bx
	return d1, f1, x1


@wp.func
def boundary_value2(cage: Cage2, f: int, xc: wp.vec2, seed: int, k: int, eps_ngon: float, max_steps_ngon: int):
	"""Linear interpolation along segment f."""
	s0 = cage.face_start[f]
	ia = cage.face_verts[s0]
	ib = cage.face_verts[s0 + 1]
	t, dd, c = seg_closest2(xc, cage.verts[ia], cage.verts[ib])
	return wp.vec4i(ia, ib, -1, -1), wp.vec4(1.0 - t, t, 0.0, 0.0)
