"""Kernel factories shared by the full solve, incremental update and point management.

One walk kernel serves the full solve, appended/reset points and REWALK (identical code path).
Cache layout: walk k of point i lives at c = k * pcap + i (coalesced over points, and adjacent
threads accumulate into different rows of the sums).
"""

import warp as wp

from cagecut.wos.funcs3d import tie_tol

# no FMA contraction: a boundary value or distance must round identically in every kernel that
# evaluates it (walk add, REWALK / ONFACE subtract, ONFACE add), otherwise subtract/add cycles drift
# by float32 ulps and orphaned columns keep residues
NO_FMA = {"fuse_fp": False}
wp.set_module_options(NO_FMA)

_CACHE = {}


# ---------------------------------------------------------------------------------------------
# accumulation helpers (float64 sums)
# ---------------------------------------------------------------------------------------------

@wp.func
def add_value(S: wp.array2d(dtype=wp.float64), i: int, ids: wp.vec4i, w: wp.vec4, sign: wp.float64):
	for j in range(4):
		if ids[j] >= 0:
			wp.atomic_add(S, i, ids[j], sign * wp.float64(w[j]))


@wp.func
def add_grad3(G: wp.array3d(dtype=wp.float64), i: int, ids: wp.vec4i, w: wp.vec4, g: wp.vec3, sign: wp.float64):
	for j in range(4):
		if ids[j] >= 0:
			sw = sign * wp.float64(w[j])
			for a in range(3):
				wp.atomic_add(G, i, ids[j], a, sw * wp.float64(g[a]))


@wp.func
def add_grad2(G: wp.array3d(dtype=wp.float64), i: int, ids: wp.vec4i, w: wp.vec4, g: wp.vec2, sign: wp.float64):
	for j in range(4):
		if ids[j] >= 0:
			sw = sign * wp.float64(w[j])
			for a in range(2):
				wp.atomic_add(G, i, ids[j], a, sw * wp.float64(g[a]))


@wp.func
def add_gm3(Gm: wp.array2d(dtype=wp.float64), i: int, g: wp.vec3, sign: wp.float64):
	for a in range(3):
		wp.atomic_add(Gm, i, a, sign * wp.float64(g[a]))


@wp.func
def add_gm2(Gm: wp.array2d(dtype=wp.float64), i: int, g: wp.vec2, sign: wp.float64):
	for a in range(2):
		wp.atomic_add(Gm, i, a, sign * wp.float64(g[a]))


# ---------------------------------------------------------------------------------------------
# outputs, sums management
# ---------------------------------------------------------------------------------------------

@wp.kernel
def weights_kernel(S: wp.array2d(dtype=wp.float64), m: wp.array(dtype=wp.float64), T: wp.array2d(dtype=wp.float32)):
	i, v = wp.tid()
	mm = m[i]
	t = wp.float32(0.0)
	if mm > wp.float64(0.0):
		t = wp.float32(S[i, v] / mm)
	T[i, v] = t


@wp.kernel
def gradients_kernel(S: wp.array2d(dtype=wp.float64), m: wp.array(dtype=wp.float64), G: wp.array3d(dtype=wp.float64),
	Gm: wp.array2d(dtype=wp.float64), dim: int, out: wp.array3d(dtype=wp.float32)):
	i, v = wp.tid()
	mm = m[i]
	for a in range(dim):
		r = wp.float32(0.0)
		if mm > wp.float64(0.0):
			t = S[i, v] / mm
			r = wp.float32((G[i, v, a] - t * Gm[i, a]) / mm)
		out[i, v, a] = r


@wp.kernel
def copy2d_f64(src: wp.array2d(dtype=wp.float64), dst: wp.array2d(dtype=wp.float64)):
	i, j = wp.tid()
	dst[i, j] = src[i, j]


@wp.kernel
def copy3d_f64(src: wp.array3d(dtype=wp.float64), dst: wp.array3d(dtype=wp.float64)):
	i, j, a = wp.tid()
	dst[i, j, a] = src[i, j, a]


@wp.kernel
def zero_rows(ids: wp.array(dtype=wp.int32), S: wp.array2d(dtype=wp.float64), m: wp.array(dtype=wp.float64),
	G: wp.array3d(dtype=wp.float64), Gm: wp.array2d(dtype=wp.float64), grads: int):
	t, v = wp.tid()
	i = ids[t]
	S[i, v] = wp.float64(0.0)
	if grads != 0:
		for a in range(G.shape[2]):
			G[i, v, a] = wp.float64(0.0)
	if v == 0:
		m[i] = wp.float64(0.0)
		if grads != 0:
			for a in range(Gm.shape[1]):
				Gm[i, a] = wp.float64(0.0)


@wp.kernel
def expand_list(ids: wp.array(dtype=wp.int32), pcap: int, n: int, out: wp.array(dtype=wp.int32)):
	# all walks of the listed points: entry t*K... ordered walk-major for coalescing
	tid = wp.tid()
	k = tid // n
	out[tid] = k * pcap + ids[tid % n]


# ---------------------------------------------------------------------------------------------
# factories
# ---------------------------------------------------------------------------------------------

def walk_kernel(B, bv_mode, gradients):
	key = ("walk", B.dim, bv_mode, bool(gradients))
	if key in _CACHE:
		return _CACHE[key]
	Cage = B.Cage
	Vec = B.Vec
	closest = B.closest
	bvalue = B.bvalue
	sample = B.sample
	proj, VA, axes = B.bv[bv_mode]
	NA = int(axes.shape[0])
	DIMF = float(B.dim)
	GRAD = bool(gradients)
	add_grad = add_grad3 if B.dim == 3 else add_grad2
	add_gm = add_gm3 if B.dim == 3 else add_gm2

	@wp.kernel(module="unique", enable_backward=False, module_options=NO_FMA)
	def kernel(cage: Cage, old: Cage, points: wp.array(dtype=Vec), seeds: wp.array(dtype=wp.int32),
		d0: wp.array(dtype=wp.float32), wlist: wp.array(dtype=wp.int32), use_list: int, p0: int, npts: int,
		pcap: int, subtract: int, eps: float, max_steps: int, eps_ngon: float, max_steps_ngon: int,
		face: wp.array(dtype=wp.int32), xcs: wp.array(dtype=Vec), gfs: wp.array(dtype=Vec),
		bv: wp.array2d(dtype=wp.float32), S: wp.array2d(dtype=wp.float64), m: wp.array(dtype=wp.float64),
		G: wp.array3d(dtype=wp.float64), Gm: wp.array2d(dtype=wp.float64)):
		tid = wp.tid()
		c = int(0)
		if use_list != 0:
			c = wlist[tid]
		else:
			c = (tid // npts) * pcap + p0 + tid % npts
		i = c % pcap
		k = c // pcap
		seed = seeds[i]

		# remove the old contribution (evaluated on the cage the walk was computed on)
		if subtract != 0:
			fo = face[c]
			if fo >= 0:
				oids, ow = bvalue(old, fo, xcs[c], seed, k, eps_ngon, max_steps_ngon)
				add_value(S, i, oids, ow, wp.float64(-1.0))
				wp.atomic_add(m, i, wp.float64(-1.0))
				if wp.static(GRAD):
					go = gfs[c]
					add_grad(G, i, oids, ow, go, wp.float64(-1.0))
					add_gm(Gm, i, go, wp.float64(-1.0))

		# the walk
		rng = wp.rand_init(seed, k)
		p = points[i]
		bound = d0[i]
		lo = VA(1.0e30)
		hi = VA(-1.0e30)
		f_land = int(-1)
		xl = Vec()
		g = Vec()
		for step in range(max_steps):
			d, f, xcl = closest(cage, p, bound)
			if f < 0:
				break
			pp = proj(p)
			lo = wp.min(lo, pp - VA(d))
			hi = wp.max(hi, pp + VA(d))
			if d < eps:
				f_land = f
				xl = xcl
				break
			dr, rng = sample(rng)
			if step == 0:
				g = (DIMF / d) * dr
			p = p + d * dr
			# 1-lipschitz bound for the next query (relative + absolute float32 slack)
			bound = 2.0 * d * (1.0 + 1.0e-5) + 1.0e-6

		face[c] = f_land
		xcs[c] = xl
		if wp.static(GRAD):
			gfs[c] = g
		for a in range(NA):
			bv[a, c] = lo[a]
			bv[NA + a, c] = hi[a]
		if f_land >= 0:
			ids, w = bvalue(cage, f_land, xl, seed, k, eps_ngon, max_steps_ngon)
			add_value(S, i, ids, w, wp.float64(1.0))
			wp.atomic_add(m, i, wp.float64(1.0))
			if wp.static(GRAD):
				add_grad(G, i, ids, w, g, wp.float64(1.0))
				add_gm(Gm, i, g, wp.float64(1.0))

	_CACHE[key] = kernel
	return kernel


def check_kernel(B, bv_mode):
	"""Classify every walk: REWALK if its bounding volume touches a swept simplex, else ONFACE if
	its landing face was modified. Appends cache indices to compact worklists."""
	key = ("check", B.dim, bv_mode)
	if key in _CACHE:
		return _CACHE[key]
	Vec = B.Vec
	proj, VA, axes = B.bv[bv_mode]
	NA = int(axes.shape[0])
	DIM = int(B.dim)

	@wp.kernel(module="unique", enable_backward=False, module_options=NO_FMA)
	def kernel(npts: int, pcap: int, face: wp.array(dtype=wp.int32), bv: wp.array2d(dtype=wp.float32),
		slo: wp.array2d(dtype=wp.float32), shi: wp.array2d(dtype=wp.float32), snrm: wp.array(dtype=Vec),
		soff: wp.array(dtype=wp.float32), ns: int, infl: float, modified: wp.array(dtype=wp.int32),
		child_start: wp.array(dtype=wp.int32), rew_list: wp.array(dtype=wp.int32), counts: wp.array(dtype=wp.int32),
		on_list: wp.array(dtype=wp.int32), mark_rew: wp.array(dtype=wp.int32), mark_on: wp.array(dtype=wp.int32)):
		tid = wp.tid()
		i = tid % npts
		k = tid // npts
		c = k * pcap + i
		lo = VA()
		hi = VA()
		for a in range(NA):
			lo[a] = bv[a, c]
			hi[a] = bv[NA + a, c]
		hit = bool(False)
		for s in range(ns):
			sep = bool(False)
			for a in range(NA):
				if hi[a] < slo[s, a] - infl or lo[a] > shi[s, a] + infl:
					sep = True
			if not sep:
				# simplex plane (line in 2D) against the walk AABB (first DIM axes)
				nr = snrm[s]
				cen = float(0.0)
				rad = float(0.0)
				for a in range(DIM):
					cen += nr[a] * 0.5 * (lo[a] + hi[a])
					rad += wp.abs(nr[a]) * 0.5 * (hi[a] - lo[a])
				if cen - rad > soff[s] + infl or cen + rad < soff[s] - infl:
					sep = True
			if not sep:
				hit = True
				break
		f = face[c]
		on = bool(False)
		if not hit and f >= 0:
			if modified[f] != 0:
				if child_start[f + 1] > child_start[f]:
					on = True
				else:
					hit = True  # face vanished without a successor: walk again
		if hit:
			idx = wp.atomic_add(counts, 0, 1)
			rew_list[idx] = c
			wp.atomic_add(mark_rew, i, 1)
		elif on:
			idx = wp.atomic_add(counts, 1, 1)
			on_list[idx] = c
			wp.atomic_add(mark_on, i, 1)

	_CACHE[key] = kernel
	return kernel


def onface_kernel(B, gradients):
	"""Re-evaluate the boundary value of walks whose landing face changed: subtract on the old
	face, relocate to the descendant face containing the landing point, add on it."""
	key = ("onface", B.dim, bool(gradients))
	if key in _CACHE:
		return _CACHE[key]
	Cage = B.Cage
	Vec = B.Vec
	bvalue = B.bvalue
	face_dist = B.face_dist
	GRAD = bool(gradients)
	add_grad = add_grad3 if B.dim == 3 else add_grad2

	@wp.kernel(module="unique", enable_backward=False, module_options=NO_FMA)
	def kernel(cage_old: Cage, cage_new: Cage, olist: wp.array(dtype=wp.int32), seeds: wp.array(dtype=wp.int32), pcap: int,
		eps_ngon: float, max_steps_ngon: int, child_start: wp.array(dtype=wp.int32), child_list: wp.array(dtype=wp.int32),
		face: wp.array(dtype=wp.int32), xcs: wp.array(dtype=Vec), gfs: wp.array(dtype=Vec),
		S: wp.array2d(dtype=wp.float64), G: wp.array3d(dtype=wp.float64)):
		tid = wp.tid()
		c = olist[tid]
		i = c % pcap
		k = c // pcap
		seed = seeds[i]
		fo = face[c]
		x = xcs[c]
		oids, ow = bvalue(cage_old, fo, x, seed, k, eps_ngon, max_steps_ngon)
		add_value(S, i, oids, ow, wp.float64(-1.0))
		# descendant containing x: the nearest one; among those within the tie window of the nearest,
		# the lowest id (children share their root and plane, so this is the closest-face tie rule)
		bd = float(1.0e30)
		for j in range(child_start[fo], child_start[fo + 1]):
			dd, cp, sd = face_dist(cage_new, child_list[j], x)
			bd = wp.min(bd, dd)
		best = int(-1)
		dlim = bd + tie_tol(bd)
		for j in range(child_start[fo], child_start[fo + 1]):
			gc = child_list[j]
			dd, cp, sd = face_dist(cage_new, gc, x)
			if best < 0 and dd <= dlim:
				best = gc
		ids, w = bvalue(cage_new, best, x, seed, k, eps_ngon, max_steps_ngon)
		add_value(S, i, ids, w, wp.float64(1.0))
		if wp.static(GRAD):
			g = gfs[c]
			add_grad(G, i, oids, ow, g, wp.float64(-1.0))
			add_grad(G, i, ids, w, g, wp.float64(1.0))
		face[c] = best

	_CACHE[key] = kernel
	return kernel


def relayout_kernel(B, bv_mode):
	"""Copy the walk cache from point capacity pcap_old to pcap_new."""
	key = ("relayout", B.dim, bv_mode)
	if key in _CACHE:
		return _CACHE[key]
	Vec = B.Vec
	NB = 2 * int(B.bv[bv_mode][2].shape[0])

	@wp.kernel(module="unique", enable_backward=False, module_options=NO_FMA)
	def kernel(npts: int, pold: int, pnew: int, face0: wp.array(dtype=wp.int32), x0: wp.array(dtype=Vec),
		g0: wp.array(dtype=Vec), bv0: wp.array2d(dtype=wp.float32), face1: wp.array(dtype=wp.int32),
		x1: wp.array(dtype=Vec), g1: wp.array(dtype=Vec), bv1: wp.array2d(dtype=wp.float32), grads: int):
		tid = wp.tid()
		i = tid % npts
		k = tid // npts
		a0 = k * pold + i
		a1 = k * pnew + i
		face1[a1] = face0[a0]
		x1[a1] = x0[a0]
		if grads != 0:
			g1[a1] = g0[a0]
		for a in range(NB):
			bv1[a, a1] = bv0[a, a0]

	_CACHE[key] = kernel
	return kernel


def bound_kernel(B):
	"""Per point initial distance bound for the first walk step (exact distance, slightly inflated)."""
	key = ("bound", B.dim)
	if key in _CACHE:
		return _CACHE[key]
	Cage = B.Cage
	Vec = B.Vec
	closest = B.closest

	@wp.kernel(module="unique", enable_backward=False, module_options=NO_FMA)
	def kernel(cage: Cage, points: wp.array(dtype=Vec), ids: wp.array(dtype=wp.int32), use_list: int, p0: int,
		guess: float, d0: wp.array(dtype=wp.float32)):
		tid = wp.tid()
		i = p0 + tid
		if use_list != 0:
			i = ids[tid]
		d, f, x = closest(cage, points[i], guess)
		d0[i] = d * (1.0 + 1.0e-5) + 1.0e-6

	_CACHE[key] = kernel
	return kernel
