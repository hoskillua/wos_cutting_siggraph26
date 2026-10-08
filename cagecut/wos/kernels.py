"""Warp kernels for the walks: full solve, incremental update after a cut step, point management.

One walk kernel serves the full solve, new / moved points and re-walks (same code path).
Per-walk cache (paper Sec. 4.4), walk k of point i at index c = k * pcap + i (pcap: point capacity):
	land_face[c], land_x[c]   landing face and landing point (-1: the walk did not land)
	gfac[c]                   gradient factor of the walk: (dim / r0) * direction of its first step
	bv[:, c]                  bounding volume of all its spheres (k-DOP or AABB interval per axis)
Per point i, float64 sums over its walks (weights = wsum / nland, Eq. 3):
	wsum[i, v]  sum of the boundary values of vertex v     nland[i]     number of landed walks
	gsum[i, v]  sum of boundary value * gradient factor    gfac_sum[i]  sum of gradient factors
"""

import warp as wp

from cagecut.wos.funcs3d import tie_tol

# no FMA contraction: a boundary value or distance must round identically in every kernel that
# evaluates it (walk add, re-walk / on-face subtract, on-face add), otherwise subtract/add cycles drift
# by float32 ulps and orphaned columns keep residues
NO_FMA = {"fuse_fp": False}
wp.set_module_options(NO_FMA)

_CACHE = {}


# ---------------------------------------------------------------------------------------------
# accumulation helpers (float64 sums); a boundary value has at most 4 vertices (ids < 0 unused)
# ---------------------------------------------------------------------------------------------

@wp.func
def add_value(wsum: wp.array2d(dtype=wp.float64), i: int, ids: wp.vec4i, w: wp.vec4, sign: wp.float64):
	for j in range(4):
		if ids[j] >= 0:
			wp.atomic_add(wsum, i, ids[j], sign * wp.float64(w[j]))


@wp.func
def add_grad3(gsum: wp.array3d(dtype=wp.float64), i: int, ids: wp.vec4i, w: wp.vec4, g: wp.vec3, sign: wp.float64):
	for j in range(4):
		if ids[j] >= 0:
			sw = sign * wp.float64(w[j])
			for a in range(3):
				wp.atomic_add(gsum, i, ids[j], a, sw * wp.float64(g[a]))


@wp.func
def add_grad2(gsum: wp.array3d(dtype=wp.float64), i: int, ids: wp.vec4i, w: wp.vec4, g: wp.vec2, sign: wp.float64):
	for j in range(4):
		if ids[j] >= 0:
			sw = sign * wp.float64(w[j])
			for a in range(2):
				wp.atomic_add(gsum, i, ids[j], a, sw * wp.float64(g[a]))


@wp.func
def add_gfac3(gfac_sum: wp.array2d(dtype=wp.float64), i: int, g: wp.vec3, sign: wp.float64):
	for a in range(3):
		wp.atomic_add(gfac_sum, i, a, sign * wp.float64(g[a]))


@wp.func
def add_gfac2(gfac_sum: wp.array2d(dtype=wp.float64), i: int, g: wp.vec2, sign: wp.float64):
	for a in range(2):
		wp.atomic_add(gfac_sum, i, a, sign * wp.float64(g[a]))


# ---------------------------------------------------------------------------------------------
# outputs, sums management
# ---------------------------------------------------------------------------------------------

@wp.kernel
def weights_kernel(wsum: wp.array2d(dtype=wp.float64), nland: wp.array(dtype=wp.float64),
	W: wp.array2d(dtype=wp.float32)):
	i, v = wp.tid()
	n = nland[i]
	t = wp.float32(0.0)
	if n > wp.float64(0.0):
		t = wp.float32(wsum[i, v] / n)
	W[i, v] = t


@wp.kernel
def gradients_kernel(wsum: wp.array2d(dtype=wp.float64), nland: wp.array(dtype=wp.float64),
	gsum: wp.array3d(dtype=wp.float64), gfac_sum: wp.array2d(dtype=wp.float64), dim: int,
	out: wp.array3d(dtype=wp.float32)):
	# boundary integral gradient (Sec. 6.1) with the weight w_v as control variate:
	# grad w_v = (gsum_v - w_v * gfac_sum) / nland
	i, v = wp.tid()
	n = nland[i]
	for a in range(dim):
		r = wp.float32(0.0)
		if n > wp.float64(0.0):
			t = wsum[i, v] / n
			r = wp.float32((gsum[i, v, a] - t * gfac_sum[i, a]) / n)
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
def zero_rows(ids: wp.array(dtype=wp.int32), wsum: wp.array2d(dtype=wp.float64), nland: wp.array(dtype=wp.float64),
	gsum: wp.array3d(dtype=wp.float64), gfac_sum: wp.array2d(dtype=wp.float64), grads: int):
	t, v = wp.tid()
	i = ids[t]
	wsum[i, v] = wp.float64(0.0)
	if grads != 0:
		for a in range(gsum.shape[2]):
			gsum[i, v, a] = wp.float64(0.0)
	if v == 0:
		nland[i] = wp.float64(0.0)
		if grads != 0:
			for a in range(gfac_sum.shape[1]):
				gfac_sum[i, a] = wp.float64(0.0)


@wp.kernel
def expand_list(ids: wp.array(dtype=wp.int32), pcap: int, n: int, out: wp.array(dtype=wp.int32)):
	# cache indices of all walks of the listed points, walk-major (coalesced)
	tid = wp.tid()
	k = tid // n
	out[tid] = k * pcap + ids[tid % n]


# ---------------------------------------------------------------------------------------------
# factories (fns: the per-dimension Warp functions, gpu_cage.dim_bundle)
# ---------------------------------------------------------------------------------------------

def walk_kernel(fns, bv_mode, gradients):
	key = ("walk", fns.dim, bv_mode, bool(gradients))
	if key in _CACHE:
		return _CACHE[key]
	Cage = fns.Cage
	Vec = fns.Vec
	closest = fns.closest
	bvalue = fns.bvalue
	sample = fns.sample
	proj, AxisVec, axes = fns.bv[bv_mode]
	N_AXES = int(axes.shape[0])
	DIMF = float(fns.dim)
	GRAD = bool(gradients)
	add_grad = add_grad3 if fns.dim == 3 else add_grad2
	add_gfac = add_gfac3 if fns.dim == 3 else add_gfac2

	@wp.kernel(module="unique", enable_backward=False, module_options=NO_FMA)
	def kernel(cage: Cage, old: Cage, points: wp.array(dtype=Vec), seeds: wp.array(dtype=wp.int32),
		dist0: wp.array(dtype=wp.float32), wlist: wp.array(dtype=wp.int32), use_list: int, p0: int, npts: int,
		pcap: int, subtract: int, eps: float, max_steps: int, eps_ngon: float, max_steps_ngon: int,
		land_face: wp.array(dtype=wp.int32), land_x: wp.array(dtype=Vec), gfac: wp.array(dtype=Vec),
		bv: wp.array2d(dtype=wp.float32), wsum: wp.array2d(dtype=wp.float64), nland: wp.array(dtype=wp.float64),
		gsum: wp.array3d(dtype=wp.float64), gfac_sum: wp.array2d(dtype=wp.float64)):
		tid = wp.tid()
		c = int(0)  # cache index of the walk
		if use_list != 0:
			c = wlist[tid]
		else:
			c = (tid // npts) * pcap + p0 + tid % npts
		i = c % pcap  # point
		k = c // pcap  # walk index of the point
		seed = seeds[i]

		# re-walk: remove the old contribution (evaluated on the cage the walk landed on)
		if subtract != 0:
			f_old = land_face[c]
			if f_old >= 0:
				old_ids, old_w = bvalue(old, f_old, land_x[c], seed, k, eps_ngon, max_steps_ngon)
				add_value(wsum, i, old_ids, old_w, wp.float64(-1.0))
				wp.atomic_add(nland, i, wp.float64(-1.0))
				if wp.static(GRAD):
					g_old = gfac[c]
					add_grad(gsum, i, old_ids, old_w, g_old, wp.float64(-1.0))
					add_gfac(gfac_sum, i, g_old, wp.float64(-1.0))

		# the walk, seeded by (point seed, walk index): a re-walk retraces the old walk on the new cage
		# exactly until its first sphere that meets the new cut, and continues from there (Sec. 4.2)
		rng = wp.rand_init(seed, k)
		p = points[i]
		bound = dist0[i]
		lo = AxisVec(1.0e30)
		hi = AxisVec(-1.0e30)
		f_land = int(-1)
		x_land = Vec()
		g = Vec()
		for step in range(max_steps):
			r, f, x_closest = closest(cage, p, bound)
			if f < 0:
				break
			pp = proj(p)
			lo = wp.min(lo, pp - AxisVec(r))
			hi = wp.max(hi, pp + AxisVec(r))
			if r < eps:
				f_land = f
				x_land = x_closest
				break
			direction, rng = sample(rng)
			if step == 0:
				g = (DIMF / r) * direction
			p = p + r * direction
			# the distance is 1-Lipschitz: bound for the next closest-face search (+ float32 slack)
			bound = 2.0 * r * (1.0 + 1.0e-5) + 1.0e-6

		land_face[c] = f_land
		land_x[c] = x_land
		if wp.static(GRAD):
			gfac[c] = g
		for a in range(N_AXES):
			bv[a, c] = lo[a]
			bv[N_AXES + a, c] = hi[a]
		if f_land >= 0:
			ids, w = bvalue(cage, f_land, x_land, seed, k, eps_ngon, max_steps_ngon)
			add_value(wsum, i, ids, w, wp.float64(1.0))
			wp.atomic_add(nland, i, wp.float64(1.0))
			if wp.static(GRAD):
				add_grad(gsum, i, ids, w, g, wp.float64(1.0))
				add_gfac(gfac_sum, i, g, wp.float64(1.0))

	_CACHE[key] = kernel
	return kernel


def check_kernel(fns, bv_mode):
	"""Classify every walk (Sec. 5.2): re-walk if its bounding volume intersects the new cut geometry,
	else on-face update if its landing face was modified. Appends cache indices to compact worklists."""
	key = ("check", fns.dim, bv_mode)
	if key in _CACHE:
		return _CACHE[key]
	Vec = fns.Vec
	proj, AxisVec, axes = fns.bv[bv_mode]
	N_AXES = int(axes.shape[0])
	DIM = int(fns.dim)

	@wp.kernel(module="unique", enable_backward=False, module_options=NO_FMA)
	def kernel(npts: int, pcap: int, land_face: wp.array(dtype=wp.int32), bv: wp.array2d(dtype=wp.float32),
		cut_lo: wp.array2d(dtype=wp.float32), cut_hi: wp.array2d(dtype=wp.float32), cut_n: wp.array(dtype=Vec),
		cut_off: wp.array(dtype=wp.float32), n_cut: int, inflate: float, modified: wp.array(dtype=wp.int32),
		child_start: wp.array(dtype=wp.int32), rewalk_list: wp.array(dtype=wp.int32), counts: wp.array(dtype=wp.int32),
		onface_list: wp.array(dtype=wp.int32), mark_rewalk: wp.array(dtype=wp.int32), mark_onface: wp.array(dtype=wp.int32)):
		tid = wp.tid()
		i = tid % npts
		k = tid // npts
		c = k * pcap + i
		lo = AxisVec()
		hi = AxisVec()
		for a in range(N_AXES):
			lo[a] = bv[a, c]
			hi[a] = bv[N_AXES + a, c]
		hit = bool(False)
		for s in range(n_cut):
			# separating axis tests: the k-DOP axes, then the cut simplex's plane (line in 2D)
			sep = bool(False)
			for a in range(N_AXES):
				if hi[a] < cut_lo[s, a] - inflate or lo[a] > cut_hi[s, a] + inflate:
					sep = True
			if not sep:
				# plane against the walk's box (its first DIM axes)
				nr = cut_n[s]
				cen = float(0.0)
				rad = float(0.0)
				for a in range(DIM):
					cen += nr[a] * 0.5 * (lo[a] + hi[a])
					rad += wp.abs(nr[a]) * 0.5 * (hi[a] - lo[a])
				if cen - rad > cut_off[s] + inflate or cen + rad < cut_off[s] - inflate:
					sep = True
			if not sep:
				hit = True
				break
		f = land_face[c]
		on = bool(False)
		if not hit and f >= 0:
			if modified[f] != 0:
				if child_start[f + 1] > child_start[f]:
					on = True
				else:
					hit = True  # face vanished without a successor: walk again
		if hit:
			idx = wp.atomic_add(counts, 0, 1)
			rewalk_list[idx] = c
			wp.atomic_add(mark_rewalk, i, 1)
		elif on:
			idx = wp.atomic_add(counts, 1, 1)
			onface_list[idx] = c
			wp.atomic_add(mark_onface, i, 1)

	_CACHE[key] = kernel
	return kernel


def onface_kernel(fns, gradients):
	"""On-face update (Sec. 4.3, 5.2) of walks whose landing face was modified: subtract the boundary value
	on the old face, find the new face containing the landing point, add the boundary value there."""
	key = ("onface", fns.dim, bool(gradients))
	if key in _CACHE:
		return _CACHE[key]
	Cage = fns.Cage
	Vec = fns.Vec
	bvalue = fns.bvalue
	face_dist = fns.face_dist
	GRAD = bool(gradients)
	add_grad = add_grad3 if fns.dim == 3 else add_grad2

	@wp.kernel(module="unique", enable_backward=False, module_options=NO_FMA)
	def kernel(cage_old: Cage, cage_new: Cage, onface_list: wp.array(dtype=wp.int32), seeds: wp.array(dtype=wp.int32),
		pcap: int, eps_ngon: float, max_steps_ngon: int, child_start: wp.array(dtype=wp.int32),
		child_list: wp.array(dtype=wp.int32), land_face: wp.array(dtype=wp.int32), land_x: wp.array(dtype=Vec),
		gfac: wp.array(dtype=Vec), wsum: wp.array2d(dtype=wp.float64), gsum: wp.array3d(dtype=wp.float64)):
		tid = wp.tid()
		c = onface_list[tid]
		i = c % pcap
		k = c // pcap
		seed = seeds[i]
		f_old = land_face[c]
		x = land_x[c]
		old_ids, old_w = bvalue(cage_old, f_old, x, seed, k, eps_ngon, max_steps_ngon)
		add_value(wsum, i, old_ids, old_w, wp.float64(-1.0))
		# new face containing x among the old face's descendants: the nearest one; among those within the
		# tie window of the nearest, the lowest id (children share their root and plane: closest-face tie rule)
		bd = float(1.0e30)
		for j in range(child_start[f_old], child_start[f_old + 1]):
			dd, cp, sd = face_dist(cage_new, child_list[j], x)
			bd = wp.min(bd, dd)
		best = int(-1)
		dlim = bd + tie_tol(bd)
		for j in range(child_start[f_old], child_start[f_old + 1]):
			gc = child_list[j]
			dd, cp, sd = face_dist(cage_new, gc, x)
			if best < 0 and dd <= dlim:
				best = gc
		ids, w = bvalue(cage_new, best, x, seed, k, eps_ngon, max_steps_ngon)
		add_value(wsum, i, ids, w, wp.float64(1.0))
		if wp.static(GRAD):
			g = gfac[c]
			add_grad(gsum, i, old_ids, old_w, g, wp.float64(-1.0))
			add_grad(gsum, i, ids, w, g, wp.float64(1.0))
		land_face[c] = best

	_CACHE[key] = kernel
	return kernel


def relayout_kernel(fns, bv_mode):
	"""Copy the walk cache from point capacity pcap_old to pcap_new."""
	key = ("relayout", fns.dim, bv_mode)
	if key in _CACHE:
		return _CACHE[key]
	Vec = fns.Vec
	NB = 2 * int(fns.bv[bv_mode][2].shape[0])

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


def bound_kernel(fns):
	"""Per point distance bound for the first walk step (exact distance, slightly inflated)."""
	key = ("bound", fns.dim)
	if key in _CACHE:
		return _CACHE[key]
	Cage = fns.Cage
	Vec = fns.Vec
	closest = fns.closest

	@wp.kernel(module="unique", enable_backward=False, module_options=NO_FMA)
	def kernel(cage: Cage, points: wp.array(dtype=Vec), ids: wp.array(dtype=wp.int32), use_list: int, p0: int,
		guess: float, dist0: wp.array(dtype=wp.float32)):
		tid = wp.tid()
		i = p0 + tid
		if use_list != 0:
			i = ids[tid]
		d, f, x = closest(cage, points[i], guess)
		dist0[i] = d * (1.0 + 1.0e-5) + 1.0e-6

	_CACHE[key] = kernel
	return kernel
