"""Incremental update of a WalkSet after one cut step (paper Sec. 4.2).

1. check: every walk whose bounding volume touches a swept simplex goes to REWALK, otherwise a
   walk landing on a modified face goes to ONFACE (compact worklists, one kernel).
2. ONFACE: re-evaluate the boundary value on the descendant face containing the landing point (for a
   face merged away this step: a descendant of the face it was merged into).
3. REWALK: subtract the old contribution and replay the walk with the same seed on the new cage.
"""

import time

import numpy as np
import warp as wp

from cagecut.wos import kernels as K
from cagecut.wos.walkset import UpdateStats, _sync

# must exceed the closest-face tie window (TIE_REL for d < 1) plus float32 rounding: a walk whose last
# ball misses new geometry by less than the tie window could still switch its landing face
SWEPT_INFLATE = 4.0e-6


def swept_bounds(swept, dim, axes):
	"""k-DOP intervals, plane normal and offset of every swept simplex (float64)."""
	s = np.asarray(swept, dtype=np.float64).reshape(-1, dim, dim)
	n = s.shape[0]
	if n == 0:
		return np.zeros((0, axes.shape[0])), np.zeros((0, axes.shape[0])), np.zeros((0, dim)), np.zeros(0)
	proj = s @ axes.T  # (n, dim, NA)
	lo = proj.min(axis=1)
	hi = proj.max(axis=1)
	if dim == 3:
		nr = np.cross(s[:, 1] - s[:, 0], s[:, 2] - s[:, 0])
	else:
		e = s[:, 1] - s[:, 0]
		nr = np.stack([e[:, 1], -e[:, 0]], axis=1)
	ln = np.linalg.norm(nr, axis=1)
	ok = ln > 1e-30
	nr[ok] /= ln[ok, None]
	nr[~ok] = 0.0  # degenerate simplex: no plane test (conservative)
	off = np.einsum("ij,ij->i", nr, s[:, 0])
	return lo, hi, nr, off


def merge_targets(face_merged_into, n_old):
	"""(n_old,) per old face f the old face whose alive descendants receive f's walks on relocation:
	face_merged_into[f] if f was merged away this step (chains followed), else f itself."""
	tgt = np.arange(n_old, dtype=np.int64)
	if face_merged_into is None:
		return tgt
	m = np.asarray(face_merged_into, dtype=np.int64).reshape(-1)[:n_old]
	sel = np.nonzero(m >= 0)[0]
	tgt[sel] = m[sel]
	for _ in range(n_old):
		nxt = tgt[tgt]
		if np.array_equal(nxt, tgt):
			break
		tgt = nxt
	return tgt


def ancestor_children(face_ancestor, new_alive, n_old, face_merged_into=None):
	"""CSR over OLD face ids f of the alive new faces g a walk landing on f may be relocated to:
	face_ancestor[g] == f', f' = face_merged_into[f] if f was merged away else f. Children sorted by id."""
	anc = np.asarray(face_ancestor, dtype=np.int64)
	alive = np.asarray(new_alive, dtype=bool)[:anc.shape[0]]
	g = np.nonzero((anc >= 0) & (anc < n_old) & alive)[0]
	a = anc[g]
	order = np.lexsort((g, a))
	g = g[order]
	a = a[order]
	counts = np.bincount(a, minlength=n_old)
	start = np.zeros(n_old + 1, dtype=np.int64)
	start[1:] = np.cumsum(counts)
	tgt = merge_targets(face_merged_into, n_old)
	if not np.array_equal(tgt, np.arange(n_old)):
		# gather the children of every face's relocation target
		cnt = start[tgt + 1] - start[tgt]
		s2 = np.zeros(n_old + 1, dtype=np.int64)
		s2[1:] = np.cumsum(cnt)
		idx = np.repeat(start[tgt], cnt) + (np.arange(int(s2[-1])) - np.repeat(s2[:-1], cnt))
		g = g[idx]
		start = s2
	return start.astype(np.int32), g.astype(np.int32)


def incremental_update(ws, old, new, delta) -> UpdateStats:
	dev = ws.device
	P, Kw = ws.num_points, ws.K
	n_walks = P * Kw
	stats = UpdateStats(n_walks=n_walks)
	zeros = np.zeros(P, np.int32)
	if delta.new is delta.old or P == 0 or delta.empty:
		ws.last_marks = {"rewalk": zeros, "onface": zeros.copy()}
		return stats
	assert (old.snapshot is delta.old or old.snapshot.version == delta.old.version) and 		(new.snapshot is delta.new or new.snapshot.version == delta.new.version), "GpuCages do not match the delta snapshots"
	_sync(dev)
	t0 = time.perf_counter()
	ws._ensure_verts(new.snapshot.num_verts)

	# host-side preparation: swept simplices, modified flags, ancestor CSR
	axes = ws.B.bv[ws.bv_mode][2]
	lo, hi, nr, off = swept_bounds(delta.swept, ws.dim, axes)
	ns = lo.shape[0]
	n_old = delta.old.num_faces
	modified = np.zeros(max(n_old, 1), dtype=np.int32)
	mf = np.asarray(delta.modified_faces, dtype=np.int64).reshape(-1)
	modified[mf[(mf >= 0) & (mf < n_old)]] = 1
	merged = getattr(delta, "face_merged_into", None)
	if merged is not None:
		mi = np.asarray(merged, dtype=np.int64).reshape(-1)[:n_old]
		modified[:mi.shape[0]][mi >= 0] = 1  # merged-away faces are dead: their walks relocate
	cstart, clist = ancestor_children(delta.face_ancestor, delta.new.face_alive, n_old, merged)

	def up(a, dtype, shape_min=1):
		a = np.ascontiguousarray(a)
		if a.shape[0] == 0:
			a = np.zeros((shape_min,) + a.shape[1:], dtype=a.dtype)
		return wp.array(a, dtype=dtype, device=dev)
	slo = up(lo.astype(np.float32).reshape(-1, axes.shape[0]), wp.float32)
	shi = up(hi.astype(np.float32).reshape(-1, axes.shape[0]), wp.float32)
	snrm = up(nr.astype(np.float32), ws.B.Vec)
	soff = up(off.astype(np.float32), wp.float32)
	mod_wp = up(modified, wp.int32)
	cs_wp = up(cstart, wp.int32)
	cl_wp = up(clist, wp.int32)

	counts = wp.zeros(2, dtype=wp.int32, device=dev)
	rew, onl = ws._worklists(n_walks)
	ws.mark_r.zero_()
	ws.mark_o.zero_()
	wp.launch(K.check_kernel(ws.B, ws.bv_mode), dim=n_walks, inputs=[P, ws.pcap, ws.face, ws.bv, slo, shi, snrm, soff,
		ns, SWEPT_INFLATE, mod_wp, cs_wp, rew, counts, onl, ws.mark_r, ws.mark_o], device=dev)
	c = counts.numpy()  # synchronizes
	n_rew, n_on = int(c[0]), int(c[1])
	t1 = time.perf_counter()

	p = ws.params
	if n_on > 0:
		wp.launch(K.onface_kernel(ws.B, ws.grads), dim=n_on, inputs=[old.data, new.data, onl, ws.seeds_wp, ws.pcap,
			float(p.eps_ngon), int(p.max_steps_ngon), cs_wp, cl_wp, ws.face, ws.xcs, ws.gfs, ws.S, ws.G], device=dev)
	_sync(dev)
	t2 = time.perf_counter()
	if n_rew > 0:
		ws._launch_walks(new, old, n_rew, wlist=rew, subtract=True)
	_sync(dev)
	t3 = time.perf_counter()

	ws.last_marks = {"rewalk": ws.mark_r.numpy()[:P].copy(), "onface": ws.mark_o.numpy()[:P].copy()}
	stats.n_rewalk = n_rew
	stats.n_onface = n_on
	stats.t_check_ms = (t1 - t0) * 1e3
	stats.t_onface_ms = (t2 - t1) * 1e3
	stats.t_rewalk_ms = (t3 - t2) * 1e3
	stats.t_total_ms = (time.perf_counter() - t0) * 1e3
	return stats
