"""Incremental update of a WalkSet after one cut step (paper Sec. 4.2-4.3, 5.2).

1. check: a walk whose bounding volume intersects the new cut geometry is re-walked; otherwise a walk
   landing on a modified face gets an on-face update (both collected in compact worklists, one kernel).
2. on-face update: re-evaluate the boundary value on the new face containing the landing point (for a
   face merged away this step: a descendant of the face it was merged into).
3. re-walk: subtract the old contribution and replay the walk with the same seed on the new cage.
"""

import time

import numpy as np
import warp as wp

from cagecut.wos import kernels
from cagecut.wos.walkset import UpdateStats, _sync

# must exceed the closest-face tie window (TIE_REL for d < 1) plus float32 rounding: a walk whose last
# sphere misses the new geometry by less than the tie window could still switch its landing face
CUT_INFLATE = 4.0e-6


def cut_bounds(new_cut, dim, axes):
	"""Per simplex of the new cut geometry: k-DOP intervals, plane normal and offset (float64)."""
	s = np.asarray(new_cut, dtype=np.float64).reshape(-1, dim, dim)
	n = s.shape[0]
	if n == 0:
		return np.zeros((0, axes.shape[0])), np.zeros((0, axes.shape[0])), np.zeros((0, dim)), np.zeros(0)
	proj = s @ axes.T  # (n, dim, n_axes)
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


def descendant_faces(face_ancestor, new_alive, n_old, face_merged_into=None):
	"""Compressed lists (start, faces) over OLD face ids f of the alive new faces a walk landing on f may move
	to: new faces g with face_ancestor[g] == f', f' = face_merged_into[f] if f was merged away else f.
	Faces of each list sorted by id."""
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
		# gather the descendants of every face's merge target
		cnt = start[tgt + 1] - start[tgt]
		s2 = np.zeros(n_old + 1, dtype=np.int64)
		s2[1:] = np.cumsum(cnt)
		idx = np.repeat(start[tgt], cnt) + (np.arange(int(s2[-1])) - np.repeat(s2[:-1], cnt))
		g = g[idx]
		start = s2
	return start.astype(np.int32), g.astype(np.int32)


def incremental_update(ws, old, new, delta) -> UpdateStats:
	dev = ws.device
	P, n_per_point = ws.num_points, ws.num_walks
	n_walks = P * n_per_point
	stats = UpdateStats(n_walks=n_walks)
	zeros = np.zeros(P, np.int32)
	if delta.new is delta.old or P == 0 or delta.empty:
		ws.last_marks = {"rewalk": zeros, "onface": zeros.copy()}
		return stats
	assert (old.snapshot is delta.old or old.snapshot.version == delta.old.version) and \
		(new.snapshot is delta.new or new.snapshot.version == delta.new.version), "GpuCages do not match the delta snapshots"
	_sync(dev)
	t0 = time.perf_counter()
	ws._ensure_verts(new.snapshot.num_verts)

	# host-side preparation: bounds of the new cut geometry, modified-face flags, descendant face lists
	axes = ws.fns.bv[ws.bv_mode][2]
	lo, hi, nr, off = cut_bounds(delta.new_cut, ws.dim, axes)
	n_cut = lo.shape[0]
	n_old = delta.old.num_faces
	modified = np.zeros(max(n_old, 1), dtype=np.int32)
	mf = np.asarray(delta.modified_faces, dtype=np.int64).reshape(-1)
	modified[mf[(mf >= 0) & (mf < n_old)]] = 1
	merged = getattr(delta, "face_merged_into", None)
	if merged is not None:
		mi = np.asarray(merged, dtype=np.int64).reshape(-1)[:n_old]
		modified[:mi.shape[0]][mi >= 0] = 1  # merged-away faces are dead: their walks move to the merge target
	child_start, child_list = descendant_faces(delta.face_ancestor, delta.new.face_alive, n_old, merged)

	def up(a, dtype, shape_min=1):
		a = np.ascontiguousarray(a)
		if a.shape[0] == 0:
			a = np.zeros((shape_min,) + a.shape[1:], dtype=a.dtype)
		return wp.array(a, dtype=dtype, device=dev)
	cut_lo = up(lo.astype(np.float32).reshape(-1, axes.shape[0]), wp.float32)
	cut_hi = up(hi.astype(np.float32).reshape(-1, axes.shape[0]), wp.float32)
	cut_n = up(nr.astype(np.float32), ws.fns.Vec)
	cut_off = up(off.astype(np.float32), wp.float32)
	modified_wp = up(modified, wp.int32)
	child_start_wp = up(child_start, wp.int32)
	child_list_wp = up(child_list, wp.int32)

	counts = wp.zeros(2, dtype=wp.int32, device=dev)
	rewalk_list, onface_list = ws._worklists(n_walks)
	ws.mark_rewalk.zero_()
	ws.mark_onface.zero_()
	wp.launch(kernels.check_kernel(ws.fns, ws.bv_mode), dim=n_walks, inputs=[P, ws.pcap, ws.land_face, ws.bv, cut_lo,
		cut_hi, cut_n, cut_off, n_cut, CUT_INFLATE, modified_wp, child_start_wp, rewalk_list, counts, onface_list,
		ws.mark_rewalk, ws.mark_onface], device=dev)
	c = counts.numpy()  # synchronizes
	n_rewalk, n_onface = int(c[0]), int(c[1])
	t1 = time.perf_counter()

	p = ws.params
	if n_onface > 0:
		wp.launch(kernels.onface_kernel(ws.fns, ws.grads), dim=n_onface, inputs=[old.data, new.data, onface_list,
			ws.seeds_wp, ws.pcap, float(p.eps_ngon), int(p.max_steps_ngon), child_start_wp, child_list_wp, ws.land_face,
			ws.land_x, ws.gfac, ws.wsum, ws.gsum], device=dev)
	_sync(dev)
	t2 = time.perf_counter()
	if n_rewalk > 0:
		ws._launch_walks(new, old, n_rewalk, wlist=rewalk_list, subtract=True)
	_sync(dev)
	t3 = time.perf_counter()

	ws.last_marks = {"rewalk": ws.mark_rewalk.numpy()[:P].copy(), "onface": ws.mark_onface.numpy()[:P].copy()}
	stats.n_rewalk = n_rewalk
	stats.n_onface = n_onface
	stats.t_check_ms = (t1 - t0) * 1e3
	stats.t_onface_ms = (t2 - t1) * 1e3
	stats.t_rewalk_ms = (t3 - t2) * 1e3
	stats.t_total_ms = (time.perf_counter() - t0) * 1e3
	return stats
