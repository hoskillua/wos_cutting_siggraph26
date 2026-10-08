"""WalkSet: all walks of a set of query points, their per-walk cache and float64 weight sums."""

import time
from dataclasses import dataclass

import numpy as np
import warp as wp

from cagecut.wos import kernels as K
from cagecut.wos.gpu_cage import GpuCage, dim_bundle


@dataclass
class SolveStats:
	n_points: int = 0
	n_walks: int = 0
	terminated_fraction: float = 0.0  # mean over points
	t_total_ms: float = 0.0


@dataclass
class UpdateStats:
	n_walks: int = 0
	n_rewalk: int = 0
	n_onface: int = 0
	t_check_ms: float = 0.0
	t_rewalk_ms: float = 0.0
	t_onface_ms: float = 0.0
	t_total_ms: float = 0.0


def point_seeds(seed, seed_base, uids):
	"""int32 seed per point from (global seed, seed_base, uid) via a splitmix64 style hash."""
	with np.errstate(over="ignore"):
		u = np.asarray(uids, dtype=np.uint64)
		x = u * np.uint64(0x9E3779B97F4A7C15)
		x ^= np.uint64(int(seed) & 0xFFFFFFFFFFFFFFFF) * np.uint64(0xBF58476D1CE4E5B9)
		x ^= np.uint64(int(seed_base) & 0xFFFFFFFFFFFFFFFF) * np.uint64(0x94D049BB133111EB)
		x += np.uint64(0x632BE59BD9B4E019)
		x ^= x >> np.uint64(30)
		x *= np.uint64(0xBF58476D1CE4E5B9)
		x ^= x >> np.uint64(27)
		x *= np.uint64(0x94D049BB133111EB)
		x ^= x >> np.uint64(31)
	return (x & np.uint64(0xFFFFFFFF)).astype(np.uint32).view(np.int32)


# walks have very different lengths: small blocks free their SM slots sooner (measured ~10% faster than 256)
WALK_BLOCK = 64


def _sync(device):
	wp.synchronize_device(device)


class WalkSet:
	def __init__(self, dim, points, params, gradients, device=None, seed_base=0):
		self.dim = int(dim)
		self.params = params
		self.grads = bool(gradients)
		self.device = wp.get_device(device)
		self.seed_base = int(seed_base)
		self.B = dim_bundle(self.dim)
		self.bv_mode = "kdop" if params.bounding_volume == "kdop" else "aabb"
		self.NA = int(self.B.bv[self.bv_mode][2].shape[0])
		self.K = int(params.num_walks)
		pts = np.asarray(points, dtype=np.float64).reshape(-1, self.dim)
		self.points = pts.copy()
		self.uids = np.arange(pts.shape[0], dtype=np.int64)
		self._next_uid = pts.shape[0]
		self.num_points = pts.shape[0]
		self.pcap = 0
		self.vcap = 0
		self.nv = 0
		self.solved = False
		self.last_marks = {"rewalk": np.zeros(self.num_points, np.int32), "onface": np.zeros(self.num_points, np.int32)}
		self._T = None
		self._dT = None
		self._lists = None
		self._alloc_points(self.num_points + max(16, self.num_points // 16))
		self._upload_points()

	# -----------------------------------------------------------------------------------------
	# storage
	# -----------------------------------------------------------------------------------------

	def _alloc_points(self, pcap):
		"""(Re)allocate everything sized by point capacity, keeping existing data."""
		dev = self.device
		Vec = self.B.Vec
		old = self.pcap
		n = self.num_points if old > 0 else 0
		C = self.K * pcap
		assert C < 2 ** 31, "num_walks * point capacity exceeds int32 cache indexing"
		face = wp.full(C, -1, dtype=wp.int32, device=dev)
		xcs = wp.zeros(C, dtype=Vec, device=dev)
		gfs = wp.zeros(C if self.grads else 1, dtype=Vec, device=dev)
		bv = wp.zeros((2 * self.NA, C), dtype=wp.float32, device=dev)
		vc = max(self.vcap, 1)
		S = wp.zeros((pcap, vc), dtype=wp.float64, device=dev)
		m = wp.zeros(pcap, dtype=wp.float64, device=dev)
		G = wp.zeros((pcap, vc, self.dim) if self.grads else (1, 1, self.dim), dtype=wp.float64, device=dev)
		Gm = wp.zeros((pcap, self.dim) if self.grads else (1, self.dim), dtype=wp.float64, device=dev)
		d0 = wp.zeros(pcap, dtype=wp.float32, device=dev)
		mark_r = wp.zeros(pcap, dtype=wp.int32, device=dev)
		mark_o = wp.zeros(pcap, dtype=wp.int32, device=dev)
		if old > 0 and n > 0:
			wp.launch(K.relayout_kernel(self.B, self.bv_mode), dim=n * self.K, inputs=[n, old, pcap, self.face, self.xcs,
				self.gfs, self.bv, face, xcs, gfs, bv, int(self.grads)], device=dev)
			wp.launch(K.copy2d_f64, dim=(n, self.vcap), inputs=[self.S, S], device=dev)
			wp.copy(m, self.m, count=n)
			wp.copy(d0, self.d0, count=n)
			if self.grads:
				wp.launch(K.copy3d_f64, dim=(n, self.vcap, self.dim), inputs=[self.G, G], device=dev)
				wp.launch(K.copy2d_f64, dim=(n, self.dim), inputs=[self.Gm, Gm], device=dev)
		self.face, self.xcs, self.gfs, self.bv = face, xcs, gfs, bv
		self.S, self.m, self.G, self.Gm, self.d0 = S, m, G, Gm, d0
		self.mark_r, self.mark_o = mark_r, mark_o
		self.vcap = vc
		self.pcap = pcap
		self.points_wp = wp.zeros(pcap, dtype=Vec, device=dev)
		self.seeds_wp = wp.zeros(pcap, dtype=wp.int32, device=dev)

	def _upload_points(self):
		n = self.num_points
		if n == 0:
			return
		self.points_wp.assign(np.ascontiguousarray(np.concatenate([self.points.astype(np.float32),
			np.zeros((self.pcap - n, self.dim), np.float32)])))
		seeds = np.zeros(self.pcap, dtype=np.int32)
		seeds[:n] = point_seeds(self.params.seed, self.seed_base, self.uids)
		self.seeds_wp.assign(seeds)

	def seeds(self):
		return point_seeds(self.params.seed, self.seed_base, self.uids)

	def _ensure_points(self, n):
		if n > self.pcap:
			self._alloc_points(max(n, int(self.pcap * 1.5) + 16))

	def _ensure_verts(self, nv):
		"""Grow the vertex (column) capacity of the sums; new columns are zero."""
		if nv > self.vcap:
			dev = self.device
			vc = max(nv, int(self.vcap * 1.5) + 16)
			S = wp.zeros((self.pcap, vc), dtype=wp.float64, device=dev)
			wp.launch(K.copy2d_f64, dim=(self.pcap, self.vcap), inputs=[self.S, S], device=dev)
			self.S = S
			if self.grads:
				G = wp.zeros((self.pcap, vc, self.dim), dtype=wp.float64, device=dev)
				wp.launch(K.copy3d_f64, dim=(self.pcap, self.vcap, self.dim), inputs=[self.G, G], device=dev)
				self.G = G
			self.vcap = vc
		self.nv = max(self.nv, nv)

	def _worklists(self, n):
		"""Reusable REWALK / ONFACE worklist buffers with room for n entries each."""
		if self._lists is None or self._lists[0].shape[0] < n:
			self._lists = (wp.empty(n, dtype=wp.int32, device=self.device), wp.empty(n, dtype=wp.int32, device=self.device))
		return self._lists

	def _new_uids(self, n):
		u = np.arange(self._next_uid, self._next_uid + n, dtype=np.int64)
		self._next_uid += n
		return u

	# -----------------------------------------------------------------------------------------
	# walking
	# -----------------------------------------------------------------------------------------

	def _bounds(self, cage, p0=0, n=None, ids=None):
		dev = self.device
		if ids is not None:
			n = len(ids)
		if n == 0:
			return
		wl = ids if ids is not None else self._dummy_i32()
		wp.launch(K.bound_kernel(self.B), dim=n, inputs=[cage.data, self.points_wp, wl, int(ids is not None), int(p0),
			float(cage.diam / 64.0), self.d0], device=dev)

	def _dummy_i32(self):
		return wp.zeros(1, dtype=wp.int32, device=self.device)

	def _launch_walks(self, cage, old, n, wlist=None, p0=0, npts=1, subtract=False):
		if n == 0:
			return
		p = self.params
		wl = wlist if wlist is not None else self._dummy_i32()
		wp.launch(K.walk_kernel(self.B, self.bv_mode, self.grads), dim=n, inputs=[cage.data, old.data, self.points_wp,
			self.seeds_wp, self.d0, wl, int(wlist is not None), int(p0), int(max(npts, 1)), self.pcap, int(subtract),
			float(p.eps), int(p.max_steps), float(p.eps_ngon), int(p.max_steps_ngon), self.face, self.xcs, self.gfs,
			self.bv, self.S, self.m, self.G, self.Gm], device=self.device, block_dim=WALK_BLOCK)

	def solve(self, cage: GpuCage) -> SolveStats:
		"""All walks of all points on `cage` (sums reset)."""
		assert cage.dim == self.dim
		_sync(self.device)
		t0 = time.perf_counter()
		self._ensure_verts(cage.snapshot.num_verts)
		self.S.zero_()
		self.m.zero_()
		self.G.zero_()
		self.Gm.zero_()
		P = self.num_points
		self._bounds(cage, 0, P)
		self._launch_walks(cage, cage, P * self.K, p0=0, npts=P)
		_sync(self.device)
		t = (time.perf_counter() - t0) * 1e3
		self.solved = True
		tf = self.terminated_fraction()
		return SolveStats(P, P * self.K, float(tf.mean()) if P > 0 else 0.0, t)

	def update(self, old: GpuCage, new: GpuCage, delta) -> "UpdateStats":
		from cagecut.wos.update import incremental_update
		return incremental_update(self, old, new, delta)

	def append_points(self, cage: GpuCage, points) -> np.ndarray:
		"""Add points with fresh seeds and walk them on `cage`. Returns their ids."""
		pts = np.asarray(points, dtype=np.float64).reshape(-1, self.dim)
		n = pts.shape[0]
		P = self.num_points
		ids = np.arange(P, P + n, dtype=np.int64)
		if n == 0:
			return ids
		self._ensure_points(P + n)
		self._ensure_verts(cage.snapshot.num_verts)
		self.points = np.concatenate([self.points, pts])
		self.uids = np.concatenate([self.uids, self._new_uids(n)])
		self.num_points = P + n
		self._upload_points()
		for key in ("rewalk", "onface"):
			self.last_marks[key] = np.concatenate([self.last_marks[key], np.zeros(n, np.int32)])
		self._bounds(cage, P, n)
		self._launch_walks(cage, cage, n * self.K, p0=P, npts=n)
		return ids

	def reset_points(self, cage: GpuCage, ids, positions) -> None:
		"""Move points, give them fresh seeds and redo all their walks on `cage`."""
		ids = np.asarray(ids, dtype=np.int64).reshape(-1)
		if ids.size == 0:
			return
		assert np.unique(ids).size == ids.size, "duplicate ids in reset_points"
		self._ensure_verts(cage.snapshot.num_verts)
		self.points[ids] = np.asarray(positions, dtype=np.float64).reshape(-1, self.dim)
		self.uids[ids] = self._new_uids(ids.size)
		self._upload_points()
		dev = self.device
		n = int(ids.size)
		ids_wp = wp.array(ids.astype(np.int32), dtype=wp.int32, device=dev)
		wp.launch(K.zero_rows, dim=(n, self.vcap), inputs=[ids_wp, self.S, self.m, self.G, self.Gm, int(self.grads)],
			device=dev)
		self._bounds(cage, ids=ids_wp)
		wl = wp.empty(n * self.K, dtype=wp.int32, device=dev)
		wp.launch(K.expand_list, dim=n * self.K, inputs=[ids_wp, self.pcap, n, wl], device=dev)
		self._launch_walks(cage, cage, n * self.K, wlist=wl)

	# -----------------------------------------------------------------------------------------
	# outputs
	# -----------------------------------------------------------------------------------------

	def weights_wp(self):
		P, V = self.num_points, self.nv
		if self._T is None or self._T.shape != (P, V):
			self._T = wp.zeros((P, V), dtype=wp.float32, device=self.device)
		if P > 0 and V > 0:
			wp.launch(K.weights_kernel, dim=(P, V), inputs=[self.S, self.m, self._T], device=self.device)
		return self._T

	def gradients_wp(self):
		assert self.grads, "WalkSet created without gradients"
		P, V = self.num_points, self.nv
		if self._dT is None or self._dT.shape != (P, V, self.dim):
			self._dT = wp.zeros((P, V, self.dim), dtype=wp.float32, device=self.device)
		if P > 0 and V > 0:
			wp.launch(K.gradients_kernel, dim=(P, V), inputs=[self.S, self.m, self.G, self.Gm, self.dim, self._dT],
				device=self.device)
		return self._dT

	def weights(self) -> np.ndarray:
		return self.weights_wp().numpy()

	def gradients(self) -> np.ndarray:
		return self.gradients_wp().numpy()

	def terminated_fraction(self) -> np.ndarray:
		return self.m.numpy()[:self.num_points] / float(self.K)

	# cache access (debug)
	def cache(self):
		"""Per-walk cache as numpy, indexed [point, walk]: face, landing point, gradient factor, bv."""
		P, Kw, pc = self.num_points, self.K, self.pcap

		def take(a):
			a = a.reshape((Kw, pc) + a.shape[1:])[:, :P]
			return np.swapaxes(a, 0, 1)
		out = {"face": take(self.face.numpy()), "x": take(self.xcs.numpy())}
		if self.grads:
			out["g"] = take(self.gfs.numpy())
		bv = self.bv.numpy().reshape(2 * self.NA, Kw, pc)[:, :, :P]
		out["bv"] = np.transpose(bv, (2, 1, 0))
		return out
