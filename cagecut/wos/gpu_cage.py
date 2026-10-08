"""Device mirror of a CageSnapshot: flat face arrays, BVH over face AABBs, closest-face queries."""

from types import SimpleNamespace

import numpy as np
import warp as wp

from cagecut.geometry.snapshot import CageSnapshot
from cagecut.wos import funcs2d as F2
from cagecut.wos import funcs3d as F3

# AABB inflation for the BVH (planar model points, float32 rounding)
BVH_PAD = 1.0e-6

# same arithmetic as the walk kernels (see kernels.NO_FMA)
wp.set_module_options({"fuse_fp": False})


@wp.kernel
def _closest3_kernel(cage: F3.Cage3, points: wp.array(dtype=wp.vec3), bound: float,
	dist: wp.array(dtype=float), face: wp.array(dtype=wp.int32), cp: wp.array(dtype=wp.vec3)):
	i = wp.tid()
	d, f, x = F3.closest_face3(cage, points[i], bound)
	dist[i] = d
	face[i] = f
	cp[i] = x


@wp.kernel
def _closest2_kernel(cage: F2.Cage2, points: wp.array(dtype=wp.vec2), bound: float,
	dist: wp.array(dtype=float), face: wp.array(dtype=wp.int32), cp: wp.array(dtype=wp.vec2)):
	i = wp.tid()
	d, f, x = F2.closest_face2(cage, points[i], bound)
	dist[i] = d
	face[i] = f
	cp[i] = x


@wp.kernel
def _bvalue3_kernel(cage: F3.Cage3, faces: wp.array(dtype=wp.int32), xs: wp.array(dtype=wp.vec3),
	seeds: wp.array(dtype=wp.int32), ks: wp.array(dtype=wp.int32), eps_ngon: float, max_steps_ngon: int,
	ids: wp.array(dtype=wp.vec4i), ws: wp.array(dtype=wp.vec4)):
	i = wp.tid()
	a, b = F3.boundary_value3(cage, faces[i], xs[i], seeds[i], ks[i], eps_ngon, max_steps_ngon)
	ids[i] = a
	ws[i] = b


@wp.kernel
def _bvalue2_kernel(cage: F2.Cage2, faces: wp.array(dtype=wp.int32), xs: wp.array(dtype=wp.vec2),
	seeds: wp.array(dtype=wp.int32), ks: wp.array(dtype=wp.int32), eps_ngon: float, max_steps_ngon: int,
	ids: wp.array(dtype=wp.vec4i), ws: wp.array(dtype=wp.vec4)):
	i = wp.tid()
	a, b = F2.boundary_value2(cage, faces[i], xs[i], seeds[i], ks[i], eps_ngon, max_steps_ngon)
	ids[i] = a
	ws[i] = b


# per-dimension bundles used by the kernel factories
DIM3 = SimpleNamespace(dim=3, Cage=F3.Cage3, Vec=wp.vec3, closest=F3.closest_face3, bvalue=F3.boundary_value3,
	face_dist=F3.face_dist3, sample=F3.sample_dir3,
	bv={"kdop": (F3.kdop13_proj, F3.vec13, F3.KDOP13_AXES), "aabb": (F3.aabb3_proj, wp.vec3, F3.AABB3_AXES)},
	closest_kernel=_closest3_kernel, bvalue_kernel=_bvalue3_kernel)
DIM2 = SimpleNamespace(dim=2, Cage=F2.Cage2, Vec=wp.vec2, closest=F2.closest_face2, bvalue=F2.boundary_value2,
	face_dist=F2.face_dist2, sample=F2.sample_dir2,
	bv={"kdop": (F2.kdop8_proj, wp.vec4, F2.KDOP8_AXES), "aabb": (F2.aabb2_proj, wp.vec2, F2.AABB2_AXES)},
	closest_kernel=_closest2_kernel, bvalue_kernel=_bvalue2_kernel)


def edge_lines(snapshot):
	"""(len(face_verts), 3): supporting line (unit normal nx, ny, offset) of every face edge in its face
	frame, through the edge's line-defining vertices (snapshot.face_edge_ref). Pieces of a split edge
	reference the original edge's vertices, so their lines are bit-identical to the parent's."""
	ref = np.asarray(snapshot.face_edge_ref, dtype=np.int64).reshape(-1, 2)
	f_of = snapshot.face_of_slot()
	u = snapshot.face_u[f_of]
	v = snapshot.face_v[f_of]
	P = [snapshot.verts[ref[:, k]] for k in range(2)]
	a = np.stack([np.einsum("ij,ij->i", P[0], u), np.einsum("ij,ij->i", P[0], v)], axis=1)
	b = np.stack([np.einsum("ij,ij->i", P[1], u), np.einsum("ij,ij->i", P[1], v)], axis=1)
	e = b - a
	n = np.stack([-e[:, 1], e[:, 0]], axis=1)
	ln = np.linalg.norm(n, axis=1)
	n = n / np.where(ln > 0.0, ln, 1.0)[:, None]
	return np.concatenate([n, np.einsum("ij,ij->i", n, a)[:, None]], axis=1)


def dim_bundle(dim):
	return DIM3 if dim == 3 else DIM2


class GpuCage:
	"""Immutable device copy of one snapshot. Rebuild (from_snapshot) after every topology change."""

	def __init__(self):
		self.snapshot = None
		self.data = None
		self.device = None
		self.bvh = None
		self.use_bvh = True
		self._keep = []

	@property
	def dim(self):
		return self.snapshot.dim

	@staticmethod
	def from_snapshot(snapshot: CageSnapshot, params, device=None) -> "GpuCage":
		dev = wp.get_device(device)
		g = GpuCage()
		g.snapshot = snapshot
		g.device = dev
		g.use_bvh = getattr(params, "distance", "bvh") != "brute"
		dim = snapshot.dim
		F = snapshot.num_faces
		alive = snapshot.face_alive.astype(bool)

		def arr(a, dtype):
			x = wp.array(np.ascontiguousarray(a), dtype=dtype, device=dev)
			g._keep.append(x)
			return x

		# face AABBs from the planar model points (dead faces get far away boxes)
		lo = np.full((max(F, 1), 3), 1.0e20)
		hi = np.full((max(F, 1), 3), 1.0e20)
		n_alive = int(alive.sum())
		diam = 1.0
		if n_alive > 0:
			pts = snapshot.planar_points()
			if dim == 2:
				pts = np.concatenate([pts, np.zeros((pts.shape[0], 1))], axis=1)
			st = snapshot.face_start.astype(np.int64)
			pl = np.minimum.reduceat(pts, st, axis=0)
			ph = np.maximum.reduceat(pts, st, axis=0)
			pad = BVH_PAD * np.maximum(1.0, np.maximum.reduceat(np.abs(pts).max(axis=1), st))
			lo[:F][alive] = (pl - pad[:, None])[alive]
			hi[:F][alive] = (ph + pad[:, None])[alive]
			if dim == 2:
				lo[:F, 2][alive] = -1.0
				hi[:F, 2][alive] = 1.0
			diam = float(np.linalg.norm(pl[alive].min(axis=0) - ph[alive].max(axis=0)))
		if n_alive == 0:
			g.use_bvh = False
		if g.use_bvh:
			g.bvh = wp.Bvh(arr(lo, wp.vec3), arr(hi, wp.vec3))

		if dim == 3:
			d = F3.Cage3()
			d.face_kind = arr(snapshot.face_kind.astype(np.int32), wp.int32)
			d.vnext = arr(np.asarray(snapshot.face_vnext, dtype=np.int32).reshape(-1), wp.int32)
			d.coords = arr(snapshot.face_coords2d.astype(np.float32).reshape(-1, 2), wp.vec2)
			d.normal = arr(snapshot.face_normal.astype(np.float32).reshape(-1, 3), wp.vec3)
			d.fu = arr(snapshot.face_u.astype(np.float32).reshape(-1, 3), wp.vec3)
			d.fv = arr(snapshot.face_v.astype(np.float32).reshape(-1, 3), wp.vec3)
			d.edge_line = arr(edge_lines(snapshot).astype(np.float32).reshape(-1, 3), wp.vec3)
		else:
			d = F2.Cage2()
			d.face_kind = arr(snapshot.face_kind.astype(np.int32), wp.int32)
			d.verts = arr(snapshot.verts.astype(np.float32).reshape(-1, 2), wp.vec2)
			d.normal = arr(snapshot.face_normal.astype(np.float32).reshape(-1, 2), wp.vec2)
		d.face_start = arr(snapshot.face_start.astype(np.int32), wp.int32)
		d.face_size = arr(snapshot.face_size.astype(np.int32), wp.int32)
		d.face_alive = arr(alive.astype(np.int32), wp.int32)
		d.face_root = arr(np.asarray(snapshot.face_root, dtype=np.int32).reshape(F), wp.int32)
		d.face_verts = arr(snapshot.face_verts.astype(np.int32), wp.int32)
		d.offset = arr(snapshot.face_offset.astype(np.float32), wp.float32)
		d.bvh = g.bvh.id if g.use_bvh else wp.uint64(0)
		d.num_faces = F
		d.diam = diam
		d.use_bvh = 1 if g.use_bvh else 0
		g.data = d
		g.diam = diam
		return g

	def closest(self, points, bound=None):
		"""Exact closest face of each point: (dist (n,), face (n,), closest_point (n, dim)), numpy."""
		B = dim_bundle(self.dim)
		pts = np.ascontiguousarray(np.asarray(points, dtype=np.float32).reshape(-1, self.dim))
		n = pts.shape[0]
		dist, face, cp = self.closest_wp(wp.array(pts, dtype=B.Vec, device=self.device), bound)
		return dist.numpy(), face.numpy(), cp.numpy().reshape(n, self.dim)

	def closest_wp(self, points_wp, bound=None, n=None):
		B = dim_bundle(self.dim)
		n = points_wp.shape[0] if n is None else n
		dist = wp.empty(n, dtype=float, device=self.device)
		face = wp.empty(n, dtype=wp.int32, device=self.device)
		cp = wp.empty(n, dtype=B.Vec, device=self.device)
		b = self.diam / 64.0 if bound is None else float(bound)
		if n > 0:
			wp.launch(B.closest_kernel, dim=n, inputs=[self.data, points_wp, b, dist, face, cp], device=self.device)
		return dist, face, cp

	def boundary_values(self, faces, points, seeds, ks, params):
		"""Boundary values (ids (n,4), weights (n,4)) at points on faces, as the walks use them."""
		B = dim_bundle(self.dim)
		n = len(faces)
		dev = self.device
		ids = wp.empty(n, dtype=wp.vec4i, device=dev)
		ws = wp.empty(n, dtype=wp.vec4, device=dev)
		wp.launch(B.bvalue_kernel, dim=n, inputs=[self.data,
			wp.array(np.asarray(faces, dtype=np.int32), dtype=wp.int32, device=dev),
			wp.array(np.asarray(points, dtype=np.float32).reshape(-1, self.dim), dtype=B.Vec, device=dev),
			wp.array(np.asarray(seeds, dtype=np.int32), dtype=wp.int32, device=dev),
			wp.array(np.asarray(ks, dtype=np.int32), dtype=wp.int32, device=dev),
			float(params.eps_ngon), int(params.max_steps_ngon), ids, ws], device=dev)
		return ids.numpy(), ws.numpy()
