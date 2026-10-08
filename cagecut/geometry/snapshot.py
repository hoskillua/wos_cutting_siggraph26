"""Immutable cage snapshot: the only cage representation the WoS solver, embedding and sim see.

Produced by the topology code (cut/topology2d.py, cut/topology3d.py) after every change.

Conventions
-----------
* Face ids and vertex ids are persistent: faces/vertices are only ever appended. A face that
  no longer exists has face_alive False; an orphaned vertex has vert_alive False.
* 3D faces are planar polygons, CCW around the OUTWARD normal. Each face carries its plane
  (unit normal n, offset o = n.x) and a tangent frame (u, v) derived from n only. All
  geometric queries use the planar model: vertex j of face f is represented by its 2D frame
  coordinates face_coords2d, i.e. its projection on the face plane. Children of a split face
  inherit the parent plane exactly, so distances computed before and after a split agree
  bit-for-bit (required for exact walk reuse).
* 2D faces are directed segments (a -> b) of CCW loops; outward normal (e_y, -e_x)/|e|.
* face_root: the id of the face a face descends from by splits (its own id for input faces and
  for faces created from nothing, e.g. cut faces). Split children keep the root of their parent.
  The walk's closest-face tie rule orders equidistant faces by (inner side, root, id), which,
  unlike the bare id, does not change when a face is split (exact walk reuse at shared edges).
* face_edge_ref (3D): for the edge from slot j to slot j+1 of a face, the two vertex ids whose
  positions define its supporting line. A piece of a split edge keeps the original edge's ids, so
  the distance to its interior is computed from the same line before and after the split
  (bit-identical walk distances). Default: the edge's own endpoints.
* Faces may have several boundary loops (3D cut faces with holes). A face's slots are stored
  contiguously, loop after loop; face_vnext[j] is the slot of the next vertex along j's loop, so
  face edges are (j, face_vnext[j]) for every slot j of the face. Outer loops are CCW and hole loops
  CW around the outward normal. Multi-loop faces are always FACE_NGON. 2D faces are single segments.
* Cut faces (3D) have the root CUT_ROOT + 2 b + (0 for the + side, 1 for the - side) of the blade b
  (0, 1, ... per cutter) that created them: merging cut faces (same blade and side) never changes the
  tie order, and neither does a later blade splitting a cut face (children inherit the root).
"""

from dataclasses import dataclass

import numpy as np

from cagecut.geometry.polygon import (FACE_NGON, FACE_SEGMENT, classify_face, frame_from_normal, newell_normal,
	to_plane_2d)

CUT_ROOT = 1 << 30  # base face_root of 3D cut faces (see above)


@dataclass
class CageSnapshot:
	dim: int
	verts: np.ndarray  # (V, dim) float64 rest positions
	vert_alive: np.ndarray  # (V,) bool
	face_verts: np.ndarray  # flat int32 vertex ids
	face_start: np.ndarray  # (F,) int32 offset into face_verts
	face_size: np.ndarray  # (F,) int32
	face_alive: np.ndarray  # (F,) bool
	face_kind: np.ndarray  # (F,) int32: FACE_TRI / FACE_QUAD / FACE_NGON / FACE_SEGMENT
	face_normal: np.ndarray  # (F, dim) unit outward normal
	face_offset: np.ndarray  # (F,) n . x for points x on the face
	face_u: np.ndarray | None  # (F, 3) tangent frame (3D only)
	face_v: np.ndarray | None
	face_coords2d: np.ndarray | None  # (len(face_verts), 2) planar coords aligned with face_verts (3D only)
	version: int = 0  # incremented by the topology on every change
	face_root: np.ndarray | None = None  # (F,) int32 split-invariant ancestor id; None means arange(F)
	face_edge_ref: np.ndarray | None = None  # (len(face_verts), 2) int32 line-defining vertex ids (3D)
	face_vnext: np.ndarray | None = None  # (len(face_verts),) int32 slot of the next vertex in the same loop
	face_nloops: np.ndarray | None = None  # (F,) int32 number of boundary loops

	def __post_init__(self):
		if self.face_root is None:
			self.face_root = np.arange(self.face_size.shape[0], dtype=np.int32)
		if self.face_vnext is None:
			self.face_vnext = single_loop_vnext(self.face_start, self.face_size, self.face_verts.shape[0])
		if self.face_nloops is None:
			self.face_nloops = np.ones(self.face_size.shape[0], dtype=np.int32)
		if self.face_edge_ref is None and self.dim == 3:
			self.face_edge_ref = default_edge_ref(self.face_verts, self.face_vnext)

	@property
	def num_verts(self):
		return self.verts.shape[0]

	@property
	def num_faces(self):
		return self.face_size.shape[0]

	def face(self, f):
		s = self.face_start[f]
		return self.face_verts[s:s + self.face_size[f]]

	def face_loops(self, f):
		"""Boundary loops of face f as a list of vertex id arrays (slot order)."""
		s, n = int(self.face_start[f]), int(self.face_size[f])
		loops, seen = [], np.zeros(n, dtype=bool)
		for j0 in range(n):
			if seen[j0]:
				continue
			loop, j = [], j0
			while not seen[j]:
				seen[j] = True
				loop.append(self.face_verts[s + j])
				j = int(self.face_vnext[s + j]) - s
			loops.append(np.array(loop, dtype=np.int32))
		return loops

	def faces(self, alive_only=True):
		return [self.face(f) for f in range(self.num_faces) if self.face_alive[f] or not alive_only]

	def face_points(self, f):
		"""Planar-model 3D positions of face f's vertices (3D) or segment endpoints (2D)."""
		if self.dim == 2:
			return self.verts[self.face(f)]
		s = self.face_start[f]
		c = self.face_coords2d[s:s + self.face_size[f]]
		n, u, v = self.face_normal[f], self.face_u[f], self.face_v[f]
		return c[:, :1] * u + c[:, 1:2] * v + self.face_offset[f] * n

	def face_of_slot(self):
		"""(len(face_verts),) face id of every slot of face_verts (faces are stored contiguously in order)."""
		return np.repeat(np.arange(self.num_faces), self.face_size)

	def planar_points(self):
		"""face_points of every face concatenated, aligned with face_verts (same arithmetic, vectorized)."""
		if self.dim == 2:
			return self.verts[self.face_verts]
		f = self.face_of_slot()
		c = self.face_coords2d
		return c[:, :1] * self.face_u[f] + c[:, 1:2] * self.face_v[f] + self.face_offset[f][:, None] * self.face_normal[f]

	def components(self):
		"""Connected component label per vertex (via shared faces); -1 for dead vertices."""
		from scipy.sparse import coo_matrix
		from scipy.sparse.csgraph import connected_components
		V = self.num_verts
		f = self.face_of_slot()
		keep = self.face_alive[f]
		first = self.face_verts[self.face_start[f]]  # every face vertex is linked to the face's first vertex
		a, b = first[keep], self.face_verts[keep]
		g = coo_matrix((np.ones(a.shape[0], dtype=np.int8), (a, b)), shape=(V, V))
		_, labels = connected_components(g, directed=False)
		labels = labels.astype(np.int64)
		labels[~self.vert_alive] = -1
		return labels


def piece_labels(snap):
	"""Material piece label per vertex: snap.components(), except that a cavity shell (3D: a surface
	component of negative volume; 2D: a loop set of negative area, i.e. a hole) or a zero-volume crack
	pocket inside the material gets the label of the smallest positive component enclosing it. A hollow
	solid, a polygon with holes or a solid with an internal crack is one piece. -1 for dead vertices."""
	lab = snap.components()
	comps = sorted(set(lab.tolist()) - {-1})
	if len(comps) < 2:
		return lab
	X = snap.verts
	prims = {k: [] for k in comps}
	for f in range(snap.num_faces):
		if not snap.face_alive[f]:
			continue
		for lp in snap.face_loops(f):
			k = int(lab[lp[0]])
			if snap.dim == 2:
				prims[k].append((lp[0], lp[1]))
			else:
				prims[k].extend((lp[0], lp[i], lp[i + 1]) for i in range(1, len(lp) - 1))
	m = snap.dim
	prims = {k: X[np.array(t, dtype=np.int64).reshape(-1, m)] for k, t in prims.items()}
	if m == 2:
		meas = {k: 0.5 * float(np.sum(t[:, 0, 0] * t[:, 1, 1] - t[:, 1, 0] * t[:, 0, 1])) for k, t in prims.items()}
	else:
		meas = {k: float(np.sum(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])))) / 6.0
			for k, t in prims.items()}
	ext = float(np.max(np.ptp(X[lab >= 0], axis=0)))
	tol = 1e-9 * max(ext, 1e-300) ** m  # slivers cut off by the 1e-7 generic position shift measure ~0
	pos = [k for k in comps if meas[k] > tol]
	out = lab.copy()
	for k in comps:
		if meas[k] > tol:
			continue
		vk = np.nonzero(lab == k)[0]
		if meas[k] >= -tol:
			# zero measure: a sliver cut off at the surface (keeps its label), or a crack pocket strictly
			# inside the material (a bounded blade plunged inside): every vertex well inside one piece
			wins = [(j, min(_winding(prims[j], X[v]) for v in vk[:8])) for j in pos]
			inner = [j for j, w in wins if w > 0.9]
			if inner:
				out[lab == k] = min(inner, key=lambda j: meas[j])
			continue
		x = X[vk[0]]
		best = None
		for j in pos:
			w = _winding(prims[j], x)
			if w > 0.5 and (best is None or meas[j] < meas[best]):
				best = j
		if best is not None:
			out[lab == k] = best
	return out


def _winding(prims, x):
	"""Winding number at x of closed oriented segments (n, 2, 2) or triangles (n, 3, 3)."""
	if prims.shape[1] == 2:
		a, b = prims[:, 0] - x, prims[:, 1] - x
		return float(np.sum(np.arctan2(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0], np.sum(a * b, axis=1)))) / (2.0 * np.pi)
	a, b, c = (prims[:, i] - x for i in range(3))
	la, lb, lc = (np.linalg.norm(y, axis=1) for y in (a, b, c))
	dot = lambda x, y: np.einsum("ij,ij->i", x, y)
	den = la * lb * lc + dot(a, b) * lc + dot(a, c) * lb + dot(b, c) * la
	det = dot(a, np.cross(b, c))
	return float(np.sum(np.arctan2(det, den))) / (2.0 * np.pi)


def single_loop_vnext(face_start, face_size, n_slots):
	nxt = np.arange(n_slots, dtype=np.int32) + 1
	if face_size.shape[0] > 0:
		nxt[face_start + face_size - 1] = face_start
	return nxt.astype(np.int32)


def default_edge_ref(face_verts, face_vnext):
	"""Own endpoints of every face edge: (face_verts[j], face_verts[vnext[j]])."""
	return np.stack([face_verts, face_verts[face_vnext]], axis=1).astype(np.int32)


def build_snapshot(dim, verts, faces, face_planes=None, vert_alive=None, face_alive=None, version=0, face_root=None,
	face_edge_ref=None):
	"""Build a snapshot.

	faces: list of int arrays (3D polygons, or 2D segments (a, b)); a 3D face with holes is given as a
		list of int arrays (its loops, outer first).
	face_planes: optional (normals (F,dim), offsets (F,)) to keep inherited planes exactly;
		rows with a zero normal are computed (Newell normal, mean offset).
	face_root: optional (F,) split-invariant ancestor ids (default: the face ids).
	face_edge_ref: optional (len(flat faces), 2) line-defining vertex ids per face edge (3D; default:
		the edge's own endpoints).
	"""
	verts = np.asarray(verts, dtype=np.float64)
	F = len(faces)
	loops_of = [f if isinstance(f, (list, tuple)) and len(f) > 0 and np.ndim(f[0]) == 1 else [f] for f in faces]
	faces = [np.concatenate([np.asarray(l, dtype=np.int32) for l in ls]) for ls in loops_of]
	sizes = np.array([len(f) for f in faces], dtype=np.int32)
	start = np.zeros(F, dtype=np.int32)
	if F > 1:
		start[1:] = np.cumsum(sizes)[:-1]
	flat = np.concatenate(faces).astype(np.int32) if F > 0 else np.zeros(0, dtype=np.int32)
	vnext = np.arange(flat.shape[0], dtype=np.int32) + 1
	nloops = np.array([len(ls) for ls in loops_of], dtype=np.int32)
	for f, ls in enumerate(loops_of):
		k = int(start[f]) if F > 0 else 0
		for l in ls:
			vnext[k + len(l) - 1] = k
			k += len(l)
	normals = np.zeros((F, dim))
	offsets = np.zeros(F)
	if face_planes is not None:
		normals[:] = face_planes[0]
		offsets[:] = face_planes[1]
	alive_f = np.ones(F, dtype=bool) if face_alive is None else np.asarray(face_alive, dtype=bool)
	alive_v = np.ones(verts.shape[0], dtype=bool) if vert_alive is None else np.asarray(vert_alive, dtype=bool)
	kind = np.zeros(F, dtype=np.int32)
	root = np.arange(F, dtype=np.int32) if face_root is None else np.asarray(face_root, dtype=np.int32).copy()

	if dim == 2:
		for f, (a, b) in enumerate(faces):
			if np.dot(normals[f], normals[f]) == 0.0:
				e = verts[b] - verts[a]
				n = np.array([e[1], -e[0]]) / np.linalg.norm(e)
				normals[f] = n
				offsets[f] = np.dot(n, verts[a])
		kind[:] = FACE_SEGMENT
		return CageSnapshot(2, verts, alive_v, flat, start, sizes, alive_f, kind, normals, offsets,
			None, None, None, version, root, None, vnext, nloops)

	us = np.zeros((F, 3))
	vs = np.zeros((F, 3))
	coords = np.zeros((flat.shape[0], 2))
	for f, fv in enumerate(faces):
		p = verts[fv]
		if np.dot(normals[f], normals[f]) == 0.0:
			n = sum(newell_normal(verts[l]) for l in loops_of[f])
			nl = np.linalg.norm(n)
			n = n / nl if nl > 0.0 else np.array([0.0, 0.0, 1.0])
			normals[f] = n
			offsets[f] = np.mean(p @ n)
		u, v = frame_from_normal(normals[f])
		us[f], vs[f] = u, v
		c = to_plane_2d(p, u, v)
		coords[start[f]:start[f] + sizes[f]] = c
		kind[f] = classify_face(c) if nloops[f] == 1 else FACE_NGON
	ref = None if face_edge_ref is None else np.asarray(face_edge_ref, dtype=np.int32).reshape(-1, 2).copy()
	return CageSnapshot(3, verts, alive_v, flat, start, sizes, alive_f, kind, normals, offsets,
		us, vs, coords, version, root, ref, vnext, nloops)
