"""Mesh file IO. Faces keep file order (no regrouping by face size)."""

import numpy as np


def read_obj(path):
	"""Read vertices and polygon faces from an OBJ file.

	Returns (verts (V,3) float64, faces list[np.ndarray int32]) with faces in file order.
	Texture/normal indices are ignored, negative indices are resolved.
	"""
	verts = []
	faces = []
	with open(path, "r") as f:
		for line in f:
			parts = line.split()
			if not parts:
				continue
			if parts[0] == "v":
				verts.append([float(x) for x in parts[1:4]])
			elif parts[0] == "f":
				face = []
				for tok in parts[1:]:
					idx = int(tok.split("/")[0])
					# OBJ indices are 1-based, negatives count from the end
					face.append(idx - 1 if idx > 0 else len(verts) + idx)
				faces.append(np.array(face, dtype=np.int32))
	return np.array(verts, dtype=np.float64).reshape(-1, 3), faces


def write_obj(path, verts, faces):
	with open(path, "w") as f:
		for v in verts:
			f.write("v " + " ".join(f"{x:.9g}" for x in v) + "\n")
		for face in faces:
			f.write("f " + " ".join(str(int(i) + 1) for i in face) + "\n")


def triangulate_faces(faces):
	"""Fan triangulation of polygon faces, for rendering only."""
	tris = []
	for face in faces:
		for i in range(1, len(face) - 1):
			tris.append([face[0], face[i], face[i + 1]])
	return np.array(tris, dtype=np.int32).reshape(-1, 3)


def legacy_face_order(faces):
	"""Face order used by the old code (meshio cells_dict): quads, then triangles, then other polygons.

	Returns an array mapping legacy face index -> file order face index.
	"""
	sizes = np.array([len(f) for f in faces])
	quads = np.nonzero(sizes == 4)[0]
	tris = np.nonzero(sizes == 3)[0]
	polys = np.nonzero((sizes != 3) & (sizes != 4))[0]
	return np.concatenate([quads, tris, polys]).astype(np.int64)


def unit_box_transform(points):
	"""Legacy normalization: center the bounding box at 0.5 and scale the longest side to 1."""
	lo = points.min(axis=0)
	hi = points.max(axis=0)
	center = 0.5 * (lo + hi)
	scale = 1.0 / np.max(hi - lo)
	offset = np.full(points.shape[1], 0.5)
	return lambda p: (np.asarray(p, dtype=np.float64) - center) * scale + offset
