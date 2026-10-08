"""Blade preview geometry for the viewer (numpy only, no polyscope).

blade_outline: where the blade plane meets the cage, as segments whose end points are expressed on cage
edges, so that they can be placed on the *deformed* cage (the cut is computed on the rest cage while the
cage on screen is deformed). A finite blade only shows the part inside its strip."""

import numpy as np


def blade_outline(snap, spec, tol=1e-9):
	"""Segments of (blade plane) ∩ (cage faces). Returns (ends (m, 2, 3), tau (m, 2)).

	ends[i, k] = (u, v, a): segment end k is the point (1 - a) x_u + a x_v on cage edge (u, v).
	The visible part of segment i is the chord between its two ends restricted to tau[i] in [0, 1]
	(a finite blade clips the chord to its strip)."""
	p1 = np.asarray(spec["p1"], dtype=np.float64)
	p2 = np.asarray(spec["p2"], dtype=np.float64)
	l = p2 - p1
	length = float(np.linalg.norm(l))
	d = np.asarray(spec["direction"], dtype=np.float64)
	l = l / length
	n = np.cross(l, d - np.dot(d, l) * l)
	n = n / np.linalg.norm(n)
	bounded = bool(spec.get("bounded", False))
	X = snap.verts
	sd = (X - p1) @ n
	sd = np.where(np.abs(sd) < tol, 0.0, sd)
	s_all = (X - p1) @ l
	ends, taus = [], []
	for f in range(snap.num_faces):
		if not snap.face_alive[f]:
			continue
		cr = []  # (position along the chord, u, v, a, s)
		w = np.cross(n, snap.face_normal[f])
		for lp in snap.face_loops(f):
			a_ids, b_ids = lp, np.roll(lp, -1)
			hit = sd[a_ids] * sd[b_ids] < 0.0
			for u, v in zip(a_ids[hit], b_ids[hit]):
				a = sd[u] / (sd[u] - sd[v])
				P = X[u] + a * (X[v] - X[u])
				cr.append((float(P @ w), int(u), int(v), float(a), float((P - p1) @ l)))
		cr.sort()
		for k in range(0, len(cr) - 1, 2):
			A, B = cr[k], cr[k + 1]
			t0, t1 = 0.0, 1.0
			if bounded:
				sa, sb = A[4], B[4]
				if abs(sb - sa) < 1e-14:
					if not 0.0 <= sa <= length:
						continue
				else:
					ta, tb = sorted(((0.0 - sa) / (sb - sa), (length - sa) / (sb - sa)))
					t0, t1 = max(t0, ta), min(t1, tb)
					if t1 <= t0:
						continue
			ends.append([[A[1], A[2], A[3]], [B[1], B[2], B[3]]])
			taus.append([t0, t1])
	return np.array(ends, dtype=np.float64).reshape(-1, 2, 3), np.array(taus, dtype=np.float64).reshape(-1, 2)


def outline_nodes(ends, tau, x):
	"""(2m, 3) node positions of the outline segments on cage positions x (V, 3)."""
	if len(ends) == 0:
		return np.zeros((0, 3))
	u, v, a = ends[..., 0].astype(np.int64), ends[..., 1].astype(np.int64), ends[..., 2:3].reshape(len(ends), 2, 1)
	P = x[u] * (1.0 - a) + x[v] * a  # (m, 2, 3)
	A, B = P[:, 0], P[:, 1]
	t0, t1 = tau[:, :1], tau[:, 1:]
	return np.stack([A + t0 * (B - A), A + t1 * (B - A)], axis=1).reshape(-1, 3)


def affine_fit(rest, now):
	"""Least-squares affine map (M (dim+1, dim)) of the rest cage onto the displayed one: x ≈ [X, 1] M."""
	R = np.concatenate([rest, np.ones((rest.shape[0], 1))], axis=1)
	M, *_ = np.linalg.lstsq(R, now, rcond=None)
	return M


def apply_affine(M, P):
	P = np.asarray(P, dtype=np.float64)
	return P @ M[:-1] + M[-1]
