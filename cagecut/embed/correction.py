"""Minimal affine weight correction (paper Eq. 5), applied when the weights are used (never written back).

For a point p with Monte Carlo weights w over the vertices v_j of its piece c, find the smallest change
dw (L2) such that sum_j (w_j + dw_j) [v_j; 1] = [p; 1]:
	dw = A_c^T (A_c A_c^T)^{-1} ([p; 1] - A_c w),   A_c = [v_j; 1]_{j in c}   ((dim+1) x |c|)
(Eq. 5 states the last row as sum_j dw_j = 0; the same, as the walk weights sum to one.)
Weights on vertices outside the point's piece (and on dead vertices) are set to zero. Pieces are the
material pieces of the cut cage (piece_labels): the shell of a cavity belongs to the solid around it.
"""

import numpy as np


def point_components(W, labels):
	"""Piece of each point = piece of its largest weight among live vertices."""
	W = np.asarray(W)
	labels = np.asarray(labels)
	Wm = np.where(labels[None, :] >= 0, W, -np.inf)
	return labels[np.argmax(Wm, axis=1)]


def affine_correction(W, points, rest_verts, labels):
	"""W (P, V) weights, points (P, dim) rest positions, rest_verts (V, dim), labels (V,) from
	geometry.snapshot.piece_labels (-1 = dead; a cavity shell / hole loop has its enclosing piece's label).
	Returns corrected weights (P, V) float64."""
	return AffineCorrector(rest_verts, labels)(W, points)


class AffineCorrector:
	"""Precomputes the per-piece (dim+1)x(dim+1) inverses (rest cage only) for repeated evaluation."""

	def __init__(self, rest_verts, labels):
		self.labels = np.asarray(labels, dtype=np.int64)
		self.verts = np.asarray(rest_verts, dtype=np.float64)[:self.labels.shape[0]]
		self._comp = {}
		for c in np.unique(self.labels):
			if c < 0:
				continue
			mask = (self.labels == c).astype(np.float64)
			# A_c over all vertices, zero columns outside the piece
			A = np.vstack([self.verts.T, np.ones(mask.size)]) * mask[None, :]
			self._comp[int(c)] = (mask, A, np.linalg.pinv(A @ A.T))

	def __call__(self, W, points):
		"""Columns beyond the labelled vertices (spare capacity) come back as zeros."""
		W = np.asarray(W)
		pts = np.asarray(points, dtype=np.float64)
		P = W.shape[0]
		out = np.zeros((P, W.shape[1]), dtype=np.float64)
		if P == 0:
			return out
		V = min(self.labels.shape[0], W.shape[1])
		W = W[:, :V]
		comp = point_components(W, self.labels[:V])
		for c, (mask, A, Minv) in self._comp.items():
			rows = np.nonzero(comp == c)[0]
			if rows.size == 0:
				continue
			Am = A[:, :V]
			if mask[V:].any():
				Minv = np.linalg.pinv(Am @ Am.T)
			allr = rows.size == P
			# masked weights: the piece's columns, zero elsewhere (row gathers only)
			Wc = (W if allr else W[rows]) * mask[None, :V]
			r = np.hstack([pts if allr else pts[rows], np.ones((rows.size, 1))]) - Wc @ Am.T
			Wc += (r @ Minv) @ Am
			if allr:
				out[:, :V] = Wc
			else:
				out[rows, :V] = Wc
		return out
