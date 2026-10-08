"""Dense stiffness assembly K = sum_q w_q B_q^T H_q B_q for cage-embedded elasticity.

B_q maps cage positions to F_q = sum_j x_j (outer) gradT_qj, so with X[(q,b), j] = gradT_qj[b]:
	K[(j,a), (k,c)] = sum_q w_q sum_{b,e} X[(q,b), j] H_q[ab, ce] X[(q,e), k]
Written as one GEMM: Y[(q,b), (a,k,c)] = w_q sum_e H_q[ab, ce] X[(q,e), k], R = X^T Y; R viewed
as (V*d, V*d) is K (row index j*d + a, column index k*d + c). On CUDA the GEMM uses warp tiles in
float32 and the per-point StVK Hessians (optionally PSD projected by cyclic Jacobi) are computed on
the device from F; on CPU everything is numpy float64.
"""

import numpy as np
import warp as wp

from cagecut.sim import gemm as _gemm


@wp.kernel(enable_backward=False)
def _build_y(H: wp.array3d(dtype=wp.float32), G: wp.array3d(dtype=wp.float32), w: wp.array(dtype=wp.float32),
	dim: int, Vp: int, Y: wp.array2d(dtype=wp.float32)):
	q, k = wp.tid()
	wq = w[q]
	for a in range(dim):
		for b in range(dim):
			for c in range(dim):
				s = float(0.0)
				for e in range(dim):
					s += H[q, a * dim + b, c * dim + e] * G[q, k, e]
				Y[q * dim + b, (a * Vp + k) * dim + c] = wq * s


@wp.kernel(enable_backward=False)
def _pad_g(Gs: wp.array3d(dtype=wp.float32), V: int, dim: int, G: wp.array3d(dtype=wp.float32),
	XT: wp.array2d(dtype=wp.float32)):
	# G (Q, Vp, d) zero padded copy of Gs (Q, V, d); XT[k, q*d + b] = G[q, k, b]
	q, k = wp.tid()
	for b in range(dim):
		g = float(0.0)
		if k < V:
			g = Gs[q, k, b]
		G[q, k, b] = g
		XT[k, q * dim + b] = g


@wp.kernel(enable_backward=False)
def _gather(R: wp.array2d(dtype=wp.float32), dofs: wp.array(dtype=wp.int32), d: int, Vp: int, scale: wp.float64,
	diag: wp.array(dtype=wp.float64), use_diag: int, out: wp.array2d(dtype=wp.float64)):
	# out[i, k] = scale * K[dofs[i], dofs[k]] (+ diag[i] on the diagonal), K = R viewed as (Vp*d, Vp*d)
	i, k = wp.tid()
	r = dofs[i]
	c = dofs[k]
	v = scale * wp.float64(R[r // d, (r % d) * Vp * d + c])
	if use_diag != 0 and i == k:
		v += diag[i]
	out[i, k] = v


_HESS = {}


def _hessian_kernel(dim):
	"""StVK Hessian per point from F and the rest strain Er (see stvk.py), optionally PSD projected by
	cyclic Jacobi (float64). Writes (Q, d*d, d*d) float32."""
	if dim in _HESS:
		return _HESS[dim]
	D = int(dim)
	N = D * D
	MatD = wp.types.matrix(shape=(D, D), dtype=wp.float64)
	MatN = wp.types.matrix(shape=(N, N), dtype=wp.float64)

	@wp.kernel(enable_backward=False, module="unique")
	def kernel(F: wp.array3d(dtype=wp.float64), Er: wp.array3d(dtype=wp.float64), w: wp.array(dtype=wp.float32),
		mu: wp.float64, lam: wp.float64, project: int, H: wp.array3d(dtype=wp.float32)):
		q = wp.tid()
		f = MatD()
		for a in range(D):
			for b in range(D):
				f[a, b] = F[q, a, b]
		# S = 2 mu (E - Er) + lam tr(E - Er) I, E = (F^T F - I) / 2
		E = wp.float64(0.5) * (wp.transpose(f) @ f)
		tr = wp.float64(0.0)
		for a in range(D):
			E[a, a] = E[a, a] - wp.float64(0.5)
			for b in range(D):
				E[a, b] = E[a, b] - Er[q, a, b]
			tr += E[a, a]
		S = wp.float64(2.0) * mu * E
		for a in range(D):
			S[a, a] = S[a, a] + lam * tr
		FFt = f @ wp.transpose(f)
		A = MatN()
		for a in range(D):
			for b in range(D):
				for k in range(D):
					for l in range(D):
						h = mu * f[a, l] * f[k, b] + lam * f[a, b] * f[k, l]
						if a == k:
							h += S[l, b]
						if b == l:
							h += mu * FFt[a, k]
						A[a * D + b, k * D + l] = h
		if project != 0 and w[q] > wp.float32(0.0):
			Vm = wp.identity(n=N, dtype=wp.float64)
			for sweep in range(16):
				off = wp.float64(0.0)
				tot = wp.float64(0.0)
				for i in range(N):
					for j in range(N):
						if i != j:
							off += A[i, j] * A[i, j]
						tot += A[i, j] * A[i, j]
				if off <= wp.float64(1.0e-28) * tot:
					break
				for p in range(N - 1):
					for r in range(p + 1, N):
						apq = A[p, r]
						if wp.abs(apq) > wp.float64(1.0e-300):
							th = (A[r, r] - A[p, p]) / (wp.float64(2.0) * apq)
							t = wp.float64(1.0) / (wp.abs(th) + wp.sqrt(th * th + wp.float64(1.0)))
							if th < wp.float64(0.0):
								t = -t
							c = wp.float64(1.0) / wp.sqrt(t * t + wp.float64(1.0))
							s = t * c
							for k in range(N):
								akp = A[k, p]
								akq = A[k, r]
								A[k, p] = c * akp - s * akq
								A[k, r] = s * akp + c * akq
							for k in range(N):
								apk = A[p, k]
								aqk = A[r, k]
								A[p, k] = c * apk - s * aqk
								A[r, k] = s * apk + c * aqk
							for k in range(N):
								vkp = Vm[k, p]
								vkq = Vm[k, r]
								Vm[k, p] = c * vkp - s * vkq
								Vm[k, r] = s * vkp + c * vkq
			# H = V diag(max(lambda, 0)) V^T
			for i in range(N):
				for j in range(i, N):
					s2 = wp.float64(0.0)
					for k in range(N):
						s2 += Vm[i, k] * wp.max(A[k, k], wp.float64(0.0)) * Vm[j, k]
					H[q, i, j] = wp.float32(s2)
					H[q, j, i] = wp.float32(s2)
		else:
			for i in range(N):
				for j in range(N):
					H[q, i, j] = wp.float32(wp.float64(0.5) * (A[i, j] + A[j, i]))

	_HESS[dim] = kernel
	return kernel


def _round_up(n, m):
	return ((n + m - 1) // m) * m


def gradient_matrix(G):
	"""(Q, V, d) -> X (Q*d, V) float64 with X[(q,b), j] = G[q, j, b] (one strided pass, any input dtype)."""
	Q, V, d = G.shape
	X = np.empty((Q, d, V), dtype=np.float64)
	X[...] = np.transpose(G, (0, 2, 1))
	return X.reshape(Q * d, V)


def assemble_reference(G, w, H):
	"""numpy float64 reference (and CPU path)."""
	G = np.asarray(G, dtype=np.float64)
	Q, V, d = G.shape
	Hq = H.reshape(Q, d, d, d, d) * np.asarray(w, dtype=np.float64)[:, None, None, None, None]
	Y = np.einsum("qabce,qke->qbakc", Hq, G).reshape(Q * d, d * V * d)
	return (gradient_matrix(G).T @ Y).reshape(V * d, V * d)


class StiffnessAssembler:
	def __init__(self, dim, device=None):
		self.dim = dim
		self.device = wp.get_device(device)
		self.Q = 0
		self.V = 0
		self._shape = None  # (Q, Vp, Kp) of the device buffers
		self._Vw = 0  # largest V whose columns were written into Y since it was zeroed

	def set_gradients(self, G, w):
		"""G (Q, V, d) weight gradients, w (Q,) quadrature weights (0 for inactive points).
		Device buffers are kept while the padded sizes do not change."""
		G = np.asarray(G)
		Q, V, d = G.shape
		self.Q, self.V = Q, V
		self._w64 = np.asarray(w, dtype=np.float64).copy()
		if not self.device.is_cuda:
			self._G64 = np.asarray(G, dtype=np.float64)
			return
		Vp = _round_up(max(V, 1), _gemm.TILE_M)
		Kp = _round_up(max(Q * d, 1), _gemm.TILE_K)
		N = d * Vp * d
		self.Vp = Vp
		dev = self.device
		if self._shape != (Q, Vp, Kp):
			self._G = wp.zeros((Q, Vp, d), dtype=wp.float32, device=dev)
			self._XT = wp.zeros((Vp, Kp), dtype=wp.float32, device=dev)  # padded columns stay zero
			self._w = wp.zeros(Q, dtype=wp.float32, device=dev)
			self._H = wp.zeros((Q, d * d, d * d), dtype=wp.float32, device=dev)
			self._Y = wp.zeros((Kp, N), dtype=wp.float32, device=dev)  # padded rows / columns stay zero
			self._R = wp.zeros((Vp, N), dtype=wp.float32, device=dev)
			self._F = wp.zeros((Q, d, d), dtype=wp.float64, device=dev)
			self._Er = wp.zeros((Q, d, d), dtype=wp.float64, device=dev)
			self._shape = (Q, Vp, Kp)
			self._Vw = 0
		elif V < self._Vw:
			self._Y.zero_()  # columns of vertices beyond V must be zero again
			self._Vw = 0
		self._Vw = max(self._Vw, V)
		Gs = wp.array(np.ascontiguousarray(G, dtype=np.float32), dtype=wp.float32, device=dev)
		wp.launch(_pad_g, dim=(Q, Vp), inputs=[Gs, V, d, self._G, self._XT], device=dev)
		self._w.assign(self._w64.astype(np.float32))

	def set_rest_strain(self, Er):
		"""Plastic rest strains (Q, d, d) for the device Hessians (assemble_free_strain)."""
		if self.device.is_cuda and self.Q > 0:
			self._Er.assign(np.ascontiguousarray(Er, dtype=np.float64))

	def _run(self):
		wp.launch(_build_y, dim=(self.Q, self.V), inputs=[self._H, self._G, self._w, self.dim, self.Vp, self._Y],
			device=self.device)
		_gemm.gemm(self._XT, self._Y, self._R, self.device)

	def assemble(self, H):
		"""H (Q, d*d, d*d) per-point Hessians -> symmetric K (V*d, V*d) float64."""
		d = self.dim
		if not self.device.is_cuda:
			K = assemble_reference(self._G64, self._w64, H)
		else:
			self._H.assign(np.ascontiguousarray(H, dtype=np.float32))
			self._run()
			n = self.V * d
			K = self._R.numpy().reshape(self.Vp * d, self.Vp * d)[:n, :n].astype(np.float64)
		return 0.5 * (K + K.T)

	def _gather_free(self, dofs, scale=1.0, diag=None):
		n = dofs.size
		dev = self.device
		out = wp.empty((n, n), dtype=wp.float64, device=dev)
		if n > 0:
			dw = wp.array(dofs.astype(np.int32), dtype=wp.int32, device=dev)
			dg = wp.array(np.zeros(1) if diag is None else np.asarray(diag, dtype=np.float64), dtype=wp.float64, device=dev)
			wp.launch(_gather, dim=(n, n), inputs=[self._R, dw, self.dim, self.Vp, float(scale), dg, int(diag is not None),
				out], device=dev)
		return out.numpy()

	def assemble_free(self, H, dofs):
		"""K restricted to the given dofs (rows and columns), float64; symmetric up to float32 rounding."""
		dofs = np.asarray(dofs, dtype=np.int64)
		if not self.device.is_cuda:
			K = assemble_reference(self._G64, self._w64, H)
			return K[np.ix_(dofs, dofs)]
		self._H.assign(np.ascontiguousarray(H, dtype=np.float32))
		self._run()
		return self._gather_free(dofs, scale, diag)

	def assemble_free_strain(self, F, mu, lam, project, dofs, scale=1.0, diag=None):
		"""scale * K + diag(diag) restricted to the given dofs (float64), with the StVK Hessians (rest strain
		from set_rest_strain) computed and optionally PSD projected on the device from the deformation
		gradients F (Q, d, d). CUDA only."""
		assert self.device.is_cuda
		dofs = np.asarray(dofs, dtype=np.int64)
		self._F.assign(np.ascontiguousarray(F, dtype=np.float64))
		wp.launch(_hessian_kernel(self.dim), dim=self.Q, inputs=[self._F, self._Er, self._w, float(mu), float(lam),
			int(bool(project)), self._H], device=self.device)
		self._run()
		return self._gather_free(dofs, scale, diag)
