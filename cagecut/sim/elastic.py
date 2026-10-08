"""Cage-driven StVK elasticity: backward Euler with one Newton step.

Degrees of freedom are the cage vertices x (V, dim). Deformation gradient at quadrature point q:
F_q = sum_j x_j (outer) gradT_qj. Energy: sum_q w_q Psi(F_q) over active points with the
plastic rest strain Er_q = E(F_q(rest)), so the rest configuration is exactly force free even
though Monte Carlo gradients make F_q(rest) != I.

Quadrature weight w_q and density rho:
	material.density given: w_q = quad_volume, rho = density.
	material.density None (legacy): w_q = 1 / Q and rho = 1 for both energy and mass. This is
	exactly the old code's scaling (forces and stiffness / Nq, masses sum_q T / Nq, i.e. total
	mass 1 on a unit-volume domain), so legacy E values give the same dynamics.

Backward Euler, single Newton step from x_n (K = Hessian of the elastic energy, D = alpha M + beta K):
	(M (1 + dt alpha) + (dt beta + dt^2) K) dv = dt (f(x_n) + M g + f_ext - D v_n - dt K v_n)
	(on CUDA the matrix on the left is built on the device, in float64 from the float32 K)
	v_{n+1} = v_n + dv,  x_{n+1} = x_n + dt v_{n+1}
Pinned and dead vertices are removed from the system and keep their position.
"""

import time

import numpy as np
import scipy.linalg as sla

from cagecut.sim import stvk
from cagecut.sim.assemble import StiffnessAssembler, gradient_matrix


def _to_numpy(a):
	if a is None:
		return None
	if hasattr(a, "numpy") and not isinstance(a, np.ndarray):
		return a.numpy()
	return np.asarray(a)


class ElasticSim:
	def __init__(self, dim, rest_verts, quad_volume: float, material, pinned, floor_y=None,
		psd_projection=True, device=None):
		self.dim = int(dim)
		self.rest = np.array(rest_verts, dtype=np.float64).reshape(-1, self.dim)
		self.x = self.rest.copy()
		self.v = np.zeros_like(self.x)
		self.quad_volume = float(quad_volume)
		self.material = material
		self.floor_y = floor_y
		self.psd_projection = bool(psd_projection)
		self.dt = float(material.dt)
		g = np.zeros(self.dim) if material.gravity is None else np.asarray(material.gravity, dtype=np.float64)
		self.gravity = g.reshape(self.dim)
		self.set_material(material.E, material.nu, material.rayleigh_alpha, material.rayleigh_beta)
		self.pinned = np.zeros(0, dtype=np.int64)
		self.set_pinned(pinned)
		self.alive = np.ones(self.num_verts, dtype=bool)
		self._assembler = StiffnessAssembler(self.dim, device)
		self._X = None  # (Q*d, V) gradient matrix, None until set_weights
		self._weights_V = -1
		self.masses = None
		self.Er = None
		self.w = None

	@property
	def num_verts(self):
		return self.x.shape[0]

	# ---------------------------------------------------------------------------------------
	# configuration

	def set_material(self, E, nu, rayleigh_alpha, rayleigh_beta):
		self.E, self.nu = float(E), float(nu)
		self.mu, self.lam = stvk.lame(self.E, self.nu)
		self.rayleigh_alpha = float(rayleigh_alpha)
		self.rayleigh_beta = float(rayleigh_beta)

	def set_dt(self, dt):
		self.dt = float(dt)

	def set_pinned(self, pinned):
		p = np.unique(np.asarray(pinned if pinned is not None else [], dtype=np.int64).reshape(-1))
		self.pinned = p
		self.v[p] = 0.0

	def reset(self):
		self.x = self.rest.copy()
		self.v = np.zeros_like(self.x)

	def add_vertices(self, rest_positions, x_init, v_init):
		"""Append vertices (e.g. created by a cut). Call set_weights before the next step."""
		d = self.dim
		r = np.asarray(rest_positions, dtype=np.float64).reshape(-1, d)
		self.rest = np.vstack([self.rest, r])
		self.x = np.vstack([self.x, np.asarray(x_init, dtype=np.float64).reshape(-1, d)])
		self.v = np.vstack([self.v, np.asarray(v_init, dtype=np.float64).reshape(-1, d)])
		self.alive = np.concatenate([self.alive, np.ones(r.shape[0], dtype=bool)])

	def set_weights(self, T, G, active, rest_verts, alive):
		"""Refresh weights after a solve or cut update.

		T (Q, V), G (Q, V, dim) numpy or warp arrays; active (Q,) bool; rest_verts (V, dim);
		alive (V,) bool (dead vertices are excluded from the solve and kept fixed).
		Recomputes lumped masses and the plastic rest strains.
		"""
		d = self.dim
		T = np.asarray(_to_numpy(T))
		G = np.asarray(_to_numpy(G))  # dtype kept (float32 from the walk sets); converted once below
		Q = T.shape[0]
		V = self.num_verts
		G = G.reshape(Q, -1, d)
		if T.shape[1] < V or G.shape[1] < V:
			raise ValueError(f"weights have {T.shape[1]} columns, sim has {V} vertices")
		# weight arrays may carry extra capacity columns
		T = T[:, :V]
		G = G[:, :V, :]
		active = np.ones(Q, dtype=bool) if active is None else np.asarray(_to_numpy(active), dtype=bool).reshape(Q)
		self.rest = np.array(rest_verts, dtype=np.float64).reshape(-1, d)[:V].copy()
		self.alive = np.ones(V, dtype=bool) if alive is None else np.asarray(_to_numpy(alive), dtype=bool).reshape(-1)[:V].copy()
		self.active = active.copy()

		legacy = self.material.density is None
		wq = (1.0 / max(Q, 1)) if legacy else self.quad_volume
		rho = 1.0 if legacy else float(self.material.density)
		self.w = np.where(active, wq, 0.0)

		m = rho * (self.w @ T)
		if np.any(self.alive):
			floor = 1e-3 * np.mean(np.abs(m[self.alive]))
			m = np.maximum(m, floor)
		self.masses = m

		self._weights_V = V
		self._X = gradient_matrix(G)
		self.Er = stvk.green_strain(self.deformation_gradients(self.rest))
		self._assembler.set_gradients(G, self.w)
		self._assembler.set_rest_strain(self.Er)
		self.v[~self.alive] = 0.0

	# ---------------------------------------------------------------------------------------
	# energy, forces, stiffness

	def deformation_gradients(self, x=None):
		"""(Q, d, d) with F[q, a, b] = sum_j x[j, a] G[q, j, b]."""
		x = self.x if x is None else np.asarray(x, dtype=np.float64)
		d = self.dim
		Q = self._X.shape[0] // d
		return (self._X @ x).reshape(Q, d, d).transpose(0, 2, 1)

	def energy(self, x=None):
		F = self.deformation_gradients(x)
		return float(np.sum(self.w * stvk.energy(F, self.Er, self.mu, self.lam)))

	def elastic_forces(self, x=None):
		"""-dE/dx, (V, d)."""
		F = self.deformation_gradients(x)
		P = stvk.stress(F, self.Er, self.mu, self.lam) * self.w[:, None, None]
		# f[j, a] = -sum_{q,b} X[(q,b), j] P[q, a, b]
		return -(self._X.T @ np.transpose(P, (0, 2, 1)).reshape(-1, self.dim))

	def point_hessians(self, x=None, project=None):
		F = self.deformation_gradients(x)
		H = stvk.hessian(F, self.Er, self.mu, self.lam)
		project = self.psd_projection if project is None else project
		if project:
			H[self.w > 0.0] = stvk.project_psd(H[self.w > 0.0])
		return H

	def stiffness(self, x=None, project=None):
		"""Energy Hessian d2E/dx2, dense (V*d, V*d), assembled on the sim device."""
		return self._assembler.assemble(self.point_hessians(x, project))

	# ---------------------------------------------------------------------------------------
	# time stepping

	def free_vertices(self):
		free = self.alive.copy()
		free[self.pinned[self.pinned < free.shape[0]]] = False
		return np.nonzero(free)[0]

	def step(self, ext_force=None):
		if self._X is None or self._weights_V != self.num_verts:
			raise RuntimeError("set_weights must be called after construction / add_vertices")
		t0 = time.perf_counter()
		d, dt = self.dim, self.dt
		f = self.elastic_forces() + self.masses[:, None] * self.gravity[None, :]
		if ext_force is not None:
			f = f + np.asarray(ext_force, dtype=np.float64).reshape(-1, d)
		fv = self.free_vertices()
		dofs = (fv[:, None] * d + np.arange(d)[None, :]).reshape(-1)
		t1 = time.perf_counter()
		mf = np.repeat(self.masses[fv], d)
		vf = self.v[fv].reshape(-1)
		alpha, beta = self.rayleigh_alpha, self.rayleigh_beta
		c = dt * beta + dt * dt
		dg = mf * (1.0 + dt * alpha)
		if self._assembler.device.is_cuda:
			# Hessians (and their PSD projection) and the system matrix A = c K + diag(dg) built on the device
			A = self._assembler.assemble_free_strain(self.deformation_gradients(), self.mu, self.lam,
				self.psd_projection, dofs, scale=c, diag=dg)
			t2 = time.perf_counter()
			cKv = A @ vf - dg * vf
		else:
			Kf = self._assembler.assemble_free(self.point_hessians(), dofs)
			t2 = time.perf_counter()
			A = c * Kf
			A[np.diag_indices_from(A)] += dg
			cKv = c * (Kf @ vf)
		# dt (beta + dt) K v = c K v
		rhs = dt * (f[fv].reshape(-1) - alpha * mf * vf) - cKv
		# Cholesky (always succeeds with PSD-projected Hessians); symmetric indefinite solve otherwise
		try:
			dv = sla.cho_solve(sla.cho_factor(A, lower=True, check_finite=False), rhs, check_finite=False)
		except sla.LinAlgError:
			dv = None
		if dv is None or not np.all(np.isfinite(dv)):
			dv = sla.solve(A, rhs, assume_a="sym", check_finite=False)
		t3 = time.perf_counter()

		self.v[fv] += dv.reshape(-1, d)
		self.x[fv] += dt * self.v[fv]
		if self.floor_y is not None:
			below = self.x[fv, 1] < self.floor_y
			ids = fv[below]
			self.x[ids, 1] = self.floor_y
			self.v[ids, 1] = np.maximum(self.v[ids, 1], 0.0)
		t4 = time.perf_counter()
		return {"t_forces_ms": 1e3 * (t1 - t0), "t_assemble_ms": 1e3 * (t2 - t1), "t_solve_ms": 1e3 * (t3 - t2),
			"t_update_ms": 1e3 * (t4 - t3), "t_total_ms": 1e3 * (t4 - t0), "num_dofs": int(dofs.size)}
