"""St. Venant-Kirchhoff energy with a plastic rest strain, per quadrature point (numpy, float64).

Psi(F) = mu |E - Er|^2 + lambda/2 tr(E - Er)^2,   E = (F^T F - I) / 2.

Derivation (index notation, F_ab = dx_a / dX_b):
	S = 2 mu (E - Er) + lambda tr(E - Er) I            (second Piola-Kirchhoff stress, symmetric)
	P = dPsi/dF = F S
	dE_cb/dF_kl = (delta_cl F_kb + delta_bl F_kc) / 2
	dS_cb/dF_kl = mu (delta_cl F_kb + delta_bl F_kc) + lambda delta_cb F_kl
	H[ab, kl] = dP_ab/dF_kl
	          = delta_ak S_lb + mu F_al F_kb + mu delta_bl (F F^T)_ak + lambda F_ab F_kl
The flattened index of F_ab is a * dim + b (row major). H is symmetric.
"""

import numpy as np


def lame(E, nu):
	mu = E / (2.0 * (1.0 + nu))
	lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
	return mu, lam


def green_strain(F):
	"""(Q, d, d) -> (Q, d, d)."""
	F = np.asarray(F, dtype=np.float64)
	d = F.shape[-1]
	return 0.5 * (np.einsum("...ka,...kb->...ab", F, F) - np.eye(d))


def energy(F, Er, mu, lam):
	"""Energy density per point, (Q,)."""
	D = green_strain(F) - Er
	tr = np.trace(D, axis1=-2, axis2=-1)
	return mu * np.sum(D * D, axis=(-2, -1)) + 0.5 * lam * tr * tr


def stress(F, Er, mu, lam):
	"""First Piola-Kirchhoff stress P = dPsi/dF, (Q, d, d)."""
	F = np.asarray(F, dtype=np.float64)
	d = F.shape[-1]
	D = green_strain(F) - Er
	tr = np.trace(D, axis1=-2, axis2=-1)
	S = 2.0 * mu * D + lam * tr[..., None, None] * np.eye(d)
	return F @ S


def hessian(F, Er, mu, lam):
	"""d2Psi/dF2 as (Q, d*d, d*d) with flat index a*d + b of F_ab."""
	F = np.asarray(F, dtype=np.float64)
	d = F.shape[-1]
	D = green_strain(F) - Er
	tr = np.trace(D, axis1=-2, axis2=-1)
	S = 2.0 * mu * D + lam * tr[..., None, None] * np.eye(d)
	I = np.eye(d)
	FFt = F @ np.swapaxes(F, -1, -2)
	# axes (a, b, k, l)
	H = np.einsum("ak,...lb->...abkl", I, S)
	H += mu * np.einsum("...al,...kb->...abkl", F, F)
	H += mu * np.einsum("bl,...ak->...abkl", I, FFt)
	H += lam * np.einsum("...ab,...kl->...abkl", F, F)
	return H.reshape(F.shape[:-2] + (d * d, d * d))


def project_psd(H):
	"""Clamp negative eigenvalues of symmetric (..., n, n) matrices to zero."""
	H = 0.5 * (H + np.swapaxes(H, -1, -2))
	w, Q = np.linalg.eigh(H)
	w = np.maximum(w, 0.0)
	return (Q * w[..., None, :]) @ np.swapaxes(Q, -1, -2)
