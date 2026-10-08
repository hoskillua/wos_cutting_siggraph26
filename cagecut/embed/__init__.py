"""Embedding: weight correction, quadrature sampling, embedded mesh splitting."""

from cagecut.embed.correction import AffineCorrector, affine_correction, point_components
from cagecut.embed.mesh_split import MeshSplitter, SplitResult, cut_face_pairs
from cagecut.embed.quadrature import boundary_distance, cage_volume, inside, sample_quadrature, winding_number

__all__ = ["AffineCorrector", "affine_correction", "point_components", "MeshSplitter", "SplitResult", "cut_face_pairs",
	"boundary_distance", "cage_volume", "inside", "sample_quadrature", "winding_number"]
