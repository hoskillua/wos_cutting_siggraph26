"""CutDelta: everything the weight updater, embedding and sim need to know about one cut step.

This is the only interface between cut topology and the rest of the system: the updater never
reads halfedges.
"""

from dataclasses import dataclass, field

import numpy as np

from cagecut.geometry.snapshot import CageSnapshot


@dataclass
class CutDelta:
	dim: int
	old: CageSnapshot  # cage before the step
	new: CageSnapshot  # cage after the step (vertex/face ids extend old ones)

	# Geometry added to the domain boundary during this step, as simplices:
	#   3D: (k, 3, 3) triangles covering the swept part of the cut plane
	#   2D: (k, 2, 2) segments covered by the growing slit
	# A walk is invalid iff one of its spheres intersects this set.
	swept: np.ndarray

	# OLD face ids whose vertex list or vertex positions changed (excluding faces that are
	# simply unchanged). Walks landing on them need an on-face re-evaluation.
	modified_faces: np.ndarray

	# For every NEW face id: the OLD face id it descends from (itself if it already existed,
	# the split parent for children created this step), or -1 for faces created from nothing
	# (e.g. the cut faces on the first contact). A walk that landed on old face f is relocated
	# to the new face g with face_ancestor[g] == f' that contains its landing point, where
	# f' = face_merged_into[f] if f was merged away this step, else f.
	face_ancestor: np.ndarray

	new_vertices: np.ndarray  # vertex ids created this step
	orphaned_vertices: np.ndarray  # vertex ids no longer used by any face
	moved_vertices: np.ndarray  # existing vertex ids whose rest position changed (cut tips)

	# Sim state transfer for new vertices: new vertex id -> (source vertex ids, weights);
	# deformed position/velocity of the new vertex = weighted sum over the sources.
	new_vertex_sources: dict = field(default_factory=dict)

	# For every OLD face id: the OLD face id it was merged into this step (3D cut faces of two
	# cut regions joining), or -1. Merged-away faces are dead in `new`. None means all -1.
	face_merged_into: np.ndarray = None

	topology_changed: bool = False
	completed: bool = False  # the blade has passed all material in its plane (3D) / the knife left the cage (2D)
	events: list = field(default_factory=list)  # event names of this step, in order (see topology docs)

	# Cut plane (3D) / cut line (2D): point and unit normal. Side "+" is n.(x - p) >= 0.
	plane_point: np.ndarray = None
	plane_normal: np.ndarray = None
	# 3D: (plus_face, minus_face) id pairs of all live cut faces of the current blade, one pair per
	# connected cut region (the plus face bounds the + side). Empty before contact.
	# 2D: (seg, seg) pairs of the slit segments of every slit of the current knife.
	cut_faces: list = field(default_factory=list)
	# Blade sweep parameter after the step (3D) / slit length (2D).
	t: float = 0.0

	def __post_init__(self):
		if self.face_merged_into is None:
			self.face_merged_into = np.full(self.old.num_faces, -1, dtype=np.int32)

	@property
	def empty(self):
		return self.swept.shape[0] == 0 and self.modified_faces.size == 0 and not self.topology_changed
