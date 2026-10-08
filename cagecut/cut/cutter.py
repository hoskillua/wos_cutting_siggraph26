"""Cutter: dimension independent front end of the cut topology.

	cutter = Cutter(dim, verts, faces_or_loops)
	cutter.begin(spec)              # scene.py cut spec dict; no geometry change
	delta = cutter.advance(amount)  # CutDelta (new is old when nothing changed)
	cutter.front()                  # live tips and front segments (vertex ids)
	cutter.check()                  # AssertionError on any invariant violation

3D: the blade cuts all material it sweeps over (any number of front segments and cut regions,
see topology3d.py). 2D: the knife continues along its ray and starts a new slit wherever it enters
material again. Specs the topology cannot handle (blade plane missing the cage or every part of it
behind the blade, faces parallel to the blade line among the faces it cuts, a zero-thickness crack
inside one piece crossed by the plane, slit pointing out of the cage) raise CutError from `begin`.
"""

import numpy as np

from cagecut.cut.blade import CutError
from cagecut.cut.topology2d import Cutter2D
from cagecut.cut.topology3d import Cutter3D

__all__ = ["Cutter", "CutError"]


class Cutter:
	def __init__(self, dim, verts, faces_or_loops):
		if dim not in (2, 3):
			raise ValueError("dim must be 2 or 3")
		self.dim = dim
		self.impl = Cutter3D(verts, faces_or_loops) if dim == 3 else Cutter2D(verts, faces_or_loops)

	@classmethod
	def from_scene(cls, scene):
		return cls(scene.dim, scene.cage_verts, scene.cage_faces)

	@property
	def snapshot(self):
		return self.impl.snapshot

	@property
	def active(self):
		return self.impl.active

	@property
	def completed(self):
		return self.impl.completed

	@property
	def t(self):
		"""3D: blade sweep time; 2D: slit length."""
		return self.impl.t

	@property
	def started(self):
		"""The blade touched the cage / the slit exists (topology already changed by this cut)."""
		return self.impl.contacted if self.dim == 3 else self.impl.started

	def begin(self, spec, **kwargs):
		"""Set the blade (3D, kwargs: start_at_contact) or slit (2D). No geometry change."""
		self.impl.begin(spec, **kwargs)

	def advance(self, amount):
		return self.impl.advance(amount)

	def front(self):
		"""Live cut front as vertex ids: (tips (k,), front segments (m, 2)).

		3D: every front segment (blade edge between two tips) of the current blade, any number at once.
		2D: the tip of the slit being cut (none while the knife travels outside material), no segments."""
		if self.dim == 3:
			segs = np.array(self.impl.fronts(), dtype=np.int64).reshape(-1, 2)
			return np.array(sorted(self.impl.tips), dtype=np.int64), segs
		tips = [self.impl.tip] if self.impl.cutting else []
		return np.array(tips, dtype=np.int64), np.zeros((0, 2), dtype=np.int64)

	def check(self):
		self.impl.check()
