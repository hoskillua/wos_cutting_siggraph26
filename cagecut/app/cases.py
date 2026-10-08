"""Cases the app can load: the scene configs found in a folder (no polyscope, no GPU).

A case is a .json file with a "cage_file" entry whose cage file exists (the scene loader resolves it
relative to the config's folder, then the repository root). Cut spec files and other JSON files in the
same folder are not cases."""

import os
from dataclasses import dataclass

from cagecut.scene import _load_json, resolve_path


@dataclass(frozen=True)
class Case:
	path: str
	dim: int

	@property
	def label(self):
		return f"{os.path.splitext(os.path.basename(self.path))[0]} ({self.dim}D)"


def _case_of(path):
	try:
		cfg = _load_json(path)
		if not isinstance(cfg, dict) or not isinstance(cfg.get("cage_file"), str):
			return None
		cage = resolve_path(cfg["cage_file"], os.path.dirname(path))
		if cage is None or not os.path.exists(cage):
			return None
		dim = int(cfg.get("dim", 0))
		if dim == 0:  # same rule as the scene loader
			dim = 2 if (cage.endswith(".npy") or "pinned_verts" in cfg or "points_file" in cfg) else 3
		return Case(os.path.abspath(path), dim)
	except (OSError, ValueError, TypeError):
		return None


def list_cases(folder):
	"""The loadable cases of a folder, sorted by name (empty when the folder does not exist)."""
	if not folder or not os.path.isdir(folder):
		return []
	out = []
	for name in sorted(os.listdir(folder), key=str.lower):
		if name.lower().endswith(".json"):
			c = _case_of(os.path.join(folder, name))
			if c is not None:
				out.append(c)
	return out
