"""
GAE persistence — JSON-backed storage for LearningState.

LearningState holds the learned factor weights and the current training step.
save_state / load_state use atomic write (temp-file + rename) to avoid
partial writes on crash.

Reference: docs/gae_design_v10_6.md §6 (Persistence).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LearningState
# ---------------------------------------------------------------------------

@dataclass
class LearningState:
    """
    Mutable container for the current learning state.

    Reference: docs/gae_design_v10_6.md §6.1.

    Attributes
    ----------
    weights : np.ndarray, shape (d_f,)
        Current factor weight vector.  Updated by the learning rule (Eq. 4b/4c).
    step : int
        Number of weight update steps completed so far.
    converged : bool
        True if the convergence monitor has declared convergence.
    metadata : dict[str, Any]
        Arbitrary caller-owned metadata (schema version, node_type, etc.).
        Must be JSON-serialisable.
    """

    weights: np.ndarray
    step: int = 0
    converged: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert isinstance(self.weights, np.ndarray), (
            "LearningState.weights must be np.ndarray"
        )
        assert self.weights.ndim == 1, (
            f"LearningState.weights must be 1-D, got shape {self.weights.shape}"
        )
        assert self.step >= 0, f"LearningState.step must be >= 0, got {self.step}"

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to a JSON-serialisable dict.

        Reference: docs/gae_design_v10_6.md §6.1 (serialisation format).

        Returns
        -------
        dict[str, Any]
        """
        return {
            "weights": self.weights.tolist(),
            "step": self.step,
            "converged": self.converged,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LearningState":
        """
        Reconstruct a LearningState from a dict produced by *to_dict*.

        Reference: docs/gae_design_v10_6.md §6.1 (deserialisation).

        Parameters
        ----------
        data : dict[str, Any]

        Returns
        -------
        LearningState

        Raises
        ------
        KeyError
            If a required key is absent.
        ValueError
            If the weights list is not 1-D.
        """
        weights = np.array(data["weights"], dtype=np.float64)
        if weights.ndim != 1:
            raise ValueError(
                f"LearningState.from_dict: weights must be 1-D, "
                f"got shape {weights.shape}"
            )
        return cls(
            weights=weights,
            step=int(data.get("step", 0)),
            converged=bool(data.get("converged", False)),
            metadata=dict(data.get("metadata", {})),
        )


# ---------------------------------------------------------------------------
# save_state / load_state
# ---------------------------------------------------------------------------

def save_state(state: LearningState, path: str | Path) -> None:
    """
    Atomically persist *state* to *path* as JSON.

    Uses a temp-file-and-rename strategy: writes to a sibling temp file first,
    then renames it over *path*.  This prevents partial-write corruption on
    crash or KeyboardInterrupt.

    Reference: docs/gae_design_v10_6.md §6.2 (atomic write).

    Parameters
    ----------
    state : LearningState
    path : str or Path
        Destination file path.  Parent directory must exist.

    Raises
    ------
    AssertionError
        If *state* is not a LearningState.
    OSError
        On filesystem errors.
    """
    assert isinstance(state, LearningState), (
        f"save_state: expected LearningState, got {type(state)}"
    )
    path = Path(path)
    payload = json.dumps(state.to_dict(), indent=2)

    dir_ = path.parent
    fd, tmp_path = tempfile.mkstemp(dir=dir_, suffix=".tmp", prefix=".gae_state_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
        os.replace(tmp_path, path)
        log.debug("save_state: wrote %s (step=%d)", path, state.step)
    except Exception:
        # Clean up orphaned temp file on error
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def load_state(path: str | Path) -> LearningState:
    """
    Load and return a LearningState from a JSON file written by *save_state*.

    Reference: docs/gae_design_v10_6.md §6.2 (load).

    Parameters
    ----------
    path : str or Path
        Path to the JSON file.

    Returns
    -------
    LearningState

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    json.JSONDecodeError
        If the file is not valid JSON.
    KeyError / ValueError
        If required fields are missing or malformed.
    """
    path = Path(path)
    log.debug("load_state: reading %s", path)
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    state = LearningState.from_dict(data)
    log.debug("load_state: loaded step=%d converged=%s", state.step, state.converged)
    return state
