"""Locate a FEMFFUSION checkout for the tests that read its reference inputs.

Search order: ``$FEMFFUSION_DIR``, a sibling ``FEMFFUSION`` next to this
repository, then ``~/FEMFFUSION``. Tests skip when a needed file is missing,
naming the file, rather than when only the directory is absent.
"""

import os

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _root():
    if "FEMFFUSION_DIR" in os.environ:
        return os.environ["FEMFFUSION_DIR"]
    for candidate in (os.path.join(os.path.dirname(_REPO), "FEMFFUSION"),
                      os.path.expanduser("~/FEMFFUSION")):
        if os.path.isdir(candidate):
            return candidate
    return os.path.join(os.path.dirname(_REPO), "FEMFFUSION")


def femffusion_path(*parts):
    """Path inside the FEMFFUSION checkout (which need not exist)."""
    return os.path.join(_root(), *parts)


def requires_femffusion(*relpaths):
    """Skip unless every FEMFFUSION file in ``relpaths`` exists."""
    missing = [p for p in relpaths if not os.path.isfile(femffusion_path(p))]
    return pytest.mark.skipif(
        bool(missing),
        reason=f"FEMFFUSION file(s) not found under {_root()}: {', '.join(missing)} "
               "(set FEMFFUSION_DIR)")
