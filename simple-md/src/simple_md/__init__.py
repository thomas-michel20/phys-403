# -*- coding: utf-8 -*-
"""Simple Molecular Dynamics."""

from . import integrator, postprocess, potential, structure, system

__all__ = ('structure', 'potential', 'system', 'integrator', 'postprocess')

__version__ = '0.1.0'
__authros__ = 'Austin Zadoks'


def get_version() -> str:
    """Current version string.

    Returns:
        str: Current version string.
    """
    return __version__
