"""Compatibility import for historical LRDC2F experiment code."""
from .lrdc2f_assigner import LRDC2FAssigner

DCFLAssigner = LRDC2FAssigner

__all__ = ['DCFLAssigner']
