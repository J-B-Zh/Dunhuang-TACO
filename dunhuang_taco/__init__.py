"""Dunhuang-TACO reference implementation."""

from .model import DunhuangTACO

__all__ = ["DunhuangTACO", "DinoV2Retriever"]


def __getattr__(name: str):
    """Load the DINO-v2 stack only when retrieval is actually requested."""
    if name == "DinoV2Retriever":
        from .retrieval import DinoV2Retriever

        return DinoV2Retriever
    raise AttributeError(name)
