"""Heuristic terrain-map image check (OpenCV, offline)."""

__all__ = ["is_terrain_map", "is_terrain_map_with_reason"]


def __getattr__(name: str):
    if name == "is_terrain_map":
        from .detector import is_terrain_map as fn

        return fn
    if name == "is_terrain_map_with_reason":
        from .detector import is_terrain_map_with_reason as fn2

        return fn2
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
