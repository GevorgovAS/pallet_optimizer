from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Box:
    instance_id: str
    content_id: str
    width: float
    length: float
    height: float
    mass: float | None = None


@dataclass(frozen=True)
class Pallet:
    width: float
    length: float
    max_height: float


@dataclass(frozen=True)
class Placement:
    instance_id: str
    content_id: str
    layer: int
    x: float
    y: float
    z: float
    rotation: int
    width: float
    length: float
    height: float
