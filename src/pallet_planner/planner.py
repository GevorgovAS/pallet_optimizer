from __future__ import annotations

from dataclasses import dataclass
from math import floor

from .models import Box, Pallet, Placement


@dataclass(frozen=True)
class PlannerConfig:
    rotations: tuple[int, ...] = (0, 90)
    gap: float = 0.01
    edge_margin: float = 0.0
    alternate_layers: bool = True
    compact_incomplete_layers: bool = False
    max_layers: int | None = None


@dataclass(frozen=True)
class _Cell:
    x: float
    y: float
    rotation: int
    width: float
    length: float


class PalletPlanner:
    def __init__(self, config: PlannerConfig | None = None):
        self.config = config or PlannerConfig()

    def plan(self, pallet: Pallet, boxes: list[Box]) -> list[Placement]:
        groups: dict[str, list[Box]] = {}
        for box in boxes:
            groups.setdefault(box.content_id, []).append(box)

        placements: list[Placement] = []
        layer = 0
        z_base = 0.0
        used: set[str] = set()

        ordered_groups = sorted(
            groups.items(),
            key=lambda item: -max(box.width * box.length for box in item[1]),
        )
        for content_id, group in ordered_groups:
            if self.config.max_layers is not None and layer >= self.config.max_layers:
                break
            remaining = list(group)
            while remaining:
                if self.config.max_layers is not None and layer >= self.config.max_layers:
                    break
                box = remaining[0]
                if box.instance_id in used:
                    raise ValueError(f"duplicate instance_id: {box.instance_id}")
                if z_base + box.height > pallet.max_height + 1e-9:
                    break

                cells = self._layout_cells(pallet, box, layer, len(remaining))
                if not cells:
                    remaining.pop(0)
                    continue

                layer_boxes = remaining[: len(cells)]
                if self.config.compact_incomplete_layers and len(layer_boxes) < len(cells):
                    cells = self._compact_cells(cells, len(layer_boxes))
                for item, cell in zip(layer_boxes, cells):
                    if item.instance_id in used:
                        raise ValueError(f"duplicate instance_id: {item.instance_id}")
                    used.add(item.instance_id)
                    placements.append(
                        Placement(
                            instance_id=item.instance_id,
                            content_id=content_id,
                            layer=layer,
                            x=cell.x,
                            y=cell.y,
                            z=z_base + item.height / 2.0,
                            rotation=cell.rotation,
                            width=cell.width,
                            length=cell.length,
                            height=item.height,
                        )
                    )

                remaining = remaining[len(layer_boxes) :]
                z_base += box.height
                layer += 1

        return placements

    def _layout_cells(
        self,
        pallet: Pallet,
        box: Box,
        layer: int,
        needed: int,
    ) -> list[_Cell]:
        preferred = 90 if layer % 2 else 0
        layouts = [
            self._grid(
                pallet,
                box,
                rotation,
                stagger=(layer % 2 == 1 and box.width == box.length),
            )
            for rotation in self.config.rotations
        ]
        layouts = [layout for layout in layouts if layout]
        if not layouts:
            return []

        best_count = max(len(layout) for layout in layouts)
        required = min(needed, best_count)
        if self.config.alternate_layers and layer > 0 and len(layouts) > 1:
            for layout in layouts:
                if layout[0].rotation == preferred and len(layout) >= required:
                    return layout
        return max(layouts, key=len)

    @staticmethod
    def _compact_cells(cells: list[_Cell], count: int) -> list[_Cell]:
        rows: dict[float, list[_Cell]] = {}
        for cell in cells:
            rows.setdefault(cell.y, []).append(cell)
        columns = max(len(row) for row in rows.values())
        row_count = min(len(rows), (count + columns - 1) // columns)
        selected_rows = sorted(
            sorted(rows, key=lambda y: abs(y))[:row_count]
        )
        base, extra = divmod(count, row_count)
        result = []
        for index, y in enumerate(selected_rows):
            take = base + (1 if index < extra else 0)
            centered = sorted(rows[y], key=lambda cell: abs(cell.x))[:take]
            result.extend(sorted(centered, key=lambda cell: cell.x))
        return result

    def _grid(self, pallet: Pallet, box: Box, rotation: int, stagger: bool = False) -> list[_Cell]:
        width, length = (box.width, box.length) if rotation == 0 else (box.length, box.width)
        gap = self.config.gap
        edge_margin = self.config.edge_margin
        usable_width = pallet.width - 2.0 * edge_margin
        usable_length = pallet.length - 2.0 * edge_margin
        if width > usable_width + 1e-9 or length > usable_length + 1e-9:
            return []

        columns = floor((usable_width + gap + 1e-9) / (width + gap))
        rows = floor((usable_length + gap + 1e-9) / (length + gap))
        cells: list[_Cell] = []
        x0 = -pallet.width / 2.0 + edge_margin + width / 2.0
        y0 = -pallet.length / 2.0 + edge_margin + length / 2.0
        for row in range(rows):
            row_offset = (width + gap) / 2.0 if stagger and row % 2 else 0.0
            for column in range(columns):
                x = x0 + column * (width + gap) + row_offset
                if x + width / 2.0 > pallet.width / 2.0 - edge_margin + 1e-9:
                    continue
                cells.append(
                    _Cell(
                        x=x,
                        y=y0 + row * (length + gap),
                        rotation=rotation,
                        width=width,
                        length=length,
                    )
                )
        return cells
