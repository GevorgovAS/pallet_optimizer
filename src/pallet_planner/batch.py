"""Offline batch packing with geometric support checks. All dimensions are metres."""
from dataclasses import dataclass, field, replace
from functools import lru_cache
from itertools import combinations
from math import ceil, floor, isfinite
from time import perf_counter

from .models import Box, Pallet, Placement

EPS = 1e-8


def bounds(box):
    return (box.x - box.width / 2, box.y - box.length / 2,
            box.x + box.width / 2, box.y + box.length / 2)


def intersection(a, b):
    left, bottom = max(a[0], b[0]), max(a[1], b[1])
    right, top = min(a[2], b[2]), min(a[3], b[3])
    return (left, bottom, right, top) if right > left + EPS and top > bottom + EPS else None


def hull(points):
    points = sorted(set(points))
    def cross(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
    halves = []
    for sequence in (points, list(reversed(points))):
        half = []
        for point in sequence:
            while len(half) >= 2 and cross(half[-2], half[-1], point) <= 0:
                half.pop()
            half.append(point)
        halves.append(half[:-1])
    return halves[0] + halves[1]


def support(box, below):
    """Area fraction and centre inside the convex hull of actual contact patches."""
    return _support_geometry(box.x, box.y, box.width, box.length,
                             tuple(bounds(other) for other in below))


@lru_cache(maxsize=32768)
def _support_geometry(x, y, width, length, below, minimum=0.0):
    rect = (x-width/2, y-length/2, x+width/2, y+length/2)
    contacts = [intersection(rect, other) for other in below]
    contacts = [rect for rect in contacts if rect is not None]
    area = sum((r[2]-r[0])*(r[3]-r[1]) for r in contacts)
    fraction = min(area / (width*length), 1.0)
    if fraction+EPS < minimum:
        return fraction, False
    polygon = hull([(x, y) for a, b, c, d in contacts for x in (a, c) for y in (b, d)])
    inside = len(polygon) >= 3 and all(
        (b[0]-a[0])*(y-a[1]) - (b[1]-a[1])*(x-a[0]) >= -EPS
        for a, b in zip(polygon, polygon[1:] + polygon[:1])
    )
    return fraction, inside


def seam_overlap(layer, below, gap=0.005):
    """Coincident internal joints, excluding outside edges of the stack."""
    if not layer or not below:
        return 0.0
    def joints(placements):
        segments = []
        for index, p in enumerate(placements):
            a, b, c, d = bounds(p)
            for other in placements[index+1:]:
                e, f, g, h = bounds(other)
                for x, u in ((c, e), (g, a)):
                    if -EPS <= u-x <= gap+EPS and min(d,h) > max(b,f)+EPS:
                        segments.append(('v', (x+u)/2, max(b,f), min(d,h)))
                for y, v in ((d, f), (h, b)):
                    if -EPS <= v-y <= gap+EPS and min(c,g) > max(a,e)+EPS:
                        segments.append(('h', (y+v)/2, max(a,e), min(c,g)))
        return segments
    total = 0.0
    lower_joints = joints(below)
    for axis, coordinate, start, end in joints(layer):
        for lower_axis, lower_coordinate, lower_start, lower_end in lower_joints:
            if axis == lower_axis and abs(coordinate-lower_coordinate) < 1e-5:
                total += max(0.0, min(end, lower_end)-max(start, lower_start))
    return total


def bonding(layer, below, gap=0.005):
    if not below:
        return {'status': 'base', 'aligned_fraction': 0.0}
    footprint = lambda boxes: sorted(tuple(round(v, 7) for v in bounds(p)) for p in boxes)
    repeated = footprint(layer) == footprint(below)
    joints = seam_overlap(layer, layer, gap)
    aligned = seam_overlap(layer, below, gap)
    fraction = min(aligned / joints, 1.0) if joints > EPS else 0.0
    return {'status': 'bonded' if not repeated and fraction <= 0.5+EPS else 'limited',
            'aligned_fraction': fraction}


@dataclass
class BatchResult:
    pallets: list[list[Placement]]
    unplaced: list[dict]
    deferred: list[dict] = field(default_factory=list)
    strategy: str = 'volume_height'
    added: list[Box] = field(default_factory=list)
    baseline_pallets: int | None = None
    addition_trials: int = 0
    search_limited: bool = False
    set_aside: list[dict] = field(default_factory=list)
    min_layer_fill: float = 0.0
    allow_partial_layers: bool = False


class BatchPlanner:
    def __init__(self, gap=0.005, edge_margin=0.01, min_support=0.85, progress=None,
                 min_layer_fill=0.0, ship_all=False):
        if not all(isfinite(v) for v in (gap, edge_margin, min_support)):
            raise ValueError("Non-finite planner settings")
        if gap < 0 or edge_margin < 0 or not 0 < min_support <= 1:
            raise ValueError("Invalid gap, margin or support threshold")
        if isinstance(min_layer_fill, bool) or not isfinite(min_layer_fill) or not 0 <= min_layer_fill <= 1:
            raise ValueError('min_layer_fill must be between 0 and 1')
        self.gap, self.edge_margin, self.min_support = gap, edge_margin, min_support
        self.min_layer_fill = min_layer_fill
        self.ship_all = ship_all
        self._delivery_addition_budget = 0
        self._delivery_addition_skus = None
        self.progress = progress
        self._deadline = None

    def _timed_out(self):
        return self._deadline is not None and perf_counter() >= self._deadline

    def plan(self, pallet: Pallet, boxes: list[Box], max_leftovers=0, max_additions=0,
             max_addition_area=None, time_limit=0) -> BatchResult:
        if isinstance(time_limit, bool) or not isfinite(time_limit) or time_limit < 0:
            raise ValueError('time_limit must be a non-negative number of seconds')
        self._deadline = perf_counter()+time_limit if time_limit else None
        if max_additions is not None and (isinstance(max_additions, bool) or not isinstance(max_additions, int) or max_additions < 0):
            raise ValueError('max_additions must be a non-negative integer')
        if max_addition_area is not None and (isinstance(max_addition_area, bool)
                or not isfinite(max_addition_area) or not 0 <= max_addition_area <= 1):
            raise ValueError('max_addition_area must be a fraction between 0 and 1')
        if max_additions is None and max_addition_area is None and not self.ship_all:
            raise ValueError('An addition count or area limit is required')
        self._delivery_addition_budget = 0 if self.ship_all else max_additions
        self._delivery_addition_skus = None
        baseline = self._plan(pallet, boxes, 0 if self.ship_all else max_leftovers)
        if self.ship_all:
            best = baseline
            if (max_additions != 0 and self.min_layer_fill and not baseline.unplaced
                    and not self._timed_out()):
                if self.progress:
                    self.progress('Full delivery: comparing pallet counts with added boxes')
                self._delivery_addition_budget = max_additions
                trial = self._plan(pallet, boxes, 0)
                if self._full_delivery_score(pallet, trial) < self._full_delivery_score(pallet, best):
                    best = trial
                selected = {b.content_id for b in best.added}
                for sku in sorted({b.content_id for b in trial.added}):
                    if self._timed_out():
                        break
                    if self.progress:
                        self.progress(f'Full delivery: optimizing additions for {sku}; best {len(best.pallets)} pallets')
                    self._delivery_addition_skus = selected ^ {sku}
                    proposal = self._plan(pallet, boxes, 0)
                    if self._full_delivery_score(pallet, proposal) < self._full_delivery_score(pallet, best):
                        best = proposal
                        selected = {b.content_id for b in best.added}
                self._delivery_addition_skus = None
            best.baseline_pallets = len(baseline.pallets)
            best.search_limited = self._timed_out()
            return best
        reference = self._apply_fill_policy(pallet, boxes, baseline)
        baseline.baseline_pallets = len(reference.pallets)
        result = baseline
        if (max_additions != 0 and max_addition_area != 0 and len(reference.pallets) > 1
                and not baseline.unplaced and not self._timed_out()):
            result = self._try_additions(pallet, boxes, baseline, max_leftovers, max_additions, max_addition_area)
        result.search_limited = self._timed_out()
        result = self._apply_fill_policy(pallet, boxes+result.added, result)
        result.baseline_pallets = len(reference.pallets)
        return result

    def _apply_fill_policy(self, pallet, boxes, result):
        if not self.min_layer_fill or result.min_layer_fill == self.min_layer_fill:
            return result
        inventory = {b.instance_id: b for b in boxes}
        available = {p.instance_id for plan in result.pallets for p in plan}
        available.update(item['instance_id'] for item in result.set_aside)
        groups = {}
        for box in boxes:
            if box.instance_id in available:
                groups.setdefault(box.content_id, []).append(box)
        pallets, separate, added = [], [], list(result.added)
        used_ids = set(inventory)
        area = pallet.width*pallet.length
        ordered = sorted(groups.values(), key=lambda g: (-g[0].width*g[0].length,
                                                         -g[0].height, -len(g), g[0].content_id))
        for group in ordered:
            box = group[0]
            box_area = box.width*box.length
            bound = max(1, floor((area+EPS)/box_area))
            capacity = len(self._layer(pallet, box, bound, 0, 0.0, []))
            minimum = max(1, ceil((self.min_layer_fill*area-EPS)/box_area))
            if (self.ship_all and capacity >= minimum and
                    (self._delivery_addition_skus is None or box.content_id in self._delivery_addition_skus)):
                missing = max(0, ceil(len(group)/capacity)*minimum-len(group))
                budget = self._delivery_addition_budget
                if missing and (budget is None or len(added)+missing <= budget):
                    for _ in range(missing):
                        serial = len(added)+1
                        instance_id = f'added_{serial:05d}'
                        while instance_id in used_ids:
                            serial += 1
                            instance_id = f'added_{serial:05d}'
                        extra = replace(box, instance_id=instance_id)
                        used_ids.add(instance_id)
                        inventory[instance_id] = extra
                        group.append(extra)
                        added.append(extra)
            # Maximize shipped boxes first, then use the fewest feasible layers.
            shipped = (len(group) if self.ship_all else
                       min(len(group), (len(group)//minimum)*capacity) if capacity >= minimum else 0)
            layers = ceil(shipped/capacity) if shipped else 0
            if self.ship_all and (capacity < minimum or layers*minimum > shipped):
                minimum = 1
            remaining, offset = shipped, 0
            while layers:
                count = min(capacity, remaining-minimum*(layers-1))
                target, cells = self._find_space(pallet, pallets, box, count, full=True)
                if target is None:
                    target = []
                    pallets.append(target)
                    cells = self._layer(pallet, box, count, 0, 0.0, [])
                if len(cells) != count:
                    raise RuntimeError('Cannot reproduce a feasible layer on an empty pallet')
                self._append_layer(target, group[offset:offset+count], cells)
                remaining, offset, layers = remaining-count, offset+count, layers-1
            for leftover in group[shipped:]:
                separate.append({'instance_id': leftover.instance_id, 'content_id': leftover.content_id,
                                 'reason': ('Maximum candidate layer area is below the minimum'
                                            if capacity < minimum else 'Insufficient boxes for another valid layer'),
                                 'max_layer_fill': capacity*box_area/area,
                                 'minimum_count': minimum})
        pallets, _ = self._consolidate(pallet, pallets, inventory, 0)
        result = replace(result, pallets=pallets, set_aside=separate, added=added,
                         min_layer_fill=self.min_layer_fill, allow_partial_layers=self.ship_all)
        if self.ship_all:
            while True:
                cleaned = self._prune_unused_additions(result)
                if len(cleaned.added) == len(result.added):
                    return cleaned
                pallets, _ = self._consolidate(pallet, cleaned.pallets, inventory, 0)
                result = replace(cleaned, pallets=pallets)
        return result

    @staticmethod
    def _prune_unused_additions(result):
        added_ids = {b.instance_id for b in result.added}
        if not added_ids:
            return result
        needed, pallets = set(), []
        for plan in result.pallets:
            layers = {}
            for p in plan:
                layers.setdefault(p.layer, []).append(p)
                if p.instance_id not in added_ids:
                    needed.add(p.instance_id)
            # Keep every original and all of its contact-support ancestors.
            for index in sorted(layers, reverse=True):
                for upper in layers[index]:
                    if upper.instance_id in needed:
                        needed.update(lower.instance_id for lower in layers.get(index-1, [])
                                      if intersection(bounds(upper), bounds(lower)) is not None)
            kept = [p for p in plan if p.instance_id in needed]
            if kept:
                pallets.append(kept)
        return replace(result, pallets=pallets,
                       added=[b for b in result.added if b.instance_id in needed])

    def _try_additions(self, pallet, boxes, baseline, max_leftovers, budget, area_limit):
        reference = self._apply_fill_policy(pallet, boxes, baseline)
        groups = {}
        for box in boxes:
            groups.setdefault(box.content_id, []).append(box)
        deficits = []
        for group in groups.values():
            box = group[0]
            bound = max(1, floor((pallet.width*pallet.length+EPS)/(box.width*box.length)))
            capacity = len(self._layer(pallet, box, bound, 0, 0.0, []))
            missing = (-len(group)) % capacity if capacity else 0
            if self.min_layer_fill and capacity:
                tail = sum(item['content_id'] == box.content_id for item in reference.set_aside)
                fraction = tail*box.width*box.length/(pallet.width*pallet.length)
                if fraction <= .5+EPS or fraction+EPS >= self.min_layer_fill:
                    continue
                missing = ceil((self.min_layer_fill*pallet.width*pallet.length-EPS)
                               /(box.width*box.length))-tail
                if tail+missing > capacity:
                    continue
            if (missing > 0 and (budget is None or missing <= budget)
                    and (area_limit is None or missing*box.width*box.length
                         <= area_limit*pallet.width*pallet.length+EPS)):
                deficits.append((box, missing))
        candidates = [(item,) for item in deficits]
        candidates.extend(combinations(deficits, 2))
        if len(deficits) > 2:
            candidates.append(tuple(deficits))
        candidates = [items for items in candidates
                      if budget is None or sum(amount for _, amount in items) <= budget]
        candidates.sort(key=lambda items: (sum(amount for _, amount in items),
                                           -sum(box.width*box.length for box, _ in items),
                                           tuple(box.content_id for box, _ in items)))
        best, trials = baseline, 0
        original_ids = {box.instance_id for box in boxes}
        # Bound trial count; candidates use the same geometric constraints as the baseline.
        for candidate_index, items in enumerate(candidates[:24], 1):
            if self._timed_out():
                break
            if self.progress:
                self.progress(f'Extra boxes: trial {candidate_index}/{min(len(candidates), 24)}; best {len(best.pallets)} pallets')
            added, used_ids = [], set(original_ids)
            for template, amount in items:
                for _ in range(amount):
                    serial = len(added)+1
                    instance_id = f'added_{serial:05d}'
                    while instance_id in used_ids:
                        serial += 1
                        instance_id = f'added_{serial:05d}'
                    used_ids.add(instance_id)
                    added.append(replace(template, instance_id=instance_id))
            trial = self._plan(pallet, boxes+added, max_leftovers, consolidate=False)
            trials += 1
            if trial.unplaced:
                continue
            added_ids = {box.instance_id for box in added}
            if added_ids & {item['instance_id'] for item in trial.deferred}:
                continue
            if not self.min_layer_fill:
                trial.pallets, trial.deferred = self._consolidate(
                    pallet, trial.pallets, {box.instance_id: box for box in boxes+added}, max_leftovers)
            if added_ids & {item['instance_id'] for item in trial.deferred}:
                continue
            if self.min_layer_fill:
                checked = self._apply_fill_policy(pallet, boxes+added, trial)
                if (added_ids & {item['instance_id'] for item in checked.set_aside}
                        or len(checked.set_aside) > len(reference.set_aside)
                        or len(checked.pallets) >= len(reference.pallets)):
                    continue
                trial = checked
                trial.added = added
                if self._delivery_score(trial) < self._delivery_score(best):
                    best = trial
                continue
            if (len(trial.pallets), len(added), len(trial.deferred),
                    *self._packing_score(trial.pallets, trial.deferred)[2:]) < (
                    len(best.pallets), len(best.added), len(best.deferred),
                    *self._packing_score(best.pallets, best.deferred)[2:]):
                trial.added = added
                best = trial
        best.baseline_pallets = len(baseline.pallets)
        best.addition_trials = trials
        return best

    def _delivery_score(self, result):
        return (len(result.unplaced)+len(result.deferred)+len(result.set_aside),
                len(result.pallets), len(result.added), *self._packing_score(result.pallets, result.deferred)[2:])

    def _full_delivery_score(self, pallet, result):
        partial = sum(sum(p.width*p.length for p in plan if p.layer == index)+EPS
                      < self.min_layer_fill*pallet.width*pallet.length
                      for plan in result.pallets for index in {p.layer for p in plan})
        return (len(result.unplaced)+len(result.deferred)+len(result.set_aside),
                len(result.pallets), partial, len(result.added),
                *self._packing_score(result.pallets, result.deferred)[2:])

    def _plan(self, pallet: Pallet, boxes: list[Box], max_leftovers=0, consolidate=True) -> BatchResult:
        if isinstance(max_leftovers, bool) or not isinstance(max_leftovers, int) or max_leftovers < 0:
            raise ValueError('max_leftovers must be a non-negative integer')
        if not all(isfinite(v) and v > 0 for v in (pallet.width, pallet.length, pallet.max_height)):
            raise ValueError("Invalid pallet dimensions")
        groups, seen = {}, set()
        for box in boxes:
            if not box.instance_id or not box.content_id or box.instance_id in seen:
                raise ValueError(f"Empty or duplicate ID: {box.instance_id}")
            seen.add(box.instance_id)
            if not all(isfinite(v) and v > 0 for v in (box.width, box.length, box.height)):
                raise ValueError(f"Invalid dimensions: {box.instance_id}")
            group = groups.setdefault(box.content_id, [])
            if group and (box.width, box.length, box.height) != (group[0].width, group[0].length, group[0].height):
                raise ValueError(f"SKU {box.content_id} has different packaging dimensions")
            group.append(box)
        groups = sorted(groups.values(), key=lambda g: (-g[0].width*g[0].length, -g[0].height, g[0].content_id))
        pallets, unplaced, pools = [], [], []
        for group in groups:
            template = group[0]
            area_bound = max(1, floor((pallet.width*pallet.length+EPS)/(template.width*template.length)))
            empty = self._layer(pallet, template, area_bound, 0, 0.0, [])
            capacity = len(empty)
            if not capacity or template.height > pallet.max_height + EPS:
                unplaced.extend({'instance_id': b.instance_id, 'reason': 'Does not fit an empty pallet'} for b in group)
                continue
            pools.append({'boxes': list(group), 'capacity': capacity})
        if self.min_layer_fill:
            pending = [{'instance_id': b.instance_id, 'content_id': b.content_id}
                       for pool in pools for b in pool['boxes']]
            seed = BatchResult([], unplaced, strategy='delivery_first', set_aside=pending)
            return self._apply_fill_policy(pallet, boxes, seed)
        alternative_pools = [{'boxes': list(pool['boxes']), 'capacity': pool['capacity']} for pool in pools]
        # Choose a combination from the whole inventory for each pallet, not a SKU queue.
        while any(len(pool['boxes']) >= pool['capacity'] for pool in pools):
            if self.progress:
                self.progress(f'Building pallet {len(pallets)+1}')
            selected = self._select_full_layers(pallet, pools)
            target, below, z = [], [], 0.0
            for pool_index in sorted(selected):
                pool = pools[pool_index]
                remaining, capacity = pool['boxes'], pool['capacity']
                box = remaining[0]
                if z+box.height > pallet.max_height+EPS:
                    continue
                cells = self._layer(pallet, box, capacity,
                                    below[0].layer+1 if below else 0, z, below)
                if len(cells) < capacity:
                    continue
                self._append_layer(target, remaining[:capacity], cells)
                del remaining[:capacity]
                below, z = target[-capacity:], z+box.height
            if not target:
                raise RuntimeError('Full layer selection made no progress')
            # Fill geometric/height gaps left when a selected layer lacked full support.
            while True:
                candidates = []
                for pool_index, pool in enumerate(pools):
                    remaining, capacity = pool['boxes'], pool['capacity']
                    if len(remaining) < capacity:
                        continue
                    box = remaining[0]
                    if box.width*box.length > below[0].width*below[0].length+EPS or z+box.height > pallet.max_height+EPS:
                        continue
                    cells = self._layer(pallet, box, capacity, below[0].layer+1, z, below)
                    if len(cells) == capacity:
                        candidates.append((capacity*box.width*box.length*box.height, pool_index, cells))
                if not candidates:
                    break
                _, pool_index, cells = max(candidates, key=lambda candidate: candidate[:2])
                pool = pools[pool_index]
                box, capacity = pool['boxes'][0], pool['capacity']
                self._append_layer(target, pool['boxes'][:capacity], cells)
                del pool['boxes'][:capacity]
                below, z = target[-capacity:], z+box.height
            pallets.append(target)
        self._finish_tails(pallet, pallets, pools)
        inventory = {b.instance_id: b for b in boxes}
        pallets, deferred = self._consolidate(pallet, pallets, inventory, max_leftovers) if consolidate else (pallets, [])
        strategy = 'volume_height'
        if consolidate and not self._timed_out():
            alternative = self._allocate_by_size(pallet, alternative_pools)
            self._finish_tails(pallet, alternative, alternative_pools)
            alternative, alt_deferred = self._consolidate(pallet, alternative, inventory, max_leftovers) if consolidate else (alternative, [])
            if self._packing_score(alternative, alt_deferred) < self._packing_score(pallets, deferred):
                pallets, deferred = alternative, alt_deferred
                strategy = 'best_fit_by_size'
        return BatchResult(pallets, unplaced, deferred, strategy)

    @staticmethod
    def _packing_score(pallets, deferred):
        heights = [max(p.z+p.height/2 for p in plan) for plan in pallets]
        return (len(pallets), len(deferred), round(sum(heights), 9), max(heights, default=0))

    def _allocate_by_size(self, pallet, pools):
        # Compare a size-first packing even when every original box must be retained.
        pallets = []
        for pool in pools:
            remaining, capacity = pool['boxes'], pool['capacity']
            while len(remaining) >= capacity:
                box = remaining[0]
                target, cells = self._find_space(pallet, pallets, box, capacity, full=True)
                if target is None:
                    target = []
                    pallets.append(target)
                    cells = self._layer(pallet, box, capacity, 0, 0.0, [])
                self._append_layer(target, remaining[:capacity], cells)
                del remaining[:capacity]
        return pallets

    def _finish_tails(self, pallet, pallets, pools):
        # Only after every full layer is allocated may a partial layer cap a stack.
        for pool in pools:
            remaining = pool['boxes']
            while remaining:
                target, cells = self._find_space(pallet, pallets, remaining[0], len(remaining), full=False)
                if target is None:
                    target = []
                    pallets.append(target)
                    cells = self._layer(pallet, remaining[0], len(remaining), 0, 0.0, [])
                count = len(cells)
                self._append_layer(target, remaining[:count], cells)
                remaining = remaining[count:]

    @staticmethod
    def _select_full_layers(pallet, pools):
        # Bounded height knapsack. Geometry is checked when selected layers are placed.
        states = {0.0: (0.0, ())}
        for index, pool in enumerate(pools):
            boxes, capacity = pool['boxes'], pool['capacity']
            if len(boxes) < capacity:
                continue
            box = boxes[0]
            layer_volume = capacity*box.width*box.length*box.height
            count = min(len(boxes)//capacity, floor((pallet.max_height+EPS)/box.height))
            for _ in range(count):
                for height, (volume, selected) in list(states.items()):
                    next_height = round(height+box.height, 9)
                    if next_height > pallet.max_height+EPS:
                        continue
                    candidate = volume+layer_volume, selected+(index,)
                    if next_height not in states or candidate[0] > states[next_height][0]+EPS:
                        states[next_height] = candidate
        return max(states.values(), key=lambda value: value[0])[1]

    def _single_pallet(self, pallet, boxes):
        identities = {}
        for box in boxes:
            identities.setdefault(box.content_id, []).append(box.instance_id)
        geometry = tuple(sorted((replace(box, instance_id='', mass=None) for box in boxes),
                                key=lambda box: box.content_id))
        packed = self._single_pallet_geometry(pallet, geometry)
        if packed is None:
            return None
        identities = {sku: iter(ids) for sku, ids in identities.items()}
        return [replace(p, instance_id=next(identities[p.content_id])) for p in packed]

    @lru_cache(maxsize=4096)
    def _single_pallet_geometry(self, pallet, boxes):
        groups = {}
        for box in boxes:
            groups.setdefault(box.content_id, []).append(box)
        ordered = sorted(groups.values(), key=lambda g: (-g[0].width*g[0].length, -g[0].height,
                                                         -len(g), g[0].content_id))
        minimum_height = 0.0
        for group in ordered:
            box = group[0]
            bound = max(1, floor((pallet.width*pallet.length+EPS)/(box.width*box.length)))
            capacity = len(self._layer(pallet, box, bound, 0, 0.0, []))
            if not capacity:
                return None
            minimum_height += ((len(group)+capacity-1)//capacity)*box.height
        if minimum_height > pallet.max_height+EPS:
            return None
        result, below, z = [], [], 0.0
        for group in ordered:
            remaining = list(group)
            if self.min_layer_fill:
                box = remaining[0]
                bound = max(1, floor((pallet.width*pallet.length+EPS)/(box.width*box.length)))
                capacity = len(self._layer(pallet, box, bound, 0, 0.0, []))
                minimum = max(1, ceil((self.min_layer_fill*pallet.width*pallet.length-EPS)
                                      /(box.width*box.length)))
                layers = ceil(len(remaining)/capacity)
                if layers*minimum > len(remaining):
                    if not self.ship_all:
                        return None
                    minimum = 1
            while remaining:
                if z+remaining[0].height > pallet.max_height+EPS:
                    return None
                index = below[0].layer+1 if below else 0
                count = (min(capacity, len(remaining)-minimum*(layers-1))
                         if self.min_layer_fill else len(remaining))
                cells = self._layer(pallet, remaining[0], count, index, z, below)
                if not cells:
                    return None
                if self.min_layer_fill and len(cells) != count:
                    return None
                if not self.ship_all and len(cells)*remaining[0].width*remaining[0].length+EPS < (
                        self.min_layer_fill*pallet.width*pallet.length):
                    return None
                before = len(result)
                self._append_layer(result, remaining[:len(cells)], cells)
                below = result[before:]
                z += remaining[0].height
                remaining = remaining[len(cells):]
                if self.min_layer_fill:
                    layers -= 1
        return result

    def _consolidate(self, pallet, pallets, inventory, max_leftovers):
        volume = lambda plan: sum(p.width*p.length*p.height for p in plan)
        deferred = []
        # Trial moves are transactional: a donor disappears only if every box found a home.
        changed = True
        while changed:
            if self._timed_out():
                return pallets, deferred
            changed = False
            for donor in sorted(range(len(pallets)), key=lambda i: volume(pallets[i])):
                # Concentrate the expensive repacking search on underfilled stacks.
                if (not self.ship_all and
                        volume(pallets[donor]) >= pallet.width*pallet.length*pallet.max_height/2):
                    continue
                if self.progress:
                    self.progress(f'Repacking pallet {donor+1}/{len(pallets)}')
                trial = [list(plan) for i, plan in enumerate(pallets) if i != donor]
                moved = sorted(pallets[donor], key=lambda p: (-p.width*p.length, -p.height, p.instance_id))
                # Move SKU bundles before individual boxes: intermediate partial layers
                # may not support the upper stack although the completed bundle does.
                groups = {}
                for placement in moved:
                    groups.setdefault(placement.content_id, []).append(inventory[placement.instance_id])
                relocated = set()
                for group in groups.values():
                    remaining = list(group)
                    for target in sorted(range(len(trial)), key=lambda i: -volume(trial[i])):
                        existing = [inventory[p.instance_id] for p in trial[target]]
                        for count in range(len(remaining), 1, -1):
                            if self._timed_out():
                                return pallets, deferred
                            packed = self._single_pallet(pallet, existing+remaining[:count])
                            if packed is not None:
                                trial[target] = packed
                                relocated.update(b.instance_id for b in remaining[:count])
                                remaining = remaining[count:]
                                break
                        if not remaining:
                            break
                moved = [p for p in moved if p.instance_id not in relocated]
                success = True
                trial_deferred = []
                for placement in moved:
                    box = inventory[placement.instance_id]
                    fitted = False
                    for target in sorted(range(len(trial)), key=lambda i: -volume(trial[i])):
                        if self._timed_out():
                            return pallets, deferred
                        if volume(trial[target])+box.width*box.length*box.height > pallet.width*pallet.length*pallet.max_height+EPS:
                            continue
                        candidates = [inventory[p.instance_id] for p in trial[target]] + [box]
                        packed = self._single_pallet(pallet, candidates)
                        if packed is not None:
                            trial[target] = packed
                            fitted = True
                            break
                    if not fitted:
                        if len(deferred)+len(trial_deferred) < max_leftovers:
                            trial_deferred.append({'instance_id': box.instance_id,
                                'content_id': box.content_id,
                                'reason': 'Deferred within explicit leftover budget to eliminate a pallet'})
                        else:
                            success = False
                            break
                if success:
                    pallets, changed = trial, True
                    deferred.extend(trial_deferred)
                    break
        return pallets, deferred

    @staticmethod
    def _append_layer(target, boxes, cells):
        target.extend(replace(cell, instance_id=box.instance_id, content_id=box.content_id)
                      for box, cell in zip(boxes, cells))

    def _find_space(self, pallet, pallets, box, count, full):
        best_target, best_cells, best_score = None, [], None
        for target in pallets:
            last_index = target[-1].layer
            below = [p for p in target if p.layer == last_index]
            z = below[0].z + below[0].height/2
            if z+box.height > pallet.max_height+EPS:
                continue
            if box.width*box.length > below[0].width*below[0].length+EPS:
                continue
            cells = self._layer(pallet, box, count, last_index+1, z, below)
            if not cells or (full and len(cells) < count):
                continue
            # Best fit in remaining height preserves taller free slots for later SKUs.
            score = (len(cells), z+box.height,
                     -seam_overlap(cells, below, self.gap))
            if best_score is None or score > best_score:
                best_target, best_cells, best_score = target, cells, score
        return best_target, best_cells

    def _layer(self, pallet, box, count, index, z, below):
        geometry = tuple(replace(p, instance_id='', content_id='', layer=0, z=0, height=0) for p in below)
        template = replace(box, instance_id='', mass=None)
        count = min(count, max(1, floor((pallet.width*pallet.length+EPS)/(box.width*box.length))))
        cells = self._layer_geometry(pallet, template, count, geometry)
        return [replace(p, layer=index, z=z+box.height/2) for p in cells]

    @lru_cache(maxsize=1024)
    def _layer_geometry(self, pallet, box, count, below):
        index, z = 0, 0.0
        best, best_score = [], None
        usable_w, usable_l = pallet.width-2*self.edge_margin, pallet.length-2*self.edge_margin
        grids = []
        for rotation in (0, 90):
            w, l = (box.width, box.length) if rotation == 0 else (box.length, box.width)
            nx = floor((usable_w+self.gap+EPS)/(w+self.gap))
            ny = floor((usable_l+self.gap+EPS)/(l+self.gap))
            if nx < 1 or ny < 1:
                continue
            # Search centred and half-pitch shifted grids; reduce one track to permit bonding.
            for columns, rows in sorted({(nx, ny), (max(nx-1, 1), ny), (nx, max(ny-1, 1))}):
                for dx, dy in ((0, 0), ((w+self.gap)/2, 0), (-(w+self.gap)/2, 0),
                               (0, (l+self.gap)/2), (0, -(l+self.gap)/2)):
                    cells = []
                    for row in range(rows):
                        for column in range(columns):
                            cell = Placement("", box.content_id, index,
                                (column-(columns-1)/2)*(w+self.gap)+dx,
                                (row-(rows-1)/2)*(l+self.gap)+dy,
                                z+box.height/2, rotation, w, l, box.height)
                            if abs(cell.x)+w/2 > usable_w/2+EPS or abs(cell.y)+l/2 > usable_l/2+EPS:
                                continue
                            cells.append(cell)
                    grids.append(cells)
        # Mixed row orientations often fill the residual strip of rectangular pallets.
        for first_rows in range(1, max(1, floor(usable_l/(box.length+self.gap)))+1):
            for second_rows in range(1, max(1, floor(usable_l/(box.width+self.gap)))+1):
                height = first_rows*box.length + second_rows*box.width + (first_rows+second_rows-1)*self.gap
                if height > usable_l+EPS:
                    continue
                row_rotations = [0]*first_rows + [90]*second_rows
                for rotations in (row_rotations, list(reversed(row_rotations))):
                    cells, y = [], -height/2
                    for rotation in rotations:
                        w, l = (box.width,box.length) if rotation == 0 else (box.length,box.width)
                        columns = floor((usable_w+self.gap+EPS)/(w+self.gap))
                        for column in range(columns):
                            cells.append(Placement('',box.content_id,index,
                                (column-(columns-1)/2)*(w+self.gap),y+l/2,
                                z+box.height/2,rotation,w,l,box.height))
                        y += l+self.gap
                    grids.append(cells)
        if below:
            shifted_grids = []
            for cells in grids:
                if not cells:
                    continue
                x_room = max(0.0, usable_w/2 - max(abs(p.x)+p.width/2 for p in cells))
                y_room = max(0.0, usable_l/2 - max(abs(p.y)+p.length/2 for p in cells))
                dx = min(x_room, min(p.width for p in cells)/8)
                dy = min(y_room, min(p.length for p in cells)/8)
                for sx, sy in ((dx, 0), (-dx, 0), (0, dy), (0, -dy),
                               (dx/2, dy/2), (-dx/2, -dy/2)):
                    if abs(sx)+abs(sy) > EPS:
                        shifted_grids.append([replace(p, x=p.x+sx, y=p.y+sy) for p in cells])
            grids.extend(shifted_grids)
        seen_grids = set()
        below_rects = tuple(bounds(p) for p in below)
        for cells in grids:
            if best_score is not None and min(len(cells), count) < best_score[0]:
                continue
            key = tuple((c.x, c.y, c.width, c.length, c.rotation) for c in cells)
            if key in seen_grids:
                continue
            seen_grids.add(key)
            supported = []
            for cell in cells:
                fraction, centre = (_support_geometry(cell.x, cell.y, cell.width, cell.length,
                                                      below_rects, self.min_support)
                                    if below else (1.0, True))
                if fraction+EPS >= self.min_support and centre:
                    supported.append(cell)
            supported.sort(key=lambda c: (c.x*c.x+c.y*c.y, c.y, c.x))
            supported = supported[:count]
            if not supported:
                continue
            linked = bonding(supported, below, self.gap)
            score = (len(supported), linked['status'] != 'limited',
                     -seam_overlap(supported, below, self.gap),
                     sum(p.rotation != below[0].rotation for p in supported) if below else 0,
                     -sum(c.x*c.x+c.y*c.y for c in supported))
            if best_score is None or score > best_score:
                best, best_score = supported, score
        return best


def validate_batch(pallet: Pallet, boxes: list[Box], result: BatchResult, min_support=0.85):
    """Audit output independently of the packing search, including inventory conservation."""
    errors, warnings, checks, seen = [], [], [], set()
    original_ids = {b.instance_id for b in boxes}
    extra_ids = [b.instance_id for b in result.added]
    if len(extra_ids) != len(set(extra_ids)) or set(extra_ids) & original_ids:
        errors.append('Added box IDs collide with inventory')
    templates = {b.content_id: (b.width, b.length, b.height) for b in boxes}
    for b in result.added:
        if templates.get(b.content_id) != (b.width, b.length, b.height):
            errors.append(f'{b.instance_id}: added box does not match an existing SKU')
    inventory = {b.instance_id: b for b in boxes+result.added}
    for pallet_index, placements in enumerate(result.pallets, 1):
        layers = {}
        for p in placements:
            label = f"pallet {pallet_index}, {p.instance_id}"
            original = inventory.get(p.instance_id)
            if original is None or p.instance_id in seen:
                errors.append(f"{label}: unknown or duplicate ID")
            seen.add(p.instance_id)
            if original:
                expected = (original.width, original.length) if p.rotation == 0 else (original.length, original.width)
                if p.rotation not in (0, 90) or (p.width, p.length) != expected or p.height != original.height or p.content_id != original.content_id:
                    errors.append(f"{label}: SKU or dimensions changed")
            if not all(isfinite(v) for v in (p.x, p.y, p.z, p.width, p.length, p.height)):
                errors.append(f"{label}: non-finite placement")
            if abs(p.x)+p.width/2 > pallet.width/2+EPS or abs(p.y)+p.length/2 > pallet.length/2+EPS or p.z-p.height/2 < -EPS or p.z+p.height/2 > pallet.max_height+EPS:
                errors.append(f"{label}: outside pallet bounds")
            layers.setdefault(p.layer, []).append(p)
        previous, expected_z, previous_area = [], 0.0, float("inf")
        for index, layer in sorted(layers.items()):
            if sum(p.width*p.length for p in layer)+EPS < result.min_layer_fill*pallet.width*pallet.length:
                messages = warnings if result.allow_partial_layers else errors
                messages.append(f'pallet {pallet_index}, layer {index}: insufficient layer coverage')
            if index != (previous[0].layer+1 if previous else 0):
                errors.append(f"pallet {pallet_index}: non-contiguous layers")
            if len({(p.content_id, p.width*p.length, p.height) for p in layer}) != 1:
                errors.append(f"pallet {pallet_index}, layer {index}: mixed SKU or dimensions")
            area = layer[0].width*layer[0].length
            if area > previous_area+EPS:
                errors.append(f"pallet {pallet_index}, layer {index}: larger boxes above smaller ones")
            for i, p in enumerate(layer):
                if abs(p.z-p.height/2-expected_z) > EPS:
                    errors.append(f"{p.instance_id}: incorrect support height")
                for other in layer[i+1:]:
                    if intersection(bounds(p), bounds(other)):
                        errors.append(f"{p.instance_id}: overlaps {other.instance_id}")
                fraction, centre = support(p, previous) if previous else (1.0, True)
                checks.append({"instance_id": p.instance_id, "support_fraction": round(fraction, 6), "centre_supported": centre})
                if fraction+EPS < min_support or not centre:
                    errors.append(f"{p.instance_id}: insufficient support ({fraction:.1%})")
            if previous and bonding(layer, previous)['status'] == 'limited':
                warnings.append(f"pallet {pallet_index}, layer {index}: bonding limited by candidate geometry and support")
            elif previous and seam_overlap(layer, previous) > EPS:
                warnings.append(f"pallet {pallet_index}, layer {index}: some edges align; inspect layer drawing")
            previous, expected_z, previous_area = layer, expected_z+layer[0].height, area
    rejected = [item['instance_id'] for item in result.unplaced+result.deferred+result.set_aside]
    if result.allow_partial_layers:
        for plan in result.pallets:
            for p in plan:
                if p.instance_id in extra_ids and not any(
                        q.layer == p.layer+1 and intersection(bounds(p), bounds(q)) is not None
                        for q in plan):
                    errors.append(f'{p.instance_id}: added box supports no upper box')
    if result.allow_partial_layers and (result.deferred or result.set_aside):
        errors.append('Full-delivery mode left boxes outside pallets')
    if set(extra_ids) & set(rejected):
        errors.append('An added box was not packed')
    if len(rejected) != len(set(rejected)) or set(rejected) & seen or seen | set(rejected) != set(inventory):
        errors.append("Inventory mismatch between input, packed and rejected boxes")
    return {"valid": not errors, "errors": errors, "warnings": warnings, "box_checks": checks}
