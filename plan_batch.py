"""Calculate, inspect and audit pallet layouts without running robot motion."""
import argparse
import colorsys
from dataclasses import asdict
from html import escape
import json
from pathlib import Path
import random
import sys
from time import perf_counter
import webbrowser

sys.path.insert(0, str(Path(__file__).parent / "src"))
from pallet_planner import Box, Pallet
from pallet_planner.batch import BatchPlanner, bonding, seam_overlap, support, validate_batch


def load_catalog():
    return json.loads((Path(__file__).parent / 'config/box_catalog.json').read_text())


def color_job(tokens):
    catalog = load_catalog()
    if len(tokens) % 2:
        raise ValueError('Expected pairs: red 200 yellow 157 green 245 black 71')
    quantities = {}
    for name, value in zip(tokens[::2], tokens[1::2]):
        name = name.lower()
        if name not in catalog:
            raise ValueError(f'Unknown box type {name!r}; choose: {", ".join(catalog)}')
        try:
            quantity = int(value)
        except ValueError:
            raise ValueError(f'Invalid quantity for {name}: {value!r}') from None
        if quantity < 0:
            raise ValueError(f'Quantity for {name} cannot be negative')
        quantities[name] = quantities.get(name, 0) + quantity
    items = [dict(sku=name, quantity=quantity,
                  **{key: catalog[name][key] for key in ('width', 'length', 'height')})
             for name, quantity in quantities.items() if quantity]
    if not items:
        raise ValueError('At least one box is required')
    return {'units': 'mm', 'pallet': {'width': 1200, 'length': 800,
            'max_height': 1800, 'base_height': 144}, 'boxes': items}


def load_job(document):
    """Input uses millimetres; internal planner uses metres."""
    spec = document.get("pallet", {"width": 1200, "length": 800, "max_height": 1800})
    base_height = float(spec.get("base_height", 0))
    if base_height < 0:
        raise ValueError("base_height cannot be negative")
    pallet = Pallet(float(spec['width'])/1000, float(spec['length'])/1000,
                    (float(spec['max_height'])-base_height)/1000)
    boxes = []
    for item in document["boxes"]:
        quantity = item.get("quantity", 1)
        if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity <= 0:
            raise ValueError("quantity must be a positive integer")
        sku = str(item["sku"])
        for index in range(quantity):
            instance_id = item.get("instance_id")
            if quantity > 1 and instance_id is not None:
                raise ValueError("Use instance_id only for individual boxes")
            boxes.append(Box(str(instance_id) if instance_id is not None else f"box_{len(boxes)+1:05d}",
                             sku, *(float(item[key])/1000 for key in ("width", "length", "height"))))
    return pallet, boxes


def demo_job(count=500, seed=7):
    rng = random.Random(seed)
    types = [
        ("SKU_A", 400, 300, 250), ("SKU_B", 300, 250, 200),
        ("SKU_C", 250, 200, 180), ("SKU_D", 200, 150, 150),
        ("SKU_E", 400, 300, 200), ("SKU_F", 180, 120, 120),
    ]
    boxes = []
    for index in range(count):
        sku, w, l, h = rng.choice(types)
        boxes.append({"instance_id": f"box_{index+1:05d}", "sku": sku,
                      "width": w, "length": l, "height": h})
    return {"units": "mm", "pallet": {"width": 1200, "length": 800,
            "max_height": 1800, "base_height": 144}, "boxes": boxes}


def example_job(count=500, seed=None):
    if count < 20:
        raise ValueError('example requires at least 20 boxes')
    rng = random.Random(seed)
    widths = rng.sample(range(180, 561, 10), 20)
    lengths = rng.sample(range(120, 381, 10), 20)
    heights = rng.sample(range(100, 301, 10), 20)
    quantities = [1]*20
    for _ in range(count-20):
        quantities[rng.randrange(20)] += 1
    items = [dict(sku=f'example_{i+1:02d}', width=widths[i], length=lengths[i],
                  height=heights[i], quantity=quantities[i]) for i in range(20)]
    return {'units': 'mm', 'pallet': {'width': 1200, 'length': 800,
            'max_height': 1800, 'base_height': 144}, 'boxes': items}


def assembly_diagram(pallet, placements, colors, added_ids=(), pattern_id='added-pattern'):
    """Static axonometric engineering drawing of the planned assembled stack."""
    def project(x, y, z):
        return ((x-y)*0.75, (x+y)*0.35-z)
    height = max(p.z+p.height/2 for p in placements)
    corners = [project(x, y, z) for x in (0, pallet.width)
               for y in (0, pallet.length) for z in (0, height)]
    xs, ys = [p[0] for p in corners], [p[1] for p in corners]
    view = (min(xs)-.03, min(ys)-.03, max(xs)-min(xs)+.06, max(ys)-min(ys)+.06)
    out = [f'<svg role="img" aria-label="Собранная паллета" style="max-width:600px;height:420px" '
           f'viewBox="{" ".join(map(str,view))}">',
           f'<defs><pattern id="{pattern_id}" width="0.035" height="0.035" patternUnits="userSpaceOnUse">'
           '<path d="M0,.035 L.035,0" stroke="white" stroke-width="0.007"/>'
           '</pattern></defs>']
    for p in sorted(placements, key=lambda p: (p.layer, p.x+p.y)):
        x0, y0 = p.x-p.width/2+pallet.width/2, p.y-p.length/2+pallet.length/2
        x1, y1, z0, z1 = x0+p.width, y0+p.length, p.z-p.height/2, p.z+p.height/2
        faces = [([(x0,y1,z0),(x1,y1,z0),(x1,y1,z1),(x0,y1,z1)], .8),
                 ([(x1,y0,z0),(x1,y1,z0),(x1,y1,z1),(x1,y0,z1)], .65),
                 ([(x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)], 1)]
        for face, opacity in faces:
            points = ' '.join(f'{u},{v}' for u,v in (project(*point) for point in face))
            color = colors[p.content_id]
            shade = '#' + ''.join(f'{int(int(color[i:i+2],16)*opacity):02x}' for i in (1,3,5))
            out.append(f'<polygon points="{points}" fill="{shade}" '
                       'stroke="#263944" stroke-width="0.0015">'
                       f'<title>{escape(p.instance_id)} · {escape(p.content_id)}'
                       f'{" · Дополнительная" if p.instance_id in added_ids else ""}</title></polygon>')
            if p.instance_id in added_ids:
                out.append(f'<polygon points="{points}" fill="url(#{pattern_id})" '
                           'stroke="#222" stroke-width="0.003" pointer-events="none"/>')
        if p.instance_id in added_ids:
            cx, cy = project((x0+x1)/2, (y0+y1)/2, z1)
            out.append(f'<text x="{cx}" y="{cy}" text-anchor="middle" dominant-baseline="middle" '
                       'font-size="0.07" font-weight="bold" fill="black" stroke="white" '
                       'stroke-width="0.006" paint-order="stroke">+</text>')
    out.append('</svg>')
    return ''.join(out)


def render_html(pallet, result, audit, input_boxes=None):
    added_ids = {box.instance_id for box in result.added}
    total_volume = sum(p.width*p.length*p.height for plan in result.pallets for p in plan)
    available_volume = len(result.pallets)*pallet.width*pallet.length*pallet.max_height
    utilization = total_volume/available_volume if available_volume else 0.0
    source = {b.content_id: b for b in input_boxes or []}
    source_ids = {b.instance_id for b in input_boxes or []}
    delivered_originals = sum(p.instance_id in source_ids for plan in result.pallets for p in plan)
    skus = sorted({p.content_id for plan in result.pallets for p in plan} | set(source))
    catalog = load_catalog()
    colors = {}
    for i, sku in enumerate(skus):
        rgb = colorsys.hsv_to_rgb((i*0.61803398875) % 1, .60, .78)
        fallback = '#' + ''.join(f'{round(channel*255):02x}' for channel in rgb)
        colors[sku] = catalog.get(sku, {}).get('color', fallback)
    out = ['<!doctype html><html lang="ru"><meta charset="utf-8">',
           '<meta name="viewport" content="width=device-width,initial-scale=1">',
           '<title>Раскладка паллет</title><style>',
           'body{font:15px system-ui;margin:24px;max-width:calc(100vw - 48px);color:#202a30;background:#f6f8f9;overflow-wrap:break-word}',
           'h1{font-size:26px}h2{font-size:21px}h3{font-size:16px}section{border-top:1px solid #ccd4d8;padding:16px 0}',
           '.layers{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,320px),1fr));gap:20px}',
           '.overview{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,240px),1fr));gap:16px}',
           '.overview svg{height:300px!important}.overview a{color:inherit;text-decoration:none}'
           '.overview>* ,.layers>*{min-width:0}svg{max-width:100%}',
           'table{border-collapse:collapse;margin:16px 0;width:100%;max-width:600px;table-layout:fixed}'
           'td,th{padding:6px 8px;text-align:left;border-bottom:1px solid #d7dfe3;overflow-wrap:anywhere}'
           '@media(max-width:600px){td,th{padding:6px 4px;font-size:13px}}',
           '.swatch{display:inline-block;width:12px;height:12px;margin-right:8px;border:1px solid #666}',
           'figure{margin:0}svg{width:100%;height:auto;background:white}figcaption{margin:6px 0}',
           'summary{cursor:pointer;padding:8px 0}li{margin:6px 0}text{pointer-events:none}</style>',
           '<h1>Раскладка паллет</h1>',
           (f'<p>Исходная партия: <strong>{len(input_boxes)}</strong> коробок, '
            f'<strong>{len(source)}</strong> SKU. На паллетах исходных: '
            f'<strong>{delivered_originals}/{len(input_boxes)}</strong>.</p>' if input_boxes is not None else ''),
           f'<p>Поддон: {pallet.width*1000:g} × {pallet.length*1000:g} мм. '
           f'Высота коробов: до {pallet.max_height*1000:g} мм. '
           f'Паллет: {len(result.pallets)}. Коробок: {sum(map(len,result.pallets))}. '
           f'Не размещено: {len(result.unplaced)}.</p>',
           f'<p>Заполнение доступного объёма: {utilization:.1%}. '
           f'Отложено на следующую партию: {len(result.deferred)}.</p>',
           f'<p>Без паллеты: <strong>{len(result.set_aside)}</strong> коробок. '
           f'Целевое заполнение слоя: {result.min_layer_fill:.0%} площади поддона.</p>',
           f'<p>Дополнительно добавить: <strong>{len(result.added)}</strong> коробок. '
           + (f'Паллет без добавления: {result.baseline_pallets} → с добавлением: {len(result.pallets)}.</p>'
              if result.baseline_pallets is not None else 'Режим полной доставки.</p>'),
           f'<p>Геометрическая проверка: {"ПРОЙДЕНА" if audit["valid"] else "ОШИБКА"}. '
           f'Предупреждений: {len(audit["warnings"])}.</p>']
    if result.allow_partial_layers:
        partial = [(number, index) for number, plan in enumerate(result.pallets, 1)
                   for index in sorted({p.layer for p in plan})
                   if sum(p.width*p.length for p in plan if p.layer == index)+1e-8
                   < result.min_layer_fill*pallet.width*pallet.length]
        out.append(f'<p>Слоёв ниже целевого заполнения: <strong>{len(partial)}</strong>. '
                   'Разрешённые остатки и типоразмеры с недостаточной площадью максимальной сетки.</p>')
    if result.search_limited:
        out.append('<p>Достигнут лимит времени поиска. Показана лучшая завершённая раскладка.</p>')
    if result.set_aside:
        out.append('<details><summary>Коробки без паллеты</summary><ul>')
        for sku in sorted({item['content_id'] for item in result.set_aside}):
            items = [item for item in result.set_aside if item['content_id'] == sku]
            maximum = items[0].get('max_layer_fill')
            minimum = items[0].get('minimum_count')
            reason = ''
            if maximum is not None and maximum+1e-8 < result.min_layer_fill:
                reason = f' Максимальная найденная схема занимает {maximum:.1%} поддона.'
            elif minimum is not None:
                reason = f' Для допустимого слоя требуется минимум {minimum} коробок.'
            out.append(f'<li>{escape(sku)}: {len(items)} шт. — '
                       f'{escape(", ".join(item["instance_id"] for item in items))}{reason}</li>')
        out.append('</ul></details>')
    if result.added:
        out.append('<p><strong>+ ▨</strong> Дополнительные коробки отмечены белой штриховкой и знаком «+».</p>')
        out.append('<details><summary>Дополнительные коробки</summary><ul>')
        for box in result.added:
            out.append(f'<li>+ {escape(box.instance_id)} · {escape(box.content_id)} · '
                       f'{box.width*1000:g} × {box.length*1000:g} × {box.height*1000:g} мм</li>')
        out.append('</ul></details>')
    out.append(f'<details {"open" if len(skus) <= 6 else ""}><summary>Типы коробок ({len(skus)})</summary>'
               '<table><thead><tr><th>Тип</th><th>Размер, мм</th>'
               '<th>Вход / на паллетах / отдельно / добавлено</th></tr></thead><tbody>')
    for sku in skus:
        placements = [p for plan in result.pallets for p in plan if p.content_id == sku]
        p = placements[0] if placements else source[sku]
        w, l = ((p.width, p.length) if not placements or p.rotation == 0 else (p.length, p.width))
        original_count = sum(b.content_id == sku for b in input_boxes or [])
        aside_count = sum(item['content_id'] == sku for item in result.set_aside)
        out.append(f'<tr><td><span class="swatch" style="background:{colors[sku]}"></span>{escape(sku)}</td>'
                   f'<td>{w*1000:g} × {l*1000:g} × {p.height*1000:g}</td>'
                   f'<td>{original_count} / {len(placements)} / {aside_count} / '
                   f'+{sum(p.instance_id in added_ids for p in placements)}</td></tr>')
    out.append('</tbody></table></details><div class="overview">')
    for number, plan in enumerate(result.pallets, 1):
        height = max(p.z+p.height/2 for p in plan)
        out.append(f'<a href="#pallet-{number}"><figure><figcaption>Паллета {number} · {len(plan)} шт. · '
                   f'{height*1000:.0f} мм</figcaption>')
        out.append(assembly_diagram(pallet, plan, colors, added_ids, f'overview-added-{number}'))
        out.append('</figure></a>')
    out.append('</div>')
    for number, plan in enumerate(result.pallets, 1):
        height = max(p.z+p.height/2 for p in plan)
        volume = sum(p.width*p.length*p.height for p in plan)
        out.append(f'<section id="pallet-{number}"><h2>Паллета {number}: {len(plan)} коробок, {height*1000:.0f} мм</h2>'
                   f'<p>Заполнение доступного объёма: {volume/(pallet.width*pallet.length*pallet.max_height):.1%}. '
                   f'Заполнение до верха груза: {volume/(pallet.width*pallet.length*height):.1%}</p>')
        out.append('<details><summary>Слои сверху</summary><div class="layers">')
        layers = sorted({p.layer for p in plan})
        for index in layers:
            layer = [p for p in plan if p.layer == index]
            below = [p for p in plan if p.layer == index-1]
            status = bonding(layer, below)
            description = {'base': 'Основание', 'bonded': 'Перевязка',
                           'limited': 'Перевязка ограничена геометрией и опорой'}[status['status']]
            w, l = pallet.width*1000, pallet.length*1000
            pattern_id = f'layer-added-{number}-{index}'
            out.append(f'<figure><figcaption>Слой {index+1} · {escape(layer[0].content_id)} · '
                       f'{len(layer)} шт. · низ {1000*(layer[0].z-layer[0].height/2):.0f} мм'
                       f' · площадь {sum(p.width*p.length for p in layer)/(pallet.width*pallet.length):.1%}'
                       f'<br>{description} · совпадение стыков {status["aligned_fraction"]:.0%}</figcaption>'
                       f'<svg role="img" aria-label="Слой {index+1}" viewBox="-10 -10 {w+20} {l+20}">'
                       f'<defs><pattern id="{pattern_id}" width="15" height="15" patternUnits="userSpaceOnUse">'
                       '<path d="M0,15 L15,0" stroke="white" stroke-width="4"/></pattern></defs>'
                       f'<rect width="{w}" height="{l}" fill="#e4e9ec" stroke="#657783"/>')
            for p in layer:
                x, y = (p.x-p.width/2+pallet.width/2)*1000, (p.y-p.length/2+pallet.length/2)*1000
                out.append(f'<rect x="{x}" y="{y}" width="{p.width*1000}" height="{p.length*1000}" '
                           f'fill="{colors[p.content_id]}" stroke="#fff" stroke-width="2">'
                           f'<title>{escape(p.instance_id)} · {escape(p.content_id)} · {p.rotation}°'
                           f'{" · Дополнительная" if p.instance_id in added_ids else ""}</title></rect>')
                if p.instance_id in added_ids:
                    out.append(f'<rect x="{x}" y="{y}" width="{p.width*1000}" height="{p.length*1000}" '
                               f'fill="url(#{pattern_id})" pointer-events="none"/>')
                cx, cy = (p.x+pallet.width/2)*1000, (p.y+pallet.length/2)*1000
                out.append(f'<text x="{cx}" y="{cy}" text-anchor="middle" dominant-baseline="middle" '
                           f'font-size="20" fill="white" stroke="#202a30" stroke-width="1" paint-order="stroke">'
                           f'{"+ " if p.instance_id in added_ids else ""}{p.rotation}°</text>')
            # Previous layer edges are overlaid so coincident joints can be inspected.
            for p in below:
                x, y = (p.x-p.width/2+pallet.width/2)*1000, (p.y-p.length/2+pallet.length/2)*1000
                out.append(f'<rect x="{x}" y="{y}" width="{p.width*1000}" height="{p.length*1000}" '
                           'fill="none" stroke="#17262e" opacity="0.55" stroke-width="2" stroke-dasharray="6 5"/>')
            out.append('</svg></figure>')
        out.append('</div></details></section>')
    out.append('<details><summary>Замечания по швам и отклонения</summary><ul>')
    for message in audit['errors'] + audit['warnings']:
        out.append(f'<li>{escape(message)}</li>')
    for item in result.unplaced+result.deferred:
        out.append(f'<li>{escape(item["instance_id"])}: {escape(item["reason"])}</li>')
    out.append('</ul></details></html>')
    return '\n'.join(out)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        argv = ['example', '--open']
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('boxes', nargs='*', metavar='TYPE COUNT', help='red 200 yellow 157 green 245 black 71')
    parser.add_argument('--input', type=Path, help='JSON inventory in millimetres')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--demo-count', type=int, default=500)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--min-support', type=float, default=0.85)
    parser.add_argument('--min-layer-fill', type=float, default=0.75,
                        help='target occupied pallet area; partial layers allowed for full delivery')
    parser.add_argument('--time-limit', type=float, default=0,
                        help='optimization time budget in seconds; 0 searches without a time limit')
    parser.add_argument('--max-additions', type=int, default=None,
                        help='optional extra box count cap; 0 disables')
    parser.add_argument('--max-addition-area', type=float, default=None,
                        help='legacy option; full-delivery mode has no added-area cap')
    parser.add_argument('--max-leftovers', type=int, default=0,
                        help='legacy option; full-delivery mode does not defer boxes')
    parser.add_argument('--list-types', action='store_true', help='show predefined types and dimensions in mm')
    parser.add_argument('--open', action='store_true', help='open assembled pallet visualization in browser')
    args = parser.parse_args(argv)
    is_example = args.boxes == ['example']
    if args.output is None:
        args.output = Path('reports/example' if is_example else 'reports/batch')
    if args.list_types:
        for name, spec in load_catalog().items():
            print(f'{name:8s} {spec["width"]} x {spec["length"]} x {spec["height"]} mm')
        return 0
    if args.input and args.boxes:
        parser.error('Use either TYPE COUNT pairs or --input')
    if args.demo_count <= 0:
        parser.error('--demo-count must be positive')
    try:
        if is_example:
            document = example_job(args.demo_count, args.seed)
        elif args.boxes:
            document = color_job(args.boxes)
        elif args.input:
            document = json.loads(args.input.read_text())
        else:
            document = demo_job(args.demo_count, args.seed if args.seed is not None else 7)
        if document.get('units', 'mm') != 'mm':
            raise ValueError('Input units must be mm')
        pallet, boxes = load_job(document)
        if not boxes:
            raise ValueError('Empty inventory')
        started = perf_counter()
        last_progress = started-2
        def show_progress(message):
            nonlocal last_progress
            now = perf_counter()
            if now-last_progress >= 2:
                print(f'[{now-started:.1f}s] {message}', flush=True)
                last_progress = now
        planner = BatchPlanner(min_support=args.min_support, progress=show_progress,
                               min_layer_fill=args.min_layer_fill, ship_all=True)
        print(f'Planning {len(boxes)} boxes...', flush=True)
        result = planner.plan(pallet, boxes, max_leftovers=args.max_leftovers, max_additions=args.max_additions,
                              max_addition_area=args.max_addition_area, time_limit=args.time_limit)
        planning_seconds = perf_counter()-started
        print(f'Packing calculated in {planning_seconds:.2f}s. Validating...', flush=True)
        audit = validate_batch(pallet, boxes, result, args.min_support)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))
    args.output.mkdir(parents=True, exist_ok=True)
    artifacts = {
        'input.json': document,
        'augmented_input.json': dict(document, boxes=[{
            'instance_id': box.instance_id, 'sku': box.content_id,
            'width': box.width*1000, 'length': box.length*1000, 'height': box.height*1000,
        } for box in boxes+result.added]),
        'plan.json': {'units': 'm', 'pallet': asdict(pallet), 'strategy': result.strategy, 'settings': {
            'gap': planner.gap, 'edge_margin': planner.edge_margin, 'min_support': planner.min_support,
            'min_layer_fill': planner.min_layer_fill,
            'ship_all': planner.ship_all,
            'max_leftovers': args.max_leftovers, 'max_additions': args.max_additions,
            'max_addition_area': args.max_addition_area, 'time_limit': args.time_limit},
            'search_limited': result.search_limited,
            'allow_partial_layers': result.allow_partial_layers,
            'partial_layers': [{'pallet': number, 'layer': index,
                'coverage': sum(p.width*p.length for p in plan if p.layer == index)/(pallet.width*pallet.length)}
                for number, plan in enumerate(result.pallets, 1) for index in sorted({p.layer for p in plan})
                if sum(p.width*p.length for p in plan if p.layer == index)+1e-8
                < result.min_layer_fill*pallet.width*pallet.length],
            'baseline_pallets': result.baseline_pallets, 'addition_trials': result.addition_trials,
            'pallets': [[asdict(p) for p in plan] for plan in result.pallets],
            'unplaced': result.unplaced, 'deferred': result.deferred, 'set_aside': result.set_aside,
            'added': [asdict(box) for box in result.added]},
        'validation.json': audit,
    }
    for filename, data in artifacts.items():
        (args.output/filename).write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (args.output/'layout.html').write_text(render_html(pallet, result, audit, boxes), encoding='utf-8')
    events = [{'event': 'job_started', 'input_boxes': len(boxes),
               'added_boxes': len(result.added), 'baseline_pallets': result.baseline_pallets,
               'strategy': result.strategy,
               'min_support': planner.min_support, 'gap_m': planner.gap,
               'edge_margin_m': planner.edge_margin}]
    for number, plan in enumerate(result.pallets, 1):
        for layer in sorted({p.layer for p in plan}):
            placements = [p for p in plan if p.layer == layer]
            below = [p for p in plan if p.layer == layer-1]
            support_min = min(support(p, below)[0] if below else 1.0 for p in placements)
            events.append({'event': 'layer_planned', 'pallet': number, 'layer': layer,
                           'sku': placements[0].content_id, 'count': len(placements),
                           'rotations': sorted({p.rotation for p in placements}),
                           'base_z_m': placements[0].z-placements[0].height/2,
                           'min_support_fraction': support_min,
                           'aligned_joint_length_m': seam_overlap(placements, below, planner.gap),
                           'bonding': bonding(placements, below, planner.gap),
                           'reason': 'Select admissible SKU combinations; check support; add tails; consolidate pallets'})
    events.extend(dict(event='box_deferred', **item) for item in result.deferred)
    events.extend(dict(event='box_set_aside', **item) for item in result.set_aside)
    events.extend(dict(event='box_added', **asdict(box)) for box in result.added)
    (args.output/'history.jsonl').write_text(''.join(json.dumps(event)+'\n' for event in events), encoding='utf-8')
    print(f'Input: {len(boxes)}; packed: {sum(map(len,result.pallets))}; pallets: {len(result.pallets)}; unplaced: {len(result.unplaced)}')
    print(f'Deferred: {len(result.deferred)} (limit: {args.max_leftovers})')
    print(f'Without pallet: {len(result.set_aside)}; minimum layer area: {planner.min_layer_fill:.0%}')
    original_ids = {b.instance_id for b in boxes}
    delivered = sum(p.instance_id in original_ids for plan in result.pallets for p in plan)
    print(f'Original boxes delivered: {delivered}/{len(boxes)}; added: {len(result.added)} (count cap: {args.max_additions})')
    print(f'Validation: {"PASS" if audit["valid"] else "FAIL"}; errors: {len(audit["errors"])}; warnings: {len(audit["warnings"])}')
    print(f'View: {args.output / "layout.html"}')
    print(f'Time: planning {planning_seconds:.2f}s; validation and export {perf_counter()-started-planning_seconds:.2f}s')
    if result.search_limited:
        print('Time budget reached: best complete layout saved. Use --time-limit 0 for unrestricted search.')
    if args.open:
        webbrowser.open((args.output/'layout.html').resolve().as_uri())
    return 0 if audit['valid'] and not result.unplaced else 1


if __name__ == '__main__':
    raise SystemExit(main())
