"""Genera el informe HTML de un entrenamiento a partir de results.csv.

Reglas de ApexCharts aplicadas (ver skill .agents/skills/apexcharts-skill):
  - CDN fijado en v7. El skill documenta v7; v8 no esta verificado aqui.
  - API core (vanilla JS), sin wrappers de framework.
  - `render()` obligatorio tras el constructor.
  - `null` para datos faltantes, nunca `undefined`.
  - Colores hex con `#`.
  - Breakpoints de `responsive` en orden ascendente.

Este modulo no importa torch ni ultralytics: lo usa la imagen runtime-api.
"""

import csv
import html
import json
import logging
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

# Versión fijada. No usar @latest: el skill cubre v7 y sus breaking changes.
APEXCHARTS_CDN = "https://cdn.jsdelivr.net/npm/apexcharts@7/dist/apexcharts.min.js"

PALETTE = [
    "#008FFB", "#00E396", "#FEB019", "#FF4560", "#775DD0",
    "#546E7A", "#26BFC6", "#FD6D6D", "#D3B2C7", "#8E6C8A",
]

# ─────────────────────────────────────────────────────────────────────────────
# Artefactos que Ultralytics escribe en <run>/train/.
#
# El prefijo `Box` es obligatorio: el informe busca "BoxPR_curve.png" y no
# "PR_curve.png". Ese nombre mal escrito estuvo en el informe desde el primer
# dia, asi que las curvas de precision-recall y de F1 nunca aparecieron en
# ninguno. Por eso la lista es una constante y verificar.py la contrasta.
# ─────────────────────────────────────────────────────────────────────────────
CURVE_ASSETS: tuple[tuple[str, str], ...] = (
    ("confusion_matrix.png", "Matriz de confusion"),
    ("confusion_matrix_normalized.png", "Matriz de confusion (normalizada)"),
    ("BoxPR_curve.png", "Curva precision-recall"),
    ("BoxF1_curve.png", "Curva F1"),
    ("BoxP_curve.png", "Precision por clase"),
    ("BoxR_curve.png", "Recall por clase"),
    ("results.png", "Curvas completas de Ultralytics"),
)

DATASET_ASSETS: tuple[tuple[str, str], ...] = (
    ("labels.jpg", "Distribucion de cajas y clases del dataset"),
)

# Fotos de lote. Glob y no lista fija porque Ultralytics decide cuantas escribe;
# el prefijo lo pone el, nunca el cliente.
BATCH_PATTERNS: tuple[str, ...] = ("train_batch*.jpg", "val_batch*.jpg")

ARGS_FILE = "args.yaml"


def build_meta(job, project_name: str = "", dataset_name: str = "") -> dict:
    """Diccionario de cabecera a partir de un job.

    Duck-typed a proposito: report.py no importa trainpipe.models para poder
    seguir usandose suelto (verificar.py lo importa sin Docker ni Postgres).
    """
    generado = getattr(job, "finished_at", None) or getattr(job, "started_at", None)
    if isinstance(generado, datetime):
        generado = generado.strftime("%Y-%m-%d %H:%M")
    return {
        "proyecto": project_name,
        "dataset": dataset_name,
        "modelo": getattr(job, "model", "") or "",
        "job_id": str(getattr(job, "id", "") or ""),
        "generado": generado or "",
        "parametros": dict(getattr(job, "params", {}) or {}),
    }


def collect_assets(run_dir: Path) -> list[Path]:
    """Artefactos que el informe referencia y que existen de verdad en el run.

    La lista es CERRADA y no acepta entrada de usuario: por eso el endpoint que
    la sirve no necesita la whitelist de /download, que existe precisamente
    porque /download si acepta un nombre de fichero del cliente.
    """
    found: list[Path] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        if path.name not in seen and path.is_file():
            seen.add(path.name)
            found.append(path)

    for name, _ in CURVE_ASSETS + DATASET_ASSETS:
        add(run_dir / name)
    for pattern in BATCH_PATTERNS:
        for path in sorted(run_dir.glob(pattern)):
            add(path)
    args = run_dir / ARGS_FILE
    if args.is_file():
        add(args)
    return found


def val_label_pairs(run_dir: Path) -> list[tuple[str, Path, Path]]:
    """Empareja cada val_batchN_labels.jpg con su _pred para ponerlos juntos.

    Comparar el ground truth con lo que predijo el modelo sobre las MISMAS
    imagenes es el diagnostico visual mas util que hay: cuando el mAP50 es malo,
    el motivo casi siempre se ve ahi.
    """
    pairs: list[tuple[str, Path, Path]] = []
    for labels in sorted(run_dir.glob("val_batch*_labels.jpg")):
        pred = labels.with_name(labels.name.replace("_labels.jpg", "_pred.jpg"))
        if pred.is_file():
            index = labels.name.removeprefix("val_batch").removesuffix("_labels.jpg")
            pairs.append((index, labels, pred))
    return pairs


def _read_results(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with csv_path.open(newline="", encoding="utf-8", errors="ignore") as fh:
        # Ultralytics escribe un espacio delante de cada columna.
        rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(fh)]
    return [r for r in rows if any(v.strip() for v in r.values())]


def _series(rows: list[dict], column: str) -> list[float | None]:
    """Serie lista para ApexCharts. `null` en huecos, nunca `undefined`:
    `undefined` se ignora en silencio y rompe la grafica."""
    out: list[float | None] = []
    for row in rows:
        raw = (row.get(column) or "").strip()
        try:
            out.append(float(raw))
        except ValueError:
            out.append(None)
    return out


def _has(rows: list[dict], column: str) -> bool:
    return any(column in r for r in rows)


def _final_metric(rows: list[dict], column: str) -> float | None:
    if not _has(rows, column):
        return None
    values = [v for v in _series(rows, column) if v is not None]
    return values[-1] if values else None


def _embed(chart: dict) -> str:
    """Serializa a JSON seguro para incrustar en <script>.

    `</script>` dentro de un string JSON cerraria el tag y romperia la pagina,
    asi que se neutraliza antes de volcar.
    """
    return json.dumps(chart).replace("</", "<\\/")


def _stat_card(label: str, value: float | None, suffix: str = "") -> str:
    shown = "n/d" if value is None else f"{value:.4f}{suffix}"
    return (
        f'<div class="stat"><span class="stat-label">{label}</span>'
        f'<span class="stat-value">{shown}</span></div>'
    )


def generate_report(run_dir: Path, output: Path, meta: dict | None = None) -> Path:
    """Construye el informe. `run_dir` es el directorio real del run.

    `meta` es opcional (proyecto, modelo, dataset, parametros) para que quien
    llama sin sesion --verificar.py-- siga funcionando igual.
    """
    rows = _read_results(run_dir / "results.csv")
    epochs = [int(float(r.get("epoch", i + 1))) for i, r in enumerate(rows)]

    charts: list[tuple[str, str, dict]] = []

    # --- Curva de mAP ---
    m50 = "metrics/mAP50(B)"
    m5095 = "metrics/mAP50-95(B)"
    if _has(rows, m50) or _has(rows, m5095):
        charts.append((
            "Calidad (mAP)",
            "mAP50 y mAP50-95 por epoca",
            {
                "chart": {"type": "line", "height": 340, "toolbar": {"show": True}},
                "series": [
                    {"name": "mAP50", "data": _series(rows, m50)},
                    {"name": "mAP50-95", "data": _series(rows, m5095)},
                ],
                "xaxis": {"categories": epochs, "title": {"text": "Epoca"}},
                "yaxis": {"min": 0, "max": 1, "title": {"text": "mAP"}},
                "colors": PALETTE,
                "stroke": {"width": 2, "curve": "smooth"},
                "tooltip": {"shared": True, "intersect": False},
                "legend": {"position": "top"},
                "responsive": [
                    {"breakpoint": 480, "options": {"legend": {"position": "bottom"}}},
                    {"breakpoint": 1024, "options": {"chart": {"height": 260}}},
                ],
            },
        ))

    # --- Precision / recall ---
    prec, rec = "metrics/precision(B)", "metrics/recall(B)"
    if _has(rows, prec) or _has(rows, rec):
        charts.append((
            "Precision y recall",
            "Precision frente a recall por epoca",
            {
                "chart": {"type": "area", "height": 340},
                "series": [
                    {"name": "Precision", "data": _series(rows, prec)},
                    {"name": "Recall", "data": _series(rows, rec)},
                ],
                "xaxis": {"categories": epochs, "title": {"text": "Epoca"}},
                "yaxis": {"min": 0, "max": 1, "title": {"text": "Proporcion"}},
                "colors": PALETTE[:2],
                "fill": {"type": "gradient", "opacity": 0.25},
                "stroke": {"width": 2},
                "tooltip": {"shared": True, "intersect": False},
                "legend": {"position": "top"},
                "responsive": [{"breakpoint": 480, "options": {"legend": {"position": "bottom"}}}],
            },
        ))

    # --- Perdidas ---
    loss_cols = [c for c in ("train/box_loss", "train/seg_loss", "train/cls_loss", "train/dfl_loss") if _has(rows, c)]
    if loss_cols:
        charts.append((
            "Perdidas",
            "Perdidas por epoca (train)",
            {
                "chart": {"type": "line", "height": 340},
                "series": [{"name": c.split("/")[-1].replace("_loss", ""), "data": _series(rows, c)} for c in loss_cols],
                "xaxis": {"categories": epochs, "title": {"text": "Epoca"}},
                "yaxis": {"title": {"text": "Loss"}},
                "colors": PALETTE,
                "stroke": {"width": 2, "curve": "smooth"},
                "tooltip": {"shared": True, "intersect": False},
                "legend": {"position": "top"},
                "responsive": [{"breakpoint": 480, "options": {"legend": {"position": "bottom"}}}],
            },
        ))

    # --- mAP por clase (barra) ---
    # Ultralytics no lo escribe en results.csv; se extrae del resumen final si
    # el run lo dejo disponible.
    per_class = _load_per_class(run_dir)
    if per_class:
        classes = [str(k) for k in per_class.keys()]
        charts.append((
            "mAP50 por clase",
            "Metrica por clase en la ultima epoca",
            {
                "chart": {"type": "bar", "height": 340},
                "series": [{"name": "mAP50", "data": list(per_class.values())}],
                "xaxis": {"categories": classes, "title": {"text": "Clase"}},
                "yaxis": {"min": 0, "max": 1, "title": {"text": "mAP50"}},
                "plotOptions": {"bar": {"horizontal": True, "borderRadius": 3}},
                "colors": [PALETTE[0]],
                "dataLabels": {"enabled": False},
                "tooltip": {"shared": False, "intersect": True},
                "responsive": [{"breakpoint": 480, "options": {"legend": {"show": False}}}],
            },
        ))

    # OJO: el local se llama `page` y no `html` a proposito. `html` es el modulo
    # de la stdlib que _render_html usa para escapar; un local con ese nombre
    # lo taparia.
    page = _render_html(run_dir, rows, charts, per_class, meta)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")
    log.info("informe escrito en %s", output)
    return output


def _load_per_class(run_dir: Path) -> dict[str, float]:
    path = run_dir / "per_class_metrics.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        return {str(k): float(v) for k, v in data.items()}
    except (ValueError, TypeError):
        return {}


def _esc(texto: object) -> str:
    """Escapa para HTML. Sin esto, un `<` dentro de args.yaml rompe la pagina."""
    return html.escape(str(texto), quote=True)


def _img(path: Path, title: str) -> str:
    """Referencia RELATIVA a un artefacto, no base64.

    El informe se descarga a una carpeta distinta de la del run, asi que la
    imagen tiene que viajar a su lado. La baja el endpoint report-assets (un ZIP)
    y luego el script de entrenamiento. Ver collect_assets.
    """
    return (
        f'<figure class="img">'
        f'<img src="{_esc(path.name)}" alt="{_esc(title)}" loading="lazy">'
        f"<figcaption>{_esc(title)}</figcaption></figure>"
    )


def _grid(figures: str, classes: str = "grid") -> str:
    return f'<div class="{classes}">{figures}</div>' if figures else ""


def _meta_chips(meta: dict | None) -> str:
    """Cabecera con datos reales del job. Antes solo ponia el UUID del run_dir,
    que no le decia nada a nadie."""
    if not meta:
        return ""
    chips: list[str] = []
    for key in ("proyecto", "dataset", "modelo", "job_id", "generado"):
        value = meta.get(key)
        if value:
            label = _esc(key.replace("_", " "))
            chips.append(f'<span class="chip"><span class="chip-k">{label}</span>{_esc(value)}</span>')
    params = meta.get("parametros") or {}
    destacados = ("epochs", "imgsz", "batch", "workers", "fraction", "patience", "device")
    for key in destacados:
        value = params.get(key)
        if value is not None:
            chips.append(f'<span class="chip"><span class="chip-k">{_esc(key)}</span>{_esc(value)}</span>')
    if not chips:
        return ""
    return '<div class="chips">' + "".join(chips) + "</div>"


def _args_block(run_dir: Path) -> str:
    """args.yaml con los hiperparametros exactos del run, plegado.

    Es lo que hay que guardar para poder reproducir el entrenamiento.
    """
    path = run_dir / ARGS_FILE
    if not path.is_file():
        return ""
    crudo = path.read_text(encoding="utf-8", errors="ignore")
    if not crudo.strip():
        return ""
    return (
        '<details class="card"><summary>Configuracion exacta (args.yaml)</summary>'
        '<p class="sub">Hiperparametros con los que se entreno este modelo. '
        "Es lo que hay que guardar para reproducir el run.</p>"
        f"<pre>{_esc(crudo)}</pre></details>"
    )


def _asset_sections(run_dir: Path) -> str:
    """Imagenes que Ultralytics ya genera. No se reconstruyen en ApexCharts: se
    referencian tal cual y las baja report-assets."""
    partes: list[str] = []

    curvas = "".join(
        _img(run_dir / name, title)
        for name, title in CURVE_ASSETS
        if (run_dir / name).is_file()
    )
    if curvas:
        partes.append(
            '<section class="card"><h2>Curvas de Ultralytics</h2>'
            '<p class="sub">Matriz de confusion y curvas por umbral, tal como las '
            "dibuja Ultralytics.</p>"
            + _grid(curvas, "grid grid-2")
            + "</section>"
        )

    pares = val_label_pairs(run_dir)
    if pares:
        figuras = ""
        for index, labels, pred in pares:
            figuras += (
                f'<div class="pair"><h3>val_batch{_esc(index)}</h3>'
                '<div class="pair-row">'
                + _img(labels, "Etiquetas reales (ground truth)")
                + _img(pred, "Predicciones del modelo")
                + "</div></div>"
            )
        partes.append(
            '<section class="card"><h2>Validacion visual</h2>'
            '<p class="sub">A la izquierda lo que hay de verdad; a la derecha lo '
            "que predijo el modelo sobre las mismas imagenes. Es el diagnostico "
            "visual mas util cuando el mAP50 no cuadra.</p>"
            + figuras
            + "</section>"
        )

    entrena = "".join(
        _img(path, f"train_batch{path.name.removeprefix('train_batch').removesuffix('.jpg')}")
        for path in sorted(run_dir.glob("train_batch*.jpg"))
    )
    if entrena:
        partes.append(
            '<section class="card"><h2>Lotes de entrenamiento</h2>'
            '<p class="sub">Imagenes de entrenamiento ya augmentadas (mosaico, '
            "volteos): lo que el modelo ve de verdad.</p>"
            + _grid(entrena)
            + "</section>"
        )

    dataset = "".join(
        _img(run_dir / name, title)
        for name, title in DATASET_ASSETS
        if (run_dir / name).is_file()
    )
    if dataset:
        partes.append(
            '<section class="card"><h2>Dataset</h2>'
            '<p class="sub">Como estan repartidas las cajas y las clases. Si una '
            "clase sale muy por debajo, el problema es de datos, no de "
            "entrenamiento.</p>"
            + _grid(dataset)
            + "</section>"
        )

    return "\n".join(partes)


def _render_html(run_dir, rows, charts, per_class, meta=None) -> str:
    m50 = _final_metric(rows, "metrics/mAP50(B)")
    m5095 = _final_metric(rows, "metrics/mAP50-95(B)")
    prec = _final_metric(rows, "metrics/precision(B)")
    rec = _final_metric(rows, "metrics/recall(B)")

    stats = "".join([
        _stat_card("mAP50", m50),
        _stat_card("mAP50-95", m5095),
        _stat_card("Precision", prec),
        _stat_card("Recall", rec),
        _stat_card("Epocas", float(len(rows)) if rows else None),
    ])

    sections = "\n".join(
        f'<section class="card"><h2>{_esc(title)}</h2><p class="sub">{_esc(subtitle)}</p>'
        f'<div id="chart-{i}"></div></section>'
        for i, (title, subtitle, _) in enumerate(charts)
    )
    payload = ",\n".join(_embed(c[2]) for c in charts)

    init = "\n".join(
        f"new ApexCharts(document.querySelector('#chart-{i}'), opts[{i}]).render();"
        for i in range(len(charts))
    )

    chips = _meta_chips(meta)
    assets = _asset_sections(run_dir)
    args_block = _args_block(run_dir)
    sin_assets = "" if assets else (
        '<section class="card"><p class="empty">Este run no tiene imagenes de '
        "Ultralytics. Suele ocurrir cuando el entrenamiento no llego a la "
        "primera validacion.</p></section>"
    )

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Informe de entrenamiento</title>
<script src="{APEXCHARTS_CDN}"></script>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: system-ui, sans-serif; margin: 0; padding: 2rem;
         background: #0b1220; color: #e6edf7; }}
  h1 {{ margin: 0 0 .25rem; font-size: 1.6rem; }}
  .sub {{ color: #8fa3bf; margin: 0 0 1.25rem; }}
  .card {{ background: #131c2e; border-radius: 12px; padding: 1.25rem 1.5rem;
           margin-bottom: 1.25rem; }}
  .card h2 {{ margin: 0 0 .25rem; font-size: 1.1rem; }}
  .stats {{ display: flex; flex-wrap: wrap; gap: 1rem; margin-bottom: 1.5rem; }}
  .stat {{ background: #131c2e; border-radius: 12px; padding: 1rem 1.5rem;
           min-width: 8rem; flex: 1; }}
  .stat-label {{ display: block; color: #8fa3bf; font-size: .8rem;
                 text-transform: uppercase; letter-spacing: .04em; }}
  .stat-value {{ display: block; font-size: 1.5rem; font-weight: 600;
                 margin-top: .25rem; }}
  img {{ max-width: 100%; border-radius: 8px; }}
  .empty {{ color: #8fa3bf; font-style: italic; }}

  /* Cabecera de metadatos del job */
  .chips {{ display: flex; flex-wrap: wrap; gap: .5rem; margin-bottom: 1.5rem; }}
  .chip {{ background: #131c2e; border: 1px solid #1e2b45; border-radius: 999px;
           padding: .3rem .8rem; font-size: .85rem; }}
  .chip-k {{ color: #8fa3bf; margin-right: .4rem; }}

  /* Rejillas de imagenes de Ultralytics */
  .grid {{ display: grid; gap: 1rem; }}
  .grid-2 {{ grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); }}
  .img {{ margin: 0; }}
  .img figcaption {{ color: #8fa3bf; font-size: .8rem; margin-top: .4rem; }}

  /* Par etiqueta-real / prediccion, uno al lado del otro */
  .pair {{ margin-bottom: 1.5rem; }}
  .pair:last-child {{ margin-bottom: 0; }}
  .pair h3 {{ margin: 0 0 .5rem; font-size: .9rem; color: #8fa3bf;
              font-weight: 600; }}
  .pair-row {{ display: grid; gap: 1rem;
               grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }}

  /* args.yaml, plegado por defecto */
  details.card > summary {{ cursor: pointer; font-size: 1.1rem; font-weight: 600; }}
  details.card[open] > summary {{ margin-bottom: .5rem; }}
  pre {{ background: #0b1220; border: 1px solid #1e2b45; border-radius: 8px;
         padding: 1rem; overflow-x: auto; font-size: .8rem; line-height: 1.45;
         margin: 0; }}

  @media (max-width: 640px) {{ body {{ padding: 1rem; }} }}
</style>
</head>
<body>
<h1>Informe de entrenamiento</h1>
<p class="sub">{_esc(run_dir.name)} &middot; {len(rows)} epochs registradas</p>
{chips}

<div class="stats">{stats}</div>
{sections}
{assets}
{sin_assets}
{args_block}

<script>
const opts = [
{payload}
];
{init}
</script>
</body>
</html>"""