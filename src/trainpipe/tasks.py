"""Catalogo de modelos y validacion task <-> dataset.

Se ejecuta en `init`, antes de encolar. Es deliberado: un job invalido debe
volver como error HTTP, nunca ocupar la cola y bloquear al worker.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from trainpipe.models import Task


class ValidationError(ValueError):
    """Error de configuracion que se devuelve al cliente como 4xx."""


@dataclass(frozen=True)
class ModelSpec:
    model: str
    task: Task
    family: str


YOLO26_DETECT = [ModelSpec(f"yolo26{n}.pt", Task.DETECT, "yolo26") for n in "nsmlx"]
YOLO26_SEGMENT = [
    ModelSpec(f"yolo26{n}-seg.pt", Task.SEGMENT, "yolo26") for n in "nsmlx"
]
# RT-DETR solo detecta: no existe rtdetr-seg.pt (ver docs de Ultralytics).
RTDETR_DETECT = [
    ModelSpec("rtdetr-l.pt", Task.DETECT, "rtdetr"),
    ModelSpec("rtdetr-x.pt", Task.DETECT, "rtdetr"),
]

CATALOG: list[ModelSpec] = YOLO26_DETECT + YOLO26_SEGMENT + RTDETR_DETECT

LABEL_COLUMNS_BOX = 5  # class x_center y_center width height
LABEL_COLUMNS_POLYGON_MIN = 7  # class x1 y1 x2 y2 ... (pares)


def validate_model(model: str, task: Task) -> ModelSpec:
    """Comprueba que el modelo exista y soporte la tarea pedida."""
    for spec in CATALOG:
        if spec.model == model:
            if spec.task != task:
                raise ValidationError(
                    f"'{model}' es un modelo de {spec.task.value}, "
                    f"pero el proyecto declara task={task.value}."
                )
            return spec
    raise ValidationError(
        f"Modelo desconocido: '{model}'. Disponibles: "
        + ", ".join(s.model for s in CATALOG)
    )


def validate_project_task(model: str, task: Task) -> None:
    """Atajo semantico para la combinacion task/familia."""
    spec = validate_model(model, task)
    if spec.family == "rtdetr" and task == Task.SEGMENT:  # pragma: no cover
        raise ValidationError("RT-DETR no soporta segmentacion.")


LAYOUT_STANDARD = "yolo"
LAYOUT_ROBOFLOW = "roboflow"


def detect_layout(dataset_dir: Path) -> str:
    """Distingue los dos layouts de dataset que se ven en la practica.

    - `yolo`:     images/train + labels/train, images/val + labels/val
    - `roboflow`: train/images + train/labels, valid/images + valid/labels
                  (Roboflow anida al reves y usa `valid`, no `val`)

    Roboflow es la fuente mas comun de datasets terminos/nocturnos, asi que
    soportarlo evita reorganizar 26 000 ficheros a mano.
    """
    if (dataset_dir / "images" / "train").is_dir():
        return LAYOUT_STANDARD
    if (dataset_dir / "train" / "images").is_dir():
        return LAYOUT_ROBOFLOW
    # Sin train/, pero con una sola raiz de imagenes: se asume YOLO plano.
    if (dataset_dir / "images").is_dir():
        return LAYOUT_STANDARD
    return "unknown"


def splits_for_layout(dataset_dir: Path, layout: str) -> dict[str, str]:
    """Rutas relativas de train/val/test segun el layout detectado."""
    if layout == LAYOUT_ROBOFLOW:
        val_dir = "valid" if (dataset_dir / "valid" / "images").is_dir() else "test"
        return {
            "train": "train/images",
            "val": f"{val_dir}/images",
            "test": "test/images",
        }
    if layout == LAYOUT_STANDARD:
        return {"train": "images/train", "val": "images/val"}
    return {"train": "images", "val": "images"}


def label_format_from_sample(label_file: Path) -> str:
    """'polygons' si alguna linea tiene mas de 5 columnas, 'boxes' si no."""
    columns = LABEL_COLUMNS_BOX
    try:
        with label_file.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                columns = len(line.split())
                break
    except OSError:
        return "unknown"
    return "polygons" if columns >= LABEL_COLUMNS_POLYGON_MIN else "boxes"


def _label_files(dataset_dir: Path) -> list[Path]:
    """Los .txt de etiquetas, este donde este el layout (yolo o roboflow)."""
    for base in (dataset_dir / "labels", dataset_dir):
        if not base.is_dir():
            continue
        found = sorted(base.rglob("*.txt"))
        if found:
            return found
    return []


def detect_dataset_label_format(dataset_dir: Path) -> str:
    """Inspecciona labels/ para deducir el formato real del dataset.

    Busca en las carpetas de labels que existen de verdad, segun el layout: en
    `yolo` es `labels/`, en Roboflow es `*/labels/`.
    """
    samples = _label_files(dataset_dir)
    if not samples:
        return "unknown"
    # Una sola muestra puede mentir (una imagen con caja y otra con poligono).
    # Se revisan las primeras 20 y gana la mayoria.
    counts = {"boxes": 0, "polygons": 0}
    for f in samples[:20]:
        counts[label_format_from_sample(f)] += 1
    return "polygons" if counts["polygons"] >= counts["boxes"] else "boxes"


def label_class_indices(dataset_dir: Path) -> set[int]:
    """Indices de clase que aparecen de verdad en las etiquetas.

    Es la UNICA fuente fiable del numero de clases. El `data.yaml` del ZIP se
    reescribe (las rutas relativas no funcionan en Ultralytics) y `--clases` lo
    elige el usuario, o no lo elige. Sin esto, un dataset de 3 clases usado sin
    `--clases` genera `nc: 1` y el entrenamiento pierde dos clases en silencio.
    """
    indices: set[int] = set()
    for f in _label_files(dataset_dir):
        try:
            with f.open("r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        indices.add(int(float(line.split()[0])))
                    except (ValueError, IndexError):
                        continue
        except OSError:
            continue
    return indices


def validate_task_matches_dataset(task: Task, label_format: str) -> None:
    """Falla en init, no a mitad del entrenamiento."""
    if label_format == "unknown":
        return  # no inspeccionable: se deja pasar
    if task == Task.SEGMENT and label_format == "boxes":
        raise ValidationError(
            "El proyecto declara task=segment pero las etiquetas del dataset son "
            "cajas (5 columnas). Un modelo -seg.pt sobre labels de caja falla "
            "durante el entrenamiento. Usa task=detect o sube etiquetas poligonales."
        )
    if task == Task.DETECT and label_format == "polygons":
        raise ValidationError(
            "El proyecto declara task=detect pero las etiquetas del dataset son "
            "poligonales. Un modelo de deteccion ignora los vertices de la "
            "mascara. Usa task=segment."
        )


def apply_family_defaults(family: str, params: dict) -> dict:
    """Aplica correcciones que la documentacion exige y que son silenciosas si
    no se fijan.

    Los hiperparametros llegan en `extra` (ver api.init_training), asi que se
    revisan tanto a nivel superior como dentro de `extra`.
    """
    params = dict(params)
    extra = dict(params.get("extra") or {})
    flat = {**params, **extra}

    if family == "rtdetr" and flat.get("deterministic") is True:
        # Su atencion deformable no tiene backward determinista en CUDA:
        # con deterministic=True el run no es reproducible y pierde throughput.
        if "deterministic" in extra:
            extra["deterministic"] = False
        else:
            params["deterministic"] = False

    # optimizer='auto' elige MuSGD por encima de 10 000 iteraciones e IGNORA
    # lr0. Si el usuario fijo lr0, hay que fijar tambien el optimizador.
    if "lr0" in flat and "optimizer" not in flat:
        if "optimizer" in extra:
            extra["optimizer"] = "AdamW"
        else:
            params["optimizer"] = "AdamW"

    params["extra"] = extra
    return params


_SAFE_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def validate_resource_name(name: str) -> str:
    """Los endpoints reciben nombres, nunca rutas. Esto evita path traversal."""
    if not _SAFE_NAME.match(name):
        raise ValidationError(
            f"Nombre de recurso invalido: '{name}'. Solo se permiten minusculas, "
            "digitos, guion y guion bajo (max 64 caracteres)."
        )
    return name