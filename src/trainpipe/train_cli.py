"""Proceso de entrenamiento. El UNICO modulo que importa ultralytics/torch.

Se ejecuta como `python -m trainpipe.train_cli` desde el worker, no como
llamada en proceso: `cancel` necesita poder matarlo con SIGTERM/SIGKILL sobre
su grupo de procesos.

Uso:
    python -m trainpipe.train_cli --config /data/runs/<job_id>/train.json
"""

import argparse
import json
import sys
from pathlib import Path

def build_train_kwargs(cfg: dict) -> dict:
    return {
        "data": cfg["data"],
        "epochs": cfg.get("epochs", 100),
        "imgsz": cfg.get("imgsz", 640),
        "batch": cfg.get("batch", 16),
        "project": cfg["project"],
        "name": cfg["name"],
        "exist_ok": True,
        "seed": cfg.get("seed", 0),
        "device": cfg.get("device", "cpu"),
        "verbose": cfg.get("verbose", True),
    } | {k: v for k, v in cfg.get("extra", {}).items() if v is not None}


def main() -> int:
    parser = argparse.ArgumentParser(description="Entrena un modelo YOLO/RT-DETR")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text())

    # Seleccion de dispositivo EN RUNTIME: la misma imagen corre en la Tesla
    # (GPU) y en equipos CPU-only. Nunca se hornea CUDA en el build.
    import torch

    if cfg.get("device", "auto") == "auto":
        cfg["device"] = 0 if torch.cuda.is_available() else "cpu"
        print(f"[train_cli] dispositivo seleccionado: {cfg['device']}", flush=True)

    from ultralytics import RTDETR, YOLO

    family = cfg["family"]
    model_name = cfg["model"]

    # El modelo debe resolverse a un peso concreto. Si es un nombre plano se
    # busca en MODELS_ROOT (pesos subida via API), si no Ultralytics lo descarga.
    candidate = Path(cfg.get("models_root", "/data/models")) / model_name
    weights = str(candidate) if candidate.exists() else model_name

    model = RTDETR(weights) if family == "rtdetr" else YOLO(weights)
    kwargs = build_train_kwargs(cfg)

    try:
        model.train(**kwargs)
    except KeyboardInterrupt:
        print("[train_cli] interrumpido", file=sys.stderr, flush=True)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())