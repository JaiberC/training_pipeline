"""Servicio de entrenamiento YOLO26 / RT-DETR.

Restriccion critica de imports (ver AGENTS.md): este paquete NO debe importar
torch ni ultralytics salvo en `train_cli`. La imagen runtime-api no los instala.
"""

__version__ = "0.1.0"