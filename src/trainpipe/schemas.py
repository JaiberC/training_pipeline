"""Esquemas Pydantic de la API. Sin dependencias de torch."""

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from trainpipe.models import JobStatus, Task


class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    task: Task
    model: str
    classes: list[str] = Field(default_factory=list)
    description: str | None = None


class ProjectUpdate(BaseModel):
    task: Task | None = None
    model: str | None = None
    classes: list[str] | None = None
    description: str | None = None


class ProjectOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    task: Task
    model: str
    classes: list[str]
    description: str | None
    created_at: datetime
    updated_at: datetime


class DatasetRegister(BaseModel):
    """Registra un dataset ya presente en el volumen (sin subirlo)."""

    name: str = Field(..., min_length=1, max_length=128)


class DatasetOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    project_id: uuid.UUID
    name: str
    rel_path: str
    num_images: int
    label_format: str
    created_at: datetime


class InitRequest(BaseModel):
    dataset_id: uuid.UUID
    params: dict[str, Any] = Field(default_factory=dict)
    #: Sobrescribe el modelo del proyecto SOLO para este job. Permite entrenar
    #: yolo26n, yolo26s, yolo26l... sobre el mismo proyecto y dataset sin
    #: tener que cambiar la definicion del proyecto cada vez.
    model: str | None = None
    #: Sobrescribe las clases SOLO para este job, y las propaga al proyecto y al
    #: dataset.yaml. A diferencia de `model`, aqui si se sincroniza: los nombres
    #: de clase los lee Ultralytics del dataset.yaml, no de la base de datos, y
    #: por eso tienen que decir lo mismo en los dos sitios.
    classes: list[str] | None = None


class InitResponse(BaseModel):
    job_id: uuid.UUID
    status: JobStatus
    run_dir: str
    #: Clases con las que va a entrenar este job, ya materializadas en el
    #: dataset.yaml. Es la respuesta de verdad: no hay que ir a mirar ficheros.
    classes: list[str] = Field(default_factory=list)


class JobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    project_id: uuid.UUID
    dataset_id: uuid.UUID
    model: str
    params: dict[str, Any]
    status: JobStatus
    epoch: int | None
    total_epochs: int | None
    metrics: dict[str, Any]
    error: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class StatusOut(JobOut):
    cancel_requested: bool
    progress: float | None = None