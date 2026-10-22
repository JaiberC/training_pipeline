"""Modelos ORM. Postgres es la fuente de verdad: nada de esto vive en git."""

import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


# SQLAlchemy persiste por defecto el NOMBRE del miembro del enum ("DETECT").
# La migracion crea el tipo de Postgres con los VALORES ("detect"), asi que sin
# values_callable el primer INSERT falla con InvalidTextRepresentation.
def _enum_values(enum_cls):
    return [member.value for member in enum_cls]


class Task(str, enum.Enum):
    DETECT = "detect"
    SEGMENT = "segment"


class JobStatus(str, enum.Enum):
    DRAFT = "draft"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


#: Transiciones legales. El worker y la API las validan contra esta tabla.
ALLOWED_TRANSITIONS: dict[JobStatus, set[JobStatus]] = {
    JobStatus.DRAFT: {JobStatus.QUEUED, JobStatus.CANCELLED},
    JobStatus.QUEUED: {JobStatus.RUNNING, JobStatus.CANCELLED},
    JobStatus.RUNNING: {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED},
    JobStatus.FAILED: {JobStatus.QUEUED},
    JobStatus.INTERRUPTED: {JobStatus.QUEUED},
    JobStatus.SUCCEEDED: set(),
    JobStatus.CANCELLED: set(),
}


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    task: Mapped[Task] = mapped_column(
        Enum(Task, name="task_enum", values_callable=_enum_values)
    )
    model: Mapped[str] = mapped_column(String(64))
    classes: Mapped[list] = mapped_column(JSON, default=list)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    datasets: Mapped[list["Dataset"]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )
    jobs: Mapped[list["TrainingJob"]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )


class Dataset(Base):
    __tablename__ = "datasets"
    __table_args__ = (
        # El nombre es unico DENTRO del proyecto, no global: dos proyectos
        # distintos pueden tener cada uno su "nocturno_v1".
        UniqueConstraint("project_id", "name", name="uq_datasets_project_name"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(128), index=True)
    #: Ruta RELATIVA a DATASETS_ROOT. Nunca absoluta: un dataset.yaml con rutas
    #: absolutas rompe al desplegar en otra maquina.
    rel_path: Mapped[str] = mapped_column(String(512))
    num_images: Mapped[int] = mapped_column(Integer, default=0)
    #: "boxes" o "polygons", detectado al inspeccionar los labels.
    label_format: Mapped[str] = mapped_column(String(16), default="unknown")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    project: Mapped[Project] = relationship(back_populates="datasets")


class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), index=True
    )

    model: Mapped[str] = mapped_column(String(64))
    params: Mapped[dict] = mapped_column(JSON, default=dict)

    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus, name="job_status_enum", values_callable=_enum_values),
        default=JobStatus.DRAFT,
        index=True,
    )
    #: Lo consulta el worker mientras corre; permite cancelar sin matar al worker.
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)

    run_dir: Mapped[str | None] = mapped_column(String(512), nullable=True)
    pid: Mapped[int | None] = mapped_column(Integer, nullable=True)

    epoch: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_epochs: Mapped[int | None] = mapped_column(Integer, nullable=True)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    project: Mapped[Project] = relationship(back_populates="jobs")