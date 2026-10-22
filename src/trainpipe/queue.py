"""Cola de trabajos sobre Postgres.

El claim usa FOR UPDATE SKIP LOCKED. Con un solo worker es redundante, pero
evita el doble entrenamiento si alguien levanta un segundo worker por error.
"""

import uuid

from sqlalchemy import func, select, text, update
from sqlalchemy.orm import Session

from trainpipe.models import ALLOWED_TRANSITIONS, JobStatus, TrainingJob

# Un job 'running' cuyo worker murio. Sin esta recuperacion el job queda
# 'running' para siempre y BLOQUEA LA COLA ENTERA: el unico worker sigue
# creyendo que esta ocupado y no reclama nada mas.
RECOVERABLE = {JobStatus.RUNNING}


class TransitionError(RuntimeError):
    pass


def transition(job: TrainingJob, new_status: JobStatus) -> None:
    if new_status not in ALLOWED_TRANSITIONS[job.status]:
        raise TransitionError(f"{job.status.value} -> {new_status.value} no permitido")
    job.status = new_status


def recover_interrupted(session: Session) -> int:
    """Al arrancar el worker. Devuelve cuantos jobs fueron recuperados."""
    result = session.execute(
        update(TrainingJob)
        .where(TrainingJob.status.in_(RECOVERABLE))
        .values(status=JobStatus.INTERRUPTED, error="Worker detenido durante el entrenamiento")
        .execution_options(synchronize_session=False)
    )
    session.commit()
    return result.rowcount


def claim_next_job(session: Session) -> TrainingJob | None:
    """Reclamar el siguiente job 'queued' de forma atomica."""
    stmt = (
        select(TrainingJob.id)
        .where(TrainingJob.status == JobStatus.QUEUED)
        .order_by(TrainingJob.created_at)
        .limit(1)
        .with_for_update(skip_locked=True)
    )
    row = session.execute(stmt).first()
    if row is None:
        session.rollback()
        return None

    job = session.get(TrainingJob, row.id)
    transition(job, JobStatus.RUNNING)
    job.started_at = func.now()
    session.commit()
    session.refresh(job)
    return job


def request_cancel(session: Session, job_id: uuid.UUID) -> TrainingJob:
    """Marca cancelacion. El worker la ve en su proximo poll y mata el proceso.

    No se mata desde aqui: la API no debe señalar procesos, y el worker esta
    bloqueado en `subprocess.wait()` del entrenamiento activo.
    """
    job = session.get(TrainingJob, job_id)
    if job is None:
        raise LookupError(job_id)
    if job.status in {JobStatus.SUCCEEDED, JobStatus.CANCELLED}:
        raise TransitionError(f"No se puede cancelar un job en {job.status.value}")
    job.cancel_requested = True
    if job.status == JobStatus.QUEUED:
        # Aun no corre: se cancela directo, sin tocar el worker.
        transition(job, JobStatus.CANCELLED)
    session.commit()
    return job


def is_cancel_requested(session: Session, job_id: uuid.UUID) -> bool:
    return bool(
        session.execute(
            select(TrainingJob.cancel_requested).where(TrainingJob.id == job_id)
        ).scalar()
    )


def heartbeat(session: Session, job_id: uuid.UUID, **fields) -> None:
    """Actualiza progreso sin tocar la transicion de estado."""
    if not fields:
        return
    session.execute(
        update(TrainingJob)
        .where(TrainingJob.id == job_id)
        .values(**fields)
        .execution_options(synchronize_session=False)
    )
    session.commit()


def database_ready(session: Session) -> bool:
    try:
        session.execute(text("SELECT 1"))
    except Exception:
        return False
    return True