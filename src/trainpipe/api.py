"""API FastAPI. NO importa torch ni ultralytics (ver AGENTS.md)."""

import io
import logging
import shutil
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import yaml
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from trainpipe import report, tasks
from trainpipe.config import get_settings
from trainpipe.db import get_session
from trainpipe.models import Dataset, JobStatus, Project, TrainingJob
from trainpipe.queue import TransitionError, request_cancel, transition
from trainpipe.schemas import (
    DatasetOut,
    DatasetRegister,
    InitRequest,
    InitResponse,
    JobOut,
    ProjectCreate,
    ProjectOut,
    ProjectUpdate,
    StatusOut,
)

log = logging.getLogger("trainpipe.api")
settings = get_settings()

app = FastAPI(
    title="training_pipeline",
    version="0.1.0",
    description="Servicio de entrenamiento YOLO26 / RT-DETR. AGPL-3.0.",
)


@app.on_event("startup")
def _startup() -> None:
    for path in (settings.datasets_root, settings.runs_root, settings.models_root):
        path.mkdir(parents=True, exist_ok=True)
    log.info("API lista. datasets=%s runs=%s", settings.datasets_root, settings.runs_root)


@app.get("/health")
def health(session: Session = Depends(get_session)) -> dict:
    """Estado del servicio.

    `schema` importa mas de lo que parece: el compose arranca el worker
    cuando esta API esta sana, asi que un `ok` con el esquema ausente haria
    que el worker se levantara contra una base sin tablas. Antes solo hacia
    SELECT 1, que funciona aunque no exista ninguna tabla.
    """
    db_ok = True
    schema_ok = True
    try:
        session.execute(select(1))
        session.execute(select(TrainingJob.id).limit(1))
        session.execute(select(Project.id).limit(1))
    except Exception:  # noqa: BLE001
        db_ok = False
        schema_ok = False

    return {
        "status": "ok" if db_ok and schema_ok else "degraded",
        "database": db_ok,
        "schema": schema_ok,
    }


# ───────────────────────────── Proyectos ─────────────────────────────

@app.post("/api/projects", response_model=ProjectOut, status_code=201)
def create_project(payload: ProjectCreate, session: Session = Depends(get_session)):
    try:
        tasks.validate_resource_name(payload.name)
        spec = tasks.validate_model(payload.model, payload.task)
    except tasks.ValidationError as exc:
        raise HTTPException(422, str(exc)) from exc

    if session.scalar(select(Project).where(Project.name == payload.name)):
        raise HTTPException(409, f"El proyecto '{payload.name}' ya existe.")

    project = Project(
        name=payload.name,
        task=payload.task,
        model=payload.model,
        classes=payload.classes,
        description=payload.description,
    )
    session.add(project)
    session.commit()
    session.refresh(project)
    (settings.dataset_dir(project.name)).mkdir(parents=True, exist_ok=True)
    return project


@app.get("/api/projects", response_model=list[ProjectOut])
def list_projects(session: Session = Depends(get_session)):
    return session.scalars(select(Project).order_by(Project.name)).all()


@app.get("/api/projects/{project_id}", response_model=ProjectOut)
def get_project(project_id: uuid.UUID, session: Session = Depends(get_session)):
    project = session.get(Project, project_id)
    if project is None:
        raise HTTPException(404, "Proyecto no encontrado.")
    return project


@app.put("/api/projects/{project_id}", response_model=ProjectOut)
def update_project(
    project_id: uuid.UUID,
    payload: ProjectUpdate,
    session: Session = Depends(get_session),
):
    project = session.get(Project, project_id)
    if project is None:
        raise HTTPException(404, "Proyecto no encontrado.")

    task = payload.task or project.task
    model = payload.model or project.model
    try:
        tasks.validate_model(model, task)
    except tasks.ValidationError as exc:
        raise HTTPException(422, str(exc)) from exc

    project.task = task
    project.model = model
    if payload.classes is not None:
        project.classes = payload.classes
    if payload.description is not None:
        project.description = payload.description
    session.commit()
    session.refresh(project)
    return project


@app.delete("/api/projects/{project_id}", status_code=204)
def delete_project(project_id: uuid.UUID, session: Session = Depends(get_session)):
    project = session.get(Project, project_id)
    if project is None:
        raise HTTPException(404, "Proyecto no encontrado.")

    # Cancelar antes de borrar: si no, el worker sigue entrenando sobre un
    # directorio que ya no existe.
    for job in project.jobs:
        if job.status in {JobStatus.QUEUED, JobStatus.RUNNING}:
            try:
                request_cancel(session, job.id)
            except TransitionError:
                pass

    shutil.rmtree(settings.dataset_dir(project.name), ignore_errors=True)
    # Tambien los runs. Sin esto, cada proyecto borrado dejaba su directorio de
    # runs (con los pesos .pt dentro) y el volumen crecia sin limite.
    for job in project.jobs:
        shutil.rmtree(settings.run_dir(str(job.id)), ignore_errors=True)

    session.delete(project)
    session.commit()
    return Response(status_code=204)


@app.get("/api/projects/{project_id}/export")
def export_project(project_id: uuid.UUID, session: Session = Depends(get_session)):
    """Descarga el proyecto como YAML. El contenedor NO sube nada a github;
    esto es para que el usuario lo commitee si quiere reproducibilidad."""
    project = session.get(Project, project_id)
    if project is None:
        raise HTTPException(404, "Proyecto no encontrado.")
    payload = yaml.safe_dump(
        {
            "name": project.name,
            "task": project.task.value,
            "model": project.model,
            "classes": project.classes,
            "description": project.description,
        },
        sort_keys=False,
        allow_unicode=True,
    )
    return Response(
        content=payload,
        media_type="application/yaml",
        headers={"Content-Disposition": f'attachment; filename="{project.name}.yaml"'},
    )


# ───────────────────────────── Datasets ──────────────────────────────

ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".txt", ".yaml", ".yml"}


@app.post("/api/projects/{project_id}/datasets", response_model=DatasetOut, status_code=201)
async def upload_dataset(
    project_id: uuid.UUID,
    archive: UploadFile = File(...),
    name: str | None = Form(
        None,
        description="Nombre del dataset. Si se omite, se usa el nombre del ZIP, "
        "que suele traer mayusculas y espacios y entonces es rechazado.",
    ),
    session: Session = Depends(get_session),
):
    project = session.get(Project, project_id)
    if project is None:
        raise HTTPException(404, "Proyecto no encontrado.")

    # El nombre explicito manda; el del ZIP es solo el respaldo. Sin esto,
    # `PeopleDetectionThermal.zip` se rechaza aunque el cliente ya haya elegido
    # un nombre valido.
    name = (name or archive.filename or "").removesuffix(".zip")
    try:
        tasks.validate_resource_name(name)
    except tasks.ValidationError as exc:
        raise HTTPException(422, str(exc)) from exc

    target = settings.dataset_dir(project.name) / name
    if target.exists():
        raise HTTPException(409, f"El dataset '{name}' ya existe.")
    target.mkdir(parents=True, exist_ok=True)

    staging = target.parent / f".upload-{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    try:
        archive.file.seek(0)
        with staging.joinpath("dataset.zip").open("wb") as out:
            shutil.copyfileobj(archive.file, out)

        extracted = staging / "extracted"
        extracted.mkdir()
        shutil.unpack_archive(staging / "dataset.zip", extracted, format="zip")

        # Defensa de zip-slip: `shutil.unpack_archive` NO sanea rutas, asi que
        # una entrada tipo "../../etc/cron.d/x" escaparia del directorio de
        # destino. Se comprueba que cada ruta resuelta siga dentro.
        root = extracted.resolve()
        for item in extracted.rglob("*"):
            if item.is_dir():
                continue  # los directorios (images/, labels/) son legitimos
            resolved = item.resolve()
            if not resolved.is_relative_to(root):
                raise HTTPException(422, f"Ruta fuera del dataset: {item.name}")
            if item.is_symlink() or item.suffix.lower() not in ALLOWED_SUFFIXES:
                raise HTTPException(
                    422, f"Contenido no permitido en el dataset: {item.name}"
                )
        # Si el ZIP trae una UNICA carpeta envoltorio (lo normal cuando se
        # comprime la carpeta y no su contenido), se entra en ella. Sin esto el
        # dataset queda con un nivel de mas, el layout no se detecta y el
        # entrenamiento falla con "missing path".
        _flatten_extracted(extracted, target)
    except HTTPException:
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(target, ignore_errors=True)
        raise
    except Exception as exc:  # noqa: BLE001
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(target, ignore_errors=True)
        raise HTTPException(400, f"No se pudo procesar el dataset: {exc}") from exc
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    return _register_dataset_row(session, project, target, name)


def _read_archive_class_names(target: Path) -> list[str]:
    """Nombres de clase que declara el data.yaml del ZIP, si viene.

    El servicio reescribe SIEMPRE su propio `dataset.yaml` porque las rutas
    relativas no funcionan en Ultralytics, asi que el YAML original del ZIP no
    se usaba para nada. Antes se ignoraba entero, y un modelo de edificios
    acababa etiquetado como `person`.
    """
    for nombre in ("data.yaml", "dataset.yaml", "data.yml", "dataset.yml"):
        path = target / nombre
        if not path.is_file():
            continue
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8", errors="ignore"))
        except (yaml.YAMLError, OSError):
            continue
        if not isinstance(data, dict):
            continue
        names = data.get("names")
        if isinstance(names, dict):
            # Ultralytics acepta `names: {0: a, 1: b}`; se ordena por indice.
            try:
                return [str(names[k]) for k in sorted(names, key=int)]
            except (TypeError, ValueError):
                continue
        if isinstance(names, (list, tuple)):
            return [str(n) for n in names]
    return []


def _resolve_classes(
    declaradas: list[str], indices: set[int], archive_names: list[str], dataset_name: str
) -> list[str]:
    """Decide los nombres de clase, y falla si no cuadran con las etiquetas.

    Precedencia:
      1. Lo que el usuario paso con `--clases`.
      2. Los `names` que declara el data.yaml del ZIP.
      3. `class_0`, `class_1`, ... generados.

    El numero de clases sale SIEMPRE de los indices que aparecen en las
    etiquetas, nunca de lo que el usuario escribio. Declarar una clase de mas
    es inocuo; declarar de menos hace que el entrenamiento ignore las demas en
    silencio, que es el fallo que esto evita.
    """
    necesarias = (max(indices) + 1) if indices else len(archive_names)
    sugerencia = ", ".join(archive_names) if archive_names else ""

    if declaradas:
        if necesarias and len(declaradas) < necesarias:
            ejemplo = f', por ejemplo:  --clases "{sugerencia}"' if sugerencia else ""
            raise HTTPException(
                422,
                f"El dataset '{dataset_name}' usa {necesarias} clases "
                f"(indices 0..{necesarias - 1}) y el proyecto solo declara "
                f"{len(declaradas)}: {', '.join(declaradas)}.\n"
                "Los nombres van en el MISMO ORDEN que los numeros de las "
                f"etiquetas. Declara las {necesarias}{ejemplo}.\n"
                "Declarar mas clases de las que hay no pasa nada; declarar menos "
                "haria que el entrenamiento ignorase las que faltan sin avisar.",
            )
        return list(declaradas)

    if archive_names:
        if necesarias and len(archive_names) < necesarias:
            raise HTTPException(
                422,
                f"El dataset '{dataset_name}' usa {necesarias} clases pero su "
                f"data.yaml solo declara {len(archive_names)}: {sugerencia}. "
                "Declara las clases con --clases.",
            )
        return archive_names

    if necesarias:
        return [f"class_{i}" for i in range(necesarias)]
    return ["class_0"]


def _flatten_extracted(extracted: Path, target: Path) -> None:
    """Mueve el contenido del ZIP al directorio del dataset.

    Si el ZIP trae una UNICA carpeta envoltorio (lo normal cuando se comprime
    la carpeta y no su contenido), se entra en ella. Sin esto el dataset
    queda con un nivel de mas, el layout no se detecta y el entrenamiento
    falla con "missing path".
    """
    contenido = list(extracted.iterdir())
    origen = (
        contenido[0] if len(contenido) == 1 and contenido[0].is_dir() else extracted
    )
    for item in origen.iterdir():
        shutil.move(str(item), str(target / item.name))


def _materialize_dataset_yaml(dataset_dir: Path, classes: list[str]) -> Path:
    """Escribe el dataset.yaml del dataset con las clases dadas.

    El dataset.yaml es lo que lee Ultralytics (`data:` apunta a el), asi que es
    el fichero que de verdad manda. Se reescribe en cada `init` para que no
    pueda quedar viejo: con un proyecto renombrado despues de subir el
    dataset, el YAML se quedaba con los nombres anteriores y el modelo salia
    etiquetado con la clase equivocada.
    """
    layout = tasks.detect_layout(dataset_dir)
    path = dataset_dir / "dataset.yaml"
    path.write_text(build_dataset_yaml(dataset_dir, classes, layout), encoding="utf-8")
    return path


def _register_dataset_row(session: Session, project: Project, target: Path, name: str):
    """Cuenta imagenes, deduce el formato de labels y materializa dataset.yaml.

    Compartido por `upload` y `register`: los dos terminan dejando el dataset
    en disco y creando su fila en la base de datos.
    """
    num_images = sum(
        1
        for p in target.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    if num_images == 0:
        raise HTTPException(
            422,
            f"No se encontraron imagenes en '{name}'. Se espera "
            "images/train/ + labels/train/ (YOLO) o train/images/ + "
            "train/labels/ (Roboflow).",
        )
    label_format = tasks.detect_dataset_label_format(target)

    classes = _resolve_classes(
        list(project.classes or []),
        tasks.label_class_indices(target),
        _read_archive_class_names(target),
        name,
    )
    # Si el proyecto se creo sin --clases, su registro queda con lo que de
    # verdad tiene el dataset en vez de una lista vacia.
    if not project.classes:
        project.classes = classes
        session.add(project)
        session.flush()

    # El YAML se materializa con la ruta ABSOLUTA dentro del contenedor
    # (/data/datasets/...), no con `path: .`: Ultralytics resuelve un `path`
    # relativo contra su ajuste global `datasets_dir` (por defecto /app/datasets),
    # NO contra la carpeta del YAML, y el entrenamiento falla con
    # "missing path '/app/images/val'".
    #
    # Es portable entre maquinas porque el punto de montaje del volumen es el
    # mismo en todas (/data/datasets), no porque el path sea relativo.
    _materialize_dataset_yaml(target, classes)

    dataset = Dataset(
        project_id=project.id,
        name=name,
        rel_path=f"{project.name}/{name}",
        num_images=num_images,
        label_format=label_format,
    )
    session.add(dataset)
    try:
        session.commit()
    except IntegrityError as exc:
        session.rollback()
        raise HTTPException(
            409, f"El proyecto ya tiene un dataset llamado '{name}'."
        ) from exc
    session.refresh(dataset)
    return dataset


def build_dataset_yaml(target: Path, classes: list[str], layout: str) -> str:
    splits = tasks.splits_for_layout(target, layout)
    return (
        f"# Generado por training_pipeline (layout={layout}). "
        "Ruta absoluta DENTRO del contenedor.\n"
        f"path: {target}\n"
        f"train: {splits['train']}\n"
        f"val: {splits['val']}\n"
        + (f"test: {splits['test']}\n" if "test" in splits else "")
        + f"nc: {len(classes)}\n"
        "names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(classes))
    )


@app.post("/api/projects/{project_id}/datasets/register", response_model=DatasetOut, status_code=201)
def register_dataset(
    project_id: uuid.UUID,
    payload: DatasetRegister,
    session: Session = Depends(get_session),
):
    """Registra un dataset YA presente en el volumen, sin subirlo por la API.

    Para cuando los datos se copian a mano (rsync, red compartida, copia desde
    el servidor de origen). La carpeta debe llamarse
    `${DOCKER_PATH}/datasets/<proyecto>/<nombre>` en el host, es decir
    `/data/datasets/<proyecto>/<nombre>` dentro del contenedor.
    """
    project = session.get(Project, project_id)
    if project is None:
        raise HTTPException(404, "Proyecto no encontrado.")

    name = payload.name
    try:
        tasks.validate_resource_name(name)
    except tasks.ValidationError as exc:
        raise HTTPException(422, str(exc)) from exc

    target = settings.dataset_dir(project.name) / name
    if not target.is_dir():
        raise HTTPException(
            404,
            f"No existe la carpeta '{name}' para el proyecto '{project.name}'. "
            f"Esperaba: {target} (en el host: {settings.datasets_root}/"
            f"{project.name}/{name}).",
        )
    if session.scalar(
        select(Dataset).where(Dataset.project_id == project_id, Dataset.name == name)
    ):
        raise HTTPException(409, f"El dataset '{name}' ya esta registrado.")

    return _register_dataset_row(session, project, target, name)


@app.get("/api/projects/{project_id}/datasets", response_model=list[DatasetOut])
def list_datasets(project_id: uuid.UUID, session: Session = Depends(get_session)):
    return (
        session.scalars(select(Dataset).where(Dataset.project_id == project_id)).all()
    )


@app.delete("/api/projects/{project_id}/datasets/{dataset_id}", status_code=204)
def delete_dataset(
    project_id: uuid.UUID,
    dataset_id: uuid.UUID,
    session: Session = Depends(get_session),
):
    dataset = session.get(Dataset, dataset_id)
    if dataset is None or dataset.project_id != project_id:
        raise HTTPException(404, "Dataset no encontrado.")

    for job in session.scalars(
        select(TrainingJob).where(TrainingJob.dataset_id == dataset_id)
    ).all():
        if job.status in {JobStatus.RUNNING, JobStatus.QUEUED}:
            raise HTTPException(
                409, f"El job {job.id} esta {job.status.value}; cancelalo primero."
            )

    shutil.rmtree(settings.datasets_root / dataset.rel_path, ignore_errors=True)
    session.delete(dataset)
    session.commit()
    return Response(status_code=204)


# ─────────────────────────── Entrenamiento ────────────────────────────

def _get_job(job_id: uuid.UUID, session: Session) -> TrainingJob:
    job = session.get(TrainingJob, job_id)
    if job is None:
        raise HTTPException(404, "Job no encontrado.")
    return job


@app.post("/api/trainings/{project_id}/init", response_model=InitResponse, status_code=201)
def init_training(
    project_id: uuid.UUID,
    payload: InitRequest,
    session: Session = Depends(get_session),
):
    """Valida y crea el job en estado draft. NO encola: un job invalido no debe
    ocupar la cola ni bloquear al worker."""
    project = session.get(Project, project_id)
    if project is None:
        raise HTTPException(404, "Proyecto no encontrado.")
    dataset = session.get(Dataset, payload.dataset_id)
    if dataset is None or dataset.project_id != project_id:
        raise HTTPException(404, "Dataset no encontrado o no pertenece al proyecto.")

    # Las clases se resuelven y se MATERIALIZAN aqui, no al subir el dataset.
    # El dataset.yaml es lo que lee Ultralytics, y se queda viejo en cuanto el
    # proyecto cambia: por eso se reescribe en cada init, y por eso la base de
    # datos y el fichero se actualizan a la vez.
    dataset_dir = settings.datasets_root / dataset.rel_path
    if not dataset_dir.is_dir():
        raise HTTPException(
            422,
            f"El dataset '{dataset.name}' no esta en disco ({dataset_dir}). "
            "Vuelve a subirlo o registralo con POST /datasets/register.",
        )
    clases = _resolve_classes(
        list(payload.classes or []) or list(project.classes or []),
        tasks.label_class_indices(dataset_dir),
        _read_archive_class_names(dataset_dir),
        dataset.name,
    )
    if list(project.classes or []) != clases:
        project.classes = clases
        session.add(project)
    _materialize_dataset_yaml(dataset_dir, clases)

    params = dict(payload.params)
    model = payload.model or project.model
    try:
        spec = tasks.validate_model(model, project.task)
        tasks.validate_task_matches_dataset(project.task, dataset.label_format)
    except tasks.ValidationError as exc:
        raise HTTPException(422, str(exc)) from exc

    # El worker solo lee estas claves de forma explicita; TODO lo demas se
    # transmite a Ultralytics en `extra`. Sin esto, un parametro como
    # `workers=2` se descartaba en silencio y Ultralytics usaba su valor por
    # defecto (8), sin avisar.
    known = {"epochs", "imgsz", "batch", "seed", "device", "family", "extra", "resume"}
    extra = dict(params.get("extra") or {})
    for key, value in params.items():
        if key not in known:
            extra[key] = value
    params = {k: v for k, v in params.items() if k in known and k != "extra"}
    params["extra"] = extra

    params["family"] = spec.family
    params = tasks.apply_family_defaults(spec.family, params)

    job = TrainingJob(
        project_id=project_id,
        dataset_id=dataset.id,
        model=model,
        params=params,
        status=JobStatus.DRAFT,
    )
    session.add(job)
    session.commit()
    session.refresh(job)

    # save_dir se usa tal cual; project/name autoincrementan (name, name2...).
    job.run_dir = str(settings.run_dir(str(job.id)))
    session.commit()
    session.refresh(job)

    return InitResponse(
        job_id=job.id, status=job.status, run_dir=job.run_dir, classes=clases
    )


@app.post("/api/trainings/{job_id}/start", status_code=202)
def start_training(job_id: uuid.UUID, session: Session = Depends(get_session)):
    job = _get_job(job_id, session)
    if job.status != JobStatus.DRAFT:
        raise HTTPException(409, f"El job esta en {job.status.value}; solo se inicia un draft.")
    try:
        transition(job, JobStatus.QUEUED)
    except TransitionError as exc:
        raise HTTPException(409, str(exc)) from exc
    session.commit()
    return {"job_id": str(job.id), "status": job.status.value}


@app.get("/api/trainings/{job_id}/status", response_model=StatusOut)
def job_status(job_id: uuid.UUID, session: Session = Depends(get_session)):
    job = _get_job(job_id, session)
    data = StatusOut.model_validate(job)
    if job.total_epochs and job.epoch:
        data.progress = round(min(job.epoch / job.total_epochs, 1.0), 4)
    return data


@app.post("/api/trainings/{job_id}/cancel", status_code=202)
def cancel_training(job_id: uuid.UUID, session: Session = Depends(get_session)):
    """Marca cancelacion. En un job en ejecucion la materializa el worker:
    mata el proceso y luego borra runs/<job_id>/.

    Descarga los artefactos ANTES de cancelar: cancelarlos descarta los pesos.
    """
    job = _get_job(job_id, session)
    try:
        job = request_cancel(session, job.id)
    except TransitionError as exc:
        raise HTTPException(409, str(exc)) from exc

    # Un job queued aun no corre: se limpia aqui, no hace falta el worker.
    if job.status == JobStatus.CANCELLED:
        shutil.rmtree(settings.run_dir(str(job.id)), ignore_errors=True)
        session.refresh(job)
    return {"job_id": str(job.id), "status": job.status.value}


@app.post("/api/trainings/{job_id}/restart", status_code=202)
def restart_training(
    job_id: uuid.UUID,
    mode: str = Query("scratch", pattern="^(scratch|resume)$"),
    session: Session = Depends(get_session),
):
    """Re-encola un job failed/interrupted.

    `scratch` empieza de cero. `resume` usa resume=True de Ultralytics, que
    restaura epoca, optimizador y LR: no es lo mismo y no se puede cambiar
    durante el run."""
    job = _get_job(job_id, session)
    if job.status not in {JobStatus.FAILED, JobStatus.INTERRUPTED, JobStatus.CANCELLED}:
        raise HTTPException(
            409, f"Solo se reinician jobs failed/interrupted/cancelled (este esta {job.status.value})."
        )
    if mode == "resume":
        last = settings.run_dir(str(job.id)) / "train" / "weights" / "last.pt"
        if not last.exists():
            raise HTTPException(
                409, "No hay checkpoint last.pt para reanudar; usa mode=scratch."
            )

    job.cancel_requested = False
    job.error = None
    job.finished_at = None
    job.epoch = None
    params = dict(job.params)
    params["resume"] = mode == "resume"
    if mode == "scratch":
        shutil.rmtree(settings.run_dir(str(job.id)), ignore_errors=True)
    job.params = params
    transition(job, JobStatus.QUEUED)
    session.commit()
    return {"job_id": str(job.id), "status": job.status.value, "mode": mode}


@app.get("/api/trainings/{job_id}/download")
def download_artifact(
    job_id: uuid.UUID,
    file: str = Query("best.pt", description="best.pt | last.pt | results.csv | report.html"),
    session: Session = Depends(get_session),
):
    job = _get_job(job_id, session)
    # Nunca una ruta arbitraria: se resuelve dentro del run_dir del job.
    allowed = {"best.pt", "last.pt", "results.csv", "report.html"}
    if file not in allowed:
        raise HTTPException(422, f"Artefacto no permitido. Disponibles: {sorted(allowed)}")

    run_dir = Path(job.run_dir or settings.run_dir(str(job.id)))
    # Ultralytics escribe los pesos en <run_dir>/train/weights/ y el resto de
    # artefactos en <run_dir>/train/; el informe lo genera el worker en
    # <run_dir>/. Se buscan las tres rutas en vez de asumir una.
    for path in (
        run_dir / "train" / "weights" / file,
        run_dir / "train" / file,
        run_dir / file,
    ):
        if path.is_file():
            return FileResponse(path, filename=file)
    raise HTTPException(404, f"El artefacto '{file}' no existe todavia.")
    return FileResponse(path, filename=file)


@app.get("/api/trainings/{job_id}/report")
def job_report(job_id: uuid.UUID, session: Session = Depends(get_session)):
    job = _get_job(job_id, session)
    run_dir = Path(job.run_dir or settings.run_dir(str(job.id)))
    html = run_dir / "report.html"
    if not html.exists():
        run_dir.mkdir(parents=True, exist_ok=True)
        project = session.get(Project, job.project_id)
        dataset = session.get(Dataset, job.dataset_id)
        report.generate_report(
            run_dir / "train",
            output=html,
            meta=report.build_meta(
                job,
                project.name if project else "",
                dataset.name if dataset else "",
            ),
        )
    return FileResponse(html, media_type="text/html")


@app.get("/api/trainings/{job_id}/report-assets")
def job_report_assets(job_id: uuid.UUID, session: Session = Depends(get_session)):
    """ZIP con las imagenes y el args.yaml que el informe referencia.

    El informe NO lleva las imagenes embebidas: las referencia por nombre
    relativo. Sin este endpoint, quien se baja el HTML a su carpeta se queda con
    iconos rotos, porque los PNG viven en runs/<job>/train/ y no en el destino.

    A diferencia de /download, aqui NO se acepta ningun nombre de fichero: la
    lista sale cerrada de report.collect_assets. Por eso este endpoint no
    necesita la whitelist, que en /download existe precisamente porque ahi si
    manda el cliente.
    """
    job = _get_job(job_id, session)
    run_dir = Path(job.run_dir or settings.run_dir(str(job.id)))
    assets = report.collect_assets(run_dir / "train")
    if not assets:
        raise HTTPException(
            404, "Este run no tiene imagenes de Ultralytics (puede fallar antes de la primera validacion)."
        )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in assets:
            # arcname = nombre plano: el informe los referencia sin ruta.
            zf.write(path, arcname=path.name)
    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{job.id}-report-assets.zip"'
        },
    )


@app.get("/api/trainings", response_model=list[JobOut])
def list_jobs(
    project_id: uuid.UUID | None = None,
    limit: int = Query(50, le=200),
    session: Session = Depends(get_session),
):
    stmt = select(TrainingJob).order_by(TrainingJob.created_at.desc()).limit(limit)
    if project_id:
        stmt = stmt.where(TrainingJob.project_id == project_id)
    return session.scalars(stmt).all()