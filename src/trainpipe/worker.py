"""Worker de entrenamiento. Concurrencia 1.

No importa torch ni ultralytics: lanza `train_cli` como subproceso. Eso
mantiene el contenedor worker sin torch en su propio proceso y, sobre todo,
hace que `cancel` pueda matar el entrenamiento de verdad.
"""

import json
import logging
import os
import re
import signal
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from trainpipe.config import get_settings
from trainpipe.db import session_factory
from trainpipe.models import JobStatus, TrainingJob
from trainpipe.queue import (
    claim_next_job,
    heartbeat,
    is_cancel_requested,
    recover_interrupted,
    transition,
)

log = logging.getLogger("trainpipe.worker")

# Fila del resumen de validacion de Ultralytics:
#   "                 person      142     1200     0.85     0.72     0.78    0.85     1200"
#   ^clase  ^imagenes  ^instancias ^P     ^R      ^mAP50  ^mAP50-95 ^... (espejo, descartado)
_PERCLASS_ROW = re.compile(
    r"^\s*(?P<name>\S[^|]*?)\s{2,}"
    r"(?P<images>\d+)\s+(?P<instances>\d+)\s+"
    r"(?P<p>[\d.]+)\s+(?P<r>[\d.]+)\s+(?P<map50>[\d.]+)\s+(?P<map5095>[\d.]+)\s+"
)


def write_per_class_metrics(run_dir: Path, tail: list[str]) -> bool:
    """Saca el mAP50 por clase del resumen de validacion y lo guarda.

    Ultralytics no escribe las metricas por clase en NINGUN fichero: solo las
    imprime por consola. El grafico 'mAP50 por clase' del informe no puede
    existir sin este parseo.

    Es parseo de texto pensado para humanos, asi que va a defensiva: si
    Ultralytics cambia el formato en una version futura, este bloque deja de
    encontrar filas, no escribe nada y el grafico simplemente no aparece. El
    entrenamiento no se ve afectado.
    """
    marker = None
    for i, line in enumerate(tail):
        if "Class Names (all)" in line:
            marker = i
    if marker is None:
        return False

    # Ultralytics imprime los nombres de dos formas distintas segun version:
    # "Class Names (all): car, truck" o "Class Names (all): ['car', 'truck']".
    # Se extraen los tokens que no sean corchetes, comillas ni comas para que
    # valgan las dos.
    raw = tail[marker].split(":", 1)[1]
    names = [t.strip() for t in re.findall(r"[^\[\]'\",]+", raw) if t.strip()]
    if not names:
        return False

    # Las filas van justo antes de la linea de nombres. La primera es "all".
    filas = []
    for line in tail[max(0, marker - 60):marker]:
        m = _PERCLASS_ROW.match(line)
        if m and m.group("name").strip() != "all":
            filas.append((m.group("name").strip(), float(m.group("map50"))))

    if len(filas) < len(names):
        return False

    per_class = dict(zip(names, [v for _, v in filas[: len(names)]]))
    if not per_class:
        return False

    try:
        (run_dir / "per_class_metrics.json").write_text(
            json.dumps(per_class), encoding="utf-8"
        )
    except OSError as exc:  # noqa: BLE001 - el informe no puede tumbar el job
        log.warning("No se pudo escribir per_class_metrics.json: %s", exc)
        return False
    log.info("metricas por clase: %s", per_class)
    return True



def _terminate(proc: subprocess.Popen, grace: float) -> None:
    """Mata el grupo de procesos: Ultralytics lanza dataloaders hijos que
    sobreviven a un SIGTERM al proceso padre.

    Orden obligatorio: primero morir el proceso, DESPUES borrar el directorio.
    """
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError:
        proc.terminate()

    try:
        proc.wait(timeout=grace)
        return
    except subprocess.TimeoutExpired:
        pass

    log.warning("SIGTERM ignorado tras %.0fs; enviando SIGKILL", grace)
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, OSError):
        proc.kill()
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        log.error("El proceso SIGKILL no responde; se continua igual")


def _parse_results_csv(run_dir: Path) -> tuple[int, int | None, dict]:
    """Ultralytics escribe results.csv con una fila por epoca. De ahi sale el
    progreso en vivo que expone /status."""
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return 0, None, {}
    lines = [ln for ln in csv_path.read_text(errors="ignore").splitlines() if ln.strip()]
    if len(lines) < 2:
        return 0, None, {}

    header = [h.strip() for h in lines[0].split(",")]
    last = [c.strip() for c in lines[-1].split(",")]

    epoch = None
    metrics: dict = {}
    for h, v in zip(header, last):
        key = h.strip()
        try:
            num = float(v)
        except ValueError:
            continue
        metrics[key] = num
        if key.startswith("epoch"):
            epoch = int(num)
    return len(lines) - 1, epoch, metrics


def _drain(proc: subprocess.Popen, tail: list[str]) -> None:
    """Vacia el pipe del proceso en segundo plano.

    Leer con `for line in proc.stdout` en el hilo principal NO vale: tqdm
    escribe con \\r y puede no emitir \\n durante minutos, con lo que la
    comprobacion de cancelacion se retrasaria justo cuando mas urge. Ademas el
    buffer del pipe (64 KB) se llenaria y bloquearia al hijo.

    `tail` conserva las ultimas lineas: si el entrenamiento falla, el error
    tiene que quedar registrado en el job, no solo en los logs del contenedor.
    """
    assert proc.stdout is not None
    for line in proc.stdout:
        text = line.rstrip()
        if text:
            tail.append(text)
            del tail[:-200]
        log.debug(text)


def _run_job(job: TrainingJob, session, settings) -> JobStatus:
    run_dir = settings.run_dir(str(job.id))
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = settings.datasets_root / _dataset_rel_path(session, job)
    cfg = {
        "model": job.model,
        "family": job.params.get("family", "yolo26"),
        "data": str(dataset_dir / "dataset.yaml"),
        "epochs": job.params.get("epochs", 100),
        "imgsz": job.params.get("imgsz", 640),
        "batch": job.params.get("batch", 16),
        "seed": job.params.get("seed", 0),
        "device": job.params.get("device", "auto"),
        "project": str(run_dir),
        "name": "train",
        "models_root": str(settings.models_root),
        "extra": job.params.get("extra", {}),
    }
    cfg_path = run_dir / "train.json"
    cfg_path.write_text(json.dumps(cfg))

    # Ultralytics usa project/name y autoincrementa (name, name2, name3).
    # Por eso exist_ok=True y un name fijo: el directorio real queda en
    # runs/<job_id>/train sin sorpresas.
    proc = subprocess.Popen(
        [sys.executable, "-m", "trainpipe.train_cli", "--config", str(cfg_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,  # grupo de procesos propio -> killpg
        # Ultralytics descarga los pesos preentrenados en el DIRECTORIO DE
        # TRABAJO (attempt_download_asset). /app es root y, con el worker
        # corriendo como el uid del host, no es escribible: el entrenamiento
        # falla con "Permission denied: .yolo26s.pt.part". Corriendo en el
        # volumen de modelos la descarga cae ahi y sobrevive al contenedor.
        cwd=str(settings.models_root),
    )
    tail: list[str] = []
    threading.Thread(target=_drain, args=(proc, tail), daemon=True).start()

    session.expire_all()
    job = session.get(TrainingJob, job.id)
    job.pid = proc.pid
    session.commit()
    log.info("job %s iniciado (pid %s)", job.id, proc.pid)

    cancelled = False
    last_epochs = 0
    while proc.poll() is None:
        if is_cancel_requested(session, job.id):
            log.info("cancelacion solicitada para %s", job.id)
            _terminate(proc, settings.cancel_grace_seconds)
            cancelled = True
            break

        epochs_done, epoch, metrics = _parse_results_csv(run_dir / "train")
        if epochs_done and epochs_done != last_epochs:
            last_epochs = epochs_done
            m50 = metrics.get("metrics/mAP50(B)")
            log.info(
                "job %s: epoca %s/%s%s",
                job.id, epoch or epochs_done, cfg["epochs"],
                f" mAP50={m50:.4f}" if m50 is not None else "",
            )
            heartbeat(
                session,
                job.id,
                epoch=epoch or epochs_done,
                total_epochs=cfg["epochs"],
                metrics=metrics,
            )
        time.sleep(settings.worker_poll_interval)

    proc.wait()

    if cancelled:
        # Matar primero, borrar despues.
        shutil.rmtree(run_dir, ignore_errors=True)
        return JobStatus.CANCELLED

    if proc.returncode == 0:
        # Antes del informe: el grafico de mAP por clase lee este fichero.
        write_per_class_metrics(run_dir / "train", tail)
        _generate_report(run_dir, job, session)
        return JobStatus.SUCCEEDED

    if tail:
        session.rollback()
        session.expire_all()
        current = session.get(TrainingJob, job.id)
        if current is not None:
            current.error = (
                f"train_cli termino con codigo {proc.returncode}\n"
                + "\n".join(tail[-40:])
            )[:4000]
            session.commit()
    return JobStatus.FAILED


def _dataset_rel_path(session, job: TrainingJob) -> str:
    from trainpipe.models import Dataset

    ds = session.get(Dataset, job.dataset_id)
    return ds.rel_path if ds else ""


def _generate_report(run_dir: Path, job: TrainingJob, session=None) -> None:
    from trainpipe.report import build_meta, generate_report

    project_name = dataset_name = ""
    if session is not None:
        from trainpipe.models import Dataset, Project

        try:
            project = session.get(Project, job.project_id)
            dataset = session.get(Dataset, job.dataset_id)
            project_name = project.name if project else ""
            dataset_name = dataset.name if dataset else ""
        except Exception as exc:  # noqa: BLE001 - la cabecera es cosmetica
            log.warning("No se pudieron resolver los nombres del informe: %s", exc)

    try:
        generate_report(
            run_dir / "train",
            output=run_dir / "report.html",
            meta=build_meta(job, project_name, dataset_name),
        )
    except Exception as exc:  # noqa: BLE001 - un informe fallido no tumba el job
        log.warning("No se pudo generar el informe de %s: %s", job.id, exc)


def main() -> int:
    # El stdout del entrenamiento (incluidos los tracebacks de los dataloaders)
    # se registra con log.debug en _drain. Con INFO se pierde entero, y un fallo
    # llega al usuario como un EOFError sin relacion con su causa. Con
    # WORKER_LOG_LEVEL=DEBUG se ve la salida completa de Ultralytics.
    level_name = os.environ.get("WORKER_LOG_LEVEL", "INFO").strip().upper()
    level = getattr(logging, level_name, None)
    if not isinstance(level, int):
        log.warning("WORKER_LOG_LEVEL=%r no es un nivel valido; se usa INFO",
                    os.environ.get("WORKER_LOG_LEVEL"))
        level = logging.INFO

    logging.basicConfig(
        level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    settings = get_settings()
    settings.runs_root.mkdir(parents=True, exist_ok=True)
    settings.models_root.mkdir(parents=True, exist_ok=True)

    log.info("worker arrancando (concurrencia 1)")
    session = session_factory()
    try:
        recovered = recover_interrupted(session)
    except Exception as exc:  # noqa: BLE001
        # El caso habitual es arrancar sin migraciones aplicadas. Decirlo claro
        # es mejor que soltar un traceback de psycopg y dejar al worker en bucle
        # de reinicio indefinidamente.
        if "does not exist" in str(exc) or "UndefinedTable" in type(exc).__name__:
            log.error(
                "Las tablas no existen. El entrypoint de la API deberia haber aplicado "
                "las migraciones; si no lo ha hecho, ejecuta: "
                "docker compose run --rm api alembic upgrade head"
            )
        else:
            log.error("No se pudo preparar la cola: %s", exc)
        return 1
    if recovered:
        log.warning("%d job(s) en 'running' marcados como 'interrupted'", recovered)

    # Aviso explicito del dispositivo: sin esto, un despliegue en la Tesla sin
    # el override GPU entrena en CPU durante horas sin dar ningun error.
    probe = subprocess.run(
        [sys.executable, "-c", "import torch;print(torch.cuda.is_available())"],
        capture_output=True,
        text=True,
    )
    cuda = probe.stdout.strip() == "True"
    log.warning("CUDA disponible: %s -> %s", cuda, "device=0" if cuda else "device=cpu")

    while True:
        session.rollback()
        session.expire_all()
        job = claim_next_job(session)
        if job is None:
            time.sleep(settings.worker_poll_interval)
            continue

        log.info("job %s reclamado (status=%s)", job.id, job.status.value)
        try:
            final = _run_job(job, session, settings)
        except Exception as exc:  # noqa: BLE001
            log.exception("job %s fallo con excepcion", job.id)
            session.rollback()
            final = JobStatus.FAILED
            session.expire_all()
            job = session.get(TrainingJob, job.id)
            if job is not None:
                job.error = str(exc)[:2000]

        session.rollback()
        session.expire_all()
        job = session.get(TrainingJob, job.id)
        if job is None:
            continue
        transition(job, final)
        from sqlalchemy import func

        job.finished_at = func.now()
        session.commit()
        log.info("job %s finalizado: %s", job.id, final.value)


if __name__ == "__main__":
    raise SystemExit(main())