# AGENTS.md — training_pipeline

Servicio de entrenamiento de modelos (YOLO26 / RT-DETR) para detección y segmentación.

## Arquitectura

Tres contenedores, un compose base + overrides:

| Servicio | Imagen | Rol |
|---|---|---|
| `db` | `postgres:17.4` | Persistencia. Sin `ports`; solo red interna |
| `api` | target `runtime-api` | FastAPI. Encola, consulta, sirve artefactos |
| `worker` | target `runtime-worker` | Loop de entrenamiento. **Concurrencia 1** |

- `compose.yaml` = despliegue **CPU** (caso base). `compose.gpu.yaml` añade la reserva de GPU.
- `compose.dev.yaml` monta `./src` y recarga en caliente.
- **Una sola imagen sirve para GPU y CPU-only.** La selección de dispositivo es en
  runtime vía `torch.cuda.is_available()`; nunca en build ni en compose.
- **Postgres es la fuente de verdad.** Los proyectos NO viven en git. El contenedor no sube
  nada a GitHub. `GET /api/projects/{id}/export` descarga el YAML para reproducible.

## Restricción de imports (crítico)

`api.py`, `queue.py`, `models.py`, `schemas.py`, `config.py` y `report.py` **no deben
importar `ultralytics` ni `torch`**, ni transitivamente.

Motivo: la imagen `runtime-api` no instala torch. Medido en este repo: **613 MB la API,
3.4 GB el worker**. Instalar PyTorch en la imagen de la API la llevaría al tamaño del
worker. `worker.py` tampoco importa torch: lanza `train_cli` como subproceso.

El único módulo que importa Ultralytics es `train_cli.py`.

## Trampas de Ultralytics verificadas en ejecución

Estas no salen de la documentación; aparecieron entrenando de verdad.

- **`path:` relativo en `dataset.yaml` NO funciona.** Ultralytics resuelve un `path`
  relativo contra su ajuste global `datasets_dir` (por defecto `/app/datasets`), **no**
  contra la carpeta del YAML. Con `path: .` falla con `missing path '/app/images/val'`.
  `api.py` escribe la ruta ABSOLUTA del contenedor (`/data/datasets/...`). Sigue siendo
  portable entre máquinas porque el punto de montaje del volumen es igual en todas.
- **Los parámetros no reconocidos se descartaban en silencio.** `init` separa las claves
  conocidas y mete el resto en `extra`, que sí se pasa a Ultralytics. Sin eso,
  `workers: 2` llegaba como 8 sin avisar.
- `save_dir` se usa **tal cual, sin autoincrementar**; `project`/`name` sí se incrementan
  (`name`, `name2`, `name3`). El run usa `project=run_dir` + `name="train"` +
  `exist_ok=True`, y queda en `runs/<job_id>/train`.
- **La segmentación necesita un compilador de C; la detección no.** Triton compila los
  kernels de GPU en tiempo de ejecución. Sin `gcc` en la imagen, un `-seg.pt` muere al
  primer batch con `RuntimeError: Failed to find C compiler`, y lo que ve el usuario es un
  `EOFError` del hilo `_pin_memory_loop`, que no menciona la causa. Verificado: `yolo26n.pt`
  y `yolo26s.pt` de detección completan el entrenamiento sin compilador. Por eso
  `runtime-worker` instala `gcc g++` y `runtime-api` no.
- **Sin `shm_size`, `workers > 0` cuelga el entrenamiento.** PyTorch pasa los tensores de
  los dataloaders por `/dev/shm`, que Docker deja en 64 MB. Con
  `8 workers × 2 prefetch × 16 batch × 640² × 3` hacen falta 315 MB, así que los procesos
  se bloquean sin consumir CPU ni GPU. `compose.yaml` pone `shm_size: ${SHM_SIZE:-2gb}`
  en el worker. **No es específico de la GPU**: es multiproceso, y afecta igual en CPU.
- **Un contenedor creado sin `-f compose.gpu.yaml` no falla, entrena en CPU.** El
  `deploy.resources.reservations.devices` solo existe en el override, y `train_cli` elige
  dispositivo en runtime con `torch.cuda.is_available()`. Sin GPU todo funciona y tarda
  ~20x más. Comprobar con
  `docker compose -f compose.yaml -f compose.gpu.yaml exec worker python -c "import torch;print(torch.cuda.is_available())"`.
  Un `restart` NO lo arregla: la reserva se fija al crear el contenedor, hay que `down` + `up`.
- **El `dataset.yaml` es lo que lee Ultralytics, y manda sobre la base de datos.** Los
  nombres de clase de la matriz de confusión, de las curvas y del checkpoint salen de ahí,
  no de `project.classes`. Por eso `init` lo **reescribe** en cada job con las clases
  vigentes, y las actualiza en el proyecto a la vez: si divergen, el modelo se etiqueta con
  la clase de un entrenamiento anterior. Antes solo se escribía al subir el dataset, y un
  `--clases` posterior no llegaba al fichero.
  El orden de resolución es `--clases` > `names` del `data.yaml` del ZIP > `class_0,
  class_1...`. El **número** de clases sale siempre de los índices que aparecen en las
  etiquetas (`tasks.label_class_indices`), nunca de lo que declara el usuario: declarar de
  menos hace que el entrenamiento ignore las demás en silencio, y eso es un 422 en `init`.
- **Un ZIP con una carpeta envoltorio se aplana al subirlo** (`_flatten_extracted`). Sin
  eso el dataset queda con un nivel de más, `detect_layout` no reconoce el layout y el
  entrenamiento falla con `missing path`. Las dos formas de ZIP valen: con carpeta
  envoltorio y con los splits en la raíz.

## Build

- `compileall` **no** detecta `NameError`. Hay que importar los módulos de verdad:
  `docker compose run --rm api python -c "import trainpipe.worker"`.
- Los stages `deps-*` crean un stub vacío de `src/trainpipe`: sin él, setuptools falla con
  `'src' does not exist`. El código real se copia en runtime y `PYTHONPATH=/app/src` tiene
  precedencia sobre el stub del site-packages.
- Rebuild tras tocar `src/`: ~3 s. Las capas de dependencias no se reinstalan.

## Por qué el entrenamiento es un subproceso

`worker` ejecuta `python -m trainpipe.train_cli` con `start_new_session=True` y guarda el
PID. El cancelado depende de poder matarlo con `SIGTERM` → `SIGKILL` sobre el **grupo de
procesos**. Entrenar en el propio proceso del worker haría imposible matarlo de forma limpia.

Orden obligatorio al cancelar: primero matar el proceso, **después** borrar `runs/<job_id>/`.
Borrar antes deja al worker escribiendo sobre directorios inexistentes.

## Comandos

```bash
# Desarrollo (monta src, recarga en caliente)
docker compose -f compose.yaml -f compose.dev.yaml up --build

# CPU-only
docker compose up -d

# GPU
docker compose -f compose.yaml -f compose.gpu.yaml up -d

# Build y publicación (desde el equipo de desarrollo)
docker buildx build --target runtime-api    -t <user>/tp-api:0.1.0 --push .
docker buildx build --target runtime-worker -t <user>/tp-worker:0.1.0 --push .
```

Sin puerto publicado en Postgres, `psql` y `pg_dump` se ejecutan **dentro** del contenedor:

```bash
docker compose exec db pg_dump -U $DBUSER $DBNAME > backup.sql
```

Postgres es la fuente de verdad: **sin `pg_dump` periódico se pierden todos los proyectos.**
La imagen es sin estado; `docker pull` no transporta datos (ni DB ni volúmenes).

## Skill de ApexCharts

Al generar o modificar el informe HTML, **consultar el skill `apexcharts-skill`** antes de
escribir el JS:

```
.agents/skills/apexcharts-skill/SKILL.md
```

`report.py` genera HTML estático con la API core vía CDN, no un framework wrapper. Puntos
que el skill marca como errores frecuentes y que ya están resueltos en `report.py`: `render()`
obligatorio, `null` (no `undefined`) para datos faltantes, colores hex con `#`, `radialBar` en
0–100, `responsive` en orden ascendente.

**El CDN está fijado en `apexcharts@7`.** No subirlo a v8: el skill documenta los cambios de
v7 y no cubre v8. Fijar versión, nunca `@latest`.

## Restricciones del dominio

- **RT-DETR solo soporta detección.** No existe `rtdetr-seg.pt`. `tasks.py` lo valida y falla
  en `init`, no a mitad del entrenamiento.
- **`project.task` debe coincidir con el dataset.** Un `-seg.pt` sobre labels de caja falla.
  `init` lo comprueba antes de encolar: así un job inválido nunca ocupa la cola ni bloquea
  al worker.
- **`optimizer='auto'` ignora `lr0`** cuando las iteraciones superan 10 000 (elige MuSGD).
  Si se fija `lr0`, hay que fijar también `optimizer`. `apply_family_defaults` lo revisa
  tanto a nivel superior como dentro de `extra`.
- **RT-DETR exige `deterministic=False`** en CUDA; su atención deformable no tiene backward
  determinista.
- **RT-DETR no supera 300 detecciones por imagen** con los checkpoints preentrenados;
  `max_det` no sube ese techo.
- Imágenes nocturnas desbalanceadas: `cls_pw` (0–1) pondera por frecuencia inversa de clase.

## Recuperación de la cola

Al arrancar el worker, todo job en `running` pasa a `interrupted`: el worker murió con él.
**Verificado**: matando el contenedor con SIGKILL a mitad de un entrenamiento, el job se
queda en `running` para siempre y bloquea la cola; al relanzar el worker el log dice
`N job(s) en 'running' marcados como 'interrupted'` y el job queda re-encolable.

El claim usa `FOR UPDATE SKIP LOCKED`. Con un solo worker es redundante, pero evita el doble
entrenamiento si alguien levanta un segundo worker por error.

## Operación

- **Postgres es la fuente de verdad.** `docker pull` no transporta ni la base ni los
  volúmenes: desplegar en otra máquina arranca un servicio vacío. Hay que restaurar el
  `pg_dump` y copiar los volúmenes a mano.
- **`pg_dump` sin puertos publicados** va dentro del contenedor:
  `docker compose exec db pg_dump -U $DBUSER $DBNAME > backup.sql`. El `pg_dump` del host
  contra `localhost:5432` no funciona.
- **Los volúmenes los escribe el uid del host.** `compose.yaml` fija
  `user: "${HOST_UID}:${HOST_GID}"` en api y worker (Postgres queda con el suyo).
  Sin eso el worker creaba directorios como root y el usuario no podía escribir
  dentro ni borrarlos (`Permission denied`). Si cambias de usuario, ajusta ambos
  valores en el `.env`.
- Ultralytics descarga los pesos preentrenados en el **directorio de trabajo**, que
  con el worker no-root no puede ser `/app`: el entrenamiento moría con
  `Permission denied: .yolo26s.pt.part`. Por eso el subproceso se lanza con
  `cwd=settings.models_root`.
- **Colocar archivos a mano no basta**: la base de datos es la fuente de verdad.
  Hay que llamar a `POST /api/projects/{id}/datasets/register` después de copiar la
  carpeta a `data/datasets/<proyecto>/<nombre>/`.
- **Los datasets tienen dos layouts y `tasks.detect_layout` los distingue solo:**
  `yolo` (`images/train` + `labels/train`) y `roboflow` (`train/images` +
  `train/labels`, con `valid` y no `val`). El `dataset.yaml` se genera con las
  rutas que corresponden a cada uno. Roboflow es la fuente más común de datasets
  térmicos/nocturnos; sin esto, un dataset de 26 000 ficheros fallaba con
  `missing path '.../images/val'`.
- **El worker corre sin `HOME`.** Con `user: ${HOST_UID}` Matplotlib cae en
  `/.config/matplotlib`, no escribible, y crea un directorio temporal en cada import
  (ralentiza cada dataloader). `compose.yaml` fija `MPLCONFIGDIR=/tmp/matplotlib` y
  `HOME=/tmp`.
- **`DELETE /api/projects` limpia dataset y runs** de todos sus jobs. Sin esa limpieza el
  volumen crecía sin límite (~12 MB por entrenamiento).
- **Cancela después de descargar**: `cancel` borra `runs/<job_id>/` y los pesos no sobreviven.
- `restart?mode=resume` reanuda desde `last.pt` restaurando época, optimizador y LR; sin
  checkpoint devuelve 409. `mode=scratch` empieza de cero. No son intercambiables.

## Depurar un job que falla

- **Con `INFO` el worker no dice por qué falla.** `worker.py` registra la salida de
  `train_cli` con `log.debug`, así que a INFO se descarta entera: el `error` del job solo
  trae las últimas 40 líneas y muchas veces son el error secundario (el hilo de
  `pin_memory` o un `EOFError`), no la causa. `WORKER_LOG_LEVEL=DEBUG` en el `.env`, o
  `docker compose exec -e WORKER_LOG_LEVEL=DEBUG worker ...` para una ejecución suelta.
  Con DEBUG se ve la salida completa de Ultralytics y la causa sale antes del síntoma.
- **El monitor del script miente sobre el estado.** `entrenar.sh` solo reprinta cuando
  cambia la época, así que un job que pasa de `queued` a `running` con época 0 se ve
  congelado en `queued epoca 0/0`. `docker logs` es la fuente fiable.
- **Un job parado tiene tres causas distintas que parecen la misma**: cuelgue por
  `/dev/shm` (0% de CPU y 0% de GPU con memoria reservada), entrenamiento en CPU por
  falta del override de GPU (0% de GPU pero CPU alta y sin proceso en `nvidia-smi`), y
  muerte al primer batch por compilador ausente (el proceso aparece y desaparece). Las
  distingue `/dev/shm` con `df -h`, `torch.cuda.is_available()` y la presencia del
  proceso en `nvidia-smi`, en ese orden.

## Arranque y esquema

- **La API aplica las migraciones al arrancar** (`docker/entrypoint-api.sh`:
  `alembic upgrade head` y luego `exec "$@"`). El `exec` es obligatorio: sin él el
  shell queda como PID 1, uvicorn no recibe el SIGTERM y `docker compose down` acaba
  en SIGKILL. El worker **no** lleva migraciones en su imagen y por eso depende de
  que la API esté sana.
- **`/health` comprueba el esquema**, no solo `SELECT 1`. El compose arranca el worker
  cuando la API está sana, así que un `ok` con el esquema ausente dejaría al worker
  contra una base sin tablas. Respuesta: `{"status","database","schema"}`.
- El healthcheck de la API usa Python porque la imagen no tiene `curl`
  (`python:3.12-slim`).

## Makefile

- `make` = version + build. Pregunta major/minor/patch y escribe `VERSION`.
- **`--push` solo se añade si `REGISTRY != local`.** Con `REGISTRY=local` (por
  defecto) fallaría siempre, porque `local` no es un registro.
- **La versión se lee dentro de la receta, no con `$(VERSION)`.** Make evalúa las
  variables al arrancar: con `release: version build` usaría el valor anterior al
  bump y etiquetaría las imágenes con la versión equivocada.
- El prompt lee de `/dev/tty` y, si viene vacío, cae a stdin: así funciona
  interactivo y también en CI. `make version VER=1.2.3` lo salta.
- `docker build` solo construye un `--target` por invocación, de ahí las dos
  llamadas (api y worker). Construye solo para `linux/amd64`: suficiente para GPU
  RTX/Tesla, pero no para arm64/Jetson (ahí haría falta buildx).

## Git

Se versiona **código**: `src/`, `scripts/`, compose, Dockerfile, migraciones,
`AGENTS.md`, `.agents/`.

No se versiona: datasets, pesos `.pt`, `runs/`, `venv/`, `.env`,
`resultados/`, `.agents/skills/**/node_modules/`.

El `.gitignore` **no** debe listar nunca ficheros de código: `git status` es la comprobación
que detecta ese error.

## scripts/entrenar.sh

Atajo de CLI que hace el flujo completo (crear proyecto → subir ZIP → init →
start → monitorizar → descargar). Todo lo de abajo aplica también a él.

- **Es idempotente**: proyecto y dataset se reutilizan si ya existen; cada
  ejecución crea un job nuevo. `--modelo` se envía como override en `init`, así
  que un mismo proyecto admite yolo26n/s/m/l sin redefinirlo.
- **`init` acepta `model` como override por job** (`InitRequest.model`), no solo
  el del proyecto. La **tarea** en cambio la fija el proyecto.
- Los errores de FastAPI se leen de `.detail`; con `jq -r .job_id` un error
  imprimiría `null` silenciosamente. El script lo distingue y aborta.
- `results.csv` y `report.html` viven en `<run_dir>/train/` y `<run_dir>/`
  respectivamente, no en la misma carpeta que los pesos. `download_artifact`
  prueba las tres rutas por eso.

## Pendientes conocidos

- La API no implementa autenticación. Antes de exponerla fuera de la red privada hace falta.
- Los endpoints aceptan **nombres de proyecto**, nunca rutas de archivo. `tasks.validate_resource_name`
  y la lista blanca de `download` evitan path traversal.
- Sin lockfile de GPU: un entrenamiento lanzado por CLI compite con el del worker y ambos
  hacen OOM en la GPU.
- `GET /api/trainings/{job_id}/report` genera el informe bajo demanda si el worker no lo dejó.