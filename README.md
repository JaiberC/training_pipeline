# training_pipeline

Servicio de entrenamiento de modelos **YOLO26** y **RT-DETR** para deteccion y
segmentacion. API FastAPI, cola en Postgres, un entrenamiento a la vez.

Licencia **AGPL-3.0-only**. Ver `LICENSE`.

## Atribuciones

Este repositorio usa datos de terceros. Si entrenas con ellos y publicas el
resultado, cita la fuente:

- **Roboflow Universe — `people-detection-thermal` v5**, licencia **CC BY 4.0**.
  <https://universe.roboflow.com/roboflow-universe-projects/people-detection-thermal/dataset/5>
  CC BY 4.0 **permite uso comercial**; exige atribucion e indicar si se hicieron cambios.

---

# Entrenar en un comando

`scripts/entrenar.sh` hace el flujo completo: crea o recupera el proyecto, sube
el ZIP, lanza el job, **se queda vigilando el avance** y descarga los resultados.

```bash
./scripts/entrenar.sh --proyecto my_project --zip my_dataset.zip \
    --modelo yolo26s.pt --epochs 100 --imgsz 640 --batch 16
```

Es **idempotente**: si el proyecto o el dataset ya existen los reutiliza en vez de
volver a subirlos, y crea un job nuevo cada ejecucion. Por eso sirve para varios
entrenamientos sobre el mismo proyecto:

```bash
# mismo proyecto y dataset, segundo modelo mas grande
./scripts/entrenar.sh --proyecto my_project --zip my_dataset.zip \
    --modelo yolo26l.pt --epochs 200 --batch 32
```

Requiere `curl` y `jq`. Lee `API_PORT` del `.env` por su cuenta.
**Todas las opciones, los casos habituales y los codigos de salida estan en la
referencia del script, al final de este documento.**

El resto del documento explica el flujo a mano con la API, que es lo que
necesitas si quieres subir datos por SSH, lanzar el job desde otro sitio o
consultar entrenamientos antiguos.

# Build y versionado con `make`

El `Makefile` cubre todo el ciclo. El `make` por defecto **pregunta el tipo de
cambio, sube la version, construye las dos imagenes** y las sube a Docker Hub si
le pasas un `REGISTRY`.

```bash
make help                       # lista todos los objetivos

make                            # version + build (REGISTRY=local, no sube nada)
make REGISTRY=<tu-usuario>      # version + build + push a Docker Hub
make push REGISTRY=<tu-usuario> # solo subir, sin tocar la version

make up / down / logs / ps      # operar el stack
make gpu                        # levantar con soporte de GPU
make test                       # verificaciones del paquete
make health                     # estado de la API
make migrate                    # aplicar migraciones a mano
make dump                       # backup de Postgres
make clean                      # borra imagenes locales
```

## Versionado

La version vive en el fichero `VERSION`. `make` pregunta que tipo de cambio es:

```
$ make
Version actual: 0.1.0
Tipo de cambio (major/minor/patch): patch
Version 0.1.0 -> 0.1.1
==> Construyendo runtime-api [0.1.1]
==> Construyendo runtime-worker [0.1.1]
```

| Entrada | Resultado |
|---|---|
| `major` | `0.1.0` → `1.0.0` |
| `minor` | `0.1.0` → `0.2.0` |
| `patch` | `0.1.0` → `0.1.1` |

Cada build deja las imagenes con **dos etiquetas**: la version y `latest`.
Ademas la version queda grabada dentro de la imagen, para saber que corre en un
servidor sin mirar el tag:

```bash
docker inspect $REGISTRY/tp-api:latest \
  --format '{{index .Config.Labels "org.opencontainers.image.version"}}'
```

En un servidor sin terminal, o en CI, el prompt no tiene con quien hablar:

```bash
make version VER=1.2.0    # fija la version sin preguntar
```

## `REGISTRY`

`REGISTRY` decide si se sube a Docker Hub:

| Comando | Que hace |
|---|---|
| `make` | Construye con `REGISTRY=local`. No sube nada. |
| `make REGISTRY=miduvusuario` | Construye **y sube** a Docker Hub |

El `--push` solo se anade cuando `REGISTRY` no es `local`, porque `local` no es un
registro y el build fallaria siempre.

> `make` usa `docker build`, no `buildx`. Construye unicamente para la arquitectura
> del host (`linux/amd64`), que es lo que necesita el servidor GPU. Si algun dia
> hay que desplegar en arm64 (Jetson), habria que usar `buildx`.

# Guia completa de entrenamiento

## 1. Preparar el entorno (una sola vez)

```bash
cd /path/to/training_pipeline

cp .env.example .env
echo "HOST_UID=$(id -u)" >> .env     # para escribir en los volumenes sin sudo
echo "HOST_GID=$(id -g)" >> .env
mkdir -p "$DOCKER_PATH/postgres/data"  # debe existir ANTES del primer arranque
```

Edita `.env` y cambia `DBPASSWORD` y `API_PORT`. El puerto 8000 suele estar
ocupado en estaciones de trabajo; si `docker compose up` falla con
`Bind for 0.0.0.0:8000 failed: port is already allocated`, cambia `API_PORT`.

Comprueba que todo levanta:

```bash
docker compose up -d --build
docker compose ps                      # api, db y worker en 'running'
curl localhost:$API_PORT/health        # {"status":"ok","database":true}
```

Documentacion interactiva de la API: `http://localhost:$API_PORT/docs`

## 2. Definir el proyecto

Un proyecto declara **la tarea** (deteccion o segmentacion) y **el modelo base**.
Se crea una vez y puede tener varios datasets y muchos entrenamientos.

```bash
API=http://localhost:8010

PROJECT=$(curl -s -X POST $API/api/projects \
  -H 'Content-Type: application/json' \
  -d '{"name":"my_project","task":"detect","model":"yolo26s.pt","classes":["person"]}' \
  | jq -r .id)
```

### Elegir el modelo base

| Modelo | Tarea | Cuando usarlo |
|---|---|---|
| `yolo26n.pt` | detect | El mas rapido. Para probar pipelines o CPU. |
| `yolo26s.pt` | detect | **Equilibrio recomendado.** Punto de partida por defecto. |
| `yolo26m.pt` / `yolo26l.pt` / `yolo26x.pt` | detect | Mas precision, mas lento y mas VRAM. |
| `yolo26{n,s,m,l,x}-seg.pt` | segment | Deteccion **con silueta/mascara** por objeto. |
| `rtdetr-l.pt` / `rtdetr-x.pt` | detect | Detector transformer, sin NMS. Solo deteccion. |

**RT-DETR no segmenta**: no existe `rtdetr-seg.pt`. Pedirlo devuelve 422.

### `detect` o `segment`

Es la decision que mas afecta al resultado, y no se puede cambiar despues:

- **`detect`** — bounding box por persona. Necesita ~5 columnas por linea de label:
  `clase x_centro y_centro ancho alto`.
- **`segment`** — ademas delimita la silueta. Necesitavertices:
  `clase x1 y1 x2 y2 ...`.

Usa `detect` salvo que tu aplicacion necesite la silueta (conteo por area,
analisis de postura, delineado de cuerpos). Segmentacion necesita mas memoria,
entrena mas lento y con escenas nocturnas la silueta es mas dificil: la misma
persona puede aparecer partida en varios recortes.

El servicio **rechaza la combinacion incoherente antes de encolar**: si declaras
`segment` y las etiquetas son cajas, el error es 422 en `init`, no un fallo a
mitad del entrenamiento.

## 3. Cargar los datos

Se aceptan dos layouts. **La API los detecta sola** y genera el `dataset.yaml`;
no hace falta reorganizar nada ni escribir ese fichero a mano.

```
LAYOUT YOLO                              LAYOUT ROBOFLOW (imagenes invertidas)
mi_dataset/                               mi_dataset/
├── images/train/*.jpg                   ├── train/images/*
├── images/val/*.jpg                     ├── train/labels/*
├── labels/train/*.txt                   ├── valid/images/*
└── labels/val/*.txt                     ├── valid/labels/*
                                          └── test/images/*
```

Los labels van en formato YOLO normalizado, con el mismo nombre que su imagen.
La clase siempre empieza en **0**, que es el indice dentro de `classes` del proyecto.

### Opcion A — subir un ZIP por la API

El ZIP lleva `images/` y `labels/` en la raiz:

```bash
cd mi_dataset && zip -qr ../dataset.zip images labels && cd ..

DS=$(curl -s -X POST $API/api/projects/$PROJECT/datasets \
  -F 'archive=@dataset.zip' | jq -r .id)
```

### Opcion B — copiar los archivos y registrarlos

Para cuando los datos ya estan en el servidor (rsync, red compartida, copia
desde el origen). La carpeta debe quedar en
`data/datasets/<proyecto>/<nombre>/` en el host:

```bash
mkdir -p data/datasets/my_project/my_dataset
cp -r /origen/* data/datasets/my_project/my_dataset/

DS=$(curl -s -X POST $API/api/projects/$PROJECT/datasets/register \
  -H 'Content-Type: application/json' -d '{"name":"my_dataset"}' | jq -r .id)
```

Ambas devuelven el mismo `dataset_id`. El registro comprueba que existen imagenes,
detecta si las etiquetas son cajas o poligonos y escribe el `dataset.yaml` con
rutas absolutas del contenedor.

## 4. Entrenar

`init` valida y devuelve un UUID **sin encolar**. `start` lo encola. La separacion
importa: un job con configuracion invalida vuelve como error HTTP y nunca ocupa
la cola, que solo admite un entrenamiento a la vez.

```bash
JOB=$(curl -s -X POST $API/api/trainings/$PROJECT/init \
  -H 'Content-Type: application/json' \
  -d "{\"dataset_id\":\"$DS\",\"params\":{\"epochs\":50,\"imgsz\":640,\"batch\":16,\"workers\":8,\"patience\":20}}" \
  | jq -r .job_id)

curl -s -X POST $API/api/trainings/$JOB/start | jq
```

Si `init` devuelve un error, `jq` imprimiria `null`: mira el mensaje con `jq .detail`.

### Parametros que importan

Todos los de Ultralytics se pasan en `params` y llegan tal cual.

| Parametro | Para que sirve |
|---|---|
| `epochs` | Epocas. 100–300 es lo habitual con early stopping. |
| `imgsz` | 640 de partida. Sube a 1280 si las personas ocupan poco en la imagen. |
| `batch` | Bajalo si sale OOM; `batch=-1` deja que Ultralytics lo ajuste a la VRAM. |
| `device` | `0` para la GPU 0, `cpu` para forzar CPU. Por defecto se autodetecta. |
| `fraction` | Entrena solo un porcentaje del dataset. Imprescindible para pruebas. |
| `cls_pw` | 0–1. Pondera clases por frecuencia inversa. Util en escenas nocturnas, donde las detecciones validas son escasas. |
| `patience` | Epocas sin mejorar antes de parar. |
| `optimizer` | **Si fijas `lr0`, fija tambien `optimizer`.** Con `auto`, Ultralytics elige MuSGD por encima de 10 000 iteraciones e **ignora `lr0` en silencio**. |
| `augmentations` | Transformaciones de Albumentations para el dataset. |

### Probar el pipeline antes de entrenar en serio

En CPU, una prueba completa:

```bash
params: {"epochs":2, "imgsz":416, "batch":16, "fraction":0.02}
```

Ojo con `fraction`: en una prueba real con 21 422 imagenes, `fraction=0.02` uso
**53 imagenes**, no 428. Es mas aggressive de lo que sugiere el numero.

## 5. Seguir el entrenamiento

```bash
watch -n5 "curl -s $API/api/trainings/$JOB/status | jq '{status,epoch,total_epochs,progress,mAP50:.metrics[\"metrics/mAP50(B)\"]}'"
```

```json
{ "status": "running", "epoch": 12, "total_epochs": 50, "progress": 0.24,
  "mAP50": 0.41 }
```

Estados posibles: `draft` → `queued` → `running` → `succeeded` | `failed` | `cancelled`.
`interrupted` significa que el worker murio a mitad (se recupera al relanzarlo).

Tambien en vivo:

```bash
docker compose logs -f worker
```

## 6. Resultados

### Metricas clave

| Metrica | Significado |
|---|---|
| `mAP50` | Precision media con IoU 0.5. **La que se mira para decidir.** |
| `mAP50-95` | Mas estricta (IoU 0.5–0.95). Indica calidad de localizacion fina. |
| `precision` | De las detecciones hechas, cuantas son correctas. |
| `recall` | De las personas reales, cuantas encuentra. Baja = se le escapan personas. |

Con poca confianza y clases minoritarias: baja `cls_pw` a `0.25`–`0.5` antes de
concluir que el dataset no sirve.

### Descargar

```bash
curl -s -o best.pt "$API/api/trainings/$JOB/download?file=best.pt"   # pesos
curl -s -o last.pt "$API/api/trainings/$JOB/download?file=last.pt"   # ultimo checkpoint
curl -s -o results.csv "$API/api/trainings/$JOB/download?file=results.csv"
curl -s -o informe.html "$API/api/trainings/$JOB/report"             # informe
```

El informe HTML lleva curvas con ApexCharts: mAP por epoca, precision/recall,
perdidas, mAP por clase, mas las imagenes que genera Ultralytics (matriz de
confusion, curvas PR y F1). Se abre directamente en el navegador.

### Cancelar, reanudar

```bash
curl -X POST $API/api/trainings/$JOB/cancel           # mata y borra runs/<job_id>/
curl -X POST "$API/api/trainings/$JOB/restart?mode=scratch"   # de cero
curl -X POST "$API/api/trainings/$JOB/restart?mode=resume"    # desde last.pt
```

**`cancel` borra el directorio del run: descarga los pesos antes de cancelar.**
`resume` restaura epoca, optimizador y learning rate; no es equivalente a
empezar de cero y solo funciona si existe `last.pt`.

## 7. Evaluar el resultado antes de publicarlo

```bash
JOB=<job-id>
DATASET=my_dataset

docker compose exec -T worker sh -c "cd /data/models && yolo val \
  model=/data/runs/$JOB/train/weights/best.pt \
  data=/data/datasets/$PROJECT/$DATASET/dataset.yaml \
  project=/data/models/val name=check"
```

El `cd /data/models` no es opcional: el worker corre con el uid del host y `/app`
no es escribible, asi que Ultralytics falla al crear el directorio de salida.

Un mAP50 alto en el conjunto de validacion no garantiza nada en produccion.
Prueba sobre imagenes del mundo real antes de integrar.

## 8. Otros endpoints

```bash
curl $API/api/projects                                   # listar
curl $API/api/projects/$PROJECT                          # ver uno
curl -X PUT  $API/api/projects/$PROJECT -d '{...}'       # actualizar
curl $API/api/projects/$PROJECT/export                   # YAML (no sube nada)
curl $API/api/projects/$PROJECT/datasets                 # datasets del proyecto
curl -X DELETE $API/api/projects/$PROJECT                # borra proyecto, dataset y runs
curl $API/api/trainings                                  # historial
curl $API/health
```

## 9. Problemas frecuentes

| Sintoma | Causa |
|---|---|
| `Port is already allocated` | `API_PORT` ocupado. Cambialo en `.env`. |
| `missing path '.../images/val'` | El `dataset.yaml` tiene rutas relativas mal resueltas. Re-registra el dataset para regenerarlo. |
| `Permission denied` al copiar archivos | `HOST_UID`/`HOST_GID` no estan en el `.env`. |
| `'.yolo26n.pt.part': Permission denied` | El subproceso de entrenamiento no puede escribir. Verifica `MODELS_ROOT` en el volumen. |
| `Permission denied: '/app/runs'` en `yolo val` | Falta el `cd /data/models` antes del comando. |
| `CUDA disponible: False` en el log del worker | Falta `compose.gpu.yaml` o el NVIDIA Container Toolkit. **Entrena en CPU sin avisar.** |
| `too many open files` | Baja `workers` (dataloaders). |
| `CUDA out of memory` | Baja `batch` o `imgsz`. |
| Entrenamiento lentisimo | Sin GPU, o `device` forzado a `cpu` por error. |

## 10. Desplegar en el servidor GPU

La imagen se construye una vez y se hace `pull` en cualquier maquina. **La misma
imagen sirve para GPU y CPU**: el dispositivo se elige en runtime, nunca en build.

```bash
# En el equipo de desarrollo
docker build --target runtime-api    -t <user>/tp-api:0.1.0 --push .
docker build --target runtime-worker -t <user>/tp-worker:0.1.0 --push .

# En el servidor GPU
docker compose -f compose.yaml -f compose.gpu.yaml pull
docker compose -f compose.yaml -f compose.gpu.yaml up -d
```

Prerrequisito del host con GPU: **NVIDIA Container Toolkit** instalado.


Verifica al arrancar que el worker reconoce la GPU:

```bash
docker compose logs worker | grep CUDA
# debe decir: CUDA disponible: True -> device=0
```

## 11. Mantenimiento

```bash
docker compose exec db pg_dump -U $DBUSER $DBNAME > "backup_$(date +%F).sql"
docker compose restart worker          # cancela el job en curso; al relanzar se recupera
docker compose down                    # para; `down -v` ademas borda volumenes
```

Los trainings guardados en `data/runs/` ocupan del orden de 12 MB cada uno, mas el
peso. `cancel` y `DELETE /api/projects` los limpian.

Ver `AGENTS.md` para las trampas internas del dominio y del toolchain.


---

# Referencia de `scripts/entrenar.sh`

## Todas las opciones

| Opcion | Que hace |
|---|---|
| `--proyecto NOMBRE` | **Obligatorio.** Nombre del proyecto (se crea si no existe) |
| `--zip RUTA.zip` | **Obligatorio.** Dataset en ZIP con `images/` y `labels/` en la raiz |
| `--task detect\|segment` | Tarea. Solo se aplica al **crear** el proyecto |
| `--modelo M` | `yolo26{n,s,m,l,x}.pt`, `yolo26{n..x}-seg.pt`, `rtdetr-l/x.pt` |
| `--clases "a,b,c"` | Nombres de clases (defecto: `person`) |
| `--datos NOMBRE` | Nombre del dataset (defecto: el del ZIP sin extension) |
| `--epochs N` | Epocas (defecto: 50) |
| `--imgsz N` | Tamano de imagen (defecto: 640) |
| `--batch N` | Lote. `-1` lo ajusta a la VRAM (defecto: 16) |
| `--workers N` | Dataloaders (defecto: 8) |
| `--patience N` | Epocas sin mejorar antes de parar (defecto: 20) |
| `--fraccion F` | Entrena solo `Fx100%` del dataset (`0.02` = 2%) |
| `--cls-pw F` | 0-1. Pondera clases por frecuencia inversa |
| `--dispositivo D` | `0`, `cpu`, `-1` (GPU mas inactiva)... |
| `--extra JSON` | Cualquier otro parametro de Ultralytics |
| `--salida RUTA` | Carpeta de resultados (defecto: `resultados/<fecha-hora>`) |
| `--intervalo N` | Segundos entre consultas de estado (defecto: 10) |
| `--sin-descarga` | No descargar al terminar |
| `--reiniciar` | Cambia la tarea o el modelo **por defecto** del proyecto |
| `--api URL` | Base de la API si no es la del `.env` |
| `-h`, `--help` | Ayuda completa |

`--extra` sirve para lo especifico de Ultralytics sin tocar el script:

```bash
./scripts/entrenar.sh --proyecto my_project --zip my_dataset.zip \
    --extra '{"lr0":0.001,"optimizer":"AdamW","mosaic":1.0,"hsv_v":0.6}'
```

## Salida en pantalla

```
Proyecto 'my_project' ya existe  (a0184a42-...)
Este job usara yolo26l.pt (el proyecto tiene yolo26s.pt).
Dataset 'my_dataset' ya registrado  (6a396b00-...)  (no se vuelve a subir)
Configuracion: {"epochs":200,"imgsz":640,"batch":32,"workers":8,"patience":20}
Job en cola. El worker lo ira reclamando (concurrencia 1).

  running   epoca 37/100   37%  mAP50=0.4127   mAP50-95=0.2501  P=0.55  R=0.48
Entrenamiento finalizado correctamente.

Descargando en resultados/20261010-231245
  best.pt  (39M)
  last.pt  (39M)
  results.csv  (4.0K)
  report.html  (6.2K)

Metricas finales
  mAP50      0.4127
  mAP50-95   0.2501
  precision  0.5512
  recall     0.4803
```

## Casos habituales

```bash
# Prueba de humo antes de gastar GPU: minutos en vez de horas
./scripts/entrenar.sh --proyecto my_project --zip my_dataset.zip \
    --modelo yolo26n.pt --epochs 2 --fraccion 0.01 --imgsz 416

# Entrenamiento largo en la GPU, con clases desbalanceadas
./scripts/entrenar.sh --proyecto my_project --zip my_dataset.zip \
    --epochs 150 --cls-pw 0.5 --batch 32 --dispositivo 0

# Segmentacion (el dataset debe traer etiquetas poligonales)
./scripts/entrenar.sh --proyecto my_project_seg --zip my_dataset_seg.zip \
    --task segment --modelo yolo26s-seg.pt --epochs 100

# Entrenar y no descargar; se recoge luego con la API
./scripts/entrenar.sh --proyecto my_project --zip my_dataset.zip --sin-descarga

# Guardar en una carpeta concreta
./scripts/entrenar.sh --proyecto my_project --zip my_dataset.zip \
    --salida /mnt/resultados/experimento_01
```

## Codigos de salida

Util si lo encadenas en otro script:

| Codigo | Significado |
|---|---|
| `0` | Entrenamiento correcto y descargado |
| `1` | Error de uso, API caida, o el job termino en `failed` |
| `2` | El job fue cancelado |
| `3` | El worker se detuvo a mitad (`interrupted`) |

## Detalles que conviene saber

- **`Ctrl-C` no cancela el entrenamiento.** El script sale pero el job sigue
  corriendo en el worker. Se puede volver con `curl $API/api/trainings/$JOB/status`
  o cancelar de verdad con `curl -X POST $API/api/trainings/$JOB/cancel`.
- **La tarea la fija el proyecto, el modelo puede cambiar por job.** `--task`
  solo cuenta al crear el proyecto (usalo con `--reiniciar` para cambiarla);
  `--modelo` se envia como override en cada `init`.
- Si declaras `--task segment` con etiquetas de cajas, el script avisa antes de
  lanzar y la API responde 422 al pedir el job.
