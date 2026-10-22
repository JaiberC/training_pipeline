#!/usr/bin/env bash
#
# entrenar.sh - Entrena un modelo YOLO26/RT-DETR contra el servicio trainpipe.
#
# El flujo es idempotente: si el proyecto o el dataset ya existen los REUTILIZA,
# y cada ejecucion crea un job nuevo. Eso permite repetir el script tantas veces
# como entrenamientos quieras sobre el mismo proyecto sin volver a subir 26 000
# imagenes.
#
#   1. crear/recuperar proyecto
#   2. subir el ZIP del dataset (o recuperarlo si ya esta)
#   3. init + start del job
#   4. monitorizar hasta que termine
#   5. descargar pesos, metricas e informe
#
# Uso: ver la ayuda con -h

set -euo pipefail

API="${API:-}"
PROYECTO=""
TASK="detect"
MODELO="yolo26l.pt"
CLASES=""
ZIP=""
DATOS=""
EPOCHS=50
IMGSZ=640
BATCH=16
WORKERS=8
PATIENCE=20
FRACCION=""
CLS_PW=""
DISPOSITIVO=""
EXTRA_JSON=""
SALIDA=""
INTERVALO=10
SIN_DESCARGA=0
REINICIAR=0
JOB=""
REINTENTO=10

# ─────────────────────────────────────────────────────────────────────────────
# Presentacion
# ─────────────────────────────────────────────────────────────────────────────
if [ -t 1 ]; then
  C_RESET=$'\033[0m'; C_OK=$'\033[32m'; C_ERR=$'\033[31m'
  C_WARN=$'\033[33m';  C_INFO=$'\033[36m'; C_DIM=$'\033[2m'; C_B=$'\033[1m'
else
  C_RESET=""; C_OK=""; C_ERR=""; C_WARN=""; C_INFO=""; C_DIM=""; C_B=""
fi

info() { printf '%s%s%s\n' "$C_INFO" "$*" "$C_RESET" >&2; }
ok()   { printf '%s%s%s\n' "$C_OK" "$*" "$C_RESET" >&2; }
warn() { printf '%s%s%s\n' "$C_WARN" "$*" "$C_RESET" >&2; }
err()  { printf '%s%s%s\n' "$C_ERR" "$*" "$C_RESET" >&2; }
die()  { err "$*"; exit 1; }

uso() {
  cat <<'EOF'
Entrena un modelo YOLO26 / RT-DETR usando el servicio trainpipe.

USO
  ./scripts/entrenar.sh --proyecto NOMBRE --zip RUTA.zip [opciones]

MODOS DE USO
  1) Lanzar un entrenamiento nuevo:
       ./scripts/entrenar.sh --proyecto NOMBRE --zip RUTA.zip [opciones]
  2) Volver a un entrenamiento ya en curso (solo monitorea y descarga):
       ./scripts/entrenar.sh --job UUID

OBLIGATORIO (salvo que se use --job)
  --proyecto NOMBRE      Nombre del proyecto (se crea si no existe)
  --zip RUTA.zip         Dataset en ZIP con images/ y labels/ en la raiz

SEGUIR UN JOB EXISTENTE
  --job UUID             Se adjunta a un entrenamiento ya creado: monitorea su
                         avance y descarga los resultados al terminar. No crea
                         un job nuevo, no sube datos y no toca el proyecto.
                         Es lo que se usa cuando la sesion anterior se cerro:
                         el entrenamiento sigue corriendo en el worker.

DEFINIR EL PROYECTO (solo se usan al crearlo)
  --task T               detect (defecto) | segment
  --modelo M             yolo26n.pt | yolo26s.pt | ... | rtdetr-l.pt
                         Segmentacion: yolo26s-seg.pt
  --clases "a,b,c"       Nombres de clase, EN ORDEN: el numero de cada uno debe
                          coincidir con el que llevan las etiquetas. Si no lo
                          pasas, se toman del data.yaml del propio dataset y, si
                          tampoco los trae, se generan como class_0, class_1...
                          Si el dataset usa MAS clases de las que declaras, el
                          servicio lo rechaza al subirlo: declararlas de menos
                          hace que el entrenamiento ignore las demas sin avisar.

DATOS
  --datos NOMBRE         Nombre del dataset (defecto: el nombre del zip sin .zip)

HIPERPARAMETROS (se envian tal cual a Ultralytics)
  --epochs N             Epocas (defecto: 50)
  --imgsz N              Tamano de imagen (defecto: 640)
  --batch N              Lote. -1 lo ajusta a la VRAM (defecto: 16)
  --workers N            Dataloaders (defecto: 8)
  --patience N           Epocas sin mejorar antes de parar (defecto: 20)
  --fraccion F           Entrena solo Fx100% del dataset (0.02 = 2%)
  --cls-pw F             0-1, pondera clases por frecuencia inversa
  --dispositivo D        0, cpu, -1 (GPU mas inactiva)...
  --extra JSON           Objetos JSON extra, p.ej. '{"lr0":0.001,"optimizer":"AdamW"}'

SALIDA
  --salida RUTA          Carpeta de resultados (defecto: ./resultados)
  --intervalo N          Segundos entre consultas de estado (defecto: 10)
  --sin-descarga         No descargar al terminar
  --reiniciar            Si el proyecto ya existe, actualizarlo con --task/--modelo

OTROS
  --api URL              Base de la API (defecto: lee API_PORT del .env, si no
                         http://localhost:8000)
  -h, --help             Esta ayuda

EJEMPLOS
  # Primera vez: proyecto nuevo con el ZIP
  ./scripts/entrenar.sh --proyecto nocturno_personas --zip thermal.zip \
      --modelo yolo26s.pt --epochs 100

  # Segundo entrenamiento del MISMO proyecto: no vuelve a subir nada
  ./scripts/entrenar.sh --proyecto nocturno_personas --zip thermal.zip \
      --modelo yolo26l.pt --epochs 200 --batch 32

  # Prueba rapida en CPU antes de gastar GPU
  ./scripts/entrenar.sh --proyecto nocturno_personas --zip thermal.zip \
      --epochs 2 --fraccion 0.01 --imgsz 416

  # Desbalance en imagenes nocturnas
  ./scripts/entrenar.sh --proyecto nocturno_personas --zip thermal.zip \
      --cls-pw 0.5 --epochs 150

  # La sesion se cerro pero el entrenamiento sigue: volver a mirarlo
  ./scripts/entrenar.sh --job 0c876710-b7ae-42dd-a1d0-4228400e62cb
EOF
}

# ─────────────────────────────────────────────────────────────────────────────
# Argumentos
# ─────────────────────────────────────────────────────────────────────────────
while [ $# -gt 0 ]; do
  case "$1" in
    --proyecto)     PROYECTO="${2:-}"; shift 2 ;;
    --zip)          ZIP="${2:-}"; shift 2 ;;
    --job)          JOB="${2:-}"; shift 2 ;;
    --reintento)    REINTENTO="${2:-}"; shift 2 ;;
    --task)         TASK="${2:-}"; shift 2 ;;
    --modelo)       MODELO="${2:-}"; shift 2 ;;
    --clases)       CLASES="${2:-}"; shift 2 ;;
    --datos)        DATOS="${2:-}"; shift 2 ;;
    --epochs)       EPOCHS="${2:-}"; shift 2 ;;
    --imgsz)        IMGSZ="${2:-}"; shift 2 ;;
    --batch)        BATCH="${2:-}"; shift 2 ;;
    --workers)      WORKERS="${2:-}"; shift 2 ;;
    --patience)     PATIENCE="${2:-}"; shift 2 ;;
    --fraccion)     FRACCION="${2:-}"; shift 2 ;;
    --cls-pw)       CLS_PW="${2:-}"; shift 2 ;;
    --dispositivo)  DISPOSITIVO="${2:-}"; shift 2 ;;
    --extra)        EXTRA_JSON="${2:-}"; shift 2 ;;
    --salida)       SALIDA="${2:-}"; shift 2 ;;
    --intervalo)    INTERVALO="${2:-}"; shift 2 ;;
    --api)          API="${2:-}"; shift 2 ;;
    --sin-descarga) SIN_DESCARGA=1; shift ;;
    --reiniciar)    REINICIAR=1; shift ;;
    -h|--help)      uso; exit 0 ;;
    *)              die "Opcion desconocida: $1  (usa -h)" ;;
  esac
done

# ─────────────────────────────────────────────────────────────────────────────
# Validaciones previas
# ─────────────────────────────────────────────────────────────────────────────
# Raiz del repo: se calcula siempre, se use o no --api, porque la carpeta de
# resultados por defecto depende de ella.
RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

command -v jq  >/dev/null 2>&1 || die "Falta 'jq'.  sudo apt install jq"
command -v curl >/dev/null 2>&1 || die "Falta 'curl'."
# unzip no es imprescindible: sin el, el informe se descarga pero se abre sin
# las imagenes de Ultralytics. Aviso, no error.
command -v unzip >/dev/null 2>&1 || warn "Falta 'unzip': el informe saldra sin imagenes.  sudo apt install unzip"

# Con --job solo se monitorea: no hacen falta proyecto ni ZIP.
if [ -z "$JOB" ]; then
  [ -n "$PROYECTO" ] || { uso; die "Falta --proyecto (o usa --job UUID para seguir uno en curso)"; }
  [ -n "$ZIP" ]      || { uso; die "Falta --zip (o usa --job UUID para seguir uno en curso)"; }
  [ -f "$ZIP" ]      || die "No existe el ZIP: $ZIP"
fi

case "$TASK" in detect|segment) ;; *) die "--task debe ser 'detect' o 'segment'" ;; esac

if [ -z "$API" ]; then
  if [ -f "$RAIZ/.env" ]; then
    PUERTO="$(grep -E '^API_PORT=' "$RAIZ/.env" | cut -d= -f2 | tr -d '[:space:]')"
    [ -n "$PUERTO" ] && API="http://localhost:$PUERTO"
  fi
  API="${API:-http://localhost:8000}"
fi

[ -n "$SALIDA" ] || SALIDA="$RAIZ/resultados/$(date +%Y%m%d-%H%M%S)"

# Convierte un nombre en algo que la API acepte (minusculas, digitos, - y _).
# Quita acentos con iconv si esta disponible; si no, cada caracter no ASCII se
# convierte en guion bajo.
sanitizar() {
  local s="$1"
  if command -v iconv >/dev/null 2>&1; then
    s=$(printf '%s' "$s" | iconv -f UTF-8 -t ASCII//TRANSLIT 2>/dev/null) || s="$1"
  fi
  printf '%s' "$s" \
    | tr '[:upper:]' '[:lower:]' \
    | sed -e 's/[^a-z0-9_-]\{1,\}/_/g' -e 's/^[-_]\{1,\}//' -e 's/[-_]\{1,\}$//' \
    | cut -c1-64
}

# El nombre del dataset se deriva del nombre del ZIP cuando el usuario no pasa
# --datos. Como ese nombre NO lo ha elegido el, se sanea aqui: los datasets
# reales llegan con mayusculas y espacios ('PeopleDetectionThermal.zip') y la
# API los rechaza a proposito, porque esa validacion evita path traversal.
if [ -z "$JOB" ] && [ -z "$DATOS" ]; then
  CRUDO="$(basename "$ZIP" .zip)"
  DATOS="$(sanitizar "$CRUDO")"
  [ -n "$DATOS" ] || DATOS="dataset"
  if [ "$DATOS" != "$CRUDO" ]; then
    info "El ZIP se llama '$CRUDO'; se usara '$DATOS' como nombre de dataset."
    info "Para elegir otro: --datos NOMBRE"
  fi
fi

# --proyecto y un --datos explicito SI los eligio el usuario, asi que se
# comprueban aqui: fallar antes de subir 1 GB de datos, no despues.
nombre_valido() {
  printf '%s' "$1" | grep -qE '^[a-z0-9][a-z0-9_-]{0,63}$'
}

if [ -z "$JOB" ]; then
  if ! nombre_valido "$PROYECTO"; then
    die "Nombre de proyecto invalido: '$PROYECTO'.
       Solo se permiten minusculas, digitos, guion y guion bajo (max 64).
       Sugerencia:  $(sanitizar "$PROYECTO")"
  fi

  if ! nombre_valido "$DATOS"; then
    die "Nombre de dataset invalido: '$DATOS'.
       Sugerencia:  $(sanitizar "$DATOS")"
  fi
fi

CLASSES_JSON="$(
  printf '%s' "$CLASES" | awk -F',' '{for(i=1;i<=NF;i++){gsub(/^[ \t]+|[ \t]+$/,"",$i); if($i!="") printf "%s\"%s\"", (i>1?",":""), $i}}'
)"
if [ -n "$CLASES" ] && [ -z "$CLASSES_JSON" ]; then
  die "--clases no puede ser solo comas o espacios"
fi
# CLASSES_JSON vacio significa "el usuario no eligio nombres": la API los saca
# del data.yaml del ZIP y, si tampoco los trae, genera class_0, class_1... Un
# default fijo tipo 'person' etiquetaba como persona un modelo de edificios.

# ─────────────────────────────────────────────────────────────────────────────
# Utilidades HTTP
# ─────────────────────────────────────────────────────────────────────────────
# ultima_llamada: cuerpo de la ultima respuesta y codigo HTTP
ULTIMO_CUERPO=""
ULTIMO_CODIGO=""

http() {
  local metodo="$1" ruta="$2" datos="${3:-}"
  local tmp; tmp="$(mktemp)"
  if [ -n "$datos" ]; then
    # `curl -w '%{http_code}'` ya emite 000 si no hay conexion, asi que no se
# encadena un `|| echo 000`: concatenaria el codigo dos veces ("000000").
ULTIMO_CODIGO="$(curl -sS -o "$tmp" -w '%{http_code}' \
      -X "$metodo" "$API$ruta" -H 'Content-Type: application/json' -d "$datos" || true)"
  else
    ULTIMO_CODIGO="$(curl -sS -o "$tmp" -w '%{http_code}' \
      -X "$metodo" "$API$ruta" || true)"
  fi
  ULTIMO_CUERPO="$(cat "$tmp")"; rm -f "$tmp"
}

# Extrae el campo 'detail' de un error de FastAPI.
detalle_error() {
  printf '%s' "$ULTIMO_CUERPO" | jq -r '.detail // empty' 2>/dev/null \
    || printf '%s' "$ULTIMO_CUERPO"
}

die_http() {
  local d; d="$(detalle_error)"
  [ -n "$d" ] || d="$(printf '%s' "$ULTIMO_CUERPO" | head -c 300)"
  err "HTTP $ULTIMO_CODIGO: $d"
  exit 1
}

# ─────────────────────────────────────────────────────────────────────────────
# 0. El servicio esta vivo
#
# Se reintenta en vez de morir: la conexion puede caerse y volver (un tunel SSH
# que se reinicia, una red intermitente). El entrenamiento corre en el worker y
# no se ve afectado, asi que perder el monitor no debe_costar el trabajo.
# ─────────────────────────────────────────────────────────────────────────────
ESPERA=0
until http GET /health && [ "$ULTIMO_CODIGO" = "200" ]; do
  if [ "$ESPERA" = "0" ]; then
    warn "Sin respuesta de la API en $API/health. Reintentando cada ${REINTENTO}s."
    warn "Si es un tunel, comprueba que siga levantado. El entrenamiento NO se ha parado."
    ESPERA=1
  fi
  sleep "$REINTENTO"
done
info "API $API  $C_DIM($(printf '%s' "$ULTIMO_CUERPO" | jq -c .))$C_RESET"

if [ -n "$JOB" ]; then
  # ───────────────────────────────────────────────────────────────────────────
  # Modo seguimiento: solo monitoreo y descarga. No se crea nada.
  # ───────────────────────────────────────────────────────────────────────────
  JOB_ID="$JOB"
  http GET "/api/trainings/$JOB_ID/status"
  [ "$ULTIMO_CODIGO" = "200" ] || die_http
  ESTADO="$(printf '%s' "$ULTIMO_CUERPO" | jq -r .status)"
  if [ "$ESTADO" = "draft" ] || [ "$ESTADO" = "queued" ]; then
    warn "El job esta en '$ESTADO': todavia no ha empezado. Se seguira esperando."
  fi
  ok "Seguimiento del job $JOB_ID  $C_DIM($(printf '%s' "$ULTIMO_CUERPO" | jq -c '{status,epoch,total_epochs}'))"
else

# ─────────────────────────────────────────────────────────────────────────────
# 1. Proyecto (crear o recuperar)
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ID="$(curl -sS "$API/api/projects" | jq -r --arg n "$PROYECTO" '.[] | select(.name==$n) | .id' | head -1)"
MODEL_OVERRIDE=""

if [ -n "$PROJECT_ID" ]; then
  ok "Proyecto '$PROYECTO' ya existe  $C_DIM($PROJECT_ID)$C_RESET"
  ACTUAL_TASK="$(curl -sS "$API/api/projects/$PROJECT_ID" | jq -r .task)"
  ACTUAL_MODEL="$(curl -sS "$API/api/projects/$PROJECT_ID" | jq -r .model)"
  if [ "$REINICIAR" = "1" ]; then
    # Sin --clases NO se tocan las clases del proyecto: dejarlas vacias
    # borraria las que dedujo el dataset al subirlo. Para cambiarlas hay que
    # pedirlas explicitamente.
    if [ -n "$CLASSES_JSON" ]; then
      http PUT "/api/projects/$PROJECT_ID" \
        "{\"task\":\"$TASK\",\"model\":\"$MODELO\",\"classes\":[$CLASSES_JSON]}"
    else
      http PUT "/api/projects/$PROJECT_ID" \
        "{\"task\":\"$TASK\",\"model\":\"$MODELO\"}"
    fi
    [ "$ULTIMO_CODIGO" = "200" ] || die_http
    ok "Proyecto actualizado a task=$TASK model=$MODELO"
    if [ -z "$CLASSES_JSON" ]; then
      info "Clases del proyecto intactas (no pasaste --clases)."
    fi
  else
    if [ "$ACTUAL_TASK" != "$TASK" ]; then
      warn "El proyecto usa task=$ACTUAL_TASK y pediste task=$TASK."
      warn "La tarea la fija el proyecto; usalo con --reiniciar para cambiarla."
      TASK="$ACTUAL_TASK"
    fi
    if [ "$ACTUAL_MODEL" != "$MODELO" ]; then
      # El modelo puede variar por job: se envia como override en init.
      info "Este job usara $MODELO (el proyecto tiene $ACTUAL_MODEL)."
      MODEL_OVERRIDE=",\"model\":\"$MODELO\""
    fi
  fi
else
  # classes:[] significa "no he elegido nombres". Lo rellena la API al subir el
  # dataset, con los names del data.yaml del ZIP o con class_0, class_1...
  http POST /api/projects \
    "{\"name\":\"$PROYECTO\",\"task\":\"$TASK\",\"model\":\"$MODELO\",\"classes\":[$CLASSES_JSON]}"
  [ "$ULTIMO_CODIGO" = "201" ] || die_http
  PROJECT_ID="$(printf '%s' "$ULTIMO_CUERPO" | jq -r .id)"
  if [ -n "$CLASES" ]; then
    ok "Proyecto creado  $C_DIM($PROJECT_ID)$C_RESET  task=$TASK model=$MODELO clases=[$CLASES]"
  else
    ok "Proyecto creado  $C_DIM($PROJECT_ID)$C_RESET  task=$TASK model=$MODELO"
    info "Sin --clases: los nombres saldran del data.yaml del dataset."
  fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# 2. Dataset (subir o recuperar)
# ─────────────────────────────────────────────────────────────────────────────
DS_ID="$(curl -sS "$API/api/projects/$PROJECT_ID/datasets" \
  | jq -r --arg n "$DATOS" '.[] | select(.name==$n) | .id' | head -1)"

if [ -n "$DS_ID" ]; then
  ok "Dataset '$DATOS' ya registrado  $C_DIM($DS_ID)$C_RESET  $C_DIM(no se vuelve a subir)$C_RESET"
else
  TAM_ZIP="$(du -h "$ZIP" | cut -f1)"
  info "Subiendo $ZIP ($TAM_ZIP) ... esto puede tardar segun el tamano y la red."
  # El upload va como multipart/form-data, no como JSON: se hace aparte.
  # Se envia `name` explicito porque la API toma, por defecto, el nombre del
  # ZIP ("PeopleDetectionThermal.zip"), que casi siempre trae mayusculas.
  ULTIMO_CODIGO="$(curl -sS -o /tmp/.tp_up.$$ -w '%{http_code}' \
    -X POST "$API/api/projects/$PROJECT_ID/datasets" \
    -F "archive=@$ZIP" \
    -F "name=$DATOS" || true)"
  ULTIMO_CUERPO="$(cat /tmp/.tp_up.$$)"; rm -f /tmp/.tp_up.$$
  [ "$ULTIMO_CODIGO" = "201" ] || die_http
  DS_ID="$(printf '%s' "$ULTIMO_CUERPO" | jq -r .id)"
  N_IMG="$(printf '%s' "$ULTIMO_CUERPO" | jq -r .num_images)"
  FORMATO="$(printf '%s' "$ULTIMO_CUERPO" | jq -r .label_format)"
  ok "Dataset subido  $C_DIM($DS_ID)$C_RESET  ${N_IMG:-?} imagenes, etiquetas $FORMATO"

  if [ "$TASK" = "segment" ] && [ "$FORMATO" = "boxes" ]; then
    warn "Has declarado task=segment pero las etiquetas son cajas. El entrenamiento"
    warn "se detendra en init con un 422 explicativo. Usa task=detect."
  fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# 3. init + start
# ─────────────────────────────────────────────────────────────────────────────
PARAMS="{\"epochs\":$EPOCHS,\"imgsz\":$IMGSZ,\"batch\":$BATCH,\"workers\":$WORKERS,\"patience\":$PATIENCE}"
[ -n "$FRACCION" ]    && PARAMS="$(jq -c --argjson v "$FRACCION"    '. + {fraction:$v}' <<<"$PARAMS")"
[ -n "$CLS_PW" ]      && PARAMS="$(jq -c --argjson v "$CLS_PW"      '. + {cls_pw:$v}'   <<<"$PARAMS")"
[ -n "$DISPOSITIVO" ] && PARAMS="$(jq -c --argjson v "$DISPOSITIVO" '. + {device:$v}'  <<<"$PARAMS")"
if [ -n "$EXTRA_JSON" ]; then
  PARAMS="$(jq -c --argjson e "$EXTRA_JSON" '. + $e' <<<"$PARAMS")" \
    || die "--extra no es JSON valido: $EXTRA_JSON"
fi

# Las clases viajan con el job. A diferencia del modelo, que es un override
# SOLO de este job, las clases se sincronizan: las escribe en el proyecto y en
# el dataset.yaml, que es el fichero del que lee los nombres Ultralytics. Sin
# esto, un --clases de un entrenamiento nuevo no llegaba al YAML y el modelo
# salia etiquetado con la clase de un entrenamiento anterior.
CLASSES_OVERRIDE=""
[ -n "$CLASSES_JSON" ] && CLASSES_OVERRIDE=",\"classes\":[$CLASSES_JSON]"

info "Configuracion: $PARAMS"

http POST "/api/trainings/$PROJECT_ID/init" \
  "{\"dataset_id\":\"$DS_ID\",\"params\":$PARAMS$MODEL_OVERRIDE$CLASSES_OVERRIDE}"
[ "$ULTIMO_CODIGO" = "201" ] || die_http
JOB_ID="$(printf '%s' "$ULTIMO_CUERPO" | jq -r .job_id)"
ok "Job creado en draft  $C_DIM($JOB_ID)$C_RESET"

# Clases con las que va a entrenar. La respuesta del init es la fuente de
# verdad: la API las resolvio (--clases > data.yaml del dataset > class_0...)
# y las escribio en el dataset.yaml, que es de donde lee los nombres
# Ultralytics.
CLASES_FINALES="$(printf '%s' "$ULTIMO_CUERPO" | jq -r '(.classes // []) | join(", ")')"
if [ -n "$CLASES_FINALES" ]; then
  ok "Clases  $C_DIM($CLASES_FINALES)$C_RESET"
  if [ -z "$CLASES" ]; then
    case "$CLASES_FINALES" in
      class_*)
        info "  (generadas: el dataset no trae data.yaml con nombres; usa --clases)"
        ;;
      *)
        info "  (salidas del data.yaml del dataset; usa --clases para fijarlas)"
        ;;
    esac
  fi
fi

http POST "/api/trainings/$JOB_ID/start"
[ "$ULTIMO_CODIGO" = "202" ] || die_http
ok "Job en cola. El worker lo ira reclamando (concurrencia 1)."

fi   # fin del else de --job

# ─────────────────────────────────────────────────────────────────────────────
# 4. Monitorizar
# ─────────────────────────────────────────────────────────────────────────────
info "Monitorizando cada ${INTERVALO}s. Ctrl-C para dejar el job corriendo."
printf '\n'

ULT_EPOCA=""
while :; do
  # Igual que el health check: si la conexion se cae se reintenta en vez de
  # abandonar. El job sigue entrenando en el worker mientras tanto.
  until http GET "/api/trainings/$JOB_ID/status" && [ "$ULTIMO_CODIGO" = "200" ]; do
    warn "Sin respuesta de la API. Reintentando cada ${REINTENTO}s; el job sigue corriendo."
    sleep "$REINTENTO"
  done

  ESTADO="$(printf '%s' "$ULTIMO_CUERPO" | jq -r .status)"
  EPOCA="$(printf '%s' "$ULTIMO_CUERPO" | jq -r '.epoch // 0')"
  TOTAL="$(printf '%s' "$ULTIMO_CUERPO" | jq -r '.total_epochs // 0')"
  # jq devuelve float ("1.0") y bash no puede hacer aritmética con decimales
  # ($((1.0 * 100)) es un error de sintaxis), asi que el porcentaje se pide
  # ya redondeado desde jq.
  PROGRESO="$(printf '%s' "$ULTIMO_CUERPO" | jq -r '((.progress // 0) * 100) | floor')"
  MAP50="$(printf '%s' "$ULTIMO_CUERPO" | jq -r '.metrics["metrics/mAP50(B)"] // empty')"
  MAP5095="$(printf '%s' "$ULTIMO_CUERPO" | jq -r '.metrics["metrics/mAP50-95(B)"] // empty')"
  PREC="$(printf '%s' "$ULTIMO_CUERPO" | jq -r '.metrics["metrics/precision(B)"] // empty')"
  RECALL="$(printf '%s' "$ULTIMO_CUERPO" | jq -r '.metrics["metrics/recall(B)"] // empty')"
  PEND="$(printf '%s' "$ULTIMO_CUERPO" | jq -r 'if .cancel_requested then "cancelacion solicitada" else "" end')"

  if [ "$EPOCA" != "$ULT_EPOCA" ] || [ -z "$ULT_EPOCA" ]; then
    ULT_EPOCA="$EPOCA"
    printf '\r  %-9s epoca %s/%s  %3d%%  mAP50=%-8s mAP50-95=%-8s P=%-7s R=%-7s %-20s' \
      "$ESTADO" "$EPOCA" "$TOTAL" "$PROGRESO" \
      "${MAP50:-}" "${MAP5095:-}" "${PREC:-}" "${RECALL:-}" "$PEND"
  fi

  case "$ESTADO" in
    succeeded) printf '\n'; ok "Entrenamiento finalizado correctamente."; break ;;
    failed)
      printf '\n'
      err "El entrenamiento fallo."
      ERR="$(printf '%s' "$ULTIMO_CUERPO" | jq -r '.error // "sin detalle"')"
      printf '%s\n' "$ERR" | tail -25 >&2
      exit 1
      ;;
    cancelled)  printf '\n'; warn "El job fue cancelado."; exit 2 ;;
    interrupted) printf '\n'; warn "El worker se detuvo a mitad. Reintenta con:"
      warn "  curl -X POST $API/api/trainings/$JOB_ID/restart?mode=scratch"; exit 3 ;;
  esac
  sleep "$INTERVALO"
done

# ─────────────────────────────────────────────────────────────────────────────
# 5. Resultados
# ─────────────────────────────────────────────────────────────────────────────
mkdir -p "$SALIDA"
ETIQUETA="${PROYECTO}_${TASK}_$(date +%Y%m%d-%H%M%S)"

printf '\n'
info "Descargando en $SALIDA"

DESCARGAR() {
  local archivo="$1"
  local codigo
  codigo="$(curl -sS -o "$SALIDA/${ETIQUETA}_$archivo" -w '%{http_code}' \
    "$API/api/trainings/$JOB_ID/download?file=$archivo" || true)"
  if [ "$codigo" = "200" ]; then
    ok "  $archivo  $C_DIM($(du -h "$SALIDA/${ETIQUETA}_$archivo" | cut -f1))$C_RESET"
  else
    warn "  $archivo no disponible (HTTP $codigo)"
    rm -f "$SALIDA/${ETIQUETA}_$archivo"
  fi
}

# El informe NO lleva las imagenes dentro: las referencia por nombre relativo.
# Por eso hay que bajarlas a la MISMA carpeta que el HTML, o se abre con todos
# los iconos rotos. Van en un ZIP porque el endpoint no acepta nombres de
# fichero del cliente (la lista es cerrada) y asi es una sola peticion.
#
# Limitacion conocida: si descargas dos jobs al MISMO --salida, la segunda
# tanda sobrescribe las imagenes de la primera. Las referencias son relativas,
# asi que renombrar los ficheros romperia el informe. Con la carpeta por
# defecto (lleva timestamp) no ocurre.
DESCARGAR_ASSETS() {
  local tmp="$SALIDA/.assets.$$.zip" codigo n

  codigo="$(curl -sS -o "$tmp" -w '%{http_code}' \
    "$API/api/trainings/$JOB_ID/report-assets" || true)"

  if [ "$codigo" != "200" ]; then
    rm -f "$tmp"
    warn "  sin imagenes de Ultralytics en este run (HTTP $codigo)"
    return 0
  fi
  if ! command -v unzip >/dev/null 2>&1; then
    rm -f "$tmp"
    warn "  el informe trae imagenes en un ZIP pero falta 'unzip' para extraerlas"
    return 0
  fi
  # -o sobrescribe sin preguntar: reejecutar el script no debe quedarse
  # esperando un "replace?".
  if ! unzip -oq "$tmp" -d "$SALIDA" 2>/dev/null; then
    rm -f "$tmp"
    warn "  no se pudieron descomprimir las imagenes del informe"
    return 0
  fi
  rm -f "$tmp"

  n="$(ls -1 "$SALIDA" 2>/dev/null | grep -cE '\.(png|jpg|yaml)$' || true)"
  ok "  imagenes del informe  $C_DIM(${n:-0} ficheros, junto al HTML)$C_RESET"
}

if [ "$SIN_DESCARGA" = "1" ]; then
  warn "--sin-descarga: no se descarga nada."
else
  DESCARGAR best.pt
  DESCARGAR last.pt
  DESCARGAR results.csv
  DESCARGAR report.html
  # Despues del HTML: las referencias del informe son relativas y solo
  # resuelven si las imagenes acaban en esta misma carpeta.
  DESCARGAR_ASSETS
fi

# Las metricas finales en texto plano, sin depender del informe HTML.
# Se consulta el estado de nuevo: las descargas anteriores ya cambiaron el
# contexto y no conviene depender de una variable de la ultima iteracion.
http GET "/api/trainings/$JOB_ID/status"
printf '\n'
info "Metricas finales"
printf '%s' "$ULTIMO_CUERPO" | jq -r '
  .metrics |
  "  mAP50      \(.["metrics/mAP50(B)"]       // "n/d")\n" +
  "  mAP50-95   \(.["metrics/mAP50-95(B)"]    // "n/d")\n" +
  "  precision  \(.["metrics/precision(B)"]   // "n/d")\n" +
  "  recall     \(.["metrics/recall(B)"]      // "n/d")\n" +
  "  box_loss   \(.["train/box_loss"]         // "n/d")\n" +
  "  tiempo     \(.time // "n/d") s"' >&2 || warn "(no se pudieron leer las metricas)"

printf '\n'
ok "Listo. Job $JOB_ID"
info "Informe HTML: $SALIDA/${ETIQUETA}_report.html"
info "Pesos:        $SALIDA/${ETIQUETA}_best.pt"
if [ "$SIN_DESCARGA" != "1" ]; then
  info "Imagenes:     $(ls -1 "$SALIDA" 2>/dev/null | grep -cE '\.(png|jpg)$' || true) ficheros en la MISMA carpeta"
  info "              (el informe las referencia por nombre; no las lleva dentro)"
fi
