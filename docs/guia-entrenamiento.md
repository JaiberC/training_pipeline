# Guía de entrenamiento

Guía de uso del servicio `training_pipeline` para personas que ya lo tienen
desplegado y quieren entrenar un modelo de detección o segmentación sobre su
propio dataset.

No explica cómo instalar el servicio: eso está en el [README](../README.md).
Da por hecho que el servicio está arrancado y que puedes hablar con él.

- [1. Qué hace esto](#1-qué-hace-esto)
- [2. Comprobar que el servicio está vivo](#2-comprobar-que-el-servicio-está-vivo)
- [3. Preparar el dataset](#3-preparar-el-dataset)
- [4. El comando mínimo](#4-el-comando-mínimo)
- [5. Referencia de parámetros](#5-referencia-de-parámetros)
- [6. Elegir los parámetros](#6-elegir-los-parámetros)
- [7. Los resultados](#7-los-resultados)
- [8. El informe HTML](#8-el-informe-html)
- [9. Ciclo de vida del job](#9-ciclo-de-vida-del-job)
- [10. Recetas](#10-recetas)
- [11. Problemas frecuentes](#11-problemas-frecuentes)
- [12. Glosario](#12-glosario)
- [Apéndice A. La API](#apéndice-a-la-api)

---

## 1. Qué hace esto

Tienes un conjunto de imágenes con personas marcadas dentro y quieres un
programa que detecte personas en imágenes nuevas, de noche, en cámara térmica.
Un modelo así no se escribe a mano: se entrena, y entrenar significa darle
muchas imágenes marcadas y que ajuste sus parámetros internos hasta que aprenda.

Este servicio se encarga del trabajo pesado:

1. Tú le das un dataset (un ZIP con imágenes y sus etiquetas).
2. Él levanta el modelo base ya entrenado por otra gente.
3. Lo ajusta con tus datos durante las épocas que le pidas.
4. Te devuelve los pesos aprendidos (`best.pt`) y un informe con métricas.

Lo que **no** hace, y conviene saber de antemano:

- No inventa etiquetas. Garbage in, garbage out: si tus cajas están mal
  marcadas, el modelo aprende mal y no hay forma de arreglarlo desde aquí.
- No elige el modelo por ti. Tú decides si quieres uno pequeño y rápido o uno
  grande y preciso, con el compromiso de tiempo que implica.
- No entrena más de un modelo a la vez. La cola es de **concurrencia 1** (ver
  [sección 9](#9-ciclo-de-vida-del-job)).

El flujo completo son **tres pasos**:

```
   preparar el ZIP  →  ejecutar el script  →  descargar resultados
          ↑                                        │
          └──── solo la primera vez ───────────────┘
```

Los pasos 2 y 3 se repiten tantas veces como quieras. El dataset se sube **una
sola vez**: el script lo reutiliza.

---

## 2. Comprobar que el servicio está vivo

Antes del primer entrenamiento, dos comprobaciones.

### 2.1 La API responde

```bash
curl -s http://localhost:8000/health
```

```json
{"status":"ok","database":true,"schema":true}
```

Los tres campos importan:

| Campo | Qué significa si no está bien |
|---|---|
| `status` | `degraded` si algo falla. Con `ok`, sigue. |
| `database` | Que hay conexión con Postgres. Sin base de datos no hay nada. |
| `schema` | Que **las tablas existen**. Sin esto el worker levantaría contra una base vacía y fallaría al leer la cola. |

Sustituye el puerto por el que tengas en `API_PORT` del `.env` (por defecto
`8000`). Si el servicio está en otra máquina, usa su dirección.

### 2.2 La GPU está detectada

Este es el paso que más gente se salta, y es el que más cuesta caro después:

```bash
docker logs training-pipeline-worker-1 2>&1 | grep "CUDA disponible"
```

Debe decir:

```
CUDA disponible: True -> device=0
```

**Si dice `False`, todo el entrenamiento se hace en CPU.** No falla: funciona,
pero es más de **10 veces más lento**. Una corrida de 6 horas pasa a ser de 60.

### 2.3 Nota sobre las horas

`docker logs` imprime las horas en **UTC**. `nvidia-smi` y `date` usan la hora
local del equipo. Entre las dos señales puede haber varias horas de diferencia
aparente del mismo proceso. No lo tomes como que algo lleva parado.

```bash
date -u              # hora UTC, comparable con docker logs
date                 # hora local
```

---

## 3. Preparar el dataset

### 3.1 Qué acepta el servicio

Un **único archivo ZIP** con dos cosas dentro:

- Las imágenes (`.jpg`, `.png`, ...)
- Un `.txt` por imagen con lo que hay marcado en ella

El ZIP se sube **una sola vez**. El servicio lo descomprime y lo guarda; a
partir de ahí, cualquier entrenamiento posterior lo reutiliza.

### 3.2 Los dos layouts admitidos

Hay dos formas de organizar lo mismo. El servicio las detecta solo, no hace
falta declararlas.

**Layout YOLO (estándar)** — imágenes y etiquetas en carpetas hermanas:

```
mi_dataset/
├── images/
│   ├── train/       0001.jpg  0002.jpg  ...
│   └── val/         0101.jpg  0102.jpg  ...
└── labels/
    ├── train/       0001.txt  0002.txt  ...
    └── val/         0101.txt  0102.txt  ...
```

**Layout Roboflow** — cada split es una carpeta que contiene sus dos cosas:

```
mi_dataset/
├── train/
│   ├── images/      0001.jpg  0002.jpg  ...
│   └── labels/      0001.txt  0002.txt  ...
├── valid/           ← ojo: "valid", no "val"
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

Roboflow existe como layout aparte porque es la fuente más común de datasets, 
y reorganizar 20.000 ficheros a mano no es razonable.

> Si tu dataset tiene `valid/` pero no `val/`, es Roboflow. Si tiene
> `images/train/`, es YOLO estándar. El servicio lo detecta; tú no tienes que
> hacer nada.

### 3.3 El formato de las etiquetas

Cada imagen tiene su `.txt` con **el mismo nombre y extensión `.txt`**:

```
images/train/0001.jpg
labels/train/0001.txt     ← mismo nombre, .txt
```

El contenido del `.txt` cambia según lo que quieras marcar.

**Cajas (detección).** Cinco columnas separadas por espacios:

```
0 0.5123 0.4471 0.1840 0.6203
│  │      │      │      └── altura, fracción de la imagen
│  │      │      └───────── anchura, fracción de la imagen
│  │      └──────────────── centro vertical, fracción de la imagen
│  └─────────────────────── centro horizontal, fracción de la imagen
└────────────────────────── número de clase
```

Todas las coordenadas van **normalizadas entre 0 y 1**, no en píxeles. El origen
(0,0) está en la esquina superior izquierda.

**Polígonos (segmentación).** Siete columnas o más, con pares de coordenadas:

```
0 0.4231 0.3352 0.6018 0.3310 0.6102 0.7594 0.4271 0.7638
└─ x1 y1 ─┘ └─ x2 y2 ─┘ └─ x3 y3 ─┘ └─ x4 y4 ─┘
```

Cada par de coordenadas es una esquina de la máscara. Da igual cuántos vértices
tenga, mientras haya al menos 3.

**El número de clase es la posición en `--clases`.** Si escribes
`--clases "persona,vehiculo"`, entonces `0` es persona y `1` es vehículo. El
orden importa y no se puede adivinar.

#### El error más común de quien empieza

**Una imagen sin objetos necesita un `.txt` vacío, no un fichero que no existe.**

```
0001.jpg          ← hay un coche
0001.txt          0 0.51 0.44 0.18 0.62

0002.jpg          ← no hay nada
0002.txt          ← fichero vacío, de 0 bytes
```

Si el `.txt` falta, el servicio asume que es un error de sincronización y el
entrenamiento falla con un error poco descriptivo. Un `.txt` vacío le dice
"esta imagen es una escena negativa", que es información valiosa: enseña al
modelo a **no** disparar cuando no hay nada.

#### Detección contra segmentación

El tipo de etiquetas y el modelo deben coincidir. El servicio lo comprueba al
iniciar el entrenamiento y **rechaza el job con un error claro**, en vez de
dejar que falle a la mitad:

| Etiquetas | `--task` | Modelos válidos |
|---|---|---|
| Cajas (5 columnas) | `detect` | `yolo26n.pt` … `yolo26x.pt`, `rtdetr-l.pt`, `rtdetr-x.pt` |
| Polígonos (7+ columnas) | `segment` | `yolo26n-seg.pt` … `yolo26x-seg.pt` |

**RT-DETR solo detecta.** No existe `rtdetr-seg.pt`; no busques cómo activarlo.

La tarea la fija el **proyecto**, no el job. Un mismo proyecto puede entrenar
con modelos distintos, pero todos de la misma tarea.

### 3.4 Clases y balance

Las clases se declaran con `--clases`, en orden:

```bash
--clases "persona"
--clases "persona,vehiculo,perro"
```

El **número de clase es la posición en la lista**. Si pones `"persona,vehiculo"`,
el `0` es persona y el `1` es vehículo. Ese orden tiene que coincidir con los
números de tus etiquetas.

#### Si no pasas `--clases`

No hace falta. El nombre de las clases se deduce solo, en este orden:

1. **Lo que declares con `--clases`.**
2. **Los `names` del `data.yaml`** que trae el propio dataset. Los datasets de
   Roboflow lo llevan, así que un `--clases` de más solo hace falta si quieres
   cambiar el nombre.
3. **`class_0`, `class_1`, ...** generados, si el dataset no trae `data.yaml`.

El script te dice siempre con qué nombres va a entrenar:

```
Clases  (Construcción)
```

**El número de clases sale de tus etiquetas, no de lo que declares.** El servicio
las lee y sabe cuántas hay. Si declaras menos de las que usas, lo rechaza con
un error que dice exactamente qué escribir:

```
El dataset 'edificios' usa 3 clases (indices 0..2) y el proyecto solo declara
1: person.
Los nombres van en el MISMO ORDEN que los numeros de las etiquetas.
Declara las 3, por ejemplo:  --clases "Edificio, Coche, Arbol".
```

Declarar **más** clases de las que hay no pasa nada: sobran y no molestan.

> **Las clases se sincronizan en cada entrenamiento**, no solo al crear el
> proyecto. Si pasas `--clases "Construcción"`, ese nombre queda en el proyecto
> **y** en el fichero que lee el entrenamiento. Puedes renombrar las clases sin
> volver a subir el dataset.

El modelo es al revés: `--modelo` solo cambia **ese** job, y el proyecto
conserva el suyo. Así puedes entrenar `yolo26n`, `yolo26s` y `yolo26l` sobre el
mismo dataset sin tocar nada.

#### Si tu dataset tiene una clase que aparece en menos del 10% de las imágenes

El modelo la va a ignorar: con tan pocos ejemplos no puede aprender a
distinguirla. Hay dos salidas:

1. **Repetir las imágenes de esa clase** en el dataset.
2. **Ponderar las clases** con `--cls-pw 0.5`:

```bash
--cls-pw 0.5    # valor entre 0 y 1
```

Le dice al entrenamiento que las clases raras importen más. Cuanto más bajo es el
número, más pesa la clase infrequent. Con `0.5` ya es una compensación
notable.

### 3.5 La trampa de `fraction` — léela antes de usarla

`--fraccion` existe para entrenar sobre una parte del dataset, por ejemplo para
iterar rápido. **Pero este servicio recorta el dataset por orden de fichero, no
al azar.**

Con el dataset `people-detection-thermal`, los ficheros están ordenados por
origen. Las imágenes FLIR están al final. Recortar el 40% se queda con las
primeras, que incluyen las FLIR. Recortar menos, no:

| `--fraccion` | Imágenes de entrenamiento | FLIR en el subconjunto |
|---|---|---|
| `0.10` – `0.20` | 2.100 – 4.300 | **0%** |
| `0.40` | ~8.500 | ~69% |

Si entrenas con `--fraccion 0.10` believing que es "una muestra representativa",
**estás entrenando sin ninguna imagen FLIR** y no lo vas a notar hasta que veas
el mAP50.

**Regla:** antes de usar una fracción pequeña, comprueba que tu subconjunto
sigue conteniendo las imágenes que te importan. Si no sabes el orden de tus
ficheros, no uses fracción.

### 3.6 Empaquetar el ZIP

```bash
cd /ruta/del/dataset
zip -r mi_dataset.zip mi_dataset/
```

Dos detalles:

**Da igual si el ZIP tiene una carpeta envoltorio o no.** El servicio lo aplana
automáticamente. Las dos formas valen:

```
mi_dataset.zip  ->  mi_dataset/train/images/...     ✓
mi_dataset.zip  ->  train/images/...                ✓
```

Lo que **no** vale es tener las etiquetas fuera de la carpeta de imágenes. El
servicio no busca: si `images/train/` no existe, dice exactamente qué esperaba
y el entrenamiento no arranca.

**El nombre del ZIP se sanea automáticamente.** Si se llama
`PeopleDetectionThermal.zip`, el servicio lo guarda como
`peopledetectionthermal` (minúsculas, sin espacios), porque los nombres de
recursos solo admiten minúsculas, dígitos, guion y guion bajo. Funciona, pero es
feo y puede confundirte. **Ponle tú el nombre que quieras** con `--datos`:

```bash
--datos person_detection_thermal
```

El nombre que elijas también es un nombre de recurso: **solo minúsculas,
dígitos, guion y guion bajo, máximo 64 caracteres**. Si el script detecta un
problema, te propone uno válido corregido.

---

## 4. El comando mínimo

```bash
./scripts/entrenar.sh --proyecto my_project \
    --zip data/datasets/mi_dataset.zip \
    --modelo yolo26n.pt \
    --api http://localhost:8000
```

Eso basta para un entrenamiento de detección de personas. Todo lo demás tiene
valores por defecto razonables.

### Lo que hace el script, por dentro

```
1.  comprueba que la API responde
2.  crea el proyecto, o lo recupera si ya existe
3.  sube el ZIP, o lo reutiliza si el dataset ya está subido
4.  crea el job y lo encola
5.  monitoriza cada 10 s mostrando el avance
6.  al terminar, descarga pesos, métricas e informe
```

### Es idempotente

Si el proyecto y el dataset ya existen, **no los vuelve a subir**. La segunda
vez que ejecutes el comando, sube 0 bytes y crea un job nuevo.

Eso es lo que hace que iterar sea barato:

```bash
# primera vez: sube el dataset (puede tardar: son ~20.000 ficheros)
./scripts/entrenar.sh --proyecto my_project --zip mi.zip --modelo yolo26n.pt

# segunda vez: NO sube nada, solo entrena
./scripts/entrenar.sh --proyecto my_project --zip mi.zip --modelo yolo26s.pt
```

### Puedes cortarlo sin perder el entrenamiento

Si pulsas `Ctrl-C`, **el job sigue corriendo en el servidor**. Solo pierdes el
monitor y la descarga automática. Lo retomas cuando quieras:

```bash
./scripts/entrenar.sh --job <UUID> --api http://localhost:8000
```

---

## 5. Referencia de parámetros

Ejecuta `./scripts/entrenar.sh -h` para ver la lista corta.

### 5.1 Identificación

| Parámetro | Por defecto | Qué hace |
|---|---|---|
| `--proyecto NOMBRE` | — | Agrupa tus entrenamientos. **Obligatorio** salvo con `--job`. Reutilizado si ya existe. |
| `--zip RUTA.zip` | — | El dataset. **Obligatorio** salvo con `--job`. Solo se sube si el dataset no existe ya. |
| `--datos NOMBRE` | el nombre del ZIP | Cómo guardar el dataset dentro del proyecto. Ponle un nombre limpio si el ZIP trae mayúsculas o espacios. |
| `--job UUID` | — | Adjuntarse a un entrenamiento ya creado. No crea nada: solo monitoriza y descarga. |
| `--api URL` | puerto de `API_PORT` | Dónde está el servicio. |

`--proyecto` y `--datos` solo se usan **al crear**. Si ya existen, el script los
reutiliza tal cual.

### 5.2 Modelo

| Parámetro | Por defecto | Qué hace |
|---|---|---|
| `--modelo M` | `yolo26l.pt` | Qué modelo base entrenar. **Se puede cambiar en cada job.** |
| `--task T` | `detect` | `detect` o `segment`. Solo se aplica al crear el proyecto. |
| `--clases "a,b"` | deducidas | Los nombres de clase, **en orden**. El orden define el número de clase de cada etiqueta. Si no lo pasas, se deducen del `data.yaml` del dataset. |
| `--reiniciar` | no | Actualiza el proyecto si ya existe (`--task`, `--modelo`). |

**El modelo puede variar entre jobs del mismo proyecto; la tarea no.** Solo se
cambia con `--reiniciar`, y eso puede invalidar los datasets ya registrados.

**Las clases, al revés: se sincronizan siempre.** Si pasas `--clases`, ese
nombre queda en el proyecto **y** en el fichero que lee el entrenamiento, de
modo que aplica a todos los jobs siguientes. Puedes renombrar las clases sin
volver a subir el dataset.

| | ¿Cambia el proyecto? | ¿Aplica a otros jobs? |
|---|---|---|
| `--modelo` | No, solo al job | No, cada job puede usar uno |
| `--clases` | Sí | Sí, quedan fijadas |
| `--task` | Con `--reiniciar` | Sí, y puede invalidar el dataset |

#### El catálogo de modelos

| Modelo | Tarea | Cuándo |
|---|---|---|
| `yolo26n.pt` | detect | El más rápido. Para pruebas y para CPU. |
| `yolo26s.pt` | detect | Equilibrium. **Buena primera opción en GPU.** |
| `yolo26m.pt` | detect | Más precisión, más lento. |
| `yolo26l.pt` | detect | El más grande de YOLO. El que usa el script por defecto. |
| `yolo26x.pt` | detect | El mayor. Solo si tienes tiempo de sobra. |
| `yolo26{n,s,m,l,x}-seg.pt` | segment | Segmentación. Las etiquetas deben ser polígonos. |
| `rtdetr-l.pt` | detect | Detector alternativo, sin anclas. |
| `rtdetr-x.pt` | detect | La versión mayor de RT-DETR. |

Cada letra es un salto grande de tamaño: `n` → `s` → `m` → `l` → `x`.

> **RT-DETR tiene un techo de unas 300 detecciones por imagen** con los pesos
> preentrenados. Si tu escena necesita más, no subirá de ahí. Para detección de
> personas no es problema.

### 5.3 Entrenamiento

| Parámetro | Por defecto | Qué hace |
|---|---|---|
| `--epochs N` | `50` | Cuántas pasadas completas por el dataset. Más = mejor, hasta que deje de mejorar. |
| `--imgsz N` | `640` | Lado de la imagen en píxeles. **El parámetro con más impacto.** |
| `--batch N` | `16` | Imágenes por paso. Más = más rápido, pero necesita más VRAM. |
| `--workers N` | `8` | Procesos que cargan imágenes en paralelo. Ver [sección 6](#6-elegir-los-parámetros). |
| `--patience N` | `20` | Épocas sin mejorar antes de parar. Ver abajo. |

#### Qué hace `patience`

Es la **parada temprana**: corta el entrenamiento si la métrica lleva N épocas
seguidas sin mejorar.

```bash
--patience 25
```

```
época 70 → mAP50 0.7800   ← nueva mejor marca
época 71 → 0.7750
...
época 95 → 0.7700   ← 25 épocas sin superar la 70 → PARA
```

**Parar antes no te cuesta calidad.** `best.pt` se guarda cada vez que la
métrica mejora, así que te quedas con los pesos de la mejor época, no con los de
la última.

`--patience 0` desactiva la parada temprana. Un `--patience` mayor o igual que
`--epochs` es lo mismo que no ponerlo: nunca se dispara.

### 5.4 Datos

| Parámetro | Por defecto | Qué hace |
|---|---|---|
| `--fraccion F` | todo | `0.02` = el 2% del dataset. **Lee [la advertencia](#35-la-trampa-de-fraction--léela-antes-de-usarla) antes de usarlo.** |
| `--cls-pw F` | — | `0` a `1`. Pondera las clases raras. Útil cuando una clase aparece en pocas imágenes. |
| `--dispositivo D` | automático | `0`, `cpu`, `-1`... Rara vez hace falta tocarlo. |
| `--extra JSON` | — | Cualquier parámetro de Ultralytics que el script no expone. |

#### `--extra`, la salida de emergencia

```bash
--extra '{"lr0":0.001,"optimizer":"AdamW"}'
--extra '{"cache":"ram"}'
--extra '{"mosaic":0.5,"degrees":10}'
```

Todo lo que pongas aquí se pasa tal cual a Ultralytics. Es potente y también es
fácil de romper sin querer: si escribes mal el JSON, el script aborta antes de
subir nada.

> **Ojo con `lr0`.** Si fijas `lr0` pero no el optimizador, Ultralytics elige
> uno automáticamente por encima de 10.000 iteraciones **e ignora tu `lr0`**.
> Si quieres controlar la tasa de aprendizaje, fija las dos cosas:
> `--extra '{"lr0":0.001,"optimizer":"AdamW"}'`. El servicio te avisa de esto
> automáticamente, pero es más limpio hacerlo a mano.

### 5.5 Salida

| Parámetro | Por defecto | Qué hace |
|---|---|---|
| `--salida RUTA` | `./resultados/<fecha>` | Dónde se descargan los pesos y el informe. |
| `--intervalo N` | `10` | Segundos entre consultas de estado. |
| `--sin-descarga` | no | No descarga nada al terminar. Para recogerlo luego con la API. |
| `--reintento N` | `10` | Segundos entre reintentos si la API no responde. |

Si la conexión se cae (un túnel que se reinicia, una red intermitente), el
script **reintenta en vez de abandonar**. El entrenamiento sigue en el worker y
no se ve afectado: perder el monitor no cuesta el trabajo.

### 5.6 Códigos de salida

| Código | Significado |
|---|---|
| `0` | El entrenamiento terminó correctamente. |
| `1` | Error: no arrancó, falló, o el dataset no es válido. |
| `2` | El job fue cancelado. |
| `3` | El worker se detuvo a mitad. Ver [sección 9](#9-ciclo-de-vida-del-job). |

---

## 6. Elegir los parámetros

### 6.1 `imgsz`: el que más importa

Es la decisión que más cambia el resultado y la más fácil de entender mal.

**Regla:** mira la resolución nativa de tus imágenes y usa ese número, o el más
alto que quepa en tu GPU.

| Dataset | Nativo | Recomendado |
|---|---|---|
| FLIR térmico | 640×512 | **640** |
| Cámara térmica moderna | 640×480 | **640** |
| CCTV color | 1920×1080 | **640** (ya estás recortando) |
| Imágenes pequeñas | 320×240 | **320** |

Subir la resolución **más allá de la nativa no aporta nada**: estás
interpolando píxeles que no existen.

Bajarla sí hace daño, y de forma **desproporcionada** en visión nocturna. Un
objeto a 20 píxeles de alto que reduces a la mitad se queda en 10, y el modelo
deixa de poder ni verlo. Es exactamente lo que pasa si bajas de 640 a 416.

#### Referencia medida en esta infraestructura

Dos entrenamientos idénticos salvo en resolución y cantidad de datos, con
`yolo26n`, `batch 16`, `workers 8` en una RTX 3080:

| Entrenamiento | Mejor mAP50 | Tiempo por época | Duración total |
|---|---|---|---|
| 416 px, 40% de los datos, 60 épocas | **0.7531** (época 44) | 82 s | 1 h 23 min |
| 640 px, 100% de los datos, 100 épocas | **0.8903** (época 30) | 229 s | ~6,4 h (estimado) |

**+14 puntos de mAP50.** Es la diferencia entre un modelo de prueba y uno
utilizable.

Dos matices honestos: el primer entrenamiento ya terminó, y su 0.7531 es la
mejor marca **definitiva** de 60 épocas. El segundo seguía corriendo cuando se
midió esto, así que su 0.8903 solo es el valor **en la época 30** y aún puede
subir. Aun así, la comparación no es justa a favor del segundo: comparar el
mejor valor de un run terminado contra un valor intermedio de otro.

Y el dato más revelador del primero: empezó en **0.5096** en la época 1. El
segundo empezó en **0.7402**. A 416 px, el modelo necesitaba 30 épocas para
compensar lo que había perdido por el recorte.

### 6.2 `batch`: lo que cabe en tu memoria

Sube `batch` mientras quepa. Más batch = menos pasos = más rápido.

| VRAM | `imgsz 640`, `yolo26n` | `imgsz 640`, `yolo26s` |
|---|---|---|
| 6 GB (Tesla P4) | 16 | 8 |
| 10 GB (RTX 3080) | 32 | 16 |
| 24 GB (RTX 4090) | 64 | 32 |

Si te pasas, el entrenamiento muere con un error de memoria CUDA. Si te quedas
muy corto, estás desperdiciando la tarjeta. **Si no estás seguro, quédate en
16**, que es seguro en todas partes.

### 6.3 `workers` y la memoria compartida — lee esto

`workers` es el número de procesos que cargan y preparan imágenes en paralelo.
Más workers = menos paradas de la GPU. Pero tiene un coste escondido que **cuelga
el entrenamiento entero**.

PyTorch pasa los tensores de esos procesos al proceso principal por
`/dev/shm`, la memoria compartida del contenedor. Docker la deja en **64 MB por
defecto**. Lo que necesitas:

```
workers × prefetch_factor(2) × batch × imgsz² × 3 bytes
```

Con los valores por defecto a 416 px:

```
8 × 2 × 16 × 416² × 3 = 126 MB      ← más del doble de lo que hay
```

Cuando la cola no cabe, **los procesos se bloquean y el entrenamiento se queda
parado: 0% de CPU, 0% de GPU, sin error y sin avanzar**. Es el fallo más
confuso que tiene este servicio, porque parece que el proceso está vivo.

**La solución es aumentar `shm_size` en el `compose.yaml`:**

```yaml
  worker:
    shm_size: 2gb
```

Va en el servicio `worker` del `compose.yaml` **base**, no en `compose.gpu.yaml`.
No tiene nada que ver con la GPU: es la memoria compartida que usa PyTorch para
cargar los datos, tanto en CPU como en GPU.

**Después de cambiarla hay que recrear el contenedor.** `docker compose up -d`
no aplica cambios de configuración a contenedores ya existentes:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml down
docker compose -f compose.yaml -f compose.gpu.yaml up -d
```

Comprueba que aplicó:

```bash
docker exec training-pipeline-worker-1 df -h /dev/shm    # debe decir 2,0G
```

Con 2 GB tienes margen hasta `imgsz 640` y `batch 32`. Si aun así se cuelga,
sácale a 4gb.

**¿Y si no puedes tocar el compose?** Baja `batch` o `imgsz` hasta que la
cuenta quepa en 64 MB, o usa `--workers 0`. Con `workers 0` funciona siempre
(el entrenamiento carga los datos en el proceso principal), pero vas más lento.

### 6.4 `fraccion`: con cuidado

Ya está explicación en [3.5](#35-la-trampa-de-fraction--léela-antes-de-usarla).
En resumen: recorta **por orden de fichero**, no al azar, y puede dejarte sin
las imágenes que más te importan.

### 6.5 Cuánto va a tardar

Medido en esta infraestructura con `yolo26n`, `batch 16`, `workers 8`:

| Entorno | Configuración | Por época |
|---|---|---|
| CPU (20 núcleos) | 320 px, 40% datos | 598 s |
| RTX 3080 | 416 px, 40% datos | 82 s |
| RTX 3080 | 640 px, 100% datos | 229 s |

El tiempo por época escala aproximadamente con **imágenes × píxeles**. La forma
fiable de saber cuánto tardará tu job concreto: **espera a la época 1, divide y
multiplica**. No hay forma de estimarlo bien antes.

---

## 7. Los resultados

### 7.1 Las cuatro métricas

Al terminar, el script imprime las métricas finales y las descarga en
`results.csv`. Son cuatro, y cada una mide algo distinto.

#### mAP50 — «¿detecta bien lo que hay?»

**La métrica principal.** Es el área bajo la curva de precisión-recall,
calculada con un umbral de confianza del 50%.

Pensarlo así: para cada objeto real de la imagen, ¿el modelo lo encontró, y con
cuánta confianza? Un mAP50 de 0.90 significa que el modelo acierta casi siempre.
Un mAP50 de 0.40 significa que se le escapa la mitad.

**Es la que debes mirar primero.** Si solo pudieras mirar una, es esta.

#### mAP50-95 — «¿los encuadra bien?»

Igual que mAP50 pero con umbrales **más estrictos** (del 50% al 95%). Exige que
la caja que predice el modelo se solape mucho con la real.

La **diferencia entre las dos te dice mucho**:

| Patrón | Lectura |
|---|---|
| mAP50 alto, mAP50-95 alto | El modelo detecta bien **y** encuadra bien. Caso normal. |
| mAP50 alto, mAP50-95 bajo | Detecta los objetos pero las cajas están **torcidas o desplazadas**. Suele ser un problema de resolución, o de etiquetas inconsistentes en tu dataset. |
| mAP50 bajo, mAP50-95 bajo | El modelo no está aprendiendo. Suele ser etiquetas malas o muy pocos datos. |

En visión térmica el segundo patrón es **frecuente**: las cajas de las personas
en contraluz son ambiguas, porque los bordes no están donde los pone el
anotador.

#### precision — «¿cuando dice que hay algo, lo hay?»

De todas las detecciones que el modelo hace, ¿cuántas son correctas?

Con precision 0.60, de cada 10 cajas que dibuja, 4 están en sitios donde no hay
nada. En el mundo real eso es ruido constante en el vídeo.

Causas típicas: el modelo dispara con umbrales bajos, o tu dataset tiene falsos
positivos sin haber marcado.

#### recall — «¿deja sin detectar algo que hay?»

De todos los objetos que hay, ¿cuántos encuentra?

Con recall 0.60, de cada 10 personas, se le escapan 4. En un sistema de
vigilancia, **esto es peor que la precision baja**: una persona que no se ve,
no se registra.

Causas típicas: objetos pequeños, lejanos, o mal iluminados. También
subir `--imgsz`.

### 7.2 Qué hacer con cada problema

| Síntoma | Interpretación | Qué probar |
|---|---|---|
| mAP50 bajo, recall peor que precision | Objetos que se escapan | Subir `--imgsz`, más datos, revisar si hay imágenes sin etiquetar |
| precision peor que recall | Detecta de más | Revisar etiquetas ruidosas, considerar subir el umbral de confianza al usar el modelo |
| mAP50 sube y baja mucho | Entrenamiento inestable | Bajar `--batch`, bajar el LR con `--extra`, más datos |
| mAP50 ≈ 0 desde la época 1 | No está aprendiendo | **Revisar el formato de las etiquetas**, el orden de `--clases`, y que el dataset se haya subido bien |
| mAP50 sube y luego baja en las últimas | Sobreajuste | Más datos, o menos épocas |
| La validación va muy por detrás del entrenamiento | Sobreajuste fuerte | Más datos. Es lo más común al entrenar con `fraction` baja |

### 7.3 ¿Qué valor es bueno?

No hay un número universal, pero para **detección de personas en imágenes
térmicas o nocturnas**, como referencia:

| mAP50 | Lectura |
|---|---|
| < 0.40 | Algo va mal. Revisa etiquetas antes de tocar nada más. |
| 0.40 – 0.60 | Funciona, pero mal. |
| 0.60 – 0.75 | Razonable para un primer resultado. |
| 0.75 – 0.88 | Bueno. |
| > 0.90 | Muy bueno. Comprueba que no estés evaluando sobre los datos de entrenamiento. |

**Lo que más importa es la comparación contra tu propia referencia**, no contra
un número de internet. El caso real de esta infraestructura:

- Primera versión: **0.7531** a 416 px con el 40% de los datos.
- Segunda versión: **0.8903** a 640 px con el 100% de los datos.

Mismo modelo base, mismos parámetros. La única diferencia fue la resolución y
cuántos datos había. **+14 puntos sin tocar nada del entrenamiento.**

### 7.4 `best.pt` y `last.pt`

| Fichero | Qué contiene |
|---|---|
| `best.pt` | Los pesos de la **mejor época** según la métrica. **Este es el que quieres.** |
| `last.pt` | Los pesos de la **última** época. Sirve para retomar un entrenamiento interrumpido. |

El script descarga los dos. En la práctica, `best.pt` es lo que se despliega.
`last.pt` solo hace falta si quieres continuar un entrenamiento a medias.

---

## 8. El informe HTML

Además de los pesos, cada job genera un informe:

```
resultados/<proyecto>_<task>_<fecha>_report.html
```

Es un HTML autónomo con gráficos. Ábrelo con doble clic, no necesitas servidor.
Útil para comparar visualmente dos entrenamientos o para pegarlo en un informe.

Qué contiene:

| Sección | Qué mirar |
|---|---|
| **Tarjetas de resumen** | mAP50, mAP50-95, precision, recall y número de épocas. Lo último que acabó. |
| **Calidad (mAP)** | mAP50 y mAP50-95 por época. La forma de la curva dice mucho: si sube y se aplana, has tocado techo. |
| **Precisión y recall** | Su evolución por época. Si se separan mucho, es el diagnóstico de la tabla de arriba. |
| **Pérdidas** | `box_loss`, `cls_loss`, `dfl_loss`. Bajan durante el entrenamiento. Si `cls_loss` **sube**, el modelo no está aprendiendo a distinguir clases. |
| **mAP50 por clase** | Solo si el run dejó datos por clase. Te dice si una clase concreta va mucho peor que las demás, que es lo que la media global esconde. |

También puedes generarlo bajo demanda para cualquier job:

```bash
curl -o informe.html http://localhost:8000/api/trainings/<UUID>/report
```

---

## 9. Ciclo de vida del job

### 9.1 Los estados

| Estado | Qué significa |
|---|---|
| `draft` | Creado, todavía no encolado. |
| `queued` | En la cola, esperando turno. |
| `running` | El worker lo está entrenando. |
| `succeeded` | Terminó correctamente. Los pesos están disponibles. |
| `failed` | Terminó con error. El mensaje está en el campo `error`. |
| `cancelled` | Lo paraste tú. **Los pesos se han borrado.** |
| `interrupted` | El worker se detuvo a mitad (se apagó, se reinició...). |

### 9.2 Solo corre uno a la vez

**La cola tiene concurrencia 1.** Si lanzas tres entrenamientos, se ejecutan
**uno detrás de otro**, no en paralelo. El segundo y el tercero se quedan en
`queued` hasta que el primero termine.

Esto es intencionado: el worker reserva la GPU entera, y dos entrenamientos
simultáneos la compartirían y ambos irían más lento. Pero conviene saberlo,
porque **lanzar tres no produce tres resultados más rápido: produce el mismo
resultado, tres veces más tarde.**

Para ver la cola:

```bash
curl -s http://localhost:8000/api/trainings | jq -r \
  '.[] | "\(.status)  \(.id)  \(.model)"'
```

### 9.3 Cuándo el worker se detiene

Si el contenedor del worker se para a mitad de un entrenamiento (un reinicio, un
`docker compose down`, un apagón), el job se queda **bloqueado en `running`
para siempre** y **bloquea la cola**: nada más arranca hasta que lo arregles.

Al volver a levantar el worker, todo job que estuviera en `running` pasa a
`interrupted` automáticamente, y la cola se desbloquea.

### 9.4 Recuperar un job interrumpido

Aquí hay **una trampa importante** que puede costarte horas de trabajo.

```bash
# mode=scratch  →  EMPIEZA DE CERO y BORRA todo lo anterior
curl -X POST "http://localhost:8000/api/trainings/<UUID>/restart?mode=scratch"

# mode=resume  →  CONTINÚA desde donde lo dejaste
curl -X POST "http://localhost:8000/api/trainings/<UUID>/restart?mode=resume"
```

| | `scratch` | `resume` |
|---|---|---|
| Qué hace | Borra `runs/<job>/` entero | Reanuda desde `last.pt` |
| Épocas | Empieza en la 0 | Continúa donde lo dejó |
| Pesos anteriores | **Se pierden** | Se conservan |
| Cuándo usarlo | Cambiaste el dataset o el modelo | Se apagó el servidor, fallo la red, te quedaste sin tiempo |

> **Aviso:** el script, al detectar un job `interrupted`, sugiere
> `mode=scratch`. **No le hagas caso.** Ese mensaje es incorrecto: `scratch`
> borra las horas de entrenamiento que ya llevabas. Usa **`mode=resume`**, que
> restaura época, optimizador y tasa de aprendizaje desde el último checkpoint.
>
> `resume` necesita que exista `last.pt`. Si no está, la API responde con un
> `409` y solo queda `scratch`.
>
> Si lo que quieres es cambiar el dataset o el modelo, entonces sí `scratch`,
> porque los pesos anteriores ya no valen.

### 9.5 Cancelar

```bash
# por el script: Ctrl-C, que deja el job corriendo
# por la API: cancela de verdad
curl -X POST http://localhost:8000/api/trainings/<UUID>/cancel
```

**Cancelar borra `runs/<job>/` entero, incluidos los pesos.**

> **Descarga antes de cancelar.** Si cancelas, se acabó. Si solo te interrumpe
> el monitor, usa `Ctrl-C`, que **no** cancela nada.

---

## 10. Recetas

### Prueba de humo

Antes de gastar horas de GPU, comprueba que el pipeline entero funciona. Un
entrenamiento diminuto:

```bash
./scripts/entrenar.sh --proyecto my_project \
    --zip data/datasets/mi_dataset.zip \
    --modelo yolo26n.pt \
    --epochs 2 --fraccion 0.40 --imgsz 416 \
    --api http://localhost:8000
```

Tarda minutos. Si termina, el dataset es válido, las rutas funcionan y el worker
entrena. **No te dice nada sobre la calidad final**, solo sobre la mecánica.

### Entrenamiento completo

```bash
./scripts/entrenar.sh --proyecto my_project \
    --zip data/datasets/mi_dataset.zip \
    --modelo yolo26n.pt \
    --epochs 100 --imgsz 640 \
    --batch 16 --workers 8 --patience 25 \
    --api http://localhost:8000
```

### Probar un modelo más grande, sin volver a subir nada

```bash
./scripts/entrenar.sh --proyecto my_project \
    --zip data/datasets/mi_dataset.zip \
    --modelo yolo26x.pt \
    --epochs 100 --imgsz 640 \
    --api http://localhost:8000
```

El dataset no se vuelve a subir. Solo cambia el modelo.

### Retomar un entrenamiento que sigue corriendo

```bash
./scripts/entrenar.sh --job <UUID> --api http://localhost:8000
```

### Entrenar y recoger los resultados más tarde

```bash
./scripts/entrenar.sh --proyecto my_project \
    --zip data/datasets/mi_dataset.zip \
    --epochs 100 --imgsz 640 \
    --sin-descarga \
    --api http://localhost:8000
```

Después:

```bash
curl -o best.pt     "http://localhost:8000/api/trainings/<UUID>/download?file=best.pt"
curl -o report.html "http://localhost:8000/api/trainings/<UUID>/download?file=report.html"
```

### Clases desbalanceadas

```bash
./scripts/entrenar.sh --proyecto my_project \
    --zip data/datasets/mi_dataset.zip \
    --clases "persona,vehiculo" \
    --cls-pw 0.5 \
    --epochs 150 --imgsz 640 \
    --api http://localhost:8000
```

### Segmentación

El dataset tiene que traer **polígonos**, y el proyecto se crea con
`task=segment`:

```bash
./scripts/entrenar.sh --proyecto mascaras \
    --zip data/datasets/mascaras.zip \
    --task segment --modelo yolo26s-seg.pt \
    --epochs 100 --imgsz 640 \
    --api http://localhost:8000
```

Si declaras `segment` y el dataset trae cajas, la API lo rechaza al iniciar con
un error explícito. No se pierde tiempo de GPU.

---

## 11. Problemas frecuentes

### El entrenamiento se queda parado: 0% de CPU y 0% de GPU

**Causa:** `workers` mayor que 0 y `/dev/shm` con los 64 MB por defecto. Los
procesos que cargan datos se bloquean pasando tensores por memoria compartida.

**Solución:** `shm_size: 2gb` en el servicio `worker` del `compose.yaml`, y
recrear el contenedor. Ver [6.3](#63-workers-y-la-memoria-compartida--lee-esto).

**Cómo confirmarlo:**

```bash
docker stats --no-stream training-pipeline-worker-1
# CPU en 0% con varios GB de memoria RESERVADOS = bloqueado, no lento
```

### El script muestra una línea congelada

El monitor **solo reprinta cuando cambia el número de época**. Si el job pasa de
`queued` a `running` con la época todavía en 0, la línea se queda ahí.

**No está colgado.** Mira el log del worker, que sí es fiable:

```bash
docker logs -f training-pipeline-worker-1
```

Ahí verás `epoca N/M mAP50=...` cada vez que avanza.

### `docker compose`: "no configuration file provided"

Estás fuera del directorio del repo. Los comandos `docker compose` lo buscan en
el directorio actual.

```bash
cd /ruta/al/training_pipeline
```

`docker exec` sí funciona desde cualquier sitio.

### La API no responde pero el entrenamiento sigue vivo

Si el monitor dice «Sin respuesta de la API», el script reintenta solo cada
`--reintento` segundos. El job **sigue entrenando en el worker**, no se ha parado
nada. Cuando la API vuelva, el monitor se reengancha solo.

### Las horas no cuadran

`docker logs` va en UTC, `nvidia-smi` en hora local. Compara con `date -u`.

### El job se queda en `queued` y no arranca

Hay un entrenamiento anterior ocupando el worker. Mira la cola:

```bash
curl -s http://localhost:8000/api/trainings | jq -r \
  '.[] | "\(.status)  \(.id)"'
```

Solo hay uno a la vez. Espera o cancela el anterior.

### El entrenamiento falla con un error de CUDA

Te has pasado de memoria de vídeo. Baja `--batch`, o baja `--imgsz`.

```bash
--batch 8
```

### El entrenamiento falla por las etiquetas

Mira el campo `error` del job:

```bash
curl -s http://localhost:8000/api/trainings/<UUID>/status | jq -r .error
```

Los más frecuentes:

- **Formato incorrecto.** Las coordenadas no están normalizadas entre 0 y 1.
- **Orden de clases equivocado.** El número de clase de las etiquetas no
  corresponde al orden de `--clases`.
- **`.txt` que falta.** Toda imagen necesita su `.txt`, aunque esté vacío.
- **Tarea y dataset no coinciden.** Un `-seg.pt` sobre etiquetas de caja.

### El modelo entrena pero da resultados raros

Si mAP50 sale ~0 desde la primera época, el problema casi nunca es el
entrenamiento: es el dataset. Comprueba, en este orden:

1. Que las coordenadas están normalizadas (0-1), no en píxeles.
2. Que el orden de `--clases` corresponde a los números de tus etiquetas.
3. Abre un `.txt` a mano y compáralo con la imagen.
4. Comprueba que las imágenes y las etiquetas se corresponden por nombre.

### `shm_size` no se aplica

`docker compose up -d` no cambia la configuración de un contenedor existente.

```bash
docker compose -f compose.yaml -f compose.gpu.yaml down
docker compose -f compose.yaml -f compose.gpu.yaml up -d
docker exec training-pipeline-worker-1 df -h /dev/shm    # debe decir 2,0G
```

---

## 12. Glosario

| Término | Qué significa |
|---|---|
| **Época** | Una pasada completa por el dataset. Entrenar 100 épocas = ver cada imagen 100 veces. |
| **`imgsz`** | Lado de la imagen en píxeles. Todas las imágenes se reducen o amplían a ese tamaño. |
| **`batch`** | Cuántas imágenes se procesan a la vez en un paso. |
| **`workers`** | Procesos paralelos que cargan y preparan imágenes. |
| **`fraction`** | Fracción del dataset con la que se entrena. Recorta **por orden de fichero**. |
| **`patience`** | Épocas sin mejorar antes de parar. |
| **mAP50** | Métrica principal de detección, con umbral de 50%. |
| **mAP50-95** | Lo mismo con umbrales más estrictos. Mide la calidad de la caja. |
| **Precision** | De lo que detecta, cuánto está bien. |
| **Recall** | De lo que hay, cuánto encuentra. |
| **Loss** | Error del modelo. Baja durante el entrenamiento. Si sube, algo va mal. |
| **`best.pt`** | Pesos de la mejor época. El que se despliega. |
| **`last.pt`** | Pesos de la última época. Para retomar un entrenamiento. |
| **Sobreajuste** | El modelo memoriza los datos de entrenamiento y peora en los que no ha visto. Se ve como mAP50 que sube y luego baja. |
| **Detección vs segmentación** | Detección = dónde está el objeto (una caja). Segmentación = su silueta exacta (un polígono). |
| **Backbone** | La red que extrae las características de la imagen. `n` es el más pequeño, `x` el mayor. |

---

## Apéndice A. La API

Todo lo de esta guía se puede hacer con `scripts/entrenar.sh`. La API está aquí
para los casos que el script no cubre.

### Registrar un dataset que ya está en el disco

**Copiar los ficheros a mano no es suficiente.** La base de datos es la fuente
de verdad del servicio: si un dataset no está registrado, el servicio no sabe
que existe, aunque los ficheros estén en su sitio.

```bash
curl -X POST http://localhost:8000/api/projects/<PROJECT_ID>/datasets/register \
  -H 'Content-Type: application/json' \
  -d '{"name":"person_detection_thermal","rel_path":"person_detection_thermal"}'
```

### Consultar el estado de un job

```bash
curl -s http://localhost:8000/api/trainings/<UUID>/status | jq
```

Devuelve el estado, la época, el progreso, las métricas y el error si lo hay.

### Ver la cola

```bash
curl -s "http://localhost:8000/api/trainings?limit=20" | jq -r \
  '.[] | "\(.status)  \(.id)  \(.model)"'
```

### Descargar un artefacto

Solo se pueden pedir cuatro ficheros. No es arbitrario: evita rutas de fichero
inventadas por el cliente.

```bash
# file ∈ {best.pt, last.pt, results.csv, report.html}
curl -o best.pt "http://localhost:8000/api/trainings/<UUID>/download?file=best.pt"
```

### Callbacks útiles

| Método y ruta | Qué hace |
|---|---|
| `GET /health` | Estado del servicio, la base de datos y el esquema. |
| `GET /api/projects` | Lista los proyectos. |
| `POST /api/projects` | Crea uno. |
| `GET /api/projects/{id}/export` | Descarga el proyecto en YAML. |
| `DELETE /api/projects/{id}` | Borra el proyecto **y su dataset y sus runs**. |
| `GET /api/trainings` | Lista los jobs, con filtro `?project_id=`. |
| `POST /api/trainings/{id}/cancel` | Cancela. **Borra los pesos.** |
| `POST /api/trainings/{id}/restart?mode=` | `scratch` o `resume`. Ver [9.4](#94-recuperar-un-job-interrumpido). |
| `GET /api/trainings/{id}/report` | El informe HTML, generándolo si hace falta. |

### Errores

Los errores de la API vienen con un `detail` explicativo:

```bash
curl -s ... | jq -r '.detail'
```

Sin `jq`, un `{"detail":"..."}` se imprimiría entero y es fácil no leerlo.
