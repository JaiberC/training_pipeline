#!/usr/bin/env python3
"""Verificaciones del paquete trainpipe que no necesitan Docker ni Postgres.

Ejecutar:  python3 scripts/verificar.py   (o `make test`, que ademas levanta el stack)

Comprueba tres cosas que ya rompieron en el pasado:
  1. Aislamiento de imports: la imagen runtime-api no instala torch, asi que
     ningun modulo salvo train_cli puede importarlo.
  2. Validaciones de dominio: RT-DETR no segmenta, la tarea debe coincidir con
     el formato de labels, y los nombres de recurso no pueden ser rutas.
  2b. Nombres de clase: el data.yaml del ZIP se reescribe, asi que sus `names`
     se usaban para nada y el defecto 'person' del script etiquetaba como
     persona un modelo de edificios. Ademas nada comprobaba que el proyecto
     declarase suficientes clases.
  3. Informe ApexCharts: formato de series, null en huecos, colores con #,
     CDN fijado en v7, y que los nombres de los artefactos de Ultralytics sean
     los que escribe DE VERDAD (una vez se pidieron PR_curve.png y F1_curve.png
     en vez de BoxPR_curve.png y BoxF1_curve.png, y esas curvas no salieron en
     ningun informe sin que nada se quejara).
  4. Parser de metricas por clase: Ultralytics solo las imprime por consola y
     el worker las parsea. Si el formato cambia, el grafico desaparece; aqui se
     avisa antes.
"""

import ast
import json
import logging
import re
import sys
import tempfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

fallos: list[str] = []


def check(etiqueta: str, condicion: bool, detalle: str = "") -> None:
    print(("ok   " if condicion else "FALLO") + " " + etiqueta
          + (f"  -> {detalle}" if detalle and not condicion else ""))
    if not condicion:
        fallos.append(etiqueta)


# ── 1. Aislamiento de imports ────────────────────────────────────────────
def verificar_imports() -> None:
    print("\n== 1. Aislamiento de imports (la imagen api no lleva torch) ==")
    pesados = {"torch", "ultralytics", "cv2", "numpy"}
    for f in sorted((RAIZ / "src").rglob("*.py")):
        arbol = ast.parse(f.read_text(encoding="utf-8"))
        mods: set[str] = set()
        for n in ast.walk(arbol):
            if isinstance(n, ast.Import):
                mods.update(a.name.split(".")[0] for a in n.names)
            elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
                mods.add(n.module.split(".")[0])
        intersec = pesados & mods
        rel = f.relative_to(RAIZ)
        if rel.name == "train_cli.py":
            check(f"{rel} puede importar torch (es el unico)", True)
        else:
            check(f"{rel} no importa torch/ultralytics", not intersec, str(intersec))


# ── 2. Validaciones de dominio ───────────────────────────────────────────
def verificar_dominio() -> None:
    print("\n== 2. Validaciones de dominio ==")
    from trainpipe import tasks
    from trainpipe.models import Task

    def lanza(fn, *a):
        try:
            fn(*a)
            return None
        except tasks.ValidationError as e:
            return str(e)

    check("RT-DETR + segment -> error",
          lanza(tasks.validate_model, "rtdetr-l.pt", Task.SEGMENT) is not None)
    check("modelo inexistente -> error",
          lanza(tasks.validate_model, "yolo99.pt", Task.DETECT) is not None)
    check("-seg.pt con task=detect -> error",
          lanza(tasks.validate_model, "yolo26s-seg.pt", Task.DETECT) is not None)

    p = tasks.apply_family_defaults("rtdetr", {"deterministic": True})
    check("rtdetr fuerza deterministic=False", p["deterministic"] is False)
    p = tasks.apply_family_defaults("yolo26", {"lr0": 0.001})
    check("lr0 sin optimizer añade AdamW", p.get("optimizer") == "AdamW")

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "labels").mkdir()
        (d / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.3\n")
        (d / "labels" / "b.txt").write_text("0 0.5 0.5 0.2 0.3\n")
        check("labels de 5 columnas -> boxes",
              tasks.detect_dataset_label_format(d) == "boxes")
        (d / "labels" / "b.txt").write_text("0 0.1 0.2 0.3 0.4 0.5 0.6\n")
        check("labels con poligono -> polygons",
              tasks.detect_dataset_label_format(d) == "polygons")

    check("segment + boxes -> error",
          lanza(tasks.validate_task_matches_dataset, Task.SEGMENT, "boxes") is not None)
    check("detect + poligonos -> error",
          lanza(tasks.validate_task_matches_dataset, Task.DETECT, "polygons") is not None)
    tasks.validate_task_matches_dataset(Task.DETECT, "boxes")
    check("detect + boxes -> valido", True)

    for malo in ["../escape", "a/b", "Mayusculas", "x" * 70, "-abc"]:
        check(f"nombre hostil rechazado: {malo[:14]}",
              lanza(tasks.validate_resource_name, malo) is not None)
    check("nocturno_personas aceptado",
          tasks.validate_resource_name("nocturno_personas") == "nocturno_personas")

    # Layouts de dataset
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "images" / "train").mkdir(parents=True)
        check("layout yolo detectado", tasks.detect_layout(d) == tasks.LAYOUT_STANDARD)
        with tempfile.TemporaryDirectory() as td2:
            d2 = Path(td2)
            (d2 / "train" / "images").mkdir(parents=True)
            check("layout roboflow detectado", tasks.detect_layout(d2) == tasks.LAYOUT_ROBOFLOW)

    # Indices de clase: la unica fuente fiable del numero de clases
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "labels" / "train").mkdir(parents=True)
        (d / "labels" / "train" / "a.txt").write_text("0 0.5 0.5 0.2 0.3\n0 0.4 0.4 0.1 0.1\n")
        (d / "labels" / "train" / "b.txt").write_text("2 0.5 0.5 0.2 0.3\n")
        (d / "labels" / "train" / "vacio.txt").write_text("")
        check("indices de clase detectados", tasks.label_class_indices(d) == {0, 2},
              str(tasks.label_class_indices(d)))
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "labels").mkdir()
        (d / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.3\n")
        check("clase unica -> {0}", tasks.label_class_indices(d) == {0})
    with tempfile.TemporaryDirectory() as td:
        check("sin labels -> vacio", tasks.label_class_indices(Path(td)) == set())


# ── 2b. Resolucion de nombres de clase ─────────────────────────────────
def verificar_clases() -> None:
    """El data.yaml del ZIP se reescribe, y antes sus `names` se ignoraban: un
    modelo de edificios salia etiquetado como `person` porque el script traia
    ese defecto. Ademas nada comprobaba que el proyecto declarase suficientes
    clases, asi que un dataset de 3 clases sin --clases entrenaba con nc:1."""
    print("\n== 2b. Nombres de clase ==")
    import yaml
    from fastapi import HTTPException

    from trainpipe import api

    def resuelve(declaradas, indices, archive, nombre="ds"):
        return api._resolve_classes(declaradas, indices, archive, nombre)

    def falla(declaradas, indices, archive, nombre="ds"):
        try:
            api._resolve_classes(declaradas, indices, archive, nombre)
        except HTTPException as exc:
            return exc.status_code, str(exc.detail)
        return None, None

    check("--clases gana si esta completo",
          resuelve(["a", "b", "c"], {0, 1, 2}, ["x"]) == ["a", "b", "c"])

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "data.yaml").write_text("nc: 1\nnames: ['Building']\n")
        check("lee names de lista del data.yaml",
              api._read_archive_class_names(d) == ["Building"],
              str(api._read_archive_class_names(d)))
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "data.yaml").write_text("nc: 2\nnames:\n  0: Edificio\n  1: vehiculo\n")
        check("lee names de diccionario, ordenado por indice",
              api._read_archive_class_names(d) == ["Edificio", "vehiculo"],
              str(api._read_archive_class_names(d)))
    with tempfile.TemporaryDirectory() as td:
        check("sin data.yaml -> vacio", api._read_archive_class_names(Path(td)) == [])
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "data.yaml").write_text("esto: [no, es, un, yaml:\n")
        check("data.yaml corrupto -> vacio, no revienta",
              api._read_archive_class_names(d) == [])

    check("sin --clases usa los names del ZIP",
          resuelve([], {0}, ["Building"]) == ["Building"])
    check("sin --clases y sin YAML genera class_0",
          resuelve([], {0}, []) == ["class_0"])
    check("sin --clases y sin YAML genera class_0..class_2",
          resuelve([], {0, 1, 2}, []) == ["class_0", "class_1", "class_2"])
    check("sin nada de nada, al menos una clase", resuelve([], set(), []) == ["class_0"])
    check("declarar de mas se respeta (inocuo)",
          resuelve(["a", "b"], {0}, []) == ["a", "b"])

    cod, det = falla(["person"], {0, 1, 2}, ["Edificio", "Coche", "Arbol"])
    check("3 clases con 1 declarada -> 422", cod == 422, str(cod))
    check("el error sugiere el --clases correcto",
          cod == 422 and '--clases "Edificio, Coche, Arbol"' in det, str(det))
    check("el error explica el orden",
          cod == 422 and "MISMO ORDEN" in det, str(det))
    check("2 clases con 1 declarada -> 422", falla(["person"], {0, 1}, [])[0] == 422)
    check("clases justas -> sin error", falla(["a", "b"], {0, 1}, [])[0] is None)
    check("YAML corto con mas indices -> 422",
          falla([], {0, 1, 2}, ["solo_una"])[0] == 422)

    # El dataset.yaml que lee Ultralytics es el unico que cuenta.
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "labels" / "train").mkdir(parents=True)
        (d / "labels" / "train" / "a.txt").write_text("0 0.5 0.5 0.2 0.3\n")
        (d / "labels" / "train" / "b.txt").write_text("1 0.5 0.5 0.2 0.3\n")
        api._materialize_dataset_yaml(d, ["Uno", "Dos"])
        yaml = (d / "dataset.yaml").read_text()
        check("el dataset.yaml lleva los nombres dados", "0: Uno" in yaml and "1: Dos" in yaml, yaml)
        check("el dataset.yaml lleva el nc correcto", "nc: 2" in yaml, yaml)
        check("el dataset.yaml lleva la ruta absoluta", "path: /" in yaml, yaml)
        # Re-materializar con otras clases las cambia: es lo que arregla el
        # caso de un proyecto renombrado despues de subir el dataset.
        api._materialize_dataset_yaml(d, ["Cambiada"])
        yaml2 = (d / "dataset.yaml").read_text()
        check("re-materializar cambia los nombres", "0: Cambiada" in yaml2, yaml2)
        check("re-materializar cambia el nc", "nc: 1" in yaml2, yaml2)

    # ZIP con carpeta envoltorio: hay que aplanarlo o el layout no se detecta.
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "zip"
        destino = Path(td) / "destino"
        (d / "ds" / "train" / "images").mkdir(parents=True)
        (d / "ds" / "train" / "images" / "a.jpg").write_bytes(b"x")
        (d / "ds" / "data.yaml").write_text("nc: 1\n")
        destino.mkdir()
        api._flatten_extracted(d / "ds", destino)
        check("ZIP con carpeta envoltorio se aplana",
              (destino / "train" / "images" / "a.jpg").is_file(),
              str(sorted(p.name for p in destino.iterdir())))
        check("el data.yaml queda en la raiz", (destino / "data.yaml").is_file())

    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "zip"
        destino = Path(td) / "destino"
        (d / "train" / "images").mkdir(parents=True)
        (d / "train" / "images" / "a.jpg").write_bytes(b"x")
        (d / "data.yaml").write_text("nc: 1\n")
        destino.mkdir()
        api._flatten_extracted(d, destino)
        check("ZIP con splits en la raiz se queda igual",
              (destino / "train" / "images" / "a.jpg").is_file(),
              str(sorted(p.name for p in destino.iterdir())))


# ── 3. Informe ApexCharts ────────────────────────────────────────────────
def verificar_informe() -> None:
    print("\n== 3. Informe ApexCharts ==")
    from trainpipe import report

    cabecera = ("epoch,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),"
                "metrics/mAP50-95(B),train/box_loss\n")
    filas = []
    for e in range(1, 21):
        caja = "" if e == 10 else f"{0.9 - e * 0.03:.4f}"   # hueco en la epoca 10
        filas.append(f"{e},0.5,0.4,0.3,0.2,{caja}")
    d = Path(tempfile.mkdtemp())
    run = d / "train"
    run.mkdir(parents=True, exist_ok=True)
    (run / "results.csv").write_text(cabecera + "\n".join(filas) + "\n")
    (run / "per_class_metrics.json").write_text(
        json.dumps({"</script><script>alert(1)</script>": 0.5}))

    # Artefactos reales de Ultralytics, con los nombres que escribe DE VERDAD.
    # Este bloque es el que evita que vuelva el bug original: el informe pedia
    # "PR_curve.png" y "F1_curve.png", que no existen (llevan prefijo Box), asi
    # que esas dos curvas nunca salieron en ningun informe.
    for name, _ in report.CURVE_ASSETS:
        (run / name).write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)
    for name, _ in report.DATASET_ASSETS:
        (run / name).write_bytes(b"\xff\xd8\xff\xe0" + b"0" * 32)
    for i in (0, 1, 2):
        (run / f"train_batch{i}.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"0" * 32)
        (run / f"val_batch{i}_labels.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"0" * 32)
        (run / f"val_batch{i}_pred.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"0" * 32)
    (run / report.ARGS_FILE).write_text(
        'model: yolo26n.pt\nname: "</pre></details><script>alert(1)</script>"\nimgsz: 640\n')

    html = report.generate_report(
        run,
        output=d / "report.html",
        meta={
            "proyecto": "nocturno",
            "modelo": "yolo26n.pt",
            "job_id": "1894c92d-cc9a-4b7d-80f4-cd8182621cd3",
            "parametros": {"epochs": 100, "imgsz": 640, "workers": 8},
        },
    ).read_text()

    check("render() presente", ".render()" in html)
    check("sin 'undefined'", "undefined" not in html)
    check("escapa '</' en los datos", "<\\/" in html)
    check("CDN fijado en apexcharts@7", "apexcharts@7" in html and "@latest" not in html)
    check("colores hex con #", not re.search(r"colors:\s*\[\s*['\"][0-9A-Fa-f]{6}", html))

    # --- Nombres de artefactos: el bug que ya rompio una vez ---
    nombres = [n for n, _ in report.CURVE_ASSETS]
    curvas = [n for n in nombres if n.endswith("_curve.png")]
    check("las 4 curvas llevan prefijo Box (BoxPR/BoxF1...)",
          len(curvas) == 4 and all(n.startswith("Box") for n in curvas), str(curvas))
    check("no quedan nombres sin prefijo Box",
          not any(n in ("PR_curve.png", "F1_curve.png") for n in nombres))
    # El nombre malo aparece escrito en el comentario que explica el bug, asi
    # que buscar texto no vale: se miran los literales del AST, no el codigo en
    # crudo.
    literales = {
        n.value
        for n in ast.walk(ast.parse(
            (RAIZ / "src" / "trainpipe" / "report.py").read_text(encoding="utf-8")))
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    }
    malos = literales & {"PR_curve.png", "F1_curve.png"}
    check("ningun literal del codigo usa el nombre sin prefijo Box",
          not malos, str(sorted(malos)))

    # --- Todas las imagenes presentes en el HTML, referenciadas por nombre ---
    srcs = set(re.findall(r'<img src="([^"]+)"', html))
    encontrados = {p.name for p in report.collect_assets(run)}
    # args.yaml viaja en el ZIP pero no es una imagen: se pinta en un <pre>.
    imagenes = encontrados - {report.ARGS_FILE}
    check("collect_assets ve los 18 artefactos", len(encontrados) == 18,
          f"{len(encontrados)} ficheros")
    check("cada imagen esta referenciada en el HTML",
          imagenes.issubset(srcs), str(sorted(imagenes - srcs)))
    check("sin imagenes huerfanas en el HTML",
          not (srcs - encontrados), str(sorted(srcs - encontrados)))
    check("referencias relativas (sin data: ni rutas absolutas)",
          all(not s.startswith(("data:", "/", "http")) for s in srcs), str(sorted(srcs)))

    # --- Emparejado labels/pred ---
    pares = report.val_label_pairs(run)
    check("3 pares labels/pred emparejados", len(pares) == 3, str(len(pares)))

    # --- args.yaml y metadatos ---
    check("args.yaml incluido", "args.yaml" in html)
    check("args.yaml no es un <img>", report.ARGS_FILE not in srcs)
    # Solo se mira el bloque <pre>: el nombre de clase de per_class_metrics
    # tambien lleva "</script><script>alert(1)" a proposito, para comprobar el
    # escape de ApexCharts, y sale como "<\/script>". Mirar el HTML entero daria
    # un falso positivo.
    pre = re.search(r"<pre>(.*?)</pre>", html, re.S)
    check("args.yaml escapado (no inyecta script)",
          pre is not None and "<script>" not in pre.group(1)
          and "&lt;script&gt;" in pre.group(1),
          (pre.group(1)[:120] if pre else "sin <pre>"))
    check("metadatos en la cabecera", 'class="chips"' in html and "nocturno" in html)

    m = re.search(r"const opts = (\[.*?\]);\n", html, re.S)
    check("bloque opts localizable", m is not None)
    if not m:
        return
    try:
        opts = json.loads(m.group(1))
        check("opts es JSON valido", isinstance(opts, list) and len(opts) >= 3)
    except json.JSONDecodeError as e:
        check("opts es JSON valido", False, str(e))
        return

    for c in opts:
        bps = [r["breakpoint"] for r in c.get("responsive", [])]
        if bps:
            check(f"responsive ascendente ({c['chart']['type']})", bps == sorted(bps), str(bps))

    perdidas = next((c for c in opts
                     if any(s["name"] == "box" for s in c.get("series", []))), None)
    check("serie de perdidas presente", perdidas is not None)
    if perdidas:
        serie = next(s for s in perdidas["series"] if s["name"] == "box")
        nulos = [i for i, v in enumerate(serie["data"]) if v is None]
        check("el hueco sale como null y no como 0", nulos == [9], str(nulos))

    todos_ok = all(
        isinstance(s["data"], list)
        and all(v is None or isinstance(v, (int, float)) for v in s["data"])
        for c in opts for s in c.get("series", [])
    )
    check("todas las series en formato [numero|null]", todos_ok)


# ── 4. Parser de metricas por clase ─────────────────────────────────────
def verificar_per_class() -> None:
    """Ultralytics no guarda las metricas por clase en ningun fichero: el worker
    las saca parseando el resumen de validacion. Es lo unico que hace que el
    grafico 'mAP50 por clase' exista, asi que el formato se comprueba aqui."""
    print("\n== 4. Parser de metricas por clase ==")
    import re as _re

    # Se ejecuta el parser sin importar trainpipe.worker entero: ese modulo
    # arrastra pydantic_settings y la comprobacion tiene que correr sin Docker.
    src = (RAIZ / "src" / "trainpipe" / "worker.py").read_text(encoding="utf-8")
    inicio = src.index("_PERCLASS_ROW = re.compile")
    fin = src.index("def _terminate(")
    ns = {"re": _re, "json": json, "Path": Path, "log": logging.getLogger("t")}
    exec(src[inicio:fin], ns)
    escribir = ns["write_per_class_metrics"]

    casos = {
        "una clase": (
            "                 all        142      1200      0.854      0.716      0.782     0.854     1200\n"
            "                person       142      1200      0.854      0.716      0.782     0.854     1200\n"
            "Class Names (all):  person\n",
            {"person": 0.782},
        ),
        "dos clases (lista de python)": (
            "                 all        142      1200      0.854      0.716      0.782     0.854     1200\n"
            "                  car         80       700      0.910      0.790      0.840     0.900      700\n"
            "              car, truck      62       500      0.780      0.610      0.700     0.780      500\n"
            "Class Names (all):  ['car', 'truck']\n",
            {"car": 0.84, "truck": 0.7},
        ),
        "tres clases": (
            "                 all        142      1200      0.854      0.716      0.782     0.854     1200\n"
            "               person       100       900      0.880      0.750      0.810     0.880      900\n"
            "                  car         30       220      0.820      0.690      0.750     0.810      220\n"
            "                perro         12        80      0.610      0.420      0.500     0.580       80\n"
            "Class Names (all):  person, car, perro\n",
            {"person": 0.81, "car": 0.75, "perro": 0.5},
        ),
    }
    for nombre, (texto, esperado) in casos.items():
        d = Path(tempfile.mkdtemp())
        ok = escribir(d, texto.splitlines())
        f = d / "per_class_metrics.json"
        check(f"parsea {nombre}", ok and f.exists() and json.loads(f.read_text()) == esperado,
              json.loads(f.read_text()) if f.exists() else "no escribio")

    d = Path(tempfile.mkdtemp())
    check("formato desconocido: no revienta ni escribe",
          escribir(d, ["nada que ver"]) is False
          and not (d / "per_class_metrics.json").exists())


def main() -> int:
    print("Verificaciones de trainpipe (sin Docker)")
    verificar_imports()
    verificar_dominio()
    verificar_clases()
    verificar_informe()
    verificar_per_class()
    print()
    if fallos:
        print(f"FALLAN {len(fallos)}:")
        for f in fallos:
            print("  -", f)
        return 1
    print("Todas las verificaciones pasan.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())