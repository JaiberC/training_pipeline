# Makefile - construccion, versionado y operacion del stack.
#
# `make` por defecto pide el tipo de cambio (major/minor/patch), sube la version
# en VERSION, construye las dos imagenes con esa version y con `latest`, y sube
# a Docker Hub si REGISTRY no es `local`.
#
#   make                          #REGISTRY=local, no sube nada
#   make REGISTRY=<tu-usuario>   #construye y sube a Docker Hub
#   make push REGISTRY=<tu-us>    #solo subir, sin tocar la version
#
# Se usa `docker build` (classico), no buildx: construye solo para la
# arquitectura del host (linux/amd64). Es suficiente para el servidor Tesla.

SHELL := /bin/bash

# ── Configuracion (sobreescribible en la linea de comandos) ──────────────
REGISTRY ?= local
TAG      ?= latest

VERSION  := $(shell cat VERSION 2>/dev/null || echo 0.0.0)

API_IMG    := $(REGISTRY)/tp-api
WORKER_IMG := $(REGISTRY)/tp-worker

TARGET_API    := runtime-api
TARGET_WORKER := runtime-worker

# `docker build --push` falla el build entero si el registro no existe o no hay
# credenciales. Como `local` no es un registro, solo se usa --push cuando el
# usuario pasa un REGISTRY real.
ifeq ($(REGISTRY),local)
  PUSH_FLAG :=
  PUSH_NOTE := (REGISTRY=local: no se sube a Docker Hub)
else
  PUSH_FLAG := --push
  PUSH_NOTE := (sube a $(REGISTRY))
endif

COMPOSE   := docker compose
COMPOSE_GPU := $(COMPOSE) -f compose.yaml -f compose.gpu.yaml

.DEFAULT_GOAL := release

# ── Ayuda ────────────────────────────────────────────────────────────────
.PHONY: help
help:
	@echo "Objetivos disponibles:"
	@echo "  make                      version + build (+ push si REGISTRY != local)"
	@echo "  make build                build sin tocar la version"
	@echo "  make push                 build + push, sin tocar la version"
	@echo "  make version              solo preguntar y subir la version"
	@echo "  make up / down / logs     operar el stack (compose)"
	@echo "  make gpu                  up con soporte de GPU (Tesla)"
	@echo "  make test                 suites de verificacion"
	@echo "  make clean                borra imagenes locales y volumenes"
	@echo
	@echo "Variables: REGISTRY=$(REGISTRY)  TAG=$(TAG)  VERSION=$(VERSION)"
	@echo "Ejemplo:   make REGISTRY=miduvusuario"

# ── Version ──────────────────────────────────────────────────────────────
.PHONY: version
version:
	@set -e; \
	if [ -n "$(VER)" ]; then \
		next="$(VER)"; \
	else \
		cur=$$(cat VERSION 2>/dev/null || echo 0.0.0); \
		printf 'Version actual: %s\n' "$$cur"; \
		printf 'Tipo de cambio (major/minor/patch): '; \
		if [ -r /dev/tty ]; then \
			{ read -r tipo < /dev/tty; } 2>/dev/null || tipo=""; \
		fi; \
		if [ -z "$$tipo" ]; then read -r tipo || tipo=""; fi; \
		tipo=$$(printf '%s' "$$tipo" | tr '[:upper:]' '[:lower:]'); \
		IFS=. read -r MA MI PA <<< "$$cur"; \
		MA=$${MA:-0}; MI=$${MI:-0}; PA=$${PA:-0}; \
		case "$$tipo" in \
			major) next="$$((MA+1)).0.0" ;; \
			minor) next="$$MA.$$((MI+1)).0" ;; \
			patch) next="$$MA.$$MI.$$((PA+1))" ;; \
			*) echo "Tipo no valido: '$$tipo'. Usa major, minor o patch." >&2; exit 1 ;; \
		esac; \
	fi; \
	echo "$$next" > VERSION; \
	printf 'Version %s -> %s\n' "$$cur" "$$next"

# ── Construccion ─────────────────────────────────────────────────────────
# Cada imagen necesita su propia invocacion: `docker build` solo construye un
# --target por vez.
#
# OJO: la version se lee DENTRO de la receta, no con $(VERSION). Make evalua las
# variables al arrancar, asi que con `make release` (que primero sube la version
# y despues construye) $(VERSION) seria el valor ANTIGUO y las imagenes
# quedarian etiquetadas con la version anterior.
.PHONY: build
build:
	@set -e; \
	ver=$$(cat VERSION 2>/dev/null || echo 0.0.0); \
	echo "==> $(PUSH_NOTE)"; \
	echo "==> Construyendo $(TARGET_API) [$$ver]"; \
	docker build $(PUSH_FLAG) --build-arg VERSION=$$ver --target $(TARGET_API) \
		-t $(API_IMG):$$ver -t $(API_IMG):$(TAG) .; \
	echo "==> Construyendo $(TARGET_WORKER) [$$ver]"; \
	docker build $(PUSH_FLAG) --build-arg VERSION=$$ver --target $(TARGET_WORKER) \
		-t $(WORKER_IMG):$$ver -t $(WORKER_IMG):$(TAG) .; \
	echo "==> Listo: $(API_IMG):$$ver  $(WORKER_IMG):$$ver  (+ :$(TAG))"

# ── Publicacion ──────────────────────────────────────────────────────────
.PHONY: push
push:
	@if [ "$(REGISTRY)" = "local" ]; then \
		echo "REGISTRY=local no es un registro de Docker Hub."; \
		echo "Uso:  make push REGISTRY=<tu-usuario-dockerhub>"; \
		exit 1; \
	fi
	$(MAKE) build REGISTRY=$(REGISTRY) TAG=$(TAG)

# ── release: version + build ─────────────────────────────────────────────
.PHONY: release
release: version build

# ── Operacion del stack ──────────────────────────────────────────────────
.PHONY: up
up:
	$(COMPOSE) up -d --build
	@$(MAKE) --no-print-directory _espera

.PHONY: down
down:
	$(COMPOSE) down

# Borra tambien los volumenes: DESTRUCTIVO, pierde datasets, runs y la base.
.PHONY: clean-vol
clean-vol:
	$(COMPOSE) down -v

.PHONY: logs
logs:
	$(COMPOSE) logs -f --tail=100

.PHONY: logs-worker
logs-worker:
	$(COMPOSE) logs -f --tail=100 worker

.PHONY: restart
restart:
	$(COMPOSE) restart

.PHONY: gpu
gpu:
	$(COMPOSE_GPU) up -d
	@$(MAKE) --no-print-directory _espera

# El worker no arranca hasta que la API este sana, asi que los tres servicios en
# `healthy` significan que el esquema ya esta creado y el bucle de entrenamiento
# tiene acceso a la base.
.PHONY: _espera
_espera:
	@echo "==> Esperando a que los servicios esten sanos..."
	@total=$$($(COMPOSE) config --services | wc -l); \
	for i in $$(seq 1 60); do \
		s=$$($(COMPOSE) ps --format '{{.Service}}:{{.Health}}' 2>/dev/null | grep -c ':healthy' || true); \
		if [ "$$s" -ge "$$total" ]; then \
			echo "    api, db y worker listos ($${i}0s aprox)"; exit 0; \
		fi; \
		sleep 2; \
	done; \
	echo "    aviso: $$s de $$total servicios sanos tras 120s. Revisa: make logs" >&2

.PHONY: ps
ps:
	$(COMPOSE) ps

.PHONY: health
health:
	@$(COMPOSE) exec -T api python -c \
		"import urllib.request;print(urllib.request.urlopen('http://localhost:8000/health').read().decode())" \
		2>/dev/null || echo "la API no responde"

# ── Migraciones ──────────────────────────────────────────────────────────
# El entrypoint las aplica solo al arrancar; esto es para ejecutarlas a mano.
.PHONY: migrate
migrate:
	$(COMPOSE) run --rm api alembic upgrade head

.PHONY: dump
dump:
	$(COMPOSE) exec -T db pg_dump -U $${DBUSER:-trainpipe} $${DBNAME:-trainpipe} \
		> "backup_$$(date +%Y%m%d-%H%M%S).sql"
	@echo "Backup creado en $$(ls -t backup_*.sql | head -1)"

# ── Verificacion ─────────────────────────────────────────────────────────
# Las verificaciones corren DENTRO del contenedor de la API, que ya tiene las
# dependencias: montar el codigo actual evita depender de un venv en el host.
.PHONY: test
test:
	@echo "==> Verificaciones del paquete (sin Docker ni Postgres)"
	$(COMPOSE) run --rm --no-deps \
		-v $(CURDIR)/src:/app/src \
		-v $(CURDIR)/scripts:/app/scripts \
		--entrypoint python api scripts/verificar.py

# ── Limpieza ────────────────────────────────────────────────────────────
.PHONY: clean
clean:
	docker rmi -f $(API_IMG):$(VERSION) $(API_IMG):$(TAG) \
	               $(WORKER_IMG):$(VERSION) $(WORKER_IMG):$(TAG) 2>/dev/null || true
	@echo "Imagenes locales eliminadas (los volumenes siguen intactos: usa clean-vol)"

.PHONY: help-clean
help-clean:
	@echo "recuerdos: clean (imagenes) / clean-vol (borra datos: DESTRUCTIVO)"