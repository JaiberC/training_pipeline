#!/bin/sh
# Entry point del contenedor de la API.
#
# Aplica las migraciones de Alembic y luego arranca el servidor. Esto evita el
# arranque manual `alembic upgrade head`, que era un paso que se olvidaba y
# dejaba al worker reventando con 'relation training_jobs does not exist'.
#
# `exec` es importante: uvicorn sustituye a este shell y asi recibe el SIGTERM
# directamente. Sin exec, el shell se queda como PID principal, el contenedor
# ignora las senales y `docker compose down` acaba con SIGKILL tras la espera.
set -e

echo "[entrypoint] aplicando migraciones..."
if ! alembic upgrade head; then
    echo "[entrypoint] ERROR: las migraciones fallaron. Revisa la conexion con la base de datos." >&2
    exit 1
fi
echo "[entrypoint] esquema listo"

exec "$@"