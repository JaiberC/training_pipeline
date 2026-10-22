"""Configuracion leida del entorno.

No importar `torch` ni `ultralytics` aqui: este modulo lo carga la imagen
runtime-api, que no los instala. Ver la restriccion de imports en AGENTS.md.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://trainpipe:trainpipe@db:5432/trainpipe"

    data_root: Path = Path("/data")
    datasets_root: Path = Path("/data/datasets")
    runs_root: Path = Path("/data/runs")
    models_root: Path = Path("/data/models")

    # Segundos entre sondeos del worker sobre la cola.
    worker_poll_interval: float = 2.0
    # Margen entre SIGTERM y SIGKILL al cancelar un entrenamiento.
    cancel_grace_seconds: float = 10.0

    def run_dir(self, job_id: str) -> Path:
        return self.runs_root / job_id

    def dataset_dir(self, project_name: str) -> Path:
        return self.datasets_root / project_name


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings