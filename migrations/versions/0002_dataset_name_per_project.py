"""datasets.name unico por proyecto, no global

Dos proyectos distintos pueden llamar "nocturno_v1" a su dataset.
Antes `datasets_name_key` lo impedia en toda la tabla.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-10
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_constraint("datasets_name_key", "datasets", type_="unique")
    op.create_unique_constraint(
        "uq_datasets_project_name", "datasets", ["project_id", "name"]
    )


def downgrade() -> None:
    op.drop_constraint("uq_datasets_project_name", "datasets", type_="unique")
    op.create_unique_constraint("datasets_name_key", "datasets", ["name"])