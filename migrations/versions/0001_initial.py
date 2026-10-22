"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-10-10
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

task_enum = postgresql.ENUM("detect", "segment", name="task_enum", create_type=False)
job_status_enum = postgresql.ENUM(
    "draft", "queued", "running", "succeeded", "failed", "cancelled", "interrupted",
    name="job_status_enum", create_type=False,
)


def upgrade() -> None:
    bind = op.get_bind()
    task_enum.create(bind, checkfirst=True)
    job_status_enum.create(bind, checkfirst=True)

    op.create_table(
        "projects",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(128), nullable=False, unique=True),
        sa.Column("task", task_enum, nullable=False),
        sa.Column("model", sa.String(64), nullable=False),
        sa.Column("classes", sa.JSON(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_projects_name", "projects", ["name"])

    op.create_table(
        "datasets",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "project_id", postgresql.UUID(as_uuid=True),
            sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False,
        ),
        sa.Column("name", sa.String(128), nullable=False, unique=True),
        sa.Column("rel_path", sa.String(512), nullable=False),
        sa.Column("num_images", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("label_format", sa.String(16), nullable=False, server_default="unknown"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_datasets_project_id", "datasets", ["project_id"])
    op.create_index("ix_datasets_name", "datasets", ["name"])

    op.create_table(
        "training_jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "project_id", postgresql.UUID(as_uuid=True),
            sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False,
        ),
        sa.Column(
            "dataset_id", postgresql.UUID(as_uuid=True),
            sa.ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False,
        ),
        sa.Column("model", sa.String(64), nullable=False),
        sa.Column("params", sa.JSON(), nullable=False),
        sa.Column("status", job_status_enum, nullable=False, server_default="draft"),
        sa.Column("cancel_requested", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("run_dir", sa.String(512), nullable=True),
        sa.Column("pid", sa.Integer(), nullable=True),
        sa.Column("epoch", sa.Integer(), nullable=True),
        sa.Column("total_epochs", sa.Integer(), nullable=True),
        sa.Column("metrics", sa.JSON(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_training_jobs_project_id", "training_jobs", ["project_id"])
    op.create_index("ix_training_jobs_dataset_id", "training_jobs", ["dataset_id"])
    op.create_index("ix_training_jobs_status", "training_jobs", ["status"])


def downgrade() -> None:
    op.drop_table("training_jobs")
    op.drop_table("datasets")
    op.drop_table("projects")
    job_status_enum.drop(op.get_bind(), checkfirst=True)
    task_enum.drop(op.get_bind(), checkfirst=True)