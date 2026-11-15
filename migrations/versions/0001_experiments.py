"""Initial experiment, run queue, and generation result schema."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    document = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")
    timestamp = sa.DateTime(timezone=True)
    op.create_table(
        "experiments",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("settings", document, nullable=False),
        sa.Column("created_at", timestamp, nullable=False),
    )
    op.create_index("ix_experiments_created_at", "experiments", ["created_at"])
    op.create_table(
        "runs",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "experiment_id",
            sa.Uuid(),
            sa.ForeignKey("experiments.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("settings", document, nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("requested_generations", sa.Integer(), nullable=False),
        sa.Column("completed_generations", sa.Integer(), nullable=False),
        sa.Column("cancel_requested", sa.Boolean(), nullable=False),
        sa.Column("created_at", timestamp, nullable=False),
        sa.Column("updated_at", timestamp, nullable=False),
        sa.Column("started_at", timestamp),
        sa.Column("finished_at", timestamp),
        sa.Column("lease_token", sa.Uuid()),
        sa.Column("lease_expires_at", timestamp),
        sa.Column("environment", document),
        sa.Column("error", sa.Text()),
        sa.Column("champion", document),
        sa.Column("champion_fitness", sa.Float()),
        sa.Column("champion_generation", sa.Integer()),
        sa.Column("champion_replay", document),
        sa.CheckConstraint(
            "status IN ('queued','running','completed','failed','cancelled')",
            name="ck_runs_status",
        ),
        sa.CheckConstraint("requested_generations > 0", name="ck_runs_requested"),
        sa.CheckConstraint(
            "completed_generations >= 0 "
            "AND completed_generations <= requested_generations",
            name="ck_runs_progress",
        ),
    )
    op.create_index("ix_runs_experiment_id", "runs", ["experiment_id"])
    op.create_index("ix_runs_queue", "runs", ["status", "created_at"])
    op.create_index("ix_runs_expiry", "runs", ["status", "lease_expires_at"])
    op.create_table(
        "generations",
        sa.Column(
            "run_id",
            sa.Uuid(),
            sa.ForeignKey("runs.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("generation", sa.Integer(), primary_key=True),
        *(
            sa.Column(name, sa.Integer(), nullable=False)
            for name in ("population_size", "completed_count", "best_checkpoints")
        ),
        *(
            sa.Column(name, sa.Float(), nullable=False)
            for name in ("best_fitness", "mean_fitness", "best_progress")
        ),
        *(
            sa.Column(name, document, nullable=False)
            for name in ("champion", "fitness", "episode")
        ),
        sa.Column("created_at", timestamp, nullable=False),
    )


def downgrade():
    op.drop_table("generations")
    op.drop_table("runs")
    op.drop_table("experiments")
