"""Save population outcomes and controllers for interactive generation replay."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade():
    document = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")
    op.add_column("generations", sa.Column("population", document, nullable=True))


def downgrade():
    op.drop_column("generations", "population")
