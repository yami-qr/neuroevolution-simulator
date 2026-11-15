"""Migration connections use DATABASE_URL, never embedded credentials."""

from alembic import context

from neuroevolution.server import models  # noqa: F401
from neuroevolution.server.database import Base, Database, database_url

config = context.config


def migrate(connection):
    context.configure(
        connection=connection, target_metadata=Base.metadata, compare_type=True
    )
    with context.begin_transaction():
        context.run_migrations()


if context.is_offline_mode():
    context.configure(
        url=database_url(),
        target_metadata=Base.metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()
elif config.attributes.get("connection") is not None:
    migrate(config.attributes["connection"])
else:
    database = Database()
    try:
        with database.engine.connect() as connection:
            migrate(connection)
    finally:
        database.close()
