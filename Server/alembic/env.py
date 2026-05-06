import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from logging.config import fileConfig
from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context
from core.config import db_config

# config của alembic
config = context.config

# Đọc log config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Ghi đè sqlalchemy.url bằng URL động từ cấu hình môi trường (.env)
db_url = f"mysql+mysqlconnector://{db_config.user}:{db_config.password}@{db_config.host}:{db_config.port}/{db_config.database}"
config.set_main_option("sqlalchemy.url", db_url)

target_metadata = None

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )
        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
