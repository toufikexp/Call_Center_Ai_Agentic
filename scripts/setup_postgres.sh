#!/usr/bin/env bash
# Idempotent PostgreSQL provisioning for the Call Center pipeline.
#
# Creates the role and database used by ResultsStore. Safe to run as part
# of deployment (CI/CD, Ansible, container init, post-install hook). Does
# nothing destructive — uses CREATE ... IF NOT EXISTS / DO blocks.
#
# Tables (calls, call_results, batch_runs) are NOT created here. They are
# created by the application on first connection via CREATE TABLE IF NOT
# EXISTS inside ResultsStore._initialize_schema(). This script only deals
# with the parts the application cannot do safely under its own privileges.
#
# Usage:
#   PG_SUPERUSER=postgres                     \
#   PG_HOST=localhost                         \
#   PG_PORT=5432                              \
#   APP_DB_NAME=call_center                   \
#   APP_DB_USER=cc_pipeline                   \
#   APP_DB_PASSWORD='change_me'               \
#       ./scripts/setup_postgres.sh
#
# Defaults (override via env): see below.
#
# Authentication: the script invokes psql as $PG_SUPERUSER. On Linux with
# a default Postgres install you typically run:
#   sudo -E -u postgres ./scripts/setup_postgres.sh
# Otherwise set PGPASSWORD or use a .pgpass file for the superuser.

set -euo pipefail

PG_SUPERUSER="${PG_SUPERUSER:-postgres}"
PG_HOST="${PG_HOST:-localhost}"
PG_PORT="${PG_PORT:-5432}"
APP_DB_NAME="${APP_DB_NAME:-call_center}"
APP_DB_USER="${APP_DB_USER:-cc_pipeline}"
APP_DB_PASSWORD="${APP_DB_PASSWORD:-}"

if [[ -z "$APP_DB_PASSWORD" ]]; then
  echo "ERROR: APP_DB_PASSWORD is required (set it via env)." >&2
  exit 1
fi

PSQL=(psql -h "$PG_HOST" -p "$PG_PORT" -U "$PG_SUPERUSER" -v ON_ERROR_STOP=1)

echo "→ Connecting to PostgreSQL at ${PG_HOST}:${PG_PORT} as ${PG_SUPERUSER}"
"${PSQL[@]}" -d postgres -c "SELECT version();" >/dev/null

# 1. Role
echo "→ Ensuring role '${APP_DB_USER}' exists"
"${PSQL[@]}" -d postgres <<SQL
DO \$\$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '${APP_DB_USER}') THEN
    CREATE ROLE ${APP_DB_USER} LOGIN PASSWORD '${APP_DB_PASSWORD}';
    RAISE NOTICE 'Role ${APP_DB_USER} created';
  ELSE
    ALTER ROLE ${APP_DB_USER} WITH LOGIN PASSWORD '${APP_DB_PASSWORD}';
    RAISE NOTICE 'Role ${APP_DB_USER} already existed; password aligned';
  END IF;
END
\$\$;
SQL

# 2. Database (CREATE DATABASE cannot be in a DO block)
echo "→ Ensuring database '${APP_DB_NAME}' exists"
DB_EXISTS=$("${PSQL[@]}" -d postgres -tAc \
  "SELECT 1 FROM pg_database WHERE datname = '${APP_DB_NAME}'")
if [[ "$DB_EXISTS" != "1" ]]; then
  "${PSQL[@]}" -d postgres -c \
    "CREATE DATABASE ${APP_DB_NAME} OWNER ${APP_DB_USER};"
  echo "  database created"
else
  echo "  database already exists; skipping CREATE"
fi

# 3. Privileges (idempotent; safe to repeat)
echo "→ Granting privileges to '${APP_DB_USER}' on '${APP_DB_NAME}'"
"${PSQL[@]}" -d postgres -c \
  "GRANT ALL PRIVILEGES ON DATABASE ${APP_DB_NAME} TO ${APP_DB_USER};"

# Schema-level privileges inside the application database
"${PSQL[@]}" -d "${APP_DB_NAME}" <<SQL
GRANT ALL ON SCHEMA public TO ${APP_DB_USER};
ALTER DEFAULT PRIVILEGES IN SCHEMA public
  GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO ${APP_DB_USER};
ALTER DEFAULT PRIVILEGES IN SCHEMA public
  GRANT USAGE, SELECT ON SEQUENCES TO ${APP_DB_USER};
SQL

cat <<EOM

✅ PostgreSQL ready.

  DATABASE_URL=postgresql://${APP_DB_USER}:<password>@${PG_HOST}:${PG_PORT}/${APP_DB_NAME}

The pipeline will create the tables (calls, call_results, batch_runs) on its
first run when STORAGE_ENABLE=1 is set. Verify with:

  PGPASSWORD='${APP_DB_PASSWORD}' psql -h ${PG_HOST} -p ${PG_PORT} \\
      -U ${APP_DB_USER} -d ${APP_DB_NAME} -c '\dt'
EOM
