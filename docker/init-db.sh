#!/bin/bash
set -e

# Create secondary database for MLflow tracking if it doesn't already exist
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    SELECT 'CREATE DATABASE mlflow_db'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mlflow_db')\gexec
    GRANT ALL PRIVILEGES ON DATABASE mlflow_db TO $POSTGRES_USER;
EOSQL
