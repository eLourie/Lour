-- =============================================================================
--  PostgreSQL initialisation script
--  Runs once when the container is first created.
--
--  Creates:
--    agent            — main application database
--    agent_checkpoint — LangGraph PostgresSaver checkpointing database
-- =============================================================================

-- The POSTGRES_DB env var already created one DB (agent by default).
-- We only need to create the checkpoint DB here.

SELECT 'CREATE DATABASE agent_checkpoint OWNER agent'
  WHERE NOT EXISTS (
    SELECT FROM pg_database WHERE datname = 'agent_checkpoint'
  )\gexec

-- Enable pg extensions in main DB
\c agent

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Enable extensions in checkpoint DB
\c agent_checkpoint

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";