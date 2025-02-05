-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create database
CREATE DATABASE docprocessor;
\c docprocessor;

-- Create schema
CREATE SCHEMA IF NOT EXISTS app;

-- Set search path
ALTER DATABASE docprocessor SET search_path TO app, public;

-- Create roles and permissions
DO $$
BEGIN
    -- Create application role if it doesn't exist
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'app_user') THEN
        CREATE ROLE app_user WITH LOGIN PASSWORD 'app_password';
    END IF;
    
    -- Grant permissions
    GRANT USAGE ON SCHEMA app TO app_user;
    GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA app TO app_user;
    GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA app TO app_user;
    
    -- Set default privileges for future objects
    ALTER DEFAULT PRIVILEGES IN SCHEMA app
        GRANT ALL PRIVILEGES ON TABLES TO app_user;
    ALTER DEFAULT PRIVILEGES IN SCHEMA app
        GRANT ALL PRIVILEGES ON SEQUENCES TO app_user;
END
$$;

-- Create audit timestamps function
CREATE OR REPLACE FUNCTION app.update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';
