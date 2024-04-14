#!/bin/bash

sudo -u postgres psql <<PGSCRIPT

DROP DATABASE holo;
DROP USER holocleanuser;
CREATE DATABASE holo;
CREATE USER holocleanuser;
ALTER USER holocleanuser WITH PASSWORD 'abcd1234';
GRANT ALL PRIVILEGES on DATABASE holo To holocleanuser;
\c holo
ALTER SCHEMA public OWNER TO holocleanuser;


