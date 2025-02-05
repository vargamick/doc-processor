FROM postgres:16

# Copy initialization scripts
COPY docker/postgres/init.sql /docker-entrypoint-initdb.d/

# Set default encoding and locale
ENV LANG en_US.utf8
