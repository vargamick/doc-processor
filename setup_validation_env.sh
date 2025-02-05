#!/bin/bash

# Create symbolic links for parser service
mkdir -p parser_service/app/processors
mkdir -p parser_service/app/embeddings
mkdir -p parser_service/app/models
mkdir -p parser_service/app/utils

# Create symbolic links
ln -sf "$(pwd)/parser-service/app/processors/factory.py" parser_service/app/processors/factory.py
ln -sf "$(pwd)/parser-service/app/processors/base.py" parser_service/app/processors/base.py
ln -sf "$(pwd)/parser-service/app/processors/pdf_processor.py" parser_service/app/processors/pdf_processor.py
ln -sf "$(pwd)/parser-service/app/processors/docx_processor.py" parser_service/app/processors/docx_processor.py
ln -sf "$(pwd)/parser-service/app/embeddings/base.py" parser_service/app/embeddings/base.py
ln -sf "$(pwd)/parser-service/app/models/document.py" parser_service/app/models/document.py
ln -sf "$(pwd)/parser-service/app/utils/text_processing.py" parser_service/app/utils/text_processing.py

# Create __init__.py files
touch parser_service/__init__.py
touch parser_service/app/__init__.py
touch parser_service/app/processors/__init__.py
touch parser_service/app/embeddings/__init__.py
touch parser_service/app/models/__init__.py
touch parser_service/app/utils/__init__.py

echo "Validation environment setup complete"
