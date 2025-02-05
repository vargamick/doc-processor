import sys
import os
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker
import psycopg2

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from backend.models import Base, Document, ProcessingStatus, Embedding
from parser_service.app.processors.factory import ProcessorFactory
from parser_service.app.embeddings.base import EmbeddingProcessor


def check_database_schema():
    """Verify database schema implementation"""
    print("\n=== Checking Database Schema ===")

    try:
        # Connect to database
        engine = create_engine(
            "postgresql://postgres:postgres@localhost:5432/docprocessor"
        )
        inspector = inspect(engine)

        # Check tables
        tables = inspector.get_table_names()
        required_tables = ["documents", "processing_statuses", "embeddings"]

        print("\nVerifying tables...")
        for table in required_tables:
            if table in tables:
                print(f"✓ Table '{table}' exists")
                # Print columns
                columns = inspector.get_columns(table)
                print(f"  Columns:")
                for col in columns:
                    print(f"    - {col['name']}: {col['type']}")
            else:
                print(f"✗ Table '{table}' missing")

        # Check pgvector extension
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT * FROM pg_extension WHERE extname = 'vector'")
            )
            if result.fetchone():
                print("\n✓ pgvector extension installed")
            else:
                print("\n✗ pgvector extension not found")

    except Exception as e:
        print(f"\n✗ Database connection failed: {str(e)}")
        return False

    return True


def check_document_processors():
    """Verify document processor implementation"""
    print("\n=== Checking Document Processors ===")

    # Check supported types
    supported = ProcessorFactory.supported_types()
    print("\nSupported document types:", supported)

    # Verify processor creation and configuration
    for doc_type in supported:
        print(f"\nTesting {doc_type} processor:")
        try:
            processor = ProcessorFactory.get_processor(doc_type)
            if processor:
                print(f"✓ Successfully created {doc_type} processor")

                # Check processor configuration
                print("Checking processor configuration:")
                print(f"  - Chunk size: {processor.chunk_size}")
                print(f"  - Chunk overlap: {processor.chunk_overlap}")

                # Verify required methods
                methods = ["_extract_text", "_extract_metadata", "process"]
                for method in methods:
                    if hasattr(processor, method) and callable(
                        getattr(processor, method)
                    ):
                        print(f"✓ Has required method: {method}")
                    else:
                        print(f"✗ Missing required method: {method}")

                # Check embedding processor configuration if available
                if processor.embedding_processor:
                    print("\nEmbedding processor configuration:")
                    model_info = processor.embedding_processor.get_model_info()
                    for key, value in model_info.items():
                        print(f"  - {key}: {value}")
            else:
                print(f"✗ Failed to create {doc_type} processor")
        except Exception as e:
            print(f"✗ Error testing {doc_type} processor: {str(e)}")


def check_embedding_system():
    """Verify embedding system implementation"""
    print("\n=== Checking Embedding System ===")

    try:
        processor = EmbeddingProcessor()
        model_info = processor.get_model_info()

        print("\nEmbedding Configuration:")
        for key, value in model_info.items():
            print(f"- {key}: {value}")

        # Test embedding generation
        test_text = "This is a test document for embedding generation."
        embedding = processor.generate_embedding(test_text)

        print(f"\n✓ Successfully generated embedding with dimension {len(embedding)}")

    except Exception as e:
        print(f"\n✗ Embedding system check failed: {str(e)}")
        return False

    return True


def main():
    """Run all validation checks"""
    print("Starting implementation validation...\n")

    results = []

    # Check database schema
    results.append(("Database Schema", check_database_schema()))

    # Check document processors
    try:
        check_document_processors()
        results.append(("Document Processors", True))
    except Exception as e:
        print(f"\n✗ Document processor check failed: {str(e)}")
        results.append(("Document Processors", False))

    # Check embedding system
    results.append(("Embedding System", check_embedding_system()))

    # Print summary
    print("\n=== Validation Summary ===")
    for component, success in results:
        status = "✓" if success else "✗"
        print(f"{status} {component}")


if __name__ == "__main__":
    main()
