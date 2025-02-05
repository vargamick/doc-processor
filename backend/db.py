from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import text

DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/doc_processor"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def verify_db_connection():
    """Verify database connection and basic vector operations."""
    try:
        with SessionLocal() as session:
            # Test basic connection
            result = session.execute(text("SELECT 1")).scalar()
            if result != 1:
                return False, "Database connection test failed"

            # Test vector extension
            result = session.execute(text("SELECT '[1,2,3]'::vector")).scalar()
            if result is None:
                return False, "Vector extension test failed"

            return True, "Database connection and vector operations verified"
    except Exception as e:
        return False, f"Database connection error: {str(e)}"


def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
