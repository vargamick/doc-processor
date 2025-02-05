# Technology Stack Documentation

## Backend Technologies

### Core
- Python 3.13
- FastAPI (latest)
- Uvicorn ASGI server
- SQLAlchemy ORM
- Alembic migrations

### Document Processing
- python-docx
- sentence-transformers (all-MiniLM-L6-v2)
- spaCy (en_core_web_sm)
- ChromaDB (latest)

### API Documentation
- OpenAPI (Swagger)
- ReDoc

## Frontend Technologies

### Core
- Node.js 20 LTS
- React 18
- TypeScript 5
- Vite (latest)

### UI/Styling
- TailwindCSS
- HeadlessUI
- React Icons

### State Management
- React Query
- Zustand

## Database Technologies

### Relational Database
- PostgreSQL 16
  - Extensions: pgvector

### Vector Database
- ChromaDB (latest)
  - Embedding storage
  - Similarity search

## Infrastructure

### Containerization
- Docker
- Docker Compose

### Reverse Proxy
- Nginx (latest)

### AI/ML
- Ollama
  - Model: Mixtral
- Claude API Integration

## Development Tools

### IDE
- Visual Studio Code
  - Python extension
  - TypeScript extension
  - Docker extension
  - ESLint
  - Prettier

### Testing
- Python: pytest
- TypeScript: Jest
- E2E: Playwright

### Code Quality
- Black (Python formatter)
- isort (Python import sorter)
- ESLint (TypeScript/JavaScript)
- Prettier (Code formatter)

## Version Control
- Git
- GitHub/GitLab integration

## Monitoring & Logging
- Python logging
- Docker logs
- PostgreSQL logs
- Application metrics (to be determined)

## Security
- Python-jose (JWT)
- Passlib (Password hashing)
- CORS middleware
- Rate limiting
- Input validation
