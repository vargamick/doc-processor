# Codebase Summary

## Database Structure
- PostgreSQL with pgvector extension for vector operations
- SQLAlchemy ORM with Alembic migrations
- Three main tables:
  - documents: Stores document metadata and content
  - processing_statuses: Tracks document processing state
  - embeddings: Stores vector embeddings with pgvector

## Core Components

## Current Project Structure

```
doc-processor/
├── backend/                 # FastAPI Backend Service
│   ├── app.py              # Main application entry
│   └── requirements.txt    # Python dependencies
│
├── frontend/               # React Frontend
│   ├── index.html         # Entry HTML
│   ├── package.json       # Node.js dependencies
│   ├── vite.config.js     # Build configuration
│   └── src/               # Source code
│       ├── App.css        # Main styles
│       ├── App.jsx        # Root component
│       └── main.jsx       # Application entry
│
├── parser-service/         # Document Parser Service
│   ├── app.py             # Service entry point
│   ├── requirements.txt   # Python dependencies
│   └── data/              # Test data directory
│
├── nginx/                 # Nginx Configuration
│   └── nginx.conf         # Server configuration
│
└── docker/                # Docker Configuration
    ├── docker-compose.yml # Service orchestration
    ├── backend/           # Backend container
    ├── frontend/          # Frontend container
    ├── parser/            # Parser container
    ├── nginx/             # Nginx container
    ├── postgres/          # Database container
    └── chromadb/          # Vector DB container
```

## Planned Project Structure
(To be implemented in subsequent phases)

```
doc-processor/
├── backend/                 # FastAPI Backend Service
│   ├── app/
│   │   ├── api/            # API endpoints
│   │   ├── core/           # Core functionality
│   │   ├── models/         # Database models
│   │   └── services/       # Business logic
│   └── tests/              # Backend tests
│
├── frontend/               # React Frontend
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── hooks/         # Custom hooks
│   │   ├── services/      # API services
│   │   └── types/         # TypeScript types
│   └── tests/             # Frontend tests
│
├── parser-service/         # Document Parser Service
│   ├── app/
│   │   ├── processors/    # Document processors
│   │   ├── extractors/    # Data extractors
│   │   └── models/        # Data models
│   └── tests/             # Parser tests
```

## Current Architecture Status

### Services

1. Backend Service (FastAPI)
   - Basic application structure ✓
   - Health check endpoint ✓
   - Database connection setup ✓
   - Configuration management ✓
   - Parser service integration ✓

2. Frontend Service (React)
   - Project initialization ✓
   - Development environment ✓
   - Basic routing setup ✓
   - API client setup ✓

3. Parser Service
   - Basic service structure ✓
   - File handling setup ✓
   - Processing pipeline (planned)
   - Integration points implemented ✓
   - Health monitoring ✓

4. Database Services
   - PostgreSQL container setup ✓
   - ChromaDB container setup ✓
   - Schema design ✓
   - Connection testing ✓
   - Health checks ✓

## Planned Architecture
(To be implemented in subsequent phases)

### Planned Components

1. Document Processing Pipeline (Phase 2)
   - File upload handling
   - Text extraction
   - Metadata generation
   - Vector embedding creation

2. Search System (Phase 3)
   - Full-text search
   - Vector similarity search
   - Hybrid search capabilities
   - Results ranking

3. Task Management (Phase 4)
   - Task extraction
   - Task tracking
   - Status updates
   - Notifications

4. Chat Interface (Phase 4)
   - Context management
   - Query processing
   - Response generation
   - History tracking

## Current Implementation Status

### Data Flow
Core infrastructure communication paths implemented and verified:
- Frontend to Backend API communication
- Backend to Parser service integration
- Backend to Database connectivity
- Health check system across all services

### Planned Data Flows
(To be implemented in subsequent phases)

1. Document Upload Flow
   ```
   Client -> Frontend -> Backend -> Parser Service
                                -> PostgreSQL (metadata)
                                -> ChromaDB (vectors)
   ```

2. Search Request Flow
   ```
   Client -> Frontend -> Backend -> PostgreSQL/ChromaDB
                               -> Response formatting
                     -> Results display
   ```

3. Chat Interaction Flow
   ```
   Client -> Frontend -> Backend -> Context retrieval
                               -> Claude API
                               -> Response processing
                     -> Chat display
   ```

## Security Architecture
(Planned for implementation)

1. Authentication (Phase 5)
   - JWT-based auth
   - Secure token storage
   - Role-based access

2. Data Protection (Phase 5)
   - Input validation
   - SQL injection prevention
   - XSS protection
   - CORS configuration

## Current Monitoring Status

1. Basic Health Checks (Completed)
   - Service availability endpoints ✓
   - Database connectivity checks ✓
   - Container health monitoring ✓
   - Inter-service communication checks ✓
   - Resource utilization monitoring ✓

2. Planned Monitoring (Phase 5)
   - Request tracking
   - Performance monitoring
   - Error logging
   - Resource usage

## Development Workflow

1. Current Setup
   - Docker Compose environment ✓
   - Basic development workflow defined ✓
   - Local environment configuration ✓
   - Debug setup ✓
   - Container health checks ✓
   - Resource limits configured ✓
   - Service discovery implemented ✓

2. Planned Workflow Enhancements
   - Hot reloading configuration
   - Test automation setup
   - CI/CD integration
   - Code review process
