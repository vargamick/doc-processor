from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from db import verify_db_connection
import httpx
import json

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Parser service URL (use environment variable in production)
PARSER_SERVICE_URL = "http://0.0.0.0:8001"


@app.get("/health")
async def health_check():
    # Verify database connection
    db_status, db_message = verify_db_connection()

    # Check parser service connection
    parser_status = "disconnected"
    parser_message = "Unable to connect to parser service"

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{PARSER_SERVICE_URL}/health")
            if response.status_code == 200:
                parser_status = "connected"
                parser_message = "Parser service is healthy"
    except httpx.RequestError:
        pass

    # If database is not connected, system is unhealthy
    if not db_status:
        raise HTTPException(status_code=503, detail=db_message)

    # If parser service is not connected, system is unhealthy
    if parser_status == "disconnected":
        raise HTTPException(status_code=503, detail=parser_message)

    return {
        "status": "healthy",
        "database": {"status": "connected", "message": db_message},
        "parser": {"status": parser_status, "message": parser_message},
    }


@app.get("/")
async def root():
    return {"message": "Doc Processor API"}


@app.post("/documents/process")
async def process_document(file: UploadFile = File(...)):
    try:
        # Forward the file to the parser service
        async with httpx.AsyncClient() as client:
            # Create form data with the file
            files = {"file": (file.filename, await file.read(), file.content_type)}

            # Send request to parser service
            response = await client.post(f"{PARSER_SERVICE_URL}/process", files=files)

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code, detail="Parser service error"
                )

            return response.json()

    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503, detail=f"Error communicating with parser service: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
