from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pytesseract
from pdf2image import convert_from_bytes
import os
from app.processors import ProcessorFactory
from app.models.document import ProcessedDocument

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Check service health and dependencies"""
    dependencies_status = {"pytesseract": False, "poppler": False}
    error_messages = []

    # Check pytesseract
    try:
        pytesseract.get_tesseract_version()
        dependencies_status["pytesseract"] = True
    except Exception as e:
        error_messages.append(f"Tesseract error: {str(e)}")

    # Check poppler (by attempting a small PDF conversion)
    try:
        sample_pdf = b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj 2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj 3 0 obj<</Type/Page/MediaBox[0 0 3 3]/Parent 2 0 R/Resources<<>>>>endobj\nxref\n0 4\n0000000000 65535 f\n0000000010 00000 n\n0000000053 00000 n\n0000000102 00000 n\ntrailer<</Size 4/Root 1 0 R>>\nstartxref\n178\n%%EOF\n"
        convert_from_bytes(sample_pdf)
        dependencies_status["poppler"] = True
    except Exception as e:
        error_messages.append(f"Poppler error: {str(e)}")

    # Check if data directory is writable
    try:
        os.makedirs("data", exist_ok=True)
        test_file = os.path.join("data", "test_write.txt")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        storage_status = "writable"
    except Exception as e:
        storage_status = f"error: {str(e)}"
        error_messages.append(f"Storage error: {str(e)}")

    # Determine overall status
    is_healthy = all(dependencies_status.values()) and storage_status == "writable"

    response = {
        "status": "healthy" if is_healthy else "unhealthy",
        "dependencies": dependencies_status,
        "storage": storage_status,
        "supported_formats": ProcessorFactory.supported_types(),
    }

    if error_messages:
        response["errors"] = error_messages

    if not is_healthy:
        raise HTTPException(status_code=503, detail=response)

    return response


@app.post("/process")
async def process_document(
    file: UploadFile = File(...),
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
) -> ProcessedDocument:
    """Process a document and return structured results"""
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        # Get file extension and validate
        file_ext = file.filename.split(".")[-1].lower()
        if file_ext not in ProcessorFactory.supported_types():
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format. Supported formats: {ProcessorFactory.supported_types()}",
            )

        # Get appropriate processor
        processor = ProcessorFactory.get_processor(
            file_ext, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        if not processor:
            raise HTTPException(
                status_code=500, detail="Failed to initialize document processor"
            )

        # Read file content
        content = await file.read()

        # Process document
        result = processor.process(content, file.filename)

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/supported-formats")
async def supported_formats():
    """Get list of supported document formats"""
    return {"formats": ProcessorFactory.supported_types()}
