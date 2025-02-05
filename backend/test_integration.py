import httpx
import asyncio
import sys
from typing import Dict, Any


async def test_health_checks() -> Dict[str, Any]:
    """Test health endpoints of all services."""
    results = {
        "backend": {"status": "unknown", "details": None},
        "parser": {"status": "unknown", "details": None},
        "database": {"status": "unknown", "details": None},
    }

    async with httpx.AsyncClient() as client:
        # Test backend health
        try:
            response = await client.get("http://localhost:8000/health")
            results["backend"]["status"] = (
                "healthy" if response.status_code == 200 else "unhealthy"
            )
            results["backend"]["details"] = response.json()
        except Exception as e:
            results["backend"]["status"] = "error"
            results["backend"]["details"] = str(e)

        # Test parser service health
        try:
            response = await client.get("http://localhost:8001/health")
            results["parser"]["status"] = (
                "healthy" if response.status_code == 200 else "unhealthy"
            )
            results["parser"]["details"] = response.json()
        except Exception as e:
            results["parser"]["status"] = "error"
            results["parser"]["details"] = str(e)

    return results


async def test_document_processing() -> Dict[str, Any]:
    """Test document processing flow between services."""
    results = {"status": "unknown", "details": None}

    # Create a simple test PDF content
    test_pdf = b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj 2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj 3 0 obj<</Type/Page/MediaBox[0 0 3 3]/Parent 2 0 R/Resources<<>>>>endobj\nxref\n0 4\n0000000000 65535 f\n0000000010 00000 n\n0000000053 00000 n\n0000000102 00000 n\ntrailer<</Size 4/Root 1 0 R>>\nstartxref\n178\n%%EOF\n"

    async with httpx.AsyncClient() as client:
        try:
            # Test document processing
            files = {"file": ("test.pdf", test_pdf, "application/pdf")}
            response = await client.post(
                "http://localhost:8000/documents/process", files=files
            )

            if response.status_code == 200:
                results["status"] = "success"
                results["details"] = response.json()
            else:
                results["status"] = "error"
                results["details"] = f"HTTP {response.status_code}: {response.text}"

        except Exception as e:
            results["status"] = "error"
            results["details"] = str(e)

    return results


async def main():
    """Run all integration tests."""
    print("\n=== Running Integration Tests ===\n")

    print("1. Testing Health Checks...")
    health_results = await test_health_checks()
    print("\nHealth Check Results:")
    for service, result in health_results.items():
        print(f"\n{service.upper()}:")
        print(f"Status: {result['status']}")
        print(f"Details: {result['details']}")

    print("\n2. Testing Document Processing...")
    processing_results = await test_document_processing()
    print("\nDocument Processing Results:")
    print(f"Status: {processing_results['status']}")
    print(f"Details: {processing_results['details']}")

    # Determine overall test status
    all_healthy = (
        health_results["backend"]["status"] == "healthy"
        and health_results["parser"]["status"] == "healthy"
        and processing_results["status"] == "success"
    )

    print("\n=== Integration Test Summary ===")
    print(f"Overall Status: {'SUCCESS' if all_healthy else 'FAILURE'}")

    # Exit with appropriate status code
    sys.exit(0 if all_healthy else 1)


if __name__ == "__main__":
    asyncio.run(main())
