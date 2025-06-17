"""Script to populate the database with document chunks.

This script reads a document, splits it into chunks based on the configuration,
and sends these chunks to the RAG API for indexing and storage.
"""

import os
import sys

import requests

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from app.config import settings
except ImportError as e:
    print(f"Failed to import settings: {e}")
    sys.exit(1)


def get_api_url() -> str:
    """Constructs the base API URL from settings."""
    return f"http://{settings.HOST}:{settings.PORT}{settings.API_ROUTER}/{settings.API_VERSION}"


def get_or_create_default_library() -> str:
    """Retrieves the ID of the 'Research' Library. If it doesn't exist, it creates it."""
    api_url = get_api_url()
    libraries_url = f"{api_url}/libraries"
    library_name = "Research"

    try:
        # Check if the library already exists
        response = requests.get(libraries_url)
        response.raise_for_status()
        libraries = response.json()
        for library in libraries:
            if library["name"] == library_name:
                return library["id"]

        # If not, create it
        payload = {
            "name": library_name,
            "description": "Library containing the most important research paper of all time: my master's thesis :)",
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(libraries_url, json=payload, headers=headers)
        response.raise_for_status()
        new_library = response.json()

        return new_library["id"]

    except requests.exceptions.RequestException as e:
        print(f"Failed to connect to API for library creation: {e}")
        print(f"API URL: {api_url}")
        sys.exit(1)


def read_document(file_path: str) -> str:
    """Reads the content of a text file."""
    try:
        with open(file_path, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Document file not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading document: {e}")
        sys.exit(1)


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Chunks the text into smaller pieces using a sliding window approach."""
    if not isinstance(text, str):
        return []

    if chunk_overlap >= chunk_size:
        msg = "chunk_overlap must be smaller than chunk_size."
        raise ValueError(msg)

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap

    return chunks


def index_document_chunks(
    chunks_content: list[str],
    document_title: str,
    document_content: str,
    library_id: str,
) -> None:
    """Sends the document chunks to the API for indexing."""
    api_url = f"{get_api_url()}/libraries/{library_id}/documents"

    structured_chunks = [
        {"content": content, "sequence_number": i}
        for i, content in enumerate(chunks_content)
    ]

    payload = {
        "title": document_title,
        "content": document_content,
        "chunks": structured_chunks,
        "index": True,
    }

    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(api_url, json=payload, headers=headers)
        response.raise_for_status()
        # print("API Response:", response.json())
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error while indexing document: {e}")
        print(f"Response body: {response.text}")
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"Request error while indexing document: {e}")
        sys.exit(1)


def main() -> None:
    """Main script to read a document, chunk it, and send it to the API."""
    print("Starting populate_db script...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    document_path = os.path.join(base_dir, "data", "curated_master-thesis.txt")
    document_title = os.path.basename(document_path)
    print(f"Document path: {document_path}")

    print("Getting or creating default library...")
    library_id = get_or_create_default_library()
    if not library_id:
        print("Failed to get or create library")
        return
    print(f"Library ID: {library_id}")

    print("Reading and chunking the mock document to index")
    document_content = read_document(document_path)

    chunks = chunk_text(
        text=document_content,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )

    if not chunks:
        print("no chucks to see here!")
        return

    print(f"Indexing {len(chunks)} chunks.")
    index_document_chunks(chunks, document_title, document_content, library_id)


if __name__ == "__main__":
    main()
