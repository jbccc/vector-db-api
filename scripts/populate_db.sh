#!/bin/bash

set -e

HOST="${HOST:-localhost}"
PORT="${PORT:-8000}"
API_ROUTER="${API_ROUTER:-/api}"
API_VERSION="${API_VERSION:-v1}"
CHUNK_SIZE="${CHUNK_SIZE:-1000}"
CHUNK_OVERLAP="${CHUNK_OVERLAP:-200}"

get_api_url() {
    echo "http://${HOST}:${PORT}${API_ROUTER}/${API_VERSION}"
}

get_or_create_default_library() {
    local api_url=$(get_api_url)
    local libraries_url="${api_url}/libraries"
    local library_name="Research"
    
    echo "Checking for existing library..." >&2
    
    local response=$(curl -s -X GET "$libraries_url")
    if [ $? -ne 0 ]; then
        echo "Failed to connect to API for library retrieval" >&2
        echo "API URL: $api_url" >&2
        exit 1
    fi
    
    if ! echo "$response" | jq . >/dev/null 2>&1; then
        echo "API returned invalid JSON. Is the server running?" >&2
        echo "API URL: $api_url" >&2
        echo "Response: $response" >&2
        echo "Hint: If you see 'Invalid host header', try running with HOST=localhost" >&2
        exit 1
    fi
    
    local existing_id=$(echo "$response" | jq -r ".[] | select(.name == \"$library_name\") | .id")
    
    if [ -n "$existing_id" ] && [ "$existing_id" != "null" ]; then
        echo "$existing_id"
        return 0
    fi
    
    echo "Creating new library..." >&2
    
    local payload=$(jq -n \
        --arg name "$library_name" \
        --arg description "Library containing the most important research paper of all time: my master's thesis :)" \
        '{name: $name, description: $description}')
    
    local create_response=$(curl -s -X POST "$libraries_url" \
        -H "Content-Type: application/json" \
        -d "$payload")
    
    if [ $? -ne 0 ]; then
        echo "Failed to create library" >&2
        exit 1
    fi
    

    if ! echo "$create_response" | jq . >/dev/null 2>&1; then
        echo "Create API returned invalid JSON. Is the server running?" >&2
        echo "API URL: $libraries_url" >&2
        echo "Response: $create_response" >&2
        echo "Hint: If you see 'Invalid host header', try running with HOST=localhost" >&2
        exit 1
    fi
    
    local new_library_id=$(echo "$create_response" | jq -r '.id')
    echo "$new_library_id"
}

read_document() {
    local file_path="$1"
    
    if [ ! -f "$file_path" ]; then
        echo "Document file not found: $file_path" >&2
        exit 1
    fi
    
    cat "$file_path"
}

chunk_text() {
    local text="$1"
    local chunk_size="$2"
    local chunk_overlap="$3"
    
    if [ "$chunk_overlap" -ge "$chunk_size" ]; then
        echo "chunk_overlap must be smaller than chunk_size" >&2
        exit 1
    fi
    
    local text_length=${#text}
    local start=0
    local step=$((chunk_size - chunk_overlap))
    
    while [ $start -lt $text_length ]; do
        local end=$((start + chunk_size))
        if [ $end -gt $text_length ]; then
            end=$text_length
        fi
        
        echo "${text:$start:$((end - start))}"
        echo "---CHUNK_SEPARATOR---"
        
        start=$((start + step))
    done
}

index_document_chunks() {
    local chunks_file="$1"
    local document_title="$2"
    local document_content="$3"
    local library_id="$4"
    
    local api_url=$(get_api_url)
    local documents_url="${api_url}/libraries/${library_id}/documents"
    
    local chunks_json="[]"
    local sequence_number=0
    
    while IFS= read -r line; do
        if [ "$line" = "---CHUNK_SEPARATOR---" ]; then
            continue
        fi
        
        if [ -n "$line" ]; then
            local chunk_obj=$(jq -n \
                --arg content "$line" \
                --argjson seq "$sequence_number" \
                '{content: $content, sequence_number: $seq}')
            
            chunks_json=$(echo "$chunks_json" | jq ". += [$chunk_obj]")
            sequence_number=$((sequence_number + 1))
        fi
    done < "$chunks_file"
    
    local payload=$(jq -n \
        --arg title "$document_title" \
        --arg content "$document_content" \
        --argjson chunks "$chunks_json" \
        --argjson index true \
        '{title: $title, content: $content, chunks: $chunks, index: $index}')
    
    echo "Sending document to API..." >&2
    
    local response=$(curl -s -X POST "$documents_url" \
        -H "Content-Type: application/json" \
        -d "$payload")
    
    if [ $? -ne 0 ]; then
        echo "Request error while indexing document" >&2
        exit 1
    fi
    
    local error_message=$(echo "$response" | jq -r '.detail // empty')
    if [ -n "$error_message" ]; then
        echo "HTTP error while indexing document: $error_message" >&2
        echo "Response: $response" >&2
        exit 1
    fi
    
    echo "Document indexed successfully!" >&2
}

main() {
    echo "Starting populate_db script..."
    
    if ! command -v jq &> /dev/null; then
        echo "Error: jq is required but not installed" >&2
        exit 1
    fi
    
    if ! command -v curl &> /dev/null; then
        echo "Error: curl is required but not installed" >&2
        exit 1
    fi
    
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local base_dir="$(dirname "$script_dir")"
    local document_path="${base_dir}/data/curated_master-thesis.txt"
    local document_title=$(basename "$document_path")
    
    echo "Document path: $document_path"
    
    echo "Getting or creating default library..."
    local library_id=$(get_or_create_default_library)
    if [ -z "$library_id" ] || [ "$library_id" = "null" ]; then
        echo "Failed to get or create library" >&2
        exit 1
    fi
    echo "Library ID: $library_id"
    
    echo "Reading and chunking the document to index"
    local document_content=$(read_document "$document_path")
    
    local chunks_file=$(mktemp)
    trap "rm -f $chunks_file" EXIT
    
    chunk_text "$document_content" "$CHUNK_SIZE" "$CHUNK_OVERLAP" > "$chunks_file"
    
    local chunk_count=$(grep -v "^---CHUNK_SEPARATOR---$" "$chunks_file" | grep -c "^")
    
    if [ "$chunk_count" -eq 0 ]; then
        echo "No chunks to process!"
        return 0
    fi
    
    echo "Indexing $chunk_count chunks."
    index_document_chunks "$chunks_file" "$document_title" "$document_content" "$library_id"
    
    echo "Document processing completed successfully!"
}

if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    main "$@"
fi