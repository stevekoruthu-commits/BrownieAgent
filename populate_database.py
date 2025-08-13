import argparse
import os
import shutil
import multiprocessing
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain.vectorstores.chroma import Chroma


CHROMA_PATH = "chroma"
DATA_PATH = "data"


def main():

    import time
    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Profile document loading
    t0 = time.time()
    documents = load_documents()
    t1 = time.time()
    print(f"⏱️ Loaded documents in {t1-t0:.2f} seconds")

    # Profile splitting
    t2 = time.time()
    chunks = split_documents(documents)
    t3 = time.time()
    print(f"⏱️ Split documents in {t3-t2:.2f} seconds")

    # Profile embedding/indexing
    t4 = time.time()
    add_to_chroma_optimized(chunks)  # Use optimized version
    t5 = time.time()
    print(f"⏱️ Embedded and indexed in {t5-t4:.2f} seconds")


def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


def split_documents(documents: list[Document]):
    # Increase chunk size for fewer embeddings (faster)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # was 400
        chunk_overlap=40,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def process_embedding_batch(chunk_batch):
    """Process a batch of chunks for parallel embedding"""
    embedding_function = get_embedding_function()
    chunk_ids = [chunk.metadata["id"] for chunk in chunk_batch]
    texts = [chunk.page_content for chunk in chunk_batch]
    embeddings = embedding_function.embed_documents(texts)
    
    return {
        'docs': chunk_batch,
        'ids': chunk_ids,
        'embeddings': embeddings
    }


def add_to_chroma_optimized(chunks: list[Document]):
    """Optimized version with parallel processing and larger batches"""
    # Load the existing database
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=None)
    
    # Calculate Page IDs
    chunks_with_ids = calculate_chunk_ids(chunks)
    
    # Check existing documents
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")
    
    # Only add documents that don't exist in the DB
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)
    
    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        
        # Increase batch size for better throughput
        BATCH_SIZE = 256  # Increased from 64
        
        # Use parallel processing for large datasets
        if len(new_chunks) > 500:
            num_cores = min(multiprocessing.cpu_count(), 4)  # Limit cores to prevent overload
            print(f"🚀 Using parallel processing with {num_cores} cores")
            
            with multiprocessing.Pool(num_cores) as pool:
                # Process embeddings in parallel
                batch_results = pool.map(process_embedding_batch, 
                                       [new_chunks[i:i+BATCH_SIZE] 
                                        for i in range(0, len(new_chunks), BATCH_SIZE)])
            
            # Flatten results and add to database
            for batch_result in batch_results:
                db.add_documents(batch_result['docs'], 
                                ids=batch_result['ids'], 
                                embeddings=batch_result['embeddings'])
        else:
            # Use regular batch processing for smaller datasets
            embedding_function = get_embedding_function()
            for i in range(0, len(new_chunks), BATCH_SIZE):
                batch_chunks = new_chunks[i:i+BATCH_SIZE]
                batch_ids = [chunk.metadata["id"] for chunk in batch_chunks]
                batch_texts = [chunk.page_content for chunk in batch_chunks]
                embeddings = embedding_function.embed_documents(batch_texts)
                db.add_documents(batch_chunks, ids=batch_ids, embeddings=embeddings)
        
        db.persist()
    else:
        print("✅ No new documents to add")


def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    embedding_function = get_embedding_function()
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=None
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        texts = [chunk.page_content for chunk in new_chunks]

        # Batch process in smaller groups to avoid memory spikes
        BATCH_SIZE = 64
        for i in range(0, len(new_chunks), BATCH_SIZE):
            batch_chunks = new_chunks[i:i+BATCH_SIZE]
            batch_ids = new_chunk_ids[i:i+BATCH_SIZE]
            batch_texts = texts[i:i+BATCH_SIZE]
            embeddings = embedding_function.embed_documents(batch_texts)
            db.add_documents(batch_chunks, ids=batch_ids, embeddings=embeddings)
        db.persist()
    else:
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()
