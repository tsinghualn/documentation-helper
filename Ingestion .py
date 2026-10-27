import asyncio
import hashlib
import os
import ssl
from typing import Any, Dict, List
import certifi
import httpx
from dotenv import load_dotenv

load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from langsmith import traceable
from logger import (Colors, log_error, log_header, log_info, log_success, log_warning)
# TavilyExtract: takes URLs and returns clean, structured content without ads, navigations or other noise. 
# TavilyMap: Website structure discovery, automatically discovers and maps website structures by crawling through links.
# TavilyCrawl: takes a URL and returns a list of URLs to follow and raw content

# Configure SSL context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = PineconeEmbeddings(
    model="llama-text-embed-v2",
    show_progress_bar=True,
    batch_size=50,
    document_params={"input_type": "passage", "truncate": "END"},
    query_params={"input_type": "query", "truncate": "END"},
)
# vector store
# chroma = Chroma(
#     embedding_function=embeddings,
#     persist_directory="chroma_db",
# )
INDEX_NAME = "langchain-doc-index"
NAMESPACE = "default"
vector_store = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embeddings,
    text_key="text",
    namespace=NAMESPACE,
)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(
    max_depth=1, 
    max_breadth=20, 
    max_pages=1000) # map depth up to 5 levels, follow up to 20 links per page, max total pages to discover, up to 1000 pages, and instructions to follow
tavily_crawl = TavilyCrawl()

def chunk_urls(urls: List[str], chunk_size: int = 20) -> List[List[str]]:
    """Split URLs into chunks of specified size."""
    chunks = []
    for i in range(0, len(urls), chunk_size):
        chunk = urls[i : i + chunk_size]
        chunks.append(chunk)
    return chunks


async def extract_batch(urls: List[str], batch_num: int) -> List[Dict[str, Any]]:
    """Extract documents from a batch of URLs."""
    try:
        log_info(
            f"🔄 TavilyExtract: Processing batch {batch_num} with {len(urls)} URLs.",
            Colors.BLUE,
        )
        docs = await tavily_extract.ainvoke(
            input={"urls": urls, "extract_depth": "advanced"}
        )
        extracted_docs_count = len(docs.get("results", []))
        if extracted_docs_count > 0:
            log_success(
                f"TavilyExtract: Completed batch {batch_num} - extracted {extracted_docs_count} documents"
            )
        else:
            log_error(
                f"TavilyExtract: Batch {batch_num} failed to extract any documents, {docs}"
            )
        return docs
    except Exception as e:
        log_error(f"TavilyExtract: Failed to extract batch {batch_num} - {e}")
        return []


async def async_extract(url_batches: List[List[str]]):
    log_header("DOCUMENT EXTRACTION PHASE")
    log_info(
        f"🔧 TavilyExtract: Starting concurrent extraction of {len(url_batches)} batches",
        Colors.DARKCYAN,
    )

    tasks = [extract_batch(batch, i + 1) for i, batch in enumerate(url_batches)]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out exceptions and flatten results
    all_pages = []
    failed_batches = 0
    for result in results:
        if isinstance(result, Exception):
            log_error(f"TavilyExtract: Batch failed with exception - {result}")
            failed_batches += 1
        else:
            for extracted_page in result["results"]:  # type: ignore
                document = Document(
                    page_content=extracted_page["raw_content"],
                    metadata={"source": extracted_page["url"]},
                )
                all_pages.append(document)

    log_success(
        f"TavilyExtract: Extraction complete! Total pages extracted: {len(all_pages)}"
    )
    if failed_batches > 0:
        log_warning(f"TavilyExtract: {failed_batches} batches failed during extraction")

    return all_pages


@traceable(name="pinecone_documents_upsert", run_type="tool")
async def upsert_pinecone_documents(documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Upsert documents into the Pinecone document index and record a LangSmith run."""
    index_host = vector_store.index.config.host.rstrip("/")
    upsert_url = f"{index_host}/namespaces/{NAMESPACE}/documents/upsert"
    headers = {
        "Api-Key": os.environ["PINECONE_API_KEY"],
        "Content-Type": "application/json",
        "X-Pinecone-Api-Version": "2026-07",
    }
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            upsert_url, headers=headers, json={"documents": documents}
        )
    if response.status_code >= 400:
        raise RuntimeError(f"{response.status_code} {response.text}")
    return {"status_code": response.status_code, "upserted_count": len(documents)}


async def index_documents_async(documents: List[Document], batch_size: int = 50):
    """Process documents in batches asynchronously."""
    log_header("VECTOR STORAGE PHASE")
    log_info(
        f"📚 VectorStore Indexing: Preparing to add {len(documents)} documents to vector store",
        Colors.DARKCYAN,
    )

    # This index embeds chunk_text itself and accepts at most 96 documents per request.
    batch_size = min(batch_size, 96)
    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]

    log_info(
        f"📦 VectorStore Indexing: Split into {len(batches)} batches of {batch_size} documents each"
    )

    async def add_batch(batch: List[Document], batch_num: int):
        try:
            documents_payload = [
                {
                    "_id": hashlib.sha256(
                        f"{doc.metadata.get('source', '')}\n{doc.page_content}".encode()
                    ).hexdigest(),
                    "chunk_text": doc.page_content,
                    "source": str(doc.metadata.get("source", "")),
                }
                for doc in batch
            ]
            await upsert_pinecone_documents(documents_payload)
            log_success(
                f"VectorStore Indexing: Successfully added batch {batch_num}/{len(batches)} ({len(batch)} documents)"
            )
        except Exception as e:
            log_error(f"VectorStore Indexing: Failed to add batch {batch_num} - {e}")
            return False
        return True

    # Process batches concurrently
    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Count successful batches
    successful = sum(1 for result in results if result is True)

    if successful == len(batches):
        log_success(
            f"VectorStore Indexing: All batches processed successfully! ({successful}/{len(batches)})"
        )
    else:
        log_warning(
            f"VectorStore Indexing: Processed {successful}/{len(batches)} batches successfully"
        )


async def main():
    """Main async function to orchestrate the entire process."""
    log_header("DOCUMENTATION INGESTION PIPELINE")

    log_info(
        "🗺️  TavilyMap: Starting to map documentation structure from https://python.langchain.com/",
        Colors.PURPLE,
    )
    site_map = tavily_map.invoke("https://python.langchain.com/")
    log_success(
        f"TavilyMap: Successfully mapped {len(site_map['results'])} URLs from documentation site"
    )

    # Split URLs into batches of 20
    url_batches = chunk_urls(list(site_map["results"]), chunk_size=20)
    log_info(
        f"📋 URL Processing: Split {len(site_map['results'])} URLs into {len(url_batches)} batches",
        Colors.BLUE,
    )

    # Extract documents from URLs
    all_docs = await async_extract(url_batches)

    # Split documents into chunks
    log_header("DOCUMENT CHUNKING PHASE")
    log_info(
        f"✂️  Text Splitter: Processing {len(all_docs)} documents with 4000 chunk size and 200 overlap",
        Colors.YELLOW,
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(
        f"Text Splitter: Created {len(splitted_docs)} chunks from {len(all_docs)} documents"
    )

    # Process documents asynchronously
    await index_documents_async(splitted_docs, batch_size=500)

    log_header("PIPELINE COMPLETE")
    log_success("🎉 Documentation ingestion pipeline finished successfully!")
    log_info("📊 Summary:", Colors.BOLD)
    log_info(f"   • URLs mapped: {len(site_map['results'])}")
    log_info(f"   • Documents extracted: {len(all_docs)}")
    log_info(f"   • Chunks created: {len(splitted_docs)}")

   

if __name__ == "__main__":
    asyncio.run(main())