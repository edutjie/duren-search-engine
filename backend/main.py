import json
import os
import math
from fastapi import BackgroundTasks, FastAPI
import redis.asyncio as aioredis
from dotenv import load_dotenv
import pandas as pd

from bsbi import BSBIIndex
from compression import VBEPostings
from letor import LambdaMart
import models


load_dotenv()

redis = aioredis.Redis(
    db=0,
    host=os.environ.get("REDIS_HOST"),
    port=os.environ.get("REDIS_PORT"),
    password=os.environ.get("REDIS_PASSWORD"),
    decode_responses=True,
)
ONE_DAY = 60 * 60 * 24

app = FastAPI()

letor = LambdaMart(dataset_dir="dataset/")
letor.fit()
ranker = letor.get_model()

BSBI_instance = BSBIIndex(
    data_dir="collections",
    postings_encoding=VBEPostings,
    output_dir="index",
)

BSBI_instance.load()


async def set_cache(data, keys):
    await redis.set(
        keys,
        json.dumps(data),
        ex=ONE_DAY,
    )


def get_documents(scores) -> list[models.Document]:
    documents = []
    for score, doc_path in scores:
        with open(doc_path, "r", encoding="UTF8") as f:
            title, text = f.read().split("\t", 1)
            did = int(os.path.splitext(os.path.basename(doc_path))[0])
            collection = os.path.basename(os.path.dirname(doc_path))
            trim = min(len(text) // 2, 30)
            documents.append(
                {
                    "id": f"{collection}-{did}",
                    "title": title,
                    "preview": text[:trim] + "...",
                    "score": score,
                }
            )
    return documents


def paginate(data, page, limit) -> models.PaginatedDocuments:
    page = 1 if page < 1 else page
    last_page = math.ceil(len(data) / limit)
    return {
        "current_page": page,
        "last_page": 1 if last_page == 0 else last_page,
        "per_page": limit,
        "total": len(data),
        "data": data[(page - 1) * limit : page * limit],
    }


@app.get("/search/tfidf")
async def get_relevant_documents_tfidf(
    background_tasks: BackgroundTasks,
    query: str,
    is_letor: bool = True,
    k: int = 100,
    page: int = 1,
    limit: int = 10,
) -> models.PaginatedDocuments:
    # check if cache exists
    keys = f"tfidf:{query}-k:{k}-limit:{limit}"
    cache = await redis.get(keys)
    if cache:
        documents = json.loads(cache)
    else:
        tfid_scores = BSBI_instance.retrieve_tfidf(query, k=k)  # [(score, doc_id)]

        # convert to docs
        documents = get_documents(tfid_scores)

        # save to cache
        background_tasks.add_task(set_cache, documents, keys)

    if is_letor:
        keys = f"tfidf-letor:{query}-k:{k}-limit:{limit}"
        cache = await redis.get(keys)
        if cache:
            documents = json.loads(cache)
        else:
            if len(tfid_scores) == 0:
                documents = []
            else:
                tfidf_df = pd.DataFrame(tfid_scores, columns=["score", "doc_path"])
                tfid_scores = letor.rerank(query, tfidf_df)

                # convert to docs
                documents = get_documents(tfid_scores)

                # save to cache
                background_tasks.add_task(set_cache, documents, keys)

    return paginate(documents, page, limit)


@app.get("/search/bm25")
async def get_relevant_documents_bm25(
    background_tasks: BackgroundTasks,
    query: str,
    is_letor: bool = True,
    k: int = 100,
    page: int = 1,
    limit: int = 10,
) -> models.PaginatedDocuments:
    # check if cache exists
    keys = f"bm25:{query}-k:{k}-limit:{limit}"
    cache = await redis.get(keys)
    if cache:
        documents = json.loads(cache)
    else:
        bm25_scores = BSBI_instance.retrieve_bm25(query, k=k)  # [(score, doc_id)]

        # convert to dict
        documents = get_documents(bm25_scores)

        # save to cache
        background_tasks.add_task(set_cache, documents, keys)

    if is_letor:
        keys = f"bm25-letor:{query}-k:{k}-limit:{limit}"
        cache = await redis.get(keys)
        if cache:
            documents = json.loads(cache)
        else:
            if len(bm25_scores) == 0:
                documents = []
            else:
                bm25_df = pd.DataFrame(bm25_scores, columns=["score", "doc_path"])
                bm25_scores = letor.rerank(query, bm25_df)

                # convert to docs
                documents = get_documents(bm25_scores)

                # save to cache
                background_tasks.add_task(set_cache, documents, keys)

    return paginate(documents, page, limit)


@app.get("/document/{doc_id}")
def get_document_detail(doc_id: str) -> models.DocumentDetail:
    collection, did = doc_id.split("-")
    doc_path = os.path.join("collections", collection, f"{did}.txt")
    with open(doc_path, "r", encoding="UTF8") as f:
        title, content = f.read().split("\t", 1)
        return {
            "id": doc_id,
            "title": title,
            "content": content,
        }


@app.get("/related/{doc_id}")
async def get_related_documents(
    background_tasks: BackgroundTasks,
    doc_id: str,
    k: int = 10,
    page: int = 1,
    limit: int = 10,
) -> models.PaginatedDocuments:
    collection, did = doc_id.split("-")
    doc_path = os.path.join("collections", collection, f"{did}.txt")
    with open(doc_path, "r", encoding="UTF8") as f:
        title, content = f.read().split("\t", 1)
        query = title + " " + content
    documents = await get_relevant_documents_tfidf(background_tasks, query, is_letor=False, k=k, page=page, limit=limit)
    documents["data"].pop(0) # remove the first document (itself)
    return documents
