import json
import os
from fastapi import BackgroundTasks, FastAPI
import redis.asyncio as aioredis
from dotenv import load_dotenv
import pandas as pd

from bsbi import BSBIIndex
from compression import VBEPostings
from letor import LambdaMart
from collections import defaultdict


load_dotenv()

redis = aioredis.Redis(
    db=0,
    host=os.environ.get("REDIS_HOST"),
    port=os.environ.get("REDIS_PORT"),
    password=os.environ.get("REDIS_PASSWORD"),
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


@app.get("/search/tfidf")
async def get_relevant_documents(
    background_tasks: BackgroundTasks,
    query: str,
    is_letor: bool = True,
    page: int = 1,
    limit: int = 100,
):
    # check if cache exists
    keys = f"tfidf:{query}"
    cache = await redis.get(keys)
    if cache:
        tfid_scores = json.loads(cache)
    else:
        tfid_scores = BSBI_instance.retrieve_tfidf(query, k=limit)  # [(score, doc_id)]
        # convert to dict
        tfid_scores = [{"score": score, "doc_path": did} for score, did in tfid_scores]

        # save to cache
        background_tasks.add_task(set_cache, tfid_scores, keys)

    if is_letor:
        keys = f"tfidf-letor:{query}"
        cache = await redis.get(keys)
        if cache:
            tfid_scores = json.loads(cache)
        else:
            tfidf_df = pd.DataFrame(tfid_scores)
            tfidf_df["doc_id"] = tfidf_df["doc_path"].apply(
                lambda x: int(x.split("\\")[-1].removesuffix(".txt"))
            )
            tfid_scores = letor.rerank(query, tfidf_df)

            # save to cache
            background_tasks.add_task(set_cache, tfid_scores, keys)

    return tfid_scores
