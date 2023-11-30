import re
import os
from bsbi import BSBIIndex
from compression import VBEPostings
from tqdm import tqdm
import math
from letor import LambdaMart
import pandas as pd

# >>>>> 3 IR metrics: RBP p = 0.8, DCG, dan AP


def rbp(ranking, p=0.8):
    """menghitung search effectiveness metric score dengan
    Rank Biased Precision (RBP)

    Parameters
    ----------
    ranking: List[int]
       vektor biner seperti [1, 0, 1, 1, 1, 0]
       gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
       Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
               di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
               di rank-6 tidak relevan

    Returns
    -------
    Float
      score RBP
    """
    score = 0.0
    for i in range(1, len(ranking) + 1):
        pos = i - 1
        score += ranking[pos] * (p ** (i - 1))
    return (1 - p) * score


def dcg(ranking):
    """menghitung search effectiveness metric score dengan
    Discounted Cumulative Gain

    Parameters
    ----------
    ranking: List[int]
       vektor biner seperti [1, 0, 1, 1, 1, 0]
       gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
       Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
               di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
               di rank-6 tidak relevan

    Returns
    -------
    Float
      score DCG
    """
    # TODO
    dcg_score = 0.0
    for i, relevance in enumerate(ranking, start=1):
        discount = 1 / (math.log2(i + 1) if i > 1 else 1)  # avoid division by zero
        dcg_score += discount * relevance

    return dcg_score


def idcg(ranking):
    sorted_ranking = sorted(ranking, reverse=True)
    return dcg(sorted_ranking)


def ndcg(ranking):
    idcg_score = idcg(ranking)
    if idcg_score == 0:
        return 0
    return dcg(ranking) / idcg_score


def prec(ranking, k):
    """menghitung search effectiveness metric score dengan
    Precision at K

    Parameters
    ----------
    ranking: List[int]
       vektor biner seperti [1, 0, 1, 1, 1, 0]
       gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
       Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
               di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
               di rank-6 tidak relevan

    k: int
      banyak dokumen yang dipertimbangkan atau diperoleh

    Returns
    -------
    Float
      score Prec@K
    """
    # TODO
    num_relevant = 0
    for i in range(k):
        if ranking[i] == 1:
            num_relevant += 1
    return num_relevant / k


def ap(ranking):
    """menghitung search effectiveness metric score dengan
    Average Precision

    Parameters
    ----------
    ranking: List[int]
       vektor biner seperti [1, 0, 1, 1, 1, 0]
       gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
       Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
               di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
               di rank-6 tidak relevan

    Returns
    -------
    Float
      score AP
    """
    # TODO
    precision_sum = 0
    num_relevant = 0

    for i in range(len(ranking)):
        if ranking[i] == 1:
            num_relevant += 1
            precision_sum += prec(ranking, i + 1)

    if num_relevant == 0:
        return 0

    average_precision = precision_sum / num_relevant
    return average_precision


# >>>>> memuat qrels


def load_qrels(qrel_file="qrels.txt"):
    """
    memuat query relevance judgment (qrels)
    dalam format dictionary of dictionary qrels[query id][document id],
    dimana hanya dokumen yang relevan (nilai 1) yang disimpan,
    sementara dokumen yang tidak relevan (nilai 0) tidak perlu disimpan,
    misal {"Q1": {500:1, 502:1}, "Q2": {150:1}}
    """
    with open(qrel_file) as file:
        content = file.readlines()

    qrels_sparse = {}

    for line in content:
        parts = line.strip().split()
        qid = parts[0]
        did = int(parts[1])
        if not (qid in qrels_sparse):
            qrels_sparse[qid] = {}
        if not (did in qrels_sparse[qid]):
            qrels_sparse[qid][did] = 0
        qrels_sparse[qid][did] = 1
    return qrels_sparse


# >>>>> EVALUASI !


def eval_retrieval(qrels, query_file="queries.txt", k=1000):
    """
    loop ke semua query, hitung score di setiap query,
    lalu hitung MEAN SCORE-nya.
    untuk setiap query, kembalikan top-1000 documents
    """
    BSBI_instance = BSBIIndex(
        data_dir="collections", postings_encoding=VBEPostings, output_dir="index"
    )
    BSBI_instance.load()

    letor = LambdaMart(dataset_dir="dataset/qrels-folder/")
    letor.fit()

    with open(query_file, encoding="UTF8") as file:
        rbp_scores_tfidf = []
        dcg_scores_tfidf = []
        ap_scores_tfidf = []
        ndcg_scores_tfidf = []

        rbp_scores_bm25 = []
        dcg_scores_bm25 = []
        ap_scores_bm25 = []
        ndcg_scores_bm25 = []

        rbp_scores_letor_tfidf = []
        dcg_scores_letor_tfidf = []
        ap_scores_letor_tfidf = []
        ndcg_scores_letor_tfidf = []

        rbp_scores_letor_bm25 = []
        dcg_scores_letor_bm25 = []
        ap_scores_letor_bm25 = []
        ndcg_scores_letor_bm25 = []

        total_queries = 0

        for qline in tqdm(file):
            parts = qline.strip().split()
            qid = parts[0]
            query = " ".join(parts[1:])

            """
            Evaluasi TF-IDF
            """
            ranking_tfidf = []
            tfidf_raw = BSBI_instance.retrieve_tfidf(query, k=k)
            for _, doc in tfidf_raw:
                did = int(os.path.splitext(os.path.basename(doc))[0])
                # Alternatif lain:
                # 1. did = int(doc.split("\\")[-1].split(".")[0])
                # 2. did = int(re.search(r'\/.*\/.*\/(.*)\.txt', doc).group(1))
                # 3. disesuaikan dengan path Anda
                if qid not in qrels:
                    continue
                if did in qrels[qid]:
                    ranking_tfidf.append(1)
                else:
                    ranking_tfidf.append(0)
            rbp_scores_tfidf.append(rbp(ranking_tfidf))
            dcg_scores_tfidf.append(dcg(ranking_tfidf))
            ap_scores_tfidf.append(ap(ranking_tfidf))
            ndcg_scores_tfidf.append(ndcg(ranking_tfidf))

            """
            Evaluasi BM25
            """
            ranking_bm25 = []
            bm25_raw = BSBI_instance.retrieve_bm25(query, k=k, k1=1.2, b=0.75)
            # nilai k1 dan b dapat diganti-ganti
            for _, doc in bm25_raw:
                did = int(os.path.splitext(os.path.basename(doc))[0])
                # Alternatif lain:
                # 1. did = int(doc.split("\\")[-1].split(".")[0])
                # 2. did = int(re.search(r'\/.*\/.*\/(.*)\.txt', doc).group(1))
                # 3. disesuaikan dengan path Anda
                if qid not in qrels:
                    continue
                if did in qrels[qid]:
                    ranking_bm25.append(1)
                else:
                    ranking_bm25.append(0)
            rbp_scores_bm25.append(rbp(ranking_bm25))
            dcg_scores_bm25.append(dcg(ranking_bm25))
            ap_scores_bm25.append(ap(ranking_bm25))
            ndcg_scores_bm25.append(ndcg(ranking_bm25))

            if len(tfidf_raw) > 0:
                tfidf_df = pd.DataFrame(tfidf_raw, columns=["score", "doc_path"])
                reranked_tfidf = letor.rerank(query, tfidf_df)
                ranking_tfidf = []
                for _, doc in reranked_tfidf:
                    did = int(os.path.splitext(os.path.basename(doc))[0])
                    if did in qrels[qid]:
                        ranking_tfidf.append(1)
                    else:
                        ranking_tfidf.append(0)
                rbp_scores_letor_tfidf.append(rbp(ranking_tfidf))
                dcg_scores_letor_tfidf.append(dcg(ranking_tfidf))
                ap_scores_letor_tfidf.append(ap(ranking_tfidf))
                ndcg_scores_letor_tfidf.append(ndcg(ranking_tfidf))

            if len(bm25_raw) > 0:
                bm25_df = pd.DataFrame(bm25_raw, columns=["score", "doc_path"])
                reranked_bm25 = letor.rerank(query, bm25_df)
                ranking_bm25 = []
                for _, doc in reranked_bm25:
                    did = int(os.path.splitext(os.path.basename(doc))[0])
                    if did in qrels[qid]:
                        ranking_bm25.append(1)
                    else:
                        ranking_bm25.append(0)
                rbp_scores_letor_bm25.append(rbp(ranking_bm25))
                dcg_scores_letor_bm25.append(dcg(ranking_bm25))
                ap_scores_letor_bm25.append(ap(ranking_bm25))
                ndcg_scores_letor_bm25.append(ndcg(ranking_bm25))

            total_queries += 1

    print(f"Hasil evaluasi TF-IDF terhadap {total_queries} queries")
    print("RBP score =", sum(rbp_scores_tfidf) / len(rbp_scores_tfidf))
    print("DCG score =", sum(dcg_scores_tfidf) / len(dcg_scores_tfidf))
    print("AP score  =", sum(ap_scores_tfidf) / len(ap_scores_tfidf))
    print("nDCG score  =", sum(ndcg_scores_tfidf) / len(ndcg_scores_tfidf))

    print(f"Hasil evaluasi BM25 terhadap {total_queries} queries")
    print("RBP score =", sum(rbp_scores_bm25) / len(rbp_scores_bm25))
    print("DCG score =", sum(dcg_scores_bm25) / len(dcg_scores_bm25))
    print("AP score  =", sum(ap_scores_bm25) / len(ap_scores_bm25))
    print("nDCG score  =", sum(ndcg_scores_bm25) / len(ndcg_scores_bm25))

    print(f"Hasil evaluasi TF-IDF dengan LETOR terhadap {total_queries} queries")
    print("RBP score =", sum(rbp_scores_letor_tfidf) / len(rbp_scores_letor_tfidf))
    print("DCG score =", sum(dcg_scores_letor_tfidf) / len(dcg_scores_letor_tfidf))
    print("AP score  =", sum(ap_scores_letor_tfidf) / len(ap_scores_letor_tfidf))
    print("nDCG score  =", sum(ndcg_scores_letor_tfidf) / len(ndcg_scores_letor_tfidf))

    print(f"Hasil evaluasi BM25 dengan LETOR terhadap {total_queries} queries")
    print("RBP score =", sum(rbp_scores_letor_bm25) / len(rbp_scores_letor_bm25))
    print("DCG score =", sum(dcg_scores_letor_bm25) / len(dcg_scores_letor_bm25))
    print("AP score  =", sum(ap_scores_letor_bm25) / len(ap_scores_letor_bm25))
    print("nDCG score  =", sum(ndcg_scores_letor_bm25) / len(ndcg_scores_letor_bm25))


if __name__ == "__main__":
    qrels = load_qrels("dataset/test.qrels")

    # assert qrels["Q1002252"][5599474] == 1, "qrels salah"
    # assert not (6998091 in qrels["Q1007972"]), "qrels salah"

    eval_retrieval(qrels, query_file="dataset/test.queries", k=100)
