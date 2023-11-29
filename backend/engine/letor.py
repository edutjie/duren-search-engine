import re, os, random
from collections import defaultdict
from gensim.models import LsiModel
from gensim.corpora import Dictionary
from scipy.spatial.distance import cosine
import lightgbm as lgb
import numpy as np
from sklearn.metrics import ndcg_score, dcg_score
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pickle as pkl
from bsbi import BSBIIndex
from compression import VBEPostings
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings("ignore")


def zero_dict():
    return defaultdict(int)


class LambdaMart:
    NUM_TOPICS = 200

    def __init__(self, dataset_dir="dataset/") -> None:
        self.dataset_dir = dataset_dir
        self.stemmer = SnowballStemmer(language="english")
        self.stop_words = set(stopwords.words("english"))
        print("Loading dataset...")
        self._load_dataset(
            train_queries_file="train.queries",
            train_qrels_file="train.qrels",
            val_queries_file="val.queries",
            val_qrels_file="val.qrels",
        )

        self.dictionary = Dictionary()
        if os.path.exists("cache/bow_corpus.pkl"):
            with open("cache/bow_corpus.pkl", "rb") as f:
                bow_corpus = pkl.load(f)
        else:
            bow_corpus = [
                self.dictionary.doc2bow(doc, allow_update=True)
                for doc in self.collections_docs.values()
            ]
            with open("cache/bow_corpus.pkl", "wb") as f:
                pkl.dump(bow_corpus, f)

        print("Init LSI Model...")
        self.lsi_model = LsiModel(
            bow_corpus, num_topics=self.NUM_TOPICS
        )  # 200 latent topics

        print("Init TF-IDF Vectorizer...")
        self.tfidf_vectorizer = TfidfVectorizer()
        train_docs_df = pd.DataFrame(self.collections_docs.items(), columns=["id", "content"])
        train_docs_df["content"] = train_docs_df["content"].str.join(" ")
        self.vectorized_train_docs = self.tfidf_vectorizer.fit_transform(
            train_docs_df["content"]
        )

        print("Init LGBM Ranker Model...")
        self.ranker = lgb.LGBMRanker(
            objective="lambdarank",
            boosting_type="gbdt",
            n_estimators=100,
            importance_type="gain",
            metric="ndcg",
            num_leaves=40,
            learning_rate=0.02,
            max_depth=-1,
        )

    def _pre_processing_text(self, content):
        """
        Melakukan preprocessing pada text, yakni stemming dan removing stopwords
        """
        # https://github.com/ariaghora/mpstemmer/tree/master/mpstemmer
        res = []
        for term in word_tokenize(content.lower()):
            term = self.stemmer.stem(term)
            if term not in self.stop_words:
                res.append(term)
        return res

    def load_qrels(self, qrel_file="train_qrels.txt", is_train=True) -> defaultdict:
        qrels = defaultdict(zero_dict)
        with open(qrel_file, encoding="UTF8") as file:
            for line in file:
                parts = line.strip().split()
                qid = parts[0]
                did = int(parts[1])
                if is_train:
                    rel = int(parts[2])
                    qrels[qid][did] = rel
                else:
                    qrels[qid][did] = 1
        return qrels

    def load_docs(
        self, doc_file="train_docs.txt", is_id_int=True, is_raw=False
    ) -> defaultdict:
        docs = defaultdict()
        with open(doc_file, encoding="UTF8") as file:
            for line in file:
                parts = line.strip().split()
                did = int(parts[0]) if is_id_int else parts[0]
                doc = " ".join(parts[1:])
                docs[did] = doc if is_raw else self._pre_processing_text(doc)
        return docs

    def load_collections(self, collections_dir="collections/"):
        if os.path.exists("cache/collections_docs.pkl"):
            with open("cache/collections_docs.pkl", "rb") as f:
                collections_docs = pkl.load(f)
        else:
            collections_docs = defaultdict()
            for folder in os.listdir(collections_dir):
                for file in os.listdir(collections_dir + folder):
                    if file.endswith(".txt"):
                        with open(
                            collections_dir + folder + "/" + file, encoding="UTF8"
                        ) as f:
                            did = int(file.removesuffix(".txt"))
                            collections_docs[did] = self._pre_processing_text(f.read())

            with open("cache/collections_docs.pkl", "wb") as f:
                pkl.dump(collections_docs, f)

        return collections_docs

    def _load_dataset(
        self,
        train_queries_file="train_queries.txt",
        train_qrels_file="train_qrels.txt",
        val_queries_file="val_queries.txt",
        val_qrels_file="val_qrels.txt",
        num_negatives=1,
    ) -> (list, list):
        if os.path.exists("cache/collections_docs.pkl"):
            with open("cache/collections_docs.pkl", "rb") as f:
                self.collections_docs = pkl.load(f)
        else:
            self.collections_docs = self.load_collections()
            with open("cache/collections_docs.pkl", "wb") as f:
                pkl.dump(self.collections_docs, f)

        if os.path.exists("cache/train_queries.pkl"):
            with open("cache/train_queries.pkl", "rb") as f:
                train_queries = pkl.load(f)
        else:
            train_queries = self.load_docs(
                os.path.join(self.dataset_dir, train_queries_file), is_id_int=False
            )
            with open("cache/train_queries.pkl", "wb") as f:
                pkl.dump(train_queries, f)

        if os.path.exists("cache/train_qrels.pkl"):
            with open("cache/train_qrels.pkl", "rb") as f:
                train_qrels = pkl.load(f)
        else:
            train_qrels = self.load_qrels(
                os.path.join(self.dataset_dir, train_qrels_file)
            )
            with open("cache/train_qrels.pkl", "wb") as f:
                pkl.dump(train_qrels, f)

        if os.path.exists("cache/val_queries.pkl"):
            with open("cache/val_queries.pkl", "rb") as f:
                val_queries = pkl.load(f)
        else:
            val_queries = self.load_docs(
                os.path.join(self.dataset_dir, val_queries_file), is_id_int=False
            )
            with open("cache/val_queries.pkl", "wb") as f:
                pkl.dump(val_queries, f)

        if os.path.exists("cache/val_qrels.pkl"):
            with open("cache/val_qrels.pkl", "rb") as f:
                val_qrels = pkl.load(f)
        else:
            val_qrels = self.load_qrels(os.path.join(self.dataset_dir, val_qrels_file))
            with open("cache/val_qrels.pkl", "wb") as f:
                pkl.dump(val_qrels, f)

        # group_qid_count untuk model LGBMRanker
        self.train_group_qid_count = []
        self.train_dataset = []
        for q_id, docs_rels in train_qrels.items():
            self.train_group_qid_count.append(len(docs_rels) + num_negatives)
            for doc_id, rel in docs_rels.items():
                self.train_dataset.append(
                    (train_queries[q_id], self.collections_docs[doc_id], rel)
                )
            # tambahkan satu negative (random sampling saja dari documents)
            self.train_dataset.append(
                (train_queries[q_id], random.choice(list(self.collections_docs.values())), 0)
            )

        assert sum(self.train_group_qid_count) == len(
            self.train_dataset
        ), "Something's wrong"

        self.val_group_qid_count = []
        self.val_dataset = []
        for q_id, docs_rels in val_qrels.items():
            self.val_group_qid_count.append(len(docs_rels))
            for doc_id, rel in docs_rels.items():
                doc = self.collections_docs[doc_id]
                self.val_dataset.append((val_queries[q_id], doc, rel))

        assert sum(self.val_group_qid_count) == len(
            self.val_dataset
        ), "Something's wrong"

        return self.train_dataset, self.val_dataset

    def vector_rep(self, text, num_topics=200) -> list:
        rep = [
            topic_value
            for (_, topic_value) in self.lsi_model[self.dictionary.doc2bow(text)]
        ]
        return rep if len(rep) == num_topics else [0.0] * num_topics

    def _create_features(self, query, doc) -> list:
        v_q = self.vector_rep(query)
        v_d = self.vector_rep(doc)
        q = set(query)
        d = set(doc)
        cosine_dist = cosine(v_q, v_d)
        jaccard = len(q & d) / len(q | d)

        # custom
        query_vector = self.tfidf_vectorizer.transform([" ".join(query)])
        doc_vector = self.tfidf_vectorizer.transform([" ".join(doc)])
        tfidf_cosine = cosine_similarity(query_vector, doc_vector).tolist()[0][0]

        return v_q + v_d + [jaccard] + [cosine_dist] + [tfidf_cosine]

    def get_lsa(self) -> (LsiModel, Dictionary):
        return self.lsi_model, self.dictionary

    def get_x_y(self, dataset) -> (list, list):
        X = []
        y = []
        for query, doc, rel in dataset:
            X.append(self._create_features(query, doc))
            y.append(rel)
        return np.array(X), np.array(y)

    def get_model(self) -> lgb.LGBMRanker:
        return self.ranker

    def fit(self) -> None:
        if (
            os.path.exists("cache/X.pkl")
            and os.path.exists("cache/y.pkl")
            and os.path.exists("cache/X_val.pkl")
            and os.path.exists("cache/y_val.pkl")
        ):
            with open("cache/X.pkl", "rb") as f:
                self.X = pkl.load(f)
            with open("cache/y.pkl", "rb") as f:
                self.y = pkl.load(f)
            with open("cache/X_val.pkl", "rb") as f:
                self.X_val = pkl.load(f)
            with open("cache/y_val.pkl", "rb") as f:
                self.y_val = pkl.load(f)
        else:
            self.X, self.y = self.get_x_y(self.train_dataset)
            self.X_val, self.y_val = self.get_x_y(self.val_dataset)

            with open("cache/X.pkl", "wb") as f:
                pkl.dump(self.X, f)
            with open("cache/y.pkl", "wb") as f:
                pkl.dump(self.y, f)
            with open("cache/X_val.pkl", "wb") as f:
                pkl.dump(self.X_val, f)
            with open("cache/y_val.pkl", "wb") as f:
                pkl.dump(self.y_val, f)

        if os.path.exists("models/ranker.pkl"):
            with open("models/ranker.pkl", "rb") as f:
                self.ranker = pkl.load(f)
        else:
            self.ranker.fit(
                self.X,
                self.y,
                group=self.train_group_qid_count,
                eval_set=[(self.X_val, self.y_val)],
                eval_group=[self.val_group_qid_count],
                eval_at=[5, 10, 20],
            )

            with open("models/ranker.pkl", "wb") as f:
                pkl.dump(self.ranker, f)

    def predict(self, X) -> list:
        return self.ranker.predict(X)

    def predict_proba(self, X) -> list:
        return self.ranker.predict_proba(X)

    # def self_evaluate(self) -> float:
    #     X_val = self.X_val
    #     y_val = self.y_val
    #     score = 0
    #     for i in self.val_group_qid_count:
    #         curr_X = X_val[:i]
    #         curr_y = y_val[:i]
    #         score += self.evaluate(curr_X, curr_y)
    #         X_val = X_val[i:]
    #         y_val = y_val[i:]
    #     return score / len(self.val_group_qid_count)

    # def evaluate(self, X, y) -> float:
    #     y_pred = self.predict(X)
    #     if len(y) > 1:
    #         return ndcg_score([y], [y_pred])
    #     else:
    #         return dcg_score([y], [y_pred])

    def rerank(self, query: str, retrieved_df: pd.DataFrame) -> list[(int, int)]:
        X = []
        query_processed = self._pre_processing_text(query)
        for _, row in retrieved_df.iterrows():
            with open(row["doc_path"], encoding="UTF8") as file:
                doc_processed = self._pre_processing_text(file.read())
                X.append(self._create_features(query_processed, doc_processed))
        X = np.array(X)
        scores = self.predict(X)
        scores_did = [x for x in zip(scores, retrieved_df["doc_path"].tolist())]
        sorted_scores_did = sorted(scores_did, key=lambda tup: tup[0], reverse=True)
        return sorted_scores_did


if __name__ == "__main__":
    letor = LambdaMart(dataset_dir="dataset/")
    print("Training...")
    letor.fit()
    ranker = letor.get_model()
    print("Best Validation Score:", ranker.best_score_.get("valid_0"))
    # print("nDCG@all Evaluation Dataset:", letor.self_evaluate())

    print("Loading test dataset...")
    test_qrels = letor.load_qrels("dataset/test.qrels", is_train=False)
    test_queries = letor.load_docs(
        "dataset/test.queries", is_id_int=False, is_raw=True
    )

    BSBI_instance = BSBIIndex(
        data_dir="collections",
        postings_encoding=VBEPostings,
        output_dir="index",
    )

    BSBI_instance.load()

    # demo rerank
    if os.path.exists("cache/reranked_tfidfs.pkl"):
        with open("cache/reranked_tfidfs.pkl", "rb") as f:
            reranked_tfidfs = pkl.load(f)
    else:
        reranked_tfidfs = []
        for qid, query in tqdm(test_queries.items()):
            tfidf_raw = BSBI_instance.retrieve_tfidf(query, k=100)
            if len(tfidf_raw) == 0:
                continue
            tfidf_df = pd.DataFrame(tfidf_raw, columns=["score", "doc_path"])
            tfidf_df["doc_id"] = tfidf_df["doc_path"].apply(
                lambda x: int(x.split("\\")[-1].removesuffix(".txt"))
            )
            reranked_tfidf = letor.rerank(query, tfidf_df)
            reranked_tfidfs.append((qid, tfidf_raw, reranked_tfidf))
        with open("cache/reranked_tfidfs.pkl", "wb") as f:
            pkl.dump(reranked_tfidfs, f)

    print("Query:", reranked_tfidfs[0][0])
    print("TFIDF before reranked:")
    for score, doc in reranked_tfidfs[0][1]:
        print(f"{doc:30} {score:>.3f}")
    print("TFIDF after reranked:")
    for score, doc in reranked_tfidfs[0][2]:
        print(f"{doc:30} {score:>.3f}")
