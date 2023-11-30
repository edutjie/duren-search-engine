from bsbi import BSBIIndex
from compression import VBEPostings

if __name__ == "__main__":
    # sebelumnya sudah dilakukan indexing
    # BSBIIndex hanya sebagai abstraksi untuk index tersebut
    BSBI_instance = BSBIIndex(
        data_dir="collections",
        postings_encoding=VBEPostings,
        output_dir="index",
    )

    BSBI_instance.load()

    print("DOC PATH", BSBI_instance.doc_id_map[451670])

    # queries = [
    #     "batman film",
    #     "beautiful sea",
    # ]
    # print("=== TF-IDF ===")
    # for query in queries:
    #     print("Query  : ", query)
    #     print("Results:")
    #     tfid_scores = BSBI_instance.retrieve_tfidf(query, k=100)
    #     # with open("tfidf.pkl", "wb") as f:
    #     #     pkl.dump(tfid_scores, f)
    #     for score, doc in tfid_scores:
    #         print(f"{doc:30} {score:>.3f}")
    #     print()

    # print("=== BM25 ===")
    # for query in queries:
    #     print("Query  : ", query)
    #     print("Results:")
    #     bm25_scores = BSBI_instance.retrieve_bm25(query, k=100)
    #     # with open("bm25.pkl", "wb") as f:
    #     #     pkl.dump(bm25_scores, f)
    #     for score, doc in bm25_scores:
    #         print(f"{doc:30} {score:>.3f}")
    #     print()
