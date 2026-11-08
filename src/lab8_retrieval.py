"""Lab 8 - evaluate word windows (overlap 0 / 10) vs sentence-aware packing
with TF-IDF and real sentence embeddings (all-MiniLM-L6-v2, CPU).

Adapted from the lab manual's reference solution for the retrieval task. Additions: token-length
audit summary, metrics table (results_summary.csv), top-3 printout with offsets,
automatic failure listing, unanswerable-query report and strategy selection on
development data only (selected_config.json, used by lab8_chatbot.py).
All files are written to outputs/.
Run: python src/lab8_retrieval.py   (needs data/, see lab8_dataset.py)
"""
import csv
import json
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

try:
    from .lab8_chunks import make_chunks
except ImportError:                     # so it also runs as: python src/<file>.py
    from lab8_chunks import make_chunks

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
STRATEGIES = ["0", "10", "sentence"]
NAMES = {"0": "window-40/overlap-0", "10": "window-40/overlap-10", "sentence": "sentence-40"}


def relevant_ranks(row, chunks, q):
    top = np.argsort(-row, kind="stable")[:3]
    return [rank for rank, i in enumerate(top, 1)
            if chunks[i]["doc"] == q["doc"] and q["evidence"] in chunks[i]["text"]]


def evaluate(scores, chunks, items):
    hits, reciprocal = [], []
    for row, q in zip(scores, items):
        if q["doc"] is None:
            continue
        good = relevant_ranks(row, chunks, q)
        hits.append(bool(good))
        reciprocal.append(1 / good[0] if good else 0)
    return (float(np.mean(hits)), float(np.mean(reciprocal))) if hits else (None, None)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    with open(DATA_DIR / "corpus.json", encoding="utf-8") as f:
        docs = json.load(f)
    with open(DATA_DIR / "queries.json", encoding="utf-8") as f:
        items = json.load(f)
    assert docs and items
    for q in items:
        assert q["split"] in {"dev", "test"}
        if q["doc"] is not None:
            assert q["evidence"] and q["evidence"] in docs[q["doc"]]
    print("Documents:", len(docs), "| queries:", len(items),
          "| answerable dev/test:",
          sum(q["doc"] is not None and q["split"] == "dev" for q in items), "/",
          sum(q["doc"] is not None and q["split"] == "test" for q in items),
          "| unanswerable:", sum(q["doc"] is None for q in items))

    t = time.perf_counter()
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    load_seconds = time.perf_counter() - t
    print(f"Model load seconds: {load_seconds:.3f} (measured separately) | "
          f"max_seq_length: {model.max_seq_length} | embedding dim: "
          f"{getattr(model, 'get_embedding_dimension', model.get_sentence_embedding_dimension)()}")

    query_texts = [q["query"] for q in items]
    logs, summary, failures = [], [], []
    all_scores = {}
    for strategy in STRATEGIES:
        chunks = make_chunks(docs, strategy)
        texts = [c["text"] for c in chunks]
        lengths = [len(model.tokenizer(x, truncation=False)["input_ids"])
                   for x in texts + query_texts]
        print(f"\nToken audit [{NAMES[strategy]}]: {len(chunks)} chunks, max chunk tokens "
              f"{max(lengths[:len(texts)])}, max query tokens {max(lengths[len(texts):])}, "
              f"limit {model.max_seq_length} -> OK")
        if max(lengths) > model.max_seq_length:
            raise ValueError("Model limit exceeded; rechunk or shorten query")
        for method in ["tfidf", "embedding"]:
            t = time.perf_counter()
            if method == "tfidf":
                vectorizer = TfidfVectorizer()
                C = vectorizer.fit_transform(texts)
                Q = vectorizer.transform(query_texts)
            else:
                C = model.encode(texts, normalize_embeddings=True)
                Q = model.encode(query_texts, normalize_embeddings=True)
            encode_seconds = time.perf_counter() - t
            t = time.perf_counter()
            scores = (Q @ C.T).toarray() if method == "tfidf" else Q @ C.T
            search_seconds = time.perf_counter() - t
            all_scores[(strategy, method)] = (scores, chunks)
            row_out = dict(strategy=NAMES[strategy], method=method, chunks=len(chunks),
                           encode_seconds=round(encode_seconds, 4),
                           search_seconds=round(search_seconds, 6))
            for split in ["dev", "test"]:
                idx = [i for i, q in enumerate(items) if q["split"] == split]
                hit, mrr = evaluate(scores[idx], chunks, [items[i] for i in idx])
                row_out[f"{split}_hit3"], row_out[f"{split}_mrr3"] = round(hit, 3), round(mrr, 3)
            summary.append(row_out)
            print(f"  {method:<9} dev Hit@3={row_out['dev_hit3']:.3f} MRR@3={row_out['dev_mrr3']:.3f} | "
                  f"test Hit@3={row_out['test_hit3']:.3f} MRR@3={row_out['test_mrr3']:.3f} | "
                  f"encode {encode_seconds:.4f}s search {search_seconds:.6f}s")
            for row, q in zip(scores, items):
                good = relevant_ranks(row, chunks, q) if q["doc"] else []
                if q["doc"] and (not good or good[0] != 1):
                    top1 = chunks[int(np.argsort(-row, kind="stable")[0])]
                    failures.append(dict(strategy=NAMES[strategy], method=method, query=q["query"],
                                         split=q["split"], expected_doc=q["doc"],
                                         first_relevant_rank=good[0] if good else None,
                                         top1=top1["id"], top1_score=round(float(row[
                                             np.argsort(-row, kind="stable")[0]]), 3)))
                for rank, i in enumerate(np.argsort(-row, kind="stable")[:3], 1):
                    c = chunks[i]
                    logs.append(dict(strategy=NAMES[strategy], method=method, query=q["query"],
                                     split=q["split"], rank=rank, score=round(float(row[i]), 4),
                                     chunk_id=c["id"], doc=c["doc"], start=c["start"],
                                     end=c["end"], relevant=bool(q["doc"]) and c["doc"] == q["doc"]
                                     and q["evidence"] in c["text"],
                                     answerable=q["doc"] is not None, text=c["text"]))

    with open(OUT_DIR / "retrieval_log.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(logs[0]))
        writer.writeheader()
        writer.writerows(logs)
    with open(OUT_DIR / "results_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    print("\n================ RESULTS TABLE ================")
    print("%-22s %-9s %6s %8s %8s %9s %9s %9s %10s" % ("strategy", "method", "chunks", "devHit",
          "devMRR", "testHit", "testMRR", "encode_s", "search_s"))
    for r in summary:
        print("%-22s %-9s %6d %8.3f %8.3f %9.3f %9.3f %9.4f %10.6f" % (
            r["strategy"], r["method"], r["chunks"], r["dev_hit3"], r["dev_mrr3"],
            r["test_hit3"], r["test_mrr3"], r["encode_seconds"], r["search_seconds"]))

    # Selection on DEV only (embedding backend): best dev MRR@3, then dev Hit@3, then fewer chunks.
    emb = [r for r in summary if r["method"] == "embedding"]
    best = sorted(emb, key=lambda r: (-r["dev_mrr3"], -r["dev_hit3"], r["chunks"]))[0]
    strategy_key = [k for k, v in NAMES.items() if v == best["strategy"]][0]
    print(f"\nSelected on development data: {best['strategy']} + embedding "
          f"(dev Hit@3={best['dev_hit3']:.3f}, dev MRR@3={best['dev_mrr3']:.3f}).")
    print(f"Held-out test result, no retuning: Hit@3={best['test_hit3']:.3f}, "
          f"MRR@3={best['test_mrr3']:.3f}")
    with open(OUT_DIR / "selected_config.json", "w", encoding="utf-8") as f:
        json.dump({"strategy": strategy_key, "backend": "embedding", "model": MODEL_NAME,
                   "selected_by": "dev MRR@3, then dev Hit@3, then fewer chunks"}, f, indent=2)

    print("\n================ TOP-3 FOR EVERY QUERY (selected config + TF-IDF same chunks) =====")
    for method in ["embedding", "tfidf"]:
        scores, chunks = all_scores[(strategy_key, method)]
        print(f"\n##### {NAMES[strategy_key]} / {method}")
        for row, q in zip(scores, items):
            label = q["doc"] or "UNANSWERABLE"
            print(f"\nQ [{q['split']}] {q['query']}  (expected: {label})")
            for rank, i in enumerate(np.argsort(-row, kind="stable")[:3], 1):
                c = chunks[i]
                rel = q["doc"] is not None and c["doc"] == q["doc"] and q["evidence"] in c["text"]
                print(f"  {rank}. {c['id']} score={float(row[i]):.3f} chars {c['start']}-{c['end']}"
                      f"{'  [RELEVANT]' if rel else ''}")
                print("     " + c["text"])

    print("\n================ FIRST RELEVANT RANK PER QUERY (- = miss in top 3) ================")
    keys = list(all_scores)
    print("%-6s %-60s " % ("split", "query") + " ".join("%-6s" % (k[0][:4] + "/" + k[1][:3]) for k in keys))
    for qi, q in enumerate(items):
        if q["doc"] is None:
            continue
        cells = []
        for k in keys:
            scores, chunks = all_scores[k]
            good = relevant_ranks(scores[qi], chunks, q)
            cells.append("%-6s" % (good[0] if good else "-"))
        print("%-6s %-60s " % (q["split"], q["query"][:60]) + " ".join(cells))

    print("\n================ UNANSWERABLE QUERIES (reported separately) ================")
    for (strategy, method), (scores, chunks) in all_scores.items():
        for row, q in zip(scores, items):
            if q["doc"] is None:
                i = int(np.argsort(-row, kind="stable")[0])
                print(f"  {NAMES[strategy]:<22} {method:<9} {q['query']!r}: top-1 {chunks[i]['id']} "
                      f"score={float(row[i]):.3f}")

    print("\n================ FAILURES (answerable, relevant chunk not at rank 1) ==========")
    for fl in failures:
        print("  ", fl)
    with open(OUT_DIR / "failures.json", "w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2)
    print("\nWrote to outputs/: retrieval_log.csv (", len(logs), "rows ), results_summary.csv, failures.json, "
          "selected_config.json")


if __name__ == "__main__":
    main()
