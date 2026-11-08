"""Lab 8 - guided example 6.5: retrieval evaluation with Hit@3 and MRR@3.
Run: python src/lab8_guided_metrics.py
"""


def retrieval_metrics(rankings, relevant):
    hits, rr = [], []
    for ids, good in zip(rankings, relevant):
        ranks = [r for r, chunk_id in enumerate(ids[:3], 1) if chunk_id in good]
        hits.append(bool(ranks))
        rr.append(1 / ranks[0] if ranks else 0)
    return sum(hits) / len(hits), sum(rr) / len(rr)


def main():
    rankings = [["a", "b", "c"], ["c", "a", "b"], ["x", "y", "z"]]
    relevant = [{"a"}, {"a"}, {"a"}]
    hit, mrr = retrieval_metrics(rankings, relevant)
    print(f"Hit@3={hit:.3f}; MRR@3={mrr:.3f}")
    assert abs(hit - 2 / 3) < 1e-9 and abs(mrr - 0.5) < 1e-9

    # Experiment: relevant item at rank 4 must count as a miss at cutoff 3
    hit4, mrr4 = retrieval_metrics([["x", "y", "z", "a"]], [{"a"}])
    print(f"Rank-4 case: Hit@3={hit4:.3f}; MRR@3={mrr4:.3f}")
    assert hit4 == 0 and mrr4 == 0


if __name__ == "__main__":
    main()
