# AI Lab 8 - Document Retrieval Chatbot Evaluation

Lab 8 for the AI lab course (CAI601400), NLP fundamentals part 3. I evaluate a small document retrieval chatbot on six synthetic help-desk documents: 3 chunking strategies (40-word windows with overlap 0 and 10, sentence-aware packing) times 2 retrieval methods (TF-IDF and all-MiniLM-L6-v2 embeddings), on labeled dev and test questions plus two unanswerable ones. The configuration is chosen on dev data only and then used by an interactive evidence assistant.

## Files

- `main.py` runs the guided examples and the full evaluation
- `src/lab8_retrieval.py` the evaluation: token-length audit, Hit@3, MRR@3, runtime, top-3 per query, failures, unanswerable queries and the dev-only selection
- `src/lab8_chatbot.py` interactive evidence assistant that uses the selected configuration and shows the source chunk with its character offsets
- `src/lab8_guided_metrics.py` guided example 6.5, Hit@3 and MRR@3
- `src/lab8_guided_limits.py` guided example 6.4, CPU setup and model input limits
- `src/lab8_chunks.py` the chunking functions from Lab 7
- `src/lab8_search.py` the `Retriever` class from the manual
- `src/lab8_dataset.py` writes `data/corpus.json` and `data/queries.json`
- `data/` 6 synthetic documents (with an archived 2023 version of the late policy as a distractor) and 10 labeled queries
- `outputs/output.txt` output of my run
- `outputs/results_summary.csv`, `outputs/retrieval_log.csv`, `outputs/failures.json`, `outputs/selected_config.json` files written by the evaluation
- `outputs/chatbot_transcript.txt` a session with the evidence assistant
- `report/` the written report (DOCX and PDF). Labs 6, 7 and 8 were written up in one report; this lab is sections 4 to 6

## How to run

Python 3.10+ is needed. The model `sentence-transformers/all-MiniLM-L6-v2` (about 90 MB) is downloaded the first time and runs on CPU.

```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
python src/lab8_chatbot.py       # interactive, type exit to stop
```

`data/` is already in the repo. To rebuild it: `python src/lab8_dataset.py --force`.

## Results

| Strategy | Method | Chunks | Dev Hit@3 | Dev MRR@3 | Test Hit@3 | Test MRR@3 |
|----------|--------|--------|-----------|-----------|------------|------------|
| window 40 / overlap 0 | TF-IDF | 31 | 0.50 | 0.500 | 0.25 | 0.125 |
| window 40 / overlap 0 | Embedding | 31 | 0.50 | 0.500 | 0.50 | 0.208 |
| window 40 / overlap 10 | TF-IDF | 39 | 0.25 | 0.250 | 0.25 | 0.125 |
| window 40 / overlap 10 | Embedding | 39 | 0.25 | 0.125 | 0.25 | 0.125 |
| sentence 40 | TF-IDF | 39 | 0.50 | 0.500 | 0.75 | 0.417 |
| sentence 40 | Embedding | 39 | 0.50 | 0.500 | 1.00 | 0.875 |

The selection rule was fixed before looking at the test set: best dev MRR@3, then dev Hit@3, then fewer chunks. Four configurations tie on dev, so the chunk count picks window 40 / overlap 0 + embedding, and that is what the chatbot uses. On the test questions sentence-aware chunks + embeddings did best (Hit@3 1.0, MRR@3 0.875), which is why I recommend it in the report. With only 4 test questions one question moves Hit@3 by 0.25, so the numbers are not very reliable.

Unanswerable questions (hostel fee, football) still get three candidates with scores close to real answers, so a single score threshold would not work here.

All documents are synthetic teaching text, not real university policy.

## Author

Rasikh Ali (42365)

MIT License, see LICENSE.
