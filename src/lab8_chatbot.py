"""Lab 8 - interactive evidence assistant using the configuration selected on dev data.
Run: python src/lab8_chatbot.py   (after lab8_retrieval.py has written outputs/selected_config.json)
"""
import json
import sys
from pathlib import Path

try:
    from .lab8_chunks import make_chunks
    from .lab8_search import Retriever
except ImportError:                     # so it also runs as: python src/<file>.py
    from lab8_chunks import make_chunks
    from lab8_search import Retriever

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def main():
    with open(DATA_DIR / "corpus.json", encoding="utf-8") as f:
        docs = json.load(f)
    config = {"strategy": "sentence", "backend": "embedding"}
    selected = ROOT / "outputs" / "selected_config.json"
    if selected.exists():
        config.update(json.loads(selected.read_text(encoding="utf-8")))
    engine = Retriever(make_chunks(docs, config["strategy"]), config["backend"])
    label = "sentence-aware" if config["strategy"] == "sentence" else f"40-word window, overlap {config['strategy']}"
    print(f"Evidence assistant ({label} chunks, {config['backend']}). Type exit to stop.")
    while True:
        try:
            query = input("Question: ")
            if not sys.stdin.isatty():
                print(query)  # echo piped input so saved transcripts are readable
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() == "exit":
            break
        if not query.strip():
            print("Please enter a question.")
            continue
        hits = engine.search(query)
        if not hits:
            print("No candidate evidence; rephrase or contact a human.")
            continue
        print("Candidate evidence; relevance must be checked:")
        for number, (r, score) in enumerate(hits, 1):
            print(number, r["id"], f"{score:.3f}", "characters", r["start"], r["end"],
                  "|", r["text"][:70] + "...")
        try:
            chosen = input("Choose 1-3 to inspect; Enter if none apply: ")
            if not sys.stdin.isatty():
                print(chosen)
        except (EOFError, KeyboardInterrupt):
            break
        if not chosen.isdigit() or not 1 <= int(chosen) <= len(hits):
            print("No answer selected. Rephrase or contact a human.")
            continue
        r, score = hits[int(chosen) - 1]
        print("Retrieved passage:", r["text"])
        print("Source:", r["doc"], "characters", r["start"], r["end"])
        print("(Candidate evidence only - not a verified answer.)")


if __name__ == "__main__":
    main()
