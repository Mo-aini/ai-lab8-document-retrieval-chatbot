"""Lab 8 - fixed word windows vs sentence-aware packing, with source offsets.

The functions below follow the lab manual's reference solution for the chunking
task and are reused by the other scripts in this repo. The comparison and the
boundary tests run only when this file is executed directly or from main.py.
Run: python src/lab8_chunks.py   (needs data/corpus.json, see lab8_dataset.py)
"""
import json
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def word_spans(text):
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def windows(text, size=40, overlap=10, offset=0):
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Require size > 0 and 0 <= overlap < size")
    spans = word_spans(text)
    result = []
    for i in range(0, len(spans), size - overlap):
        j = min(i + size, len(spans))
        result.append((offset + spans[i][0], offset + spans[j - 1][1]))
        if j == len(spans):
            break
    return result


def sentence_cuts(text):
    # Teaching splitter: punctuation followed by whitespace (fails on "Dr." and "e.g.").
    return [0] + [m.end() for m in re.finditer(r"[.!?]\s+", text)] + [len(text)]


def sentence_chunks(text, size=40):
    if size <= 0:
        raise ValueError("size must be positive")
    cuts = sentence_cuts(text)
    out, pending, count = [], None, 0
    for a, b in zip(cuts, cuts[1:]):
        segment = text[a:b]
        n = len(segment.split())
        if not n:
            continue
        if n > size:
            if pending is not None:
                out.append(pending)
            out.extend(windows(segment, size, 0, a))   # fallback for a long sentence
            pending, count = None, 0
        elif pending is not None and count + n <= size:
            pending, count = (pending[0], b), count + n
        else:
            if pending is not None:
                out.append(pending)
            pending, count = (a, b), n
    if pending is not None:
        out.append(pending)
    # strip trailing whitespace so that text == source[start:end] is tidy
    return [(a, a + len(text[a:b].rstrip())) for a, b in out]


def make_chunks(docs, strategy, size=40):
    result = []
    for doc_id, text in docs.items():
        spans = (sentence_chunks(text, size) if strategy == "sentence"
                 else windows(text, size, int(strategy)))
        for i, (a, b) in enumerate(spans):
            result.append(dict(id=f"{doc_id}:{strategy}:{i}", doc=doc_id,
                               start=a, end=b, text=text[a:b]))
    return result


# Rule/exception (or warning/step) spans labeled by hand BEFORE chunking.
LABELED = {
    "handbook_late": "Late submission is penalized at 10% of the marks per day. However, a student "
                     "with a documented illness may request an extension if medical evidence is "
                     "sent to the module coordinator before the deadline.",
    "it_password": "If you forget your password, reset a forgotten password using the recovery "
                   "link on the login page; the link is sent to the personal email address "
                   "registered during enrolment and expires after 30 minutes.",
    "printer_help": "Warning: switch the printer off and wait for the fuser to cool before opening "
                    "any internal panel. For blank printer pages, check ink levels and run the "
                    "cleaning utility from the printer menu, then print a test page afterwards.",
}


def load_chunk_docs():
    path = DATA_DIR / "corpus.json"
    if path.exists():
        corpus = json.loads(path.read_text(encoding="utf-8"))
    else:
        try:
            from .lab8_dataset import DOCS as corpus
        except ImportError:
            from lab8_dataset import DOCS as corpus
    return {k: corpus[k] for k in LABELED}


def run_tests():
    assert windows("") == []
    assert sentence_chunks("") == []
    assert windows("short text") == [(0, 10)]
    assert sentence_chunks("Short document. Two sentences.") == [(0, 30)]
    for bad in [(2, 2), (2, 5), (0, 0), (5, -1)]:
        try:
            windows("a b c", *bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid size/overlap accepted: {bad}")
    long = " ".join(["word"] * 95) + "."
    spans = sentence_chunks(long)
    assert len(spans) == 3 and all(len(long[a:b].split()) <= 40 for a, b in spans)
    docs = {"example": "Reset your password. Keep recovery codes private."}
    for strategy in ["0", "10", "sentence"]:
        for c in make_chunks(docs, strategy):
            assert c["text"] == docs[c["doc"]][c["start"]:c["end"]] and c["text"].strip()
    print("Boundary tests passed: empty text, short document (shorter than budget), "
          "invalid overlap x4, 95-word sentence -> 3 fallback windows, offsets == source")


def compare(docs):
    original_words = sum(len(t.split()) for t in docs.values())
    print("\nDocuments:", {k: len(v.split()) for k, v in docs.items()},
          "total words:", original_words)
    print("\n%-10s %7s %9s %9s %9s %10s %s" % ("strategy", "chunks", "min_w", "mean_w",
                                              "max_w", "dup_words", "labeled spans kept whole"))
    for strategy, name in [("0", "window/0"), ("10", "window/10"), ("sentence", "sentence")]:
        chunks = make_chunks(docs, strategy)
        for c in chunks:
            assert c["text"] == docs[c["doc"]][c["start"]:c["end"]]
            assert len(c["text"].split()) <= 40
        sizes = [len(c["text"].split()) for c in chunks]
        dup = sum(sizes) - original_words
        kept = [d for d, span in LABELED.items()
                if any(span in c["text"] for c in chunks if c["doc"] == d)]
        print("%-10s %7d %9d %9.1f %9d %10d %d/3 %s" % (
            name, len(chunks), min(sizes), sum(sizes) / len(sizes), max(sizes), dup,
            len(kept), kept))
    for strategy in ["0", "10", "sentence"]:
        print(f"\n--- {strategy}: chunks of handbook_late (id, start, end, words, first/last words)")
        for c in make_chunks({"handbook_late": docs["handbook_late"]}, strategy):
            w = c["text"].split()
            print(f"  {c['id']:<26} {c['start']:>5} {c['end']:>5} {len(w):>3}  "
                  f"{' '.join(w[:5])} ... {' '.join(w[-4:])}")


def abbreviation_failure(docs):
    text = docs["handbook_late"]
    cuts = sentence_cuts(text)
    print("\nSentence-splitter failures (abbreviations):")
    for a, b in zip(cuts, cuts[1:]):
        seg = text[a:b].strip()
        if len(seg.split()) <= 3 or seg.startswith("Khan") or seg.startswith("a signed"):
            print(f"   [{a}:{b}] {seg!r}")
    longest = max((text[a:b] for a, b in zip(cuts, cuts[1:])), key=lambda s: len(s.split()))
    print(f"\nLongest sentence: {len(longest.split())} words (> 40 budget, uses window fallback)")


def main():
    run_tests()
    chunk_docs = load_chunk_docs()
    compare(chunk_docs)
    abbreviation_failure(chunk_docs)


if __name__ == "__main__":
    main()
