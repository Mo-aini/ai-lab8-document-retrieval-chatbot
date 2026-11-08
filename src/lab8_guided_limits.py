"""Lab 8 - guided example 6.4: CPU setup and model input limits.
Run: python src/lab8_guided_limits.py
"""
from sentence_transformers import SentenceTransformer


def main():
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    for label, text in [("short", "Reset my password."),
                        ("long", "password " * 300),
                        ("identifier", "CS4944_42365_Lab6.pdf POST /v2/token/refresh"),
                        ("punctuation", "Wait... what?! (really) -- yes.")]:
        ids = model.tokenizer(text, truncation=False)["input_ids"]
        fits = len(ids) <= model.max_seq_length
        print(label, "| whitespace words:", len(text.split()), "| model tokens:", len(ids),
              "| limit:", model.max_seq_length, "| fits:", fits)
        if label in ("identifier", "punctuation"):
            print("   subwords:", model.tokenizer.tokenize(text))


if __name__ == "__main__":
    main()
