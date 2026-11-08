# Lab 8 - runs the retrieval evaluation of the document chatbot
import platform

from src import lab8_guided_limits, lab8_guided_metrics, lab8_retrieval


def section(title):
    print("=" * 72)
    print(title)
    print("=" * 72)


def main():
    print("AI Lab 8: Document Retrieval Chatbot Evaluation | Rasikh Ali (42365)")
    print("Python %s, all-MiniLM-L6-v2 on CPU" % platform.python_version())
    print()
    section("LAB 8 / 6.5  Guided example: Hit@3 and MRR@3")
    lab8_guided_metrics.main()
    print()
    section("LAB 8 / 6.4  Guided example: CPU setup and model input limits")
    lab8_guided_limits.main()
    print()
    section("LAB 8 / Task  3 chunking strategies x TF-IDF / embeddings")
    lab8_retrieval.main()


if __name__ == "__main__":
    main()
