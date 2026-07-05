import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "documents")

def load_documents():
    documents = []
    filenames = os.listdir(DATA_DIR)
    sorted_filenames = sorted(filenames)  # Sort the filenames alphabetically
    for filename in sorted_filenames:
        if not filename.endswith(".md"):
            continue
        path = os.path.join(DATA_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        document = {"source": filename, "text": text}
        documents.append(document)
    return documents


def print_header(title):
    print("\n" + "=" * len(title))
    print(title)
    print("=" * len(title))