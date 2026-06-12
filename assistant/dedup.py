import hashlib, json, os

HASH_STORE = "data/sidecar/_hashes.json"

def file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()

def _load() -> dict:
    if not os.path.exists(HASH_STORE):
        return {}
    with open(HASH_STORE, "r") as f:
        return json.load(f)

def is_duplicate(path: str) -> bool:
    return file_hash(path) in _load().values()

def record(path: str, document_id: str) -> None:
    os.makedirs(os.path.dirname(HASH_STORE), exist_ok=True)
    store = _load()
    store[document_id] = file_hash(path)
    with open(HASH_STORE, "w") as f:
        json.dump(store, f, indent=2)