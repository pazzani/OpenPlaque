from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

CACHE_VERSION = 'cache-controls-v1'

def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def hash_array(a) -> str:
    return hash_bytes(np.ascontiguousarray(a).tobytes())

def hash_df(df: pd.DataFrame) -> str:
    return hash_bytes(df.to_csv(index=False).encode('utf-8'))

def save_json(obj, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding='utf-8')

def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))
