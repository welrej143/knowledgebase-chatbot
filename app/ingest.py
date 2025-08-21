# app/ingest.py
from __future__ import annotations
import os
from typing import Dict, Any, List
from .rag import ingest_folder

DATA_DIR = "./data"

def _scan_files(data_dir: str) -> Dict[str, List[str]]:
    pdfs, docs, docxs, xlsxs, others = [], [], [], [], []
    for name in sorted(os.listdir(data_dir)):
        low = name.lower()
        if low.endswith(".pdf"):
            pdfs.append(name)
        elif low.endswith(".doc"):
            docs.append(name)
        elif low.endswith(".docx"):
            docxs.append(name)
        elif low.endswith(".xlsx"):
            xlsxs.append(name)
        else:
            others.append(name)
    return {"pdf": pdfs, "doc": docs, "docx": docxs, "xlsx": xlsxs, "other": others}

def rebuild_from_data() -> Dict[str, Any]:
    if not os.path.isdir(DATA_DIR):
        print(f"[WARN] No data dir: {DATA_DIR}")
        return {"chunks_added": 0, "note": f"No data dir: {DATA_DIR}"}

    groups = _scan_files(DATA_DIR)

    def _print_group(title: str, items: List[str]):
        if not items: return
        print(f"Found {len(items)} {title}:")
        for n in items:
            print(f" - {n}")

    _print_group("PDF(s)", groups["pdf"])
    _print_group(".doc (will convert → PDF)", groups["doc"])
    _print_group(".docx", groups["docx"])
    _print_group(".xlsx", groups["xlsx"])
    if groups["other"]:
        print(f"[INFO] Skipping unsupported types: {', '.join(groups['other'])}")

    res = ingest_folder(DATA_DIR)
    print(res)
    return res

def save_and_ingest_uploads(files) -> Dict[str, Any]:
    """
    Save UploadFile(s) into DATA_DIR, run ingestion once,
    return dict with 'chunks_added' and 'uploaded'.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    saved = []
    for f in files:
        path = os.path.join(DATA_DIR, f.filename)
        with open(path, "wb") as w:
            # f.file may be a bytes-like obj or a file-like object
            if hasattr(f.file, "read"):
                w.write(f.file.read())
            else:
                w.write(f.file)
        saved.append(f.filename)

    out = ingest_folder(DATA_DIR)
    out["uploaded"] = saved
    return out

if __name__ == "__main__":
    rebuild_from_data()
