"""本地语料 RAG：切块、DashScope OpenAI 兼容 embeddings、FAISS 检索。"""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests

from config import Config

_EMBED_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"

_index = None
_chunks_meta: list[dict[str, Any]] | None = None
_load_lock = threading.Lock()


def _faiss_write_index_unicode_safe(index, dest: Path) -> None:
    """Windows 下路径含中文时 faiss.write_index 会 fopen 失败，先写到临时目录再移动。"""
    import faiss

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".faiss")
    os.close(fd)
    try:
        faiss.write_index(index, tmp)
        shutil.move(tmp, str(dest))
    except BaseException:
        if os.path.isfile(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass
        raise


def _faiss_read_index_unicode_safe(src: Path):
    """同上，读取时经 ASCII 临时路径再交给 faiss。"""
    import faiss

    fd, tmp = tempfile.mkstemp(suffix=".faiss")
    os.close(fd)
    try:
        shutil.copy2(Path(src), tmp)
        return faiss.read_index(tmp)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _corpus_dir() -> Path:
    return Path(Config.CORPUS_DIR)


def _rag_data_dir() -> Path:
    return Path(Config.RAG_DATA_DIR)


def parse_corpus_file(path: Path) -> tuple[dict[str, str], str]:
    """解析 #SOURCE_* 头与正文。"""
    raw = path.read_text(encoding="utf-8")
    lines = raw.split("\n")
    meta = {"title": "", "ref": "", "domain": "", "lang": "zh-CN"}
    i = 0
    header_re = re.compile(
        r"^#(SOURCE_TITLE|SOURCE_URL_OR_REF|DOMAIN_HINT|LANG):\s*(.*)$"
    )
    while i < len(lines):
        line = lines[i]
        m = header_re.match(line.strip())
        if m:
            key, val = m.group(1), m.group(2).strip()
            if key == "SOURCE_TITLE":
                meta["title"] = val
            elif key == "SOURCE_URL_OR_REF":
                meta["ref"] = val
            elif key == "DOMAIN_HINT":
                meta["domain"] = val
            elif key == "LANG":
                meta["lang"] = val
            i += 1
            continue
        if line.startswith("#") and ":" in line:
            i += 1
            continue
        break
    body = "\n".join(lines[i:]).strip()
    return meta, body


def chunk_text(body: str, chunk_size: int, overlap: int) -> list[str]:
    if not body:
        return []
    size = max(80, chunk_size)
    ov = max(0, min(overlap, size // 2))
    chunks: list[str] = []
    start = 0
    n = len(body)
    while start < n:
        end = min(start + size, n)
        piece = body[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = end - ov
    return chunks


def load_all_chunks() -> list[dict[str, Any]]:
    """扫描 corpus 目录下 txt，返回带 text 与元数据的 chunk 列表。"""
    root = _corpus_dir()
    if not root.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.txt")):
        if path.name.lower() == "meta_sources.txt":
            continue
        file_meta, body = parse_corpus_file(path)
        for piece in chunk_text(body, Config.RAG_CHUNK_SIZE, Config.RAG_CHUNK_OVERLAP):
            out.append(
                {
                    "text": piece,
                    "title": file_meta["title"] or path.stem,
                    "ref": file_meta["ref"],
                    "domain": file_meta["domain"],
                    "source_file": path.name,
                    "lang": file_meta["lang"],
                }
            )
    return out


def embed_texts(texts: list[str]) -> list[list[float]]:
    """单次请求最多 10 条 input（DashScope 兼容接口）。"""
    if not texts:
        return []
    key = (Config.AI_API_KEY or "").strip()
    if not key or key == "your-api-key-here-please-set-in-env":
        print("RAG: 未配置有效 AI_API_KEY，跳过向量请求")
        return []
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    payload = {"model": Config.EMBEDDING_MODEL, "input": texts}
    try:
        r = requests.post(_EMBED_URL, headers=headers, json=payload, timeout=90)
        if r.status_code != 200:
            print(f"RAG embeddings HTTP {r.status_code}: {r.text[:500]}")
            return []
        data = r.json()
        items = data.get("data") or []
        items.sort(key=lambda x: x.get("index", 0))
        return [it["embedding"] for it in items if isinstance(it.get("embedding"), list)]
    except Exception as e:
        print(f"RAG embeddings 异常: {e}")
        return []


def embed_texts_batched(all_texts: list[str], batch_size: int = 10) -> list[list[float]]:
    embeddings: list[list[float]] = []
    for i in range(0, len(all_texts), batch_size):
        batch = all_texts[i : i + batch_size]
        got = embed_texts(batch)
        if len(got) != len(batch):
            print(f"RAG: 批次嵌入数量不匹配 ({len(got)}/{len(batch)})，索引可能不完整")
            break
        embeddings.extend(got)
        time.sleep(0.08)
    return embeddings


def save_faiss_index(chunks: list[dict[str, Any]], embeddings: list[list[float]]) -> bool:
    """写入 rag_data/index.faiss 与 meta.json。"""
    import faiss

    if len(chunks) != len(embeddings) or not chunks:
        print("RAG: chunks 与 embeddings 数量不一致或为空")
        return False
    out = _rag_data_dir()
    out.mkdir(parents=True, exist_ok=True)
    xb = np.array(embeddings, dtype=np.float32)
    faiss.normalize_L2(xb)
    dim = xb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(xb)
    _faiss_write_index_unicode_safe(index, out / "index.faiss")
    meta = {
        "embedding_model": Config.EMBEDDING_MODEL,
        "chunks": [
            {
                "text": c["text"],
                "title": c.get("title") or "",
                "ref": c.get("ref") or "",
                "domain": c.get("domain") or "",
                "source_file": c.get("source_file") or "",
                "lang": c.get("lang") or "zh-CN",
            }
            for c in chunks
        ],
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    print(f"RAG: 已写入 {out}，向量数={index.ntotal}，维度={dim}")
    return True


def _lazy_load_index():
    global _index, _chunks_meta
    with _load_lock:
        if _index is not None or _chunks_meta is not None:
            return
        import faiss

        idx_path = _rag_data_dir() / "index.faiss"
        meta_path = _rag_data_dir() / "meta.json"
        if not idx_path.is_file() or not meta_path.is_file():
            _index = None
            _chunks_meta = []
            return
        try:
            _index = _faiss_read_index_unicode_safe(idx_path)
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            _chunks_meta = data.get("chunks") or []
            if _index.ntotal != len(_chunks_meta):
                print(
                    f"RAG: 警告 index.ntotal={_index.ntotal} 与 meta 条数 {len(_chunks_meta)} 不一致"
                )
        except Exception as e:
            print(f"RAG: 加载索引失败 {e}")
            _index = None
            _chunks_meta = []


def search_chunks(query: str, top_k: int | None = None) -> list[dict[str, Any]]:
    """向量检索，返回 chunk 字典列表（含完整 text）。"""
    import faiss

    _lazy_load_index()
    if _index is None or not _chunks_meta:
        return []
    q = (query or "").strip()
    if not q:
        return []
    k = top_k if top_k is not None else Config.RAG_TOP_K
    k = max(1, min(k, _index.ntotal))
    emb = embed_texts([q])
    if not emb:
        return []
    qv = np.array([emb[0]], dtype=np.float32)
    faiss.normalize_L2(qv)
    _, indices = _index.search(qv, k)
    rows: list[dict[str, Any]] = []
    for idx in indices[0]:
        if idx < 0 or idx >= len(_chunks_meta):
            continue
        rows.append(dict(_chunks_meta[idx]))
    return rows


def format_rag_context(chunks: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for i, c in enumerate(chunks, 1):
        title = (c.get("title") or "医学资料").strip()
        ref = (c.get("ref") or "").strip()
        head = f"【资料{i}】{title}"
        if ref:
            head += f"（{ref}）"
        body = (c.get("text") or "").strip()
        parts.append(f"{head}\n{body}")
    return "\n\n".join(parts)


def retrieve_for_query(query: str) -> tuple[str, list[dict[str, Any]]]:
    """返回 (注入模型的上下文文本, 供 API/前端展示的精简来源列表)。"""
    chunks = search_chunks(query)
    if not chunks:
        return "", []
    ctx = format_rag_context(chunks)
    sources: list[dict[str, Any]] = []
    for c in chunks:
        t = (c.get("text") or "").strip()
        excerpt = t if len(t) <= 320 else t[:320] + "…"
        sources.append(
            {
                "title": c.get("title") or "",
                "ref": c.get("ref") or "",
                "domain": c.get("domain") or "",
                "source_file": c.get("source_file") or "",
                "excerpt": excerpt,
            }
        )
    return ctx, sources


def rag_prefix_for_prompt(rag_context: str) -> str:
    if not rag_context:
        return ""
    return (
        "【知识库参考摘录】以下段落来自本地医学资料检索，请优先据此归纳；"
        "若资料未覆盖患者情况请明确说明「资料未涉及」，勿编造出处。\n\n"
        f"{rag_context}\n\n---\n\n"
    )
