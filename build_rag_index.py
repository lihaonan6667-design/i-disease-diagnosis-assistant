#!/usr/bin/env python3
"""从 corpus/ 构建 FAISS 向量索引到 rag_data/。需要：有效 AI_API_KEY、faiss-cpu、numpy。"""
import sys

from rag_service import embed_texts_batched, load_all_chunks, save_faiss_index


def main() -> int:
    chunks = load_all_chunks()
    if not chunks:
        print("未找到语料：请确认 corpus/ 下存在 .txt 文件")
        return 1
    texts = [c["text"] for c in chunks]
    print(f"共 {len(texts)} 条切块，开始请求 embeddings …")
    embeddings = embed_texts_batched(texts)
    if len(embeddings) != len(chunks):
        print("嵌入未完成，未写入索引")
        return 1
    if not save_faiss_index(chunks, embeddings):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
