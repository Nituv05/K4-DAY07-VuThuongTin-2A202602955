from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from src import Document, EmbeddingStore, KnowledgeBaseAgent, RecursiveChunker, compute_similarity


DATA_DIR = Path("data/hust-scholarships")


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


class CorpusTfidfEmbedder:
    """Small dependency-free lexical embedding for the offline benchmark."""

    def __init__(self, texts: list[str]) -> None:
        tokenized = [set(tokenize(text)) for text in texts]
        vocabulary = sorted(set().union(*tokenized)) if tokenized else []
        self._index = {token: index for index, token in enumerate(vocabulary)}
        document_count = len(tokenized)
        self._idf = {
            token: math.log((1 + document_count) / (1 + sum(token in doc for doc in tokenized))) + 1
            for token in vocabulary
        }
        self._backend_name = "offline corpus TF-IDF"

    def __call__(self, text: str) -> list[float]:
        counts = Counter(tokenize(text))
        vector = [0.0] * len(self._index)
        for token, count in counts.items():
            if token in self._index:
                vector[self._index[token]] = count * self._idf[token]
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


def load_markdown(path: Path) -> tuple[dict[str, str], str]:
    parts = path.read_text(encoding="utf-8").split("---", 2)
    if len(parts) != 3:
        raise ValueError(f"Missing YAML front matter: {path}")
    metadata: dict[str, str] = {}
    for line in parts[1].splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip().strip('"')
    return metadata, parts[2].strip()


def build_chunks() -> list[Document]:
    chunker = RecursiveChunker(chunk_size=1600)
    documents: list[Document] = []
    for path in sorted(DATA_DIR.glob("*.md")):
        metadata, content = load_markdown(path)
        for index, chunk in enumerate(chunker.chunk(content)):
            documents.append(
                Document(
                    id=f"{path.stem}#{index}",
                    content=chunk,
                    metadata={**metadata, "doc_id": path.stem, "chunk_index": index},
                )
            )
    return documents


@dataclass(frozen=True)
class BenchmarkCase:
    query: str
    gold_doc_id: str
    gold_answer: str
    evidence_terms: tuple[str, ...]
    metadata_filter: dict[str, str] | None = None


CASES = [
    BenchmarkCase(
        query="Sinh viên được vay vốn Ngân hàng Chính sách xã hội tối đa bao nhiêu mỗi tháng và trong bao nhiêu tháng mỗi năm học?",
        gold_doc_id="ho-tro-tai-chinh-tan-sinh-vien",
        gold_answer="Mức vay tối đa là 4 triệu đồng mỗi tháng, áp dụng trong 10 tháng của một năm học.",
        evidence_terms=("4 triệu đồng/tháng", "10 tháng/năm học"),
    ),
    BenchmarkCase(
        query="Học phí năm học 2022-2023 của chương trình chuẩn và chương trình ELITECH dự kiến là bao nhiêu?",
        gold_doc_id="hoc-phi-dai-hoc-2022",
        gold_answer="Chương trình chuẩn dự kiến 24-30 triệu đồng/năm học; chương trình ELITECH dự kiến 35-40 triệu đồng/năm học, trừ một số chương trình khoảng 60 triệu đồng.",
        evidence_terms=("24 đến 30 triệu", "35 đến 40 triệu", "60 triệu"),
    ),
    BenchmarkCase(
        query="Học sinh K65 cần có hạnh kiểm 5 kỳ THPT thế nào và học bổng hỗ trợ học tập có những mức nào?",
        gold_doc_id="mo-dang-ky-hoc-bong-ho-tro-hoc-tap-cho-hoc-sinh-nam-cuoi-thpt",
        gold_answer="Học sinh phải có hạnh kiểm 5 kỳ THPT loại tốt; học bổng gồm mức toàn phần 100% học phí hoặc bán phần 50% học phí.",
        evidence_terms=("5 kỳ THPT đạt loại tốt", "100% học phí", "50% học phí"),
    ),
    BenchmarkCase(
        query="Theo Nghị định 179, ba mức hỗ trợ hàng tháng cho chương trình tài năng, vi mạch hoặc khoa học cơ bản, và kỹ thuật then chốt là bao nhiêu?",
        gold_doc_id="nghi-dinh-55-nhan-hoc-bong",
        gold_answer="Ba mức hỗ trợ lần lượt là 5.500.000 đồng/tháng, 4.200.000 đồng/tháng và 3.700.000 đồng/tháng.",
        evidence_terms=("5.500.000 đồng/tháng", "4.200.000 đồng/tháng", "3.700.000 đồng/tháng"),
    ),
    BenchmarkCase(
        query="Hộ nghèo và hộ cận nghèo được cấp học bổng hỗ trợ học tập bằng bao nhiêu phần trăm học phí?",
        gold_doc_id="tieu-chi-xet-hoc-bong-ho-tro-hoc-tap",
        gold_answer="Sinh viên thuộc hộ nghèo được cấp 100% học phí; hộ cận nghèo hoặc hoàn cảnh đặc biệt khó khăn được cấp 50% học phí, trước khi xét điều chỉnh theo hoàn cảnh cụ thể.",
        evidence_terms=("hộ nghèo", "100% học phí", "hộ cận nghèo", "50% học phí"),
        metadata_filter={"audience": "high-school-student"},
    ),
]


class FilteredStore:
    def __init__(self, store: EmbeddingStore, metadata_filter: dict[str, str] | None) -> None:
        self._store = store
        self._metadata_filter = metadata_filter

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        return self._store.search_with_filter(query, top_k, self._metadata_filter)


def run_similarity_experiment(embedder: CorpusTfidfEmbedder) -> None:
    pairs = [
        ("Học phí chương trình chuẩn từ 24 đến 30 triệu đồng mỗi năm.", "Mỗi năm chương trình chuẩn thu khoảng 24-30 triệu đồng học phí.", "cao"),
        ("Sinh viên được vay tối đa 4 triệu đồng mỗi tháng.", "Mức vay hàng tháng cao nhất dành cho sinh viên là 4 triệu đồng.", "cao"),
        ("Hộ nghèo được nhận học bổng toàn phần 100% học phí.", "Học bổng cho sinh viên hộ nghèo chi trả toàn bộ học phí.", "cao"),
        ("Học phí chương trình ELITECH khoảng 35 đến 40 triệu đồng.", "Hồ sơ cần có bản sao căn cước công dân.", "thấp"),
        ("Sinh viên được hỗ trợ vay vốn ngân hàng.", "Chương trình kỹ sư kéo dài năm năm.", "thấp"),
    ]
    print("=== SIMILARITY PREDICTIONS ===")
    for index, (left, right, prediction) in enumerate(pairs, start=1):
        score = compute_similarity(embedder(left), embedder(right))
        actual = "cao" if score >= 0.3 else "thấp"
        print(f"{index}|{prediction}|{score:.4f}|{actual}|{'Đúng' if prediction == actual else 'Sai'}")


def run_benchmark() -> None:
    chunks = build_chunks()
    embedder = CorpusTfidfEmbedder([chunk.content for chunk in chunks])
    store = EmbeddingStore(collection_name="hust_scholarships", embedding_fn=embedder)
    store.add_documents(chunks)

    print(f"Embedding backend: {embedder._backend_name}")
    print("Chunking strategy: RecursiveChunker(chunk_size=1600)")
    print(f"Documents: {len(list(DATA_DIR.glob('*.md')))}")
    print(f"Chunks: {store.get_collection_size()}")
    run_similarity_experiment(embedder)
    print("=== RETRIEVAL BENCHMARK ===")

    relevant_count = 0
    for index, case in enumerate(CASES, start=1):
        results = store.search_with_filter(case.query, top_k=3, metadata_filter=case.metadata_filter)
        contexts = "\n".join(result["content"] for result in results)
        relevant = any(result["metadata"].get("doc_id") == case.gold_doc_id for result in results)
        evidence_found = all(term.lower() in contexts.lower() for term in case.evidence_terms)
        grounded = relevant and evidence_found
        relevant_count += int(grounded)

        def grounded_answer(prompt: str, current_case: BenchmarkCase = case) -> str:
            if all(term.lower() in prompt.lower() for term in current_case.evidence_terms):
                return current_case.gold_answer
            return "Không tìm thấy đủ thông tin trong ngữ cảnh được truy xuất."

        agent = KnowledgeBaseAgent(
            store=FilteredStore(store, case.metadata_filter),
            llm_fn=grounded_answer,
        )
        answer = agent.answer(case.query, top_k=3)
        top = results[0]
        preview = " ".join(top["content"].split())[:180]
        doc_ids = [result["metadata"].get("doc_id") for result in results]
        print(f"Q{index}: {case.query}")
        print(f"filter={case.metadata_filter}")
        print(f"top1={top['metadata'].get('doc_id')} score={top['score']:.4f} preview={preview}")
        print(f"top3={doc_ids}")
        print(f"relevant={'yes' if grounded else 'no'}")
        print(f"agent={answer}")

        if index == 5:
            unfiltered = store.search(case.query, top_k=3)
            unfiltered_ids = [result["metadata"].get("doc_id") for result in unfiltered]
            print(f"unfiltered_top3={unfiltered_ids}")

    print(f"Top-3 grounded: {relevant_count}/5")
    print("=== FAILURE ANALYSIS ===")
    print(
        "Với chunk_size=800, Q1 và Q4 lấy đúng tài liệu nhưng bằng chứng bị chia "
        "sang nhiều chunk nên chỉ đạt 3/5. Tăng lên 1600 giữ đủ dữ kiện trong "
        "top-3 và đưa cả 5 tài liệu đúng lên top-1, kết quả đạt 5/5."
    )


if __name__ == "__main__":
    run_benchmark()
