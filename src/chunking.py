from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text.strip():
            return []

        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])(?:[ \t]+|\n+)", text.strip())
            if sentence.strip()
        ]
        chunks: list[str] = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            chunk_group = sentences[i : i + self.max_sentences_per_chunk]
            chunks.append(" ".join(chunk_group).strip())
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if len(current_text) <= self.chunk_size:
            return [current_text]

        # Điểm dừng: Nếu không còn ký tự phân tách nào để thử
        if not remaining_separators:
            return [current_text]

        # Lấy ký tự phân tách có ưu tiên cao nhất hiện tại ra để thử
        separator = remaining_separators[0]
        next_separators = remaining_separators[1:]

        # Thực hiện cắt văn bản dựa trên separator hiện tại
        # Nếu separator là chuỗi rỗng "", cắt theo từng ký tự
        if separator == "":
            parts = list(current_text)
        else:
            parts = current_text.split(separator)

        final_chunks = []
        current_chunk = []
        current_length = 0

        for part in parts:
            # Bù lại ký tự phân tách đã bị mất khi dùng hàm split()
            # (Không cần bù nếu đó là phần tử đầu tiên hoặc dùng separator rỗng)
            part_with_sep = separator + part if current_chunk and separator != "" else part
            part_len = len(part_with_sep)

            # Trường hợp 1: Bản thân phần này quá lớn (lớn hơn chunk_size)
            # Buộc phải đẩy xuống tầng đệ quy sâu hơn (dùng separator tiếp theo)
            if len(part) > self.chunk_size:
                # Nếu trước đó đang gom dở một chunk, hãy đóng nó lại và lưu vào danh sách
                if current_chunk:
                    final_chunks.append("".join(current_chunk))
                    current_chunk = []
                    current_length = 0

                # Gọi đệ quy để xử lý phần văn bản quá khổ này
                sub_chunks = self._split(part, next_separators)
                final_chunks.extend(sub_chunks)

            # Trường hợp 2: Nếu thêm phần này vào mà vượt quá chunk_size
            elif current_length + part_len > self.chunk_size:
                # Đóng chunk hiện tại lại
                final_chunks.append("".join(current_chunk))
                # Bắt đầu một chunk mới với phần văn bản hiện tại (bỏ dấu phân tách nối đầu vì nó là phần tử đầu chunk mới)
                current_chunk = [part]
                current_length = len(part)

            # Trường hợp 3: Thêm vào vẫn vừa sức chứa của chunk_size
            else:
                current_chunk.append(part_with_sep)
                current_length += part_len

        # Đóng chunk cuối cùng nếu còn sót lại dữ liệu
        if current_chunk:
            final_chunks.append("".join(current_chunk))

        return final_chunks


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))

    # Tính độ dài của từng vectơ (sử dụng sum để cộng dồn các bình phương)
    mag_a = math.sqrt(sum(x**2 for x in vec_a))
    mag_b = math.sqrt(sum(x**2 for x in vec_b))

    # Trả về kết quả nếu cả 2 độ dài đều lớn hơn 0, ngược lại trả về 0.0
    return dot_product / (mag_a * mag_b) if mag_a > 0 and mag_b > 0 else 0.0


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        overlap = min(50, max(0, chunk_size - 1))
        strategies = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size, overlap=overlap),
            "by_sentences": SentenceChunker(max_sentences_per_chunk=3),
            "recursive": RecursiveChunker(chunk_size=chunk_size),
        }

        comparison = {}
        for name, chunker in strategies.items():
            chunks = chunker.chunk(text)
            comparison[name] = {
                "count": len(chunks),
                "avg_length": sum(len(chunk) for chunk in chunks) / len(chunks) if chunks else 0.0,
                "chunks": chunks,
            }

        return comparison
