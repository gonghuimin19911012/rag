# 专业知识库问答:
# 处理数据pdf/docx，进入数据库和向量库，向量库名kgdata
from __future__ import annotations

import base64
import json
import os
import re
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable
# 自定义代码
from config import MYSQL_CONFIG,llm_model,api_key,llm_url,bge_model,collection_name
from embedmodel import BGEEmbedder, get_embed
from milvus_main import insert_batch

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover
    fitz = None

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

try:
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph
except ImportError:  # pragma: no cover
    Document = None
    Table = None
    Paragraph = None

# 初始化向量模型
embedder = BGEEmbedder(bge_model["path"]) #embedding=get_embed(,bge_model["path"])

class DocumentParser:
    """Parse PDF/DOCX into paragraph chunks (list[str])."""

    def __init__(self,max_chunk_size: int = 500,validate_text: bool = True,use_llm_image_parser: bool = True):
        self.max_chunk_size = max_chunk_size
        self._sentence_delimiters = "。"
        self.validate_text = validate_text
        self.use_llm_image_parser = use_llm_image_parser
        self.llm_model_name = llm_model
        self.llm_api_key = api_key
        self.llm_base_url = llm_url
        self._llm_client = None
        if OpenAI is not None and self.llm_api_key and self.llm_base_url:
            try:
                self._llm_client = OpenAI(api_key=self.llm_api_key,base_url=self.llm_base_url,)
            except Exception:
                self._llm_client = None

    def parse(self, file_path: str) -> list[str]:
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            units = self._parse_pdf_units(file_path)
        elif suffix == ".docx":
            units = self._parse_docx_units(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        units = self._drop_standalone_noise_units(units)
        units = self._merge_colon_ended_units(units)
        units = self._merge_table_title_with_next_table(units)
        units = self._merge_standalone_continuation_labels(units)
        units = self._merge_table_title_with_previous_table(units)
        #units = self._merge_pure_table_units_upward(units)
        units = self._split_units_by_section_boundaries(units)
        units = self._merge_unterminated_paragraphs(units)
        units = self._merge_short_units_downward(units)

        if not units and self.use_llm_image_parser and self.llm_model_name and self.llm_api_key and self.llm_base_url:
            units = self._parse_pdf_with_llm_fallback(file_path)

        return self._chunk_units(units)

    def _drop_standalone_noise_units(self, units: list[tuple[str, bool]]) -> list[tuple[str, bool]]:
        filtered: list[tuple[str, bool]] = []
        for text, is_table in units:
            if not text:
                continue
            if not is_table and self._is_standalone_page_marker(text):
                continue
            filtered.append((text, is_table))
        return filtered

    def _is_standalone_page_marker(self, text: str) -> bool:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) != 1:
            return False

        compact = re.sub(r"\s+", "", lines[0])
        if re.match(r"^第?\d{1,4}页$", compact):
            return True
        if re.match(r"^[•·\.\-—_]+\d{1,4}[•·\.\-—_]+$", compact):
            return True
        return False

    def _merge_short_units_downward(self, units: list[tuple[str, bool]]) -> list[tuple[str, bool]]:
        merged: list[tuple[str, bool]] = []
        i = 0

        while i < len(units):
            text, is_table = units[i]
            if not text:
                i += 1
                continue

            if is_table or len(text) >= 100:
                merged.append((text, is_table))
                i += 1
                continue

            if self._looks_like_table_title(text) or self._starts_with_section_boundary(text):
                merged.append((text, is_table))
                i += 1
                continue

            parts = [text]
            j = i + 1
            while j < len(units):
                next_text, next_is_table = units[j]
                if not next_text:
                    j += 1
                    continue
                if next_is_table:
                    break
                if self._starts_with_section_boundary(next_text):
                    break
                parts.append(next_text)
                j += 1
                if len("\n".join(parts)) >= 100:
                    break

            merged.append(("\n".join(parts).strip(), False))
            i = j

        return merged

    def _merge_colon_ended_units(self, units: list[tuple[str, bool]]) -> list[tuple[str, bool]]:
        merged: list[tuple[str, bool]] = []
        i = 0

        while i < len(units):
            text, is_table = units[i]
            if not text or is_table:
                merged.append((text, is_table))
                i += 1
                continue

            if not text.rstrip().endswith(("：", ":")):
                merged.append((text, is_table))
                i += 1
                continue

            parts = [text]
            j = i + 1
            while j < len(units):
                next_text, next_is_table = units[j]
                if next_is_table:
                    break
                if next_text:
                    parts.append(next_text)
                    if next_text.rstrip().endswith(("。")):
                        j += 1
                        break
                j += 1

            merged.append(("\n".join(parts).strip(), False))
            i = j

        return merged

    def _merge_table_title_with_next_table(self, units: list[tuple[str, bool]]) -> list[tuple[str, bool]]:
        merged: list[tuple[str, bool]] = []
        i = 0

        while i < len(units):
            text, is_table = units[i]
            if not text or is_table:
                merged.append((text, is_table))
                i += 1
                continue

            if i + 1 >= len(units):
                merged.append((text, is_table))
                i += 1
                continue

            next_text, next_is_table = units[i + 1]
            next_is_table_like = next_is_table or self._looks_like_table_text(next_text)

            if (
                self._looks_like_table_title(text)
                or self._looks_like_inline_table_title(text)
                or (
                    self._looks_like_heading(text)
                    and not self._starts_with_section_boundary(text)
                )
            ) and next_is_table_like:
                combined = f"{text.rstrip()}\n{next_text.lstrip()}".strip()
                merged.append((combined, True))
                i += 2
                continue

            merged.append((text, is_table))
            i += 1

        return merged

    def _merge_standalone_continuation_labels(self, units: list[tuple[str, bool]]) -> list[tuple[str, bool]]:
        merged: list[tuple[str, bool]] = []
        i = 0

        while i < len(units):
            text, is_table = units[i]
            if not text:
                i += 1
                continue

            if self._looks_like_standalone_continuation_label(text):
                if i + 1 < len(units):
                    next_text, next_is_table = units[i + 1]
                    if next_is_table or self._looks_like_table_text(next_text):
                        merged.append(
                            (f"{text.strip()}\n{next_text.lstrip()}".strip(), True)
                        )
                        i += 2
                        continue

            merged.append((text, is_table))
            i += 1

        return merged

    def _merge_table_title_with_previous_table(self, units: list[tuple[str, bool]]) -> list[tuple[str, bool]]:
        merged: list[tuple[str, bool]] = []

        for i, (text, is_table) in enumerate(units):
            if (
                text
                and self._looks_like_table_title(text)
                and merged
                and (merged[-1][1] or self._looks_like_table_text(merged[-1][0]))
            ):
                next_is_table_like = False
                if i + 1 < len(units):
                    next_text, next_is_table = units[i + 1]
                    next_is_table_like = next_is_table or self._looks_like_table_text(next_text)
                if not next_is_table_like:
                    prev_text, _ = merged[-1]
                    merged[-1] = (f"{text.strip()}\n{prev_text.lstrip()}".strip(), True)
                    continue

            merged.append((text, is_table))

        return merged

    def _merge_pure_table_units_upward(self, units: list[tuple[str, bool]]) -> list[tuple[str, bool]]:
        merged: list[tuple[str, bool]] = []

        for text, is_table in units:
            if not text:
                continue

            is_table_like = is_table or self._looks_like_table_text(text)
            if is_table_like and merged:
                prev_text, _ = merged[-1]
                merged[-1] = (f"{prev_text.rstrip()}\n{text.lstrip()}".strip(), True)
                continue

            merged.append((text, is_table_like))

        return merged

    def _merge_unterminated_paragraphs(self, units: list[tuple[str, bool]]) -> list[tuple[str, bool]]:
        merged: list[tuple[str, bool]] = []
        i = 0

        while i < len(units):
            text, is_table = units[i]
            if not text:
                i += 1
                continue

            if is_table or self._ends_with_sentence_punctuation(text):
                merged.append((text, is_table))
                i += 1
                continue

            if self._looks_like_table_title(text):
                merged.append((text, is_table))
                i += 1
                continue

            parts = [text]
            j = i + 1
            while j < len(units):
                next_text, next_is_table = units[j]
                if not next_text:
                    j += 1
                    continue
                if next_is_table:
                    break
                if self._starts_with_section_boundary(next_text):
                    break
                if self._looks_like_heading(next_text) and not parts[-1].endswith(("：", ":")):
                    break

                parts.append(next_text)
                j += 1
                if self._ends_with_sentence_punctuation(next_text):
                    break

            merged.append(("\n".join(parts).strip(), False))
            i = j

        return merged

    def _split_units_by_section_boundaries(self, units: list[tuple[str, bool]]) -> list[tuple[str, bool]]:
        split_units: list[tuple[str, bool]] = []

        for text, is_table in units:
            if not text or is_table:
                split_units.append((text, is_table))
                continue

            lines = [line for line in text.splitlines() if line.strip()]
            if len(lines) <= 1:
                split_units.append((text, is_table))
                continue

            parts: list[str] = []
            current: list[str] = []
            for line in lines:
                stripped = line.strip()
                if current and self._is_section_boundary_line(stripped):
                    parts.append("\n".join(current).strip())
                    current = [stripped]
                else:
                    current.append(stripped)
            if current:
                parts.append("\n".join(current).strip())

            for part in parts:
                if part:
                    split_units.append((part, False))

        return split_units
    def _split_large_table(self,table_text: str) -> list[str]:
        lines = table_text.splitlines()
        if not lines:
            return []
        # 默认第一行作为表头
        header = lines[0]
        chunks = []
        current_lines = [header]
        current_len = len(header)
        for line in lines[1:]:
            line_len = len(line)
            # 单行已经超长,2000,10240
            if line_len > 2000:
                # 先提交已有chunk
                if len(current_lines) > 1:
                    chunks.append("\n".join(current_lines))

                # 强制切割超长行
                for i in range(0,line_len,2000):
                    chunks.append(line[i:i+2000])
                current_lines = [header]
                current_len = len(header)
                continue

            # 超过chunk大小
            if current_len + line_len > 2000:
                chunks.append("\n".join(current_lines))
                # 新chunk保留header
                current_lines = [header, line]
                current_len = len(header) + line_len
            else:
                current_lines.append(line)
                current_len += line_len
        if len(current_lines) > 1:
            chunks.append("\n".join(current_lines))

        return chunks
    def _parse_pdf_units(self, file_path: str) -> list[tuple[str, bool]]:
        if fitz is None:
            raise ImportError("Please install PyMuPDF: pip install pymupdf")

        units: list[tuple[str, bool]] = []
        page_blocks: list[list[dict]] = []

        with fitz.open(file_path) as doc:
            for page in doc:
                table_texts, table_boxes = self._extract_pdf_tables(page)
                page_units: list[dict] = [
                    {"text": t, "is_table": True, "y0": 0.0, "y1": 0.0, "page_h": 0.0}
                    for t, _ in table_texts
                ]
                page_text_chars = 0

                blocks = page.get_text("blocks")
                for block in blocks:
                    text = (block[4] or "").strip()
                    if not text:
                        continue

                    block_box = fitz.Rect(block[:4])
                    if self._inside_any_table(block_box, table_boxes):
                        continue

                    cleaned = self._clean_text(text)
                    cleaned = self._clean_pdf_block_text(cleaned)
                    if not cleaned:
                        continue
                    if self.validate_text:
                        self._raise_if_garbled(cleaned, file_type="PDF")
                    if cleaned:
                        page_text_chars += len(cleaned)
                        page_units.append(
                            {
                                "text": cleaned,
                                "is_table": False,
                                "y0": float(block[1]),
                                "y1": float(block[3]),
                                "page_h": float(page.rect.height),
                            }
                        )

                if self._should_use_llm_for_page(page, page_text_chars, len(table_texts)):
                    llm_text = self._parse_pdf_page_image_with_llm(page)
                    if llm_text:
                        page_h = float(page.rect.height)
                        page_units.append(
                            {
                                "text": llm_text,
                                "is_table": False,
                                "y0": page_h * 0.2,
                                "y1": page_h * 0.8,
                                "page_h": page_h,
                            }
                        )

                page_blocks.append(page_units)

        header_footer_patterns = self._detect_pdf_header_footer_patterns(page_blocks)
        for page_units in page_blocks:
            for item in page_units:
                text = item["text"]
                if not text:
                    continue
                if not item["is_table"]:
                    if self._is_pdf_header_footer_line(text, item, header_footer_patterns):
                        continue
                units.append((text, bool(item["is_table"])))

        return units

    def _should_use_llm_for_page(self, page, page_text_chars: int, table_count: int) -> bool:
        return False

    def _parse_pdf_page_image_with_llm(self, page) -> str:
        try:
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            img_bytes = pix.tobytes("png")
        except Exception:
            return ""

        prompt = (
            "请提取图片中的全部文字，保持阅读顺序。"
            "若存在表格，请按行输出并使用' | '分隔列。"
            "不要添加解释。"
        )
        b64 = base64.b64encode(img_bytes).decode("ascii")
        data_url = f"data:image/png;base64,{b64}"
        if self._llm_client is not None:
            try:
                resp = self._llm_client.chat.completions.create(
                    model=self.llm_model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": data_url}},
                            ],
                        }
                    ],
                    temperature=0,
                )
                content = resp.choices[0].message.content
                if isinstance(content, str):
                    return self._clean_text(content)
                if isinstance(content, list):
                    texts = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            texts.append(item.get("text", ""))
                    return self._clean_text("\n".join(x for x in texts if x))
            except Exception:
                pass

        # Fallback: raw HTTP OpenAI-compatible endpoint.
        payload = {
            "model": self.llm_model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            "temperature": 0,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        req = urllib.request.Request(
            f"{self.llm_base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.llm_api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw = resp.read().decode("utf-8", errors="ignore")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
            return ""

        try:
            obj = json.loads(raw)
            content = obj["choices"][0]["message"]["content"]
        except Exception:
            return ""

        if isinstance(content, str):
            return self._clean_text(content)
        if isinstance(content, list):
            texts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    texts.append(item.get("text", ""))
            return self._clean_text("\n".join(x for x in texts if x))
        return ""

    def _parse_pdf_with_llm_fallback(self, file_path: str) -> list[tuple[str, bool]]:
        if fitz is None:
            return []

        units: list[tuple[str, bool]] = []
        with fitz.open(file_path) as doc:
            for page in doc:
                llm_text = self._parse_pdf_page_image_with_llm(page)
                if llm_text:
                    units.append((llm_text, False))
        return units

    def _extract_pdf_tables(self, page) -> tuple[list[tuple[str, bool]], list]:
        table_texts: list[tuple[str, bool]] = []
        table_boxes: list = []

        try:
            finder = page.find_tables()
            tables = finder.tables if finder else []
        except Exception:
            tables = []

        for table in tables:
            rows = table.extract() or []
            row_lines = []
            for row in rows:
                cells = [(c or "").strip() for c in row]
                if any(cells):
                    row_lines.append(" | ".join(cells))
            text = "\n".join(row_lines).strip()
            if self.validate_text:
                self._raise_if_garbled(text, file_type="PDF table")
            if text:
                table_texts.append((text, True))
                table_boxes.append(table.bbox)

        return table_texts, table_boxes

    def _inside_any_table(self, block_rect, table_boxes: list) -> bool:
        for box in table_boxes:
            table_rect = fitz.Rect(box)
            inter = block_rect & table_rect
            if inter.is_empty:
                continue
            inter_area = inter.get_area()
            block_area = block_rect.get_area() or 1.0
            if inter_area / block_area >= 0.6:
                return True
        return False

    def _parse_docx_units(self, file_path: str) -> list[tuple[str, bool]]:
        if Document is None or Paragraph is None or Table is None:
            raise ImportError("Please install python-docx: pip install python-docx")

        doc = Document(file_path)
        units: list[tuple[str, bool]] = []

        for block in self._iter_docx_blocks(doc):
            if isinstance(block, Paragraph):
                text = self._clean_text(block.text)
                if self.validate_text:
                    self._raise_if_garbled(text, file_type="DOCX")
                if text:
                    units.append((text, False))
            elif isinstance(block, Table):
                table_text = self._docx_table_to_text(block)
                if self.validate_text:
                    self._raise_if_garbled(table_text, file_type="DOCX table")
                if table_text:
                    units.append((table_text, True))

        return units

    def _iter_docx_blocks(self, doc) -> Iterable:
        body = doc.element.body
        for child in body.iterchildren():
            if child.tag.endswith("}p"):
                yield Paragraph(child, doc)
            elif child.tag.endswith("}tbl"):
                yield Table(child, doc)

    def _docx_table_to_text(self, table) -> str:
        lines = []
        for row in table.rows:
            cells = [self._clean_text(cell.text) for cell in row.cells]
            if any(cells):
                lines.append(" | ".join(cells))
        return "\n".join(lines).strip()

    def _chunk_units(self, units: list[tuple[str, bool]]) -> list[str]:
        chunks: list[str] = []
        '''pending_table_parts: list[str] = []

        def flush_pending_tables() -> None:
            if not pending_table_parts:
                return
            merged = "\n".join(part for part in pending_table_parts if part).strip()
            if merged:
                chunks.append(merged)
            pending_table_parts.clear()'''

        for unit, is_table in units:
            unit = unit.strip()
            if not unit:
                continue
            
            if is_table or self._looks_like_table_text(unit):
                '''if pending_table_parts and self._looks_like_standalone_continuation_label( unit.splitlines()[0]):
                    flush_pending_tables()
                pending_table_parts.append(unit)'''
                # 表格不过长：直接保存完整
                if len(unit) < 2000: # 10240
                    chunks.append(unit)
                else:
                    print(f"表格 {unit} 过长2000，10240字符无法直接保存")
                    chunks.extend(self._split_large_table(unit))
                continue

            #flush_pending_tables()
            # 普通文本：按 max_chunk_size 切分
            if len(unit) <= self.max_chunk_size:
                chunks.append(unit)
            else:
                chunks.extend(self._split_long_paragraph(unit))

        #flush_pending_tables()
        return chunks

    def _looks_like_table_text(self, text: str) -> bool:
        if not text:
            return False
        if "|" in text:
            return True

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) < 2:
            return False

        column_gap_lines = sum(
            1 for line in lines if re.search(r"\S+\s{2,}\S+", line) is not None
        )
        if column_gap_lines >= max(2, len(lines) // 2):
            return True

        code_like_lines = sum(
            1 for line in lines if re.match(r"^\d{3,}\s+\S+", line) is not None
        )
        return code_like_lines >= 2

    def _looks_like_table_title(self, text: str) -> bool:
        if not text:
            return False

        cleaned = text.strip()
        if not cleaned or self._ends_with_sentence_punctuation(cleaned) or cleaned.endswith(("：", ":")):
            return False

        return re.match(
            r"^(?:续\s*附?\s*表|附\s*表|表)\s*[A-Za-z]?\s*[\.,，。]?\s*\d",
            cleaned,
        ) is not None

    def _looks_like_inline_table_title(self, text: str) -> bool:
        cleaned = text.strip()
        if not cleaned or self._ends_with_sentence_punctuation(cleaned):
            return False
        return (
            re.search(r"表\s*[A-Za-z]?\s*\d+(?:\s*[-—－\.．]\s*\d+)*\s*$", cleaned)
            is not None
        )

    def _looks_like_standalone_continuation_label(self, text: str) -> bool:
        compact = re.sub(r"\s+", "", text)
        return re.match(
            r"^续(?:附)?表(?:[A-Za-z]?[.,，。]?\d+(?:[.,，。]\d+)*)?$",
            compact,
        ) is not None

    def _looks_like_heading(self, text: str) -> bool:
        cleaned = text.strip()
        if not cleaned or len(cleaned) > 24:
            return False
        if self._ends_with_sentence_punctuation(cleaned) or cleaned.endswith(("：", ":")):
            return False

        return re.match(r"^(?:续\s*附?\s*表|附\s*表|表)\s*[A-Za-z]?\s*[\.,，。]?\s*\d", cleaned) is not None or re.match(
            r"^[A-Z]\.\d+(?:\.\d+)*\s+\S+$", cleaned
        ) is not None or re.match(r"^\d+(?:\.\d+)*\s+\S+$", cleaned) is not None

    def _ends_with_sentence_punctuation(self, text: str) -> bool:
        cleaned = text.rstrip()
        return cleaned.endswith(("。", "！", "？", ".", "!", "?"))

    def _is_section_boundary_line(self, line: str) -> bool:
        compact = re.sub(r"\s+", "", line)
        if compact == "前言":
            return True
        if re.match(r"^\d+总则$", compact) is not None:
            return True
        if re.match(r"^\d+术语$", compact) is not None:
            return True
        if re.match(r"^\d+工程计量$", compact) is not None:
            return True
        if re.match(r"^\d+工程量清单编制$", compact) is not None:
            return True
        return False

    def _starts_with_section_boundary(self, text: str) -> bool:
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            return self._is_section_boundary_line(stripped)
        return False

    def _split_long_paragraph(self, text: str) -> list[str]:
        text = text.strip()
        parts: list[str] = []

        while len(text) > self.max_chunk_size:
            cut = self._find_near_sentence_break(text, self.max_chunk_size)
            part = text[:cut].strip()
            if part:
                parts.append(part)
            text = text[cut:].strip()

        if text:
            parts.append(text)
        return parts

    def _find_near_sentence_break(self, text: str, target: int) -> int:
        if len(text) <= target:
            return len(text)

        window = 80
        start = max(0, target - window)
        end = min(len(text), target + window)

        for i in range(target, start, -1):
            ch = text[i - 1]
            if ch in self._sentence_delimiters:
                return i

        for i in range(target + 1, end + 1):
            if text[i - 1] in self._sentence_delimiters:
                return i

        return target

    def _clean_text(self, text: str) -> str:
        text = text.replace("\r", "\n")
        text = re.sub(r"[\t\f\v]+", " ", text)
        text = re.sub(r" *\n+ *", "\n", text)
        text = re.sub(r"[ ]{2,}", " ", text)
        return text.strip()

    def _clean_pdf_block_text(self, text: str) -> str:
        if not text:
            return ""

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        kept: list[str] = []
        for line in lines:
            if self._is_pdf_noise_line(line):
                continue
            kept.append(line)

        return "\n".join(kept).strip()

    def _is_pdf_noise_line(self, line: str) -> bool:
        if not line:
            return True

        compact = re.sub(r"\s+", "", line)
        plain = re.sub(r"[^\w\u4e00-\u9fff]", "", compact)
        digit_only = re.sub(r"\D", "", compact)

        # Pure page numbers or isolated chapter numbers.
        if re.match(r"^\(?\d{1,4}\)?$", compact):
            return True
        if re.match(r"^[•·\.\-—_]*\d{1,4}[•·\.\-—_]*$", compact):
            return True
        # Page marker with any decorative symbols around a number: "• 51 •", "?51?"
        if 1 <= len(digit_only) <= 3 and len(plain) == len(digit_only):
            if len(compact) > len(digit_only):
                return True
        if re.match(r"^[A-Za-z]?\d{1,4}$", compact):
            return True

        # Typical TOC leader lines, e.g. "...... (5)".
        if re.search(r"[\.·•…]{4,}", line) and re.search(r"[（(]\s*\d+\s*[)）]?\s*$", line):
            return True

        if re.match(r"^\d{1,4}$", compact):
            return True
        if re.match(r"^第?\d{1,4}页$", compact):
            return True

        useful = 0
        noisy = 0
        for ch in line:
            if ch.isspace():
                continue
            if ch.isalnum() or ("\u4e00" <= ch <= "\u9fff"):
                useful += 1
            else:
                cat = unicodedata.category(ch)
                if cat.startswith(("P", "S")):
                    noisy += 1
                else:
                    useful += 1

        total = useful + noisy
        if total == 0:
            return True
        if useful == 0 and noisy >= 2:
            return True
        if len(line) <= 12 and noisy / total >= 0.75:
            return True
        if len(line) <= 20 and noisy / total >= 0.55 and useful <= 4:
            return True

        return False

    def _normalize_pdf_repeated_line(self, text: str) -> str:
        compact = re.sub(r"\s+", "", text)
        compact = re.sub(r"[·•…．\.\-—_]+", "", compact)
        return compact

    def _detect_pdf_header_footer_patterns(self, page_blocks: list[list[dict]]) -> set[str]:
        counts: dict[str, int] = {}
        for page_units in page_blocks:
            seen_in_page: set[str] = set()
            for item in page_units:
                if item["is_table"]:
                    continue
                if not self._is_pdf_edge_item(item):
                    continue
                norm = self._normalize_pdf_repeated_line(item["text"])
                if not norm or len(norm) > 40:
                    continue
                seen_in_page.add(norm)
            for norm in seen_in_page:
                counts[norm] = counts.get(norm, 0) + 1

        return {k for k, v in counts.items() if v >= 3}

    def _is_pdf_edge_item(self, item: dict) -> bool:
        page_h = float(item.get("page_h", 0.0) or 0.0)
        y0 = float(item.get("y0", 0.0) or 0.0)
        y1 = float(item.get("y1", 0.0) or 0.0)
        if page_h <= 0:
            return False
        return y1 <= page_h * 0.12 or y0 >= page_h * 0.88

    def _is_pdf_header_footer_line(self,text: str,item: dict,header_footer_patterns: set[str],) -> bool:
        if self._is_pdf_noise_line(text):
            return True

        if not self._is_pdf_edge_item(item):
            return False

        norm = self._normalize_pdf_repeated_line(text)
        if norm in header_footer_patterns:
            return True

        return False

    def _raise_if_garbled(self, text: str, file_type: str) -> None:
        if not text:
            return

        bad_chars = text.count("\ufffd")
        control_chars = sum(
            1 for ch in text if ord(ch) < 32 and ch not in "\n\r\t"
        )
        bad_ratio = (bad_chars + control_chars) / max(len(text), 1)

        if bad_chars or bad_ratio > 0.01:
            raise ValueError(
                f"{file_type} text looks garbled. "
                "Please use a DOCX source or enable OCR before exporting results."
            )


def parse_document(file_path: str,max_chunk_size: int = 500,validate_text: bool = True,) -> list[str]:
    parser = DocumentParser(max_chunk_size=max_chunk_size, validate_text=validate_text)
    return parser.parse(file_path)

def data2embed(data: list[str],filename: str) -> list[dict[str, float]]:
    doc_id=str(uuid.uuid4())
    vec_list=[]
    i=0
    for item in data:
        i+=1
        item = item.strip() 
        vecres=get_embed(item,embedder) 
        #vec_list=[]    # list_dict: id,vector,text,doc_id
        if vecres['success']:
            vec=vecres['embeddings']
            vec_list.append({"id":f"{doc_id}_{i}","vector":vec,"text":item,"doc_id":doc_id,"doc_name":filename})
    return vec_list

if __name__ == "__main__":
    parser = DocumentParser(max_chunk_size=500)
    # 需要查询目录下所有的docx文件
    root_dir="/data/gonghuimin/agent/kg_QA/data/over"
    file_list = list(Path(root_dir).glob("*.docx"))
    import uuid
    for sample in file_list:
        if Path(sample).exists():
            filename=Path(sample).name #sample.name.split(".")[0]
            print(f"{filename}:{Path(sample).name}")
            result = parser.parse(sample)
            print(f"{sample}:{type(result)}:::::::: {len(result)} chunks")
            vec_list=data2embed(result,filename)
            print(f"{len(vec_list)} chunks")
            insert_batch(vec_list,collection_name=collection_name,alias_name="default",env='')
            '''for i, item in enumerate(result[:3], 1):
                print(f"[{i}] {item[:100]}")'''
