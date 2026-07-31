"""
Corpus loading, chunking and BM25 retrieval.

Retrieval is BM25 over section-level chunks, deliberately, not embeddings. For 24 documents a
lexical index matches as well as a vector one on the queries this system asks — which are full
of distinctive proper nouns like "LangGraph", "Fly.io", "FedRAMP" that lexical search handles
better than semantic similarity — and it costs no embedding API, no ~2 GB torch dependency, and
no index build step. The Week 3 deployment was slowed by exactly that dependency.

The retrieval interface is kept narrow (:meth:`CorpusIndex.search` returns ranked
:class:`Chunk` objects) so a vector or hybrid backend can be added later without touching the
tools or agents that call it.

Chunking is by markdown section. Sections are the natural evidence unit here: a chunk that spans
two headings tends to produce evidence whose supporting text does not actually contain the claim.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from pathlib import Path

from app.config import ROOT

CORPUS_DIR = ROOT / "corpus"

_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_LIST_VALUE = re.compile(r"^\[(.*)\]$")
_TOKEN = re.compile(r"[a-z0-9][a-z0-9._+-]*")

# Words too common in this corpus to discriminate. Kept short on purpose: an aggressive stop list
# would strip meaningful query terms like "state" or "cost".
_STOPWORDS = frozenset("""
a an the and or but if then than that this these those of in on at to for from with without by
is are was were be been being it its as not no so such can may might will would should could
""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]


@dataclass(frozen=True)
class Document:
    """One corpus file with its parsed frontmatter."""

    doc_id: str
    title: str
    domain: str
    source_type: str
    publisher: str
    published: str
    covers: tuple[str, ...]
    reliability: str
    body: str
    path: Path

    def published_date(self) -> date | None:
        try:
            return date.fromisoformat(self.published)
        except (ValueError, TypeError):
            return None


@dataclass(frozen=True)
class Chunk:
    """A retrievable section of a document.

    Carries its source document's metadata so a retrieval hit can become an Evidence record
    without a second lookup — and, importantly, so ``reliability`` and ``source_type`` reach the
    Researcher. An agent that cannot tell vendor marketing from an independent benchmark cannot
    assign confidence honestly.
    """

    chunk_id: str
    doc_id: str
    doc_title: str
    domain: str
    source_type: str
    publisher: str
    published: str
    reliability: str
    heading: str
    text: str
    score: float = 0.0

    def with_score(self, score: float) -> Chunk:
        return Chunk(**{**self.__dict__, "score": score})

    def word_count(self) -> int:
        return len(self.text.split())


def _parse_frontmatter(raw: str) -> tuple[dict[str, str | tuple[str, ...]], str]:
    m = _FRONTMATTER.match(raw)
    if not m:
        return {}, raw
    meta: dict[str, str | tuple[str, ...]] = {}
    for line in m.group(1).splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        value = value.strip()
        lm = _LIST_VALUE.match(value)
        if lm:
            meta[key.strip()] = tuple(
                v.strip() for v in lm.group(1).split(",") if v.strip()
            )
        else:
            meta[key.strip()] = value
    return meta, raw[m.end():]


def load_documents(corpus_dir: Path | None = None) -> list[Document]:
    """Read every ``.md`` file under the corpus directory."""
    base = corpus_dir or CORPUS_DIR
    docs: list[Document] = []
    for path in sorted(base.rglob("*.md")):
        meta, body = _parse_frontmatter(path.read_text(encoding="utf-8"))
        covers = meta.get("covers", ())
        docs.append(
            Document(
                doc_id=str(meta.get("doc_id") or path.stem),
                title=str(meta.get("title") or path.stem),
                domain=str(meta.get("domain") or path.parent.name),
                source_type=str(meta.get("source_type") or "unknown"),
                publisher=str(meta.get("publisher") or "unknown"),
                published=str(meta.get("published") or ""),
                covers=covers if isinstance(covers, tuple) else (str(covers),),
                reliability=str(meta.get("reliability") or "unknown"),
                body=body.strip(),
                path=path,
            )
        )
    return docs


def chunk_document(doc: Document, min_words: int = 12) -> list[Chunk]:
    """Split a document into section-level chunks by markdown heading.

    Sections shorter than ``min_words`` are folded into the previous chunk rather than emitted,
    so a bare heading never becomes a retrievable "finding" with no content behind it.
    """
    lines = doc.body.splitlines()
    sections: list[tuple[str, list[str]]] = []
    heading = doc.title
    buf: list[str] = []

    for line in lines:
        if line.startswith("#"):
            if buf:
                sections.append((heading, buf))
                buf = []
            heading = line.lstrip("#").strip() or doc.title
        else:
            buf.append(line)
    if buf:
        sections.append((heading, buf))

    chunks: list[Chunk] = []
    for head, body_lines in sections:
        text = "\n".join(body_lines).strip()
        if not text:
            continue
        if len(text.split()) < min_words and chunks:
            prev = chunks[-1]
            chunks[-1] = Chunk(**{**prev.__dict__,
                                  "text": f"{prev.text}\n\n{head}\n{text}"})
            continue
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}#{len(chunks)}",
                doc_id=doc.doc_id,
                doc_title=doc.title,
                domain=doc.domain,
                source_type=doc.source_type,
                publisher=doc.publisher,
                published=doc.published,
                reliability=doc.reliability,
                heading=head,
                text=text,
            )
        )
    return chunks


class CorpusIndex:
    """BM25 index over section chunks.

    BM25 is implemented here rather than pulled from ``rank_bm25`` for one reason worth stating:
    the whole index is ~60 chunks, the algorithm is twenty lines, and owning it means the scoring
    is inspectable when a retrieval result looks wrong. It also removes a dependency from the
    deployment.
    """

    K1 = 1.5   # term-frequency saturation
    B = 0.75   # length normalisation

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self._tokens = [tokenize(f"{c.heading} {c.text}") for c in chunks]
        self._lengths = [len(t) for t in self._tokens]
        self._avg_len = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0
        self._tf: list[dict[str, int]] = []
        self._df: dict[str, int] = {}
        for toks in self._tokens:
            counts: dict[str, int] = {}
            for t in toks:
                counts[t] = counts.get(t, 0) + 1
            self._tf.append(counts)
            for t in counts:
                self._df[t] = self._df.get(t, 0) + 1

    def __len__(self) -> int:
        return len(self.chunks)

    def _idf(self, term: str) -> float:
        n = len(self.chunks)
        df = self._df.get(term, 0)
        if df == 0:
            return 0.0
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(
        self,
        query: str,
        top_k: int = 5,
        domain: str | None = None,
        min_score: float = 0.0,
    ) -> list[Chunk]:
        """Ranked chunks for ``query``.

        Returns an empty list when nothing scores above ``min_score``. That empty result is a
        real answer — §22 requires "empty research results" to be handled, and silently returning
        the least-bad chunk is how a system ends up citing an irrelevant source for a claim.
        """
        terms = tokenize(query)
        if not terms:
            return []

        scored: list[tuple[float, int]] = []
        for i, chunk in enumerate(self.chunks):
            if domain and chunk.domain != domain:
                continue
            tf, length = self._tf[i], self._lengths[i]
            score = 0.0
            for term in terms:
                f = tf.get(term, 0)
                if not f:
                    continue
                norm = 1 - self.B + self.B * (length / self._avg_len if self._avg_len else 1)
                score += self._idf(term) * (f * (self.K1 + 1)) / (f + self.K1 * norm)
            if score > min_score:
                scored.append((score, i))

        scored.sort(key=lambda p: (-p[0], p[1]))
        return [self.chunks[i].with_score(round(s, 4)) for s, i in scored[:top_k]]

    def get_document(self, doc_id: str) -> list[Chunk]:
        return [c for c in self.chunks if c.doc_id == doc_id]

    def domains(self) -> list[str]:
        return sorted({c.domain for c in self.chunks})

    def doc_ids(self) -> list[str]:
        seen: list[str] = []
        for c in self.chunks:
            if c.doc_id not in seen:
                seen.append(c.doc_id)
        return seen


def build_index(corpus_dir: Path | None = None) -> CorpusIndex:
    chunks: list[Chunk] = []
    for doc in load_documents(corpus_dir):
        chunks.extend(chunk_document(doc))
    return CorpusIndex(chunks)


@lru_cache(maxsize=1)
def get_index() -> CorpusIndex:
    """Process-wide cached index. The corpus is static, so one build per process is enough."""
    return build_index()
