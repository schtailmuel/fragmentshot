import hashlib
import json
import logging
import random
import re
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union


MaskPattern = Union[str, re.Pattern]
MaskPatterns = Union[MaskPattern, Sequence[MaskPattern]]
MaskRulesInput = Union[
    Mapping[str, MaskPatterns],
    Sequence[Tuple[str, MaskPatterns]],
    str,
    Path,
]


_BASE_LOGGER = logging.getLogger("fragmentshot")
_MASK_LOGGER = _BASE_LOGGER.getChild("masking")


def _remove_punctuation(text):
    text = re.sub(r"[ ]+", " ", text)
    text = re.sub(r"[.,!?:;]", "", text)
    return text


def _normalize_text(text):
    return _remove_punctuation(text)


def _create_fragments(xs, n):
    return [xs[i : i + n] for i in range(len(xs) - n + 1)]


def _normalize_mask_key(key):
    cleaned_key = key.strip()
    if not cleaned_key:
        raise ValueError("Mask key cannot be empty.")
    if cleaned_key.startswith("[") and cleaned_key.endswith("]"):
        return cleaned_key
    return f"[{cleaned_key}]"


def _wrap_with_word_boundaries(pattern_text):
    stripped = pattern_text.strip()
    if not stripped:
        return stripped
    prefix = "" if stripped.startswith(r"\b") else r"\b"
    suffix = "" if stripped.endswith(r"\b") else r"\b"
    return rf"{prefix}(?:{stripped}){suffix}"


def _extract_patterns(pattern_value):
    if isinstance(pattern_value, re.Pattern):
        return [pattern_value]

    if isinstance(pattern_value, (list, tuple)):
        patterns = []
        for pattern in pattern_value:
            if isinstance(pattern, re.Pattern):
                patterns.append(pattern)
                continue
            normalized_pattern = str(pattern).strip()
            if normalized_pattern:
                patterns.append(normalized_pattern)
        if patterns:
            return patterns
        raise ValueError("Invalid mask definition: regex list cannot be empty.")

    pattern_text = str(pattern_value).strip()
    if not pattern_text:
        raise ValueError("Invalid mask definition: regex cannot be empty.")
    return [pattern_text]


def _serialize_mask_rules(mask_rules):
    serialized = []
    for replacement, pattern in mask_rules:
        serialized.append(
            {
                "replacement": replacement,
                "pattern": pattern.pattern,
                "flags": pattern.flags,
            }
        )
    return serialized


def _deserialize_mask_rules(serialized_rules):
    prepared_rules = []
    for rule in serialized_rules:
        prepared_rules.append(
            (
                str(rule["replacement"]),
                re.compile(str(rule["pattern"]), int(rule["flags"])),
            )
        )
    return prepared_rules


def _mask_rules_fingerprint(mask_rules):
    serialized = _serialize_mask_rules(mask_rules)
    payload = json.dumps(serialized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _apply_mask_rules_to_normalized_text(normalized_text, mask_rules):
    masked_text = normalized_text
    for replacement, pattern in mask_rules:
        masked_text = pattern.sub(replacement, masked_text)
    masked_text = re.sub(r"\s+", " ", masked_text).strip()
    return masked_text


def _tokenize_with_masks(text, mask_rules):
    normalized = _normalize_text(text)
    if not mask_rules:
        return normalized.split()
    return _apply_mask_rules_to_normalized_text(normalized, mask_rules).split()


def _load_mask_rules_from_file(mask_file):
    mask_path = Path(mask_file)
    _MASK_LOGGER.info("Loading mask rules from %s", mask_path)
    raw_content = mask_path.read_text(encoding="utf-8")
    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid mask JSON in '{mask_path}': {exc.msg} (line {exc.lineno}, column {exc.colno})"
        ) from exc

    if not isinstance(parsed, dict):
        raise ValueError("Mask JSON must be an object with key -> array of regex strings.")

    for key, patterns in parsed.items():
        if isinstance(patterns, str):
            patterns = [patterns]
        if not isinstance(patterns, list):
            raise ValueError(
                f"Mask value for key '{key}' must be a regex string or an array of regex strings."
            )
        if not patterns:
            raise ValueError(f"Mask value for key '{key}' must contain at least one regex.")
        for pattern in patterns:
            if not isinstance(pattern, str) or not pattern.strip():
                raise ValueError(
                    f"Mask value for key '{key}' must contain only non-empty regex strings."
                )

    prepared = _prepare_mask_rules(parsed)
    _MASK_LOGGER.info("Loaded %d compiled mask rule patterns from %s", len(prepared), mask_path)
    return prepared


def _prepare_mask_rules(mask_rules):
    if mask_rules is None:
        return []

    if isinstance(mask_rules, (str, Path)):
        return _load_mask_rules_from_file(mask_rules)

    mask_entries = mask_rules.items() if isinstance(mask_rules, Mapping) else mask_rules
    prepared_rules = []
    for entry in mask_entries:
        if len(entry) != 2:
            raise ValueError(
                "Each mask rule must contain exactly two elements: (key, regex or list[regex])."
            )
        key, patterns = entry
        replacement = _normalize_mask_key(str(key))
        for pattern in _extract_patterns(patterns):
            if isinstance(pattern, re.Pattern):
                bounded_pattern = _wrap_with_word_boundaries(pattern.pattern)
                compiled_pattern = re.compile(bounded_pattern, pattern.flags)
            else:
                bounded_pattern = _wrap_with_word_boundaries(str(pattern))
                compiled_pattern = re.compile(bounded_pattern)
            prepared_rules.append((replacement, compiled_pattern))
    return prepared_rules


@dataclass
class FragmentExample:
    src: str
    tgt: str
    src_masked: Optional[str] = None
    tgt_masked: Optional[str] = None

    @property
    def src_text(self):
        return self.src

    @property
    def tgt_text(self):
        return self.tgt


@dataclass
class FragmentShot:
    index: int
    fragment: str
    fragment_masked: str
    match_type: str
    examples: List[FragmentExample]


@dataclass
class FragmentSearchResult:
    shots: List[FragmentShot]
    num_words: int
    unknown: List[str]

    def to_dict(self):
        return {
            "shots": [
                {
                    "index": shot.index,
                    "fragment": shot.fragment,
                    "fragment_masked": shot.fragment_masked,
                    "match_type": shot.match_type,
                    "examples": [self._example_to_dict(example) for example in shot.examples],
                }
                for shot in self.shots
            ],
            "num_words": self.num_words,
            "unknown": self.unknown,
        }

    def to_legacy_dict(self):
        return {
            "shots": [
                {
                    "index": shot.index,
                    "fragment": shot.fragment,
                    "fragment_masked": shot.fragment_masked,
                    "match_type": shot.match_type,
                    "examples": [self._example_to_legacy_dict(example) for example in shot.examples],
                }
                for shot in self.shots
            ],
            "num_words": self.num_words,
            "unknown": self.unknown,
        }

    @staticmethod
    def _example_to_dict(example):
        payload = {"src": example.src, "tgt": example.tgt}
        if example.src_masked is not None:
            payload["src_masked"] = example.src_masked
        if example.tgt_masked is not None:
            payload["tgt_masked"] = example.tgt_masked
        return payload

    @staticmethod
    def _example_to_legacy_dict(example):
        payload = {"src_text": example.src, "tgt_text": example.tgt}
        if example.src_masked is not None:
            payload["src_masked"] = example.src_masked
        if example.tgt_masked is not None:
            payload["tgt_masked"] = example.tgt_masked
        return payload


class Indexer:
    def __init__(self, max_fragment_size=7, overlaps=False, mask_rules=None, logger=None):
        if max_fragment_size < 1:
            raise ValueError("max_fragment_size must be at least 1.")
        self.max_fragment_size = max_fragment_size
        self.overlaps = overlaps
        self.mask_rules = _prepare_mask_rules(mask_rules)
        self._log = logger or _BASE_LOGGER.getChild("indexer")
        self._log.debug(
            "Indexer initialized: max_fragment_size=%d overlaps=%s mask_rule_patterns=%d",
            self.max_fragment_size,
            self.overlaps,
            len(self.mask_rules),
        )

    def _initialize_db(self, output_db):
        db_path = Path(output_db)
        self._log.info("Preparing index database at %s", db_path)
        if db_path.exists():
            self._log.warning("Index path already exists and will be replaced: %s", db_path)
            db_path.unlink()
        db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute(
            """
            CREATE TABLE sentences (
                id INTEGER PRIMARY KEY,
                src TEXT NOT NULL,
                tgt TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE fragments (
                variant TEXT NOT NULL,
                size INTEGER NOT NULL,
                fragment TEXT NOT NULL,
                sentence_id INTEGER NOT NULL,
                FOREIGN KEY(sentence_id) REFERENCES sentences(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX idx_fragments_variant_size_fragment ON fragments(variant, size, fragment)"
        )
        conn.execute("CREATE INDEX idx_fragments_sentence_id ON fragments(sentence_id)")
        self._log.debug("Database schema initialized at %s", db_path)
        return conn

    def index_from_iterables(self, src_texts, tgt_texts, output_db, commit_every=2000):
        if commit_every < 1:
            raise ValueError("commit_every must be at least 1.")

        started_at = time.perf_counter()
        conn = self._initialize_db(output_db)
        sentence_id = 0
        sentence_rows = []
        fragment_rows = []
        total_fragments_raw = 0
        total_fragments_masked = 0
        last_progress_at = started_at

        self._log.info(
            "Indexing started: output_db=%s commit_every=%d max_fragment_size=%d mask_rule_patterns=%d",
            output_db,
            commit_every,
            self.max_fragment_size,
            len(self.mask_rules),
        )

        src_iter = iter(src_texts)
        tgt_iter = iter(tgt_texts)

        while True:
            src = next(src_iter, None)
            tgt = next(tgt_iter, None)
            if src is None and tgt is None:
                break
            if src is None or tgt is None:
                self._log.error("Indexing aborted: source and target iterables have different lengths.")
                conn.close()
                raise ValueError("Source and target files must have the same number of lines.")

            src = src.rstrip("\n")
            tgt = tgt.rstrip("\n")
            sentence_rows.append((sentence_id, src, tgt))

            tokens_raw = _normalize_text(src).split()
            for size in range(1, self.max_fragment_size + 1):
                for fragment in _create_fragments(tokens_raw, size):
                    fragment_rows.append(
                        ("raw", size, " ".join(word.lower() for word in fragment), sentence_id)
                    )
                    total_fragments_raw += 1

            if self.mask_rules:
                tokens_masked = _tokenize_with_masks(src, self.mask_rules)
                for size in range(1, self.max_fragment_size + 1):
                    for fragment in _create_fragments(tokens_masked, size):
                        fragment_rows.append(
                            ("masked", size, " ".join(word.lower() for word in fragment), sentence_id)
                        )
                        total_fragments_masked += 1

            sentence_id += 1

            if len(sentence_rows) >= commit_every:
                conn.executemany("INSERT INTO sentences(id, src, tgt) VALUES (?, ?, ?)", sentence_rows)
                conn.executemany(
                    "INSERT INTO fragments(variant, size, fragment, sentence_id) VALUES (?, ?, ?, ?)",
                    fragment_rows,
                )
                conn.commit()
                sentence_rows = []
                fragment_rows = []
                now = time.perf_counter()
                if now - last_progress_at >= 5.0:
                    self._log.info(
                        "Indexing progress: sentences=%d fragments_raw=%d fragments_masked=%d elapsed=%.1fs",
                        sentence_id,
                        total_fragments_raw,
                        total_fragments_masked,
                        now - started_at,
                    )
                    last_progress_at = now

        if sentence_rows:
            conn.executemany("INSERT INTO sentences(id, src, tgt) VALUES (?, ?, ?)", sentence_rows)
            conn.executemany(
                "INSERT INTO fragments(variant, size, fragment, sentence_id) VALUES (?, ?, ?, ?)",
                fragment_rows,
            )
            conn.commit()

        mask_rules_json = json.dumps(_serialize_mask_rules(self.mask_rules), sort_keys=True)
        conn.executemany(
            "INSERT INTO meta(key, value) VALUES (?, ?)",
            [
                ("schema_version", "3"),
                ("max_fragment_size", str(self.max_fragment_size)),
                ("overlaps", "1" if self.overlaps else "0"),
                ("num_sentences", str(sentence_id)),
                ("mask_rules_json", mask_rules_json),
                ("mask_rules_fingerprint", _mask_rules_fingerprint(self.mask_rules)),
            ],
        )
        conn.commit()
        conn.close()
        elapsed = time.perf_counter() - started_at
        rate = sentence_id / elapsed if elapsed > 0 else 0.0
        self._log.info(
            "Indexing finished: output_db=%s sentences=%d fragments_raw=%d fragments_masked=%d elapsed=%.2fs rate=%.2f_sentences/s",
            output_db,
            sentence_id,
            total_fragments_raw,
            total_fragments_masked,
            elapsed,
            rate,
        )
        return sentence_id

    def index_from_file(self, src_path, tgt_path, output_db, commit_every=2000):
        self._log.info("Indexing from files: src=%s tgt=%s", src_path, tgt_path)
        with open(src_path, encoding="utf-8") as src_handle, open(
            tgt_path, encoding="utf-8"
        ) as tgt_handle:
            return self.index_from_iterables(
                src_handle, tgt_handle, output_db, commit_every=commit_every
            )


class _InMemoryBackend:
    kind = "memory"

    def __init__(self, max_fragment_size, mask_rules, logger=None):
        self.max_fragment_size = max_fragment_size
        self.mask_rules = mask_rules
        self._log = logger or _BASE_LOGGER.getChild("backend.memory")
        self.src_texts = []
        self.tgt_texts = []
        self.raw_fragments_map = {
            size: defaultdict(list) for size in range(1, self.max_fragment_size + 1)
        }
        self.masked_fragments_map = {
            size: defaultdict(list) for size in range(1, self.max_fragment_size + 1)
        }
        # Backward-compatible alias.
        self.corpus_fragments_map = self.raw_fragments_map

    def add_parallel_corpus(self, src_texts, tgt_texts):
        started_at = time.perf_counter()
        src_list = list(src_texts)
        tgt_list = list(tgt_texts)
        if len(src_list) != len(tgt_list):
            raise ValueError("Source and target files must have the same number of lines.")

        start_index = len(self.src_texts)
        self.src_texts.extend(src_list)
        self.tgt_texts.extend(tgt_list)

        for offset, src_text in enumerate(src_list):
            sentence_id = start_index + offset
            tokens_raw = _normalize_text(src_text).split()
            for size in range(1, self.max_fragment_size + 1):
                for fragment in _create_fragments(tokens_raw, size):
                    fragment_tuple = tuple(word.lower() for word in fragment)
                    self.raw_fragments_map[size][fragment_tuple].append(sentence_id)

            if self.mask_rules:
                tokens_masked = _tokenize_with_masks(src_text, self.mask_rules)
                for size in range(1, self.max_fragment_size + 1):
                    for fragment in _create_fragments(tokens_masked, size):
                        fragment_tuple = tuple(word.lower() for word in fragment)
                        self.masked_fragments_map[size][fragment_tuple].append(sentence_id)

        self._log.info(
            "In-memory corpus updated: added_sentences=%d total_sentences=%d elapsed=%.2fs",
            len(src_list),
            len(self.src_texts),
            time.perf_counter() - started_at,
        )
        return len(src_list)

    def has_corpus(self):
        return bool(self.src_texts)

    def get_sentence_ids_for_fragment(self, size, fragment_words, masked=False):
        map_ref = self.masked_fragments_map if masked else self.raw_fragments_map
        fragment_tuple = tuple(word.lower() for word in fragment_words)
        return list(map_ref[size].get(fragment_tuple, []))

    def get_sentence(self, sentence_id):
        return self.src_texts[sentence_id], self.tgt_texts[sentence_id]


class _SQLiteBackend:
    kind = "sqlite"

    def __init__(self, index_path, logger=None):
        self._log = logger or _BASE_LOGGER.getChild("backend.sqlite")
        started_at = time.perf_counter()
        self.index_path = Path(index_path)
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index file not found: {self.index_path}")
        self._log.info("Opening SQLite index at %s", self.index_path)
        self.conn = sqlite3.connect(str(self.index_path))
        self.conn.row_factory = sqlite3.Row
        self._fragments_has_variant = self._detect_fragments_variant_column()
        self.max_fragment_size = self._load_meta_int("max_fragment_size")

        mask_rules_json = self._load_meta_value("mask_rules_json", "[]")
        try:
            serialized_rules = json.loads(mask_rules_json)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Index contains invalid 'mask_rules_json': {exc.msg} (line {exc.lineno}, column {exc.colno})"
            ) from exc
        if not isinstance(serialized_rules, list):
            raise ValueError("Index metadata 'mask_rules_json' must be a JSON array.")

        self.mask_rules = _deserialize_mask_rules(serialized_rules)
        self.mask_rules_fingerprint = self._load_meta_value(
            "mask_rules_fingerprint", _mask_rules_fingerprint(self.mask_rules)
        )
        self._log.info(
            "SQLite index loaded: path=%s max_fragment_size=%d mask_rule_patterns=%d variant_support=%s elapsed=%.2fs",
            self.index_path,
            self.max_fragment_size,
            len(self.mask_rules),
            self._fragments_has_variant,
            time.perf_counter() - started_at,
        )

    def _detect_fragments_variant_column(self):
        rows = self.conn.execute("PRAGMA table_info(fragments)").fetchall()
        return any(row["name"] == "variant" for row in rows)

    def _load_meta_value(self, key, default=None):
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        if row is None:
            return default
        return row["value"]

    def _load_meta_int(self, key):
        value = self._load_meta_value(key)
        if value is None:
            raise ValueError(f"Index is missing required metadata key: {key}")
        return int(value)

    def has_corpus(self):
        row = self.conn.execute("SELECT COUNT(1) AS n FROM sentences").fetchone()
        return bool(row["n"])

    def get_sentence_ids_for_fragment(self, size, fragment_words, masked=False):
        fragment = " ".join(word.lower() for word in fragment_words)
        if self._fragments_has_variant:
            variant = "masked" if masked else "raw"
            rows = self.conn.execute(
                "SELECT sentence_id FROM fragments WHERE variant = ? AND size = ? AND fragment = ?",
                (variant, size, fragment),
            ).fetchall()
        else:
            # Legacy index compatibility (single fragment namespace).
            rows = self.conn.execute(
                "SELECT sentence_id FROM fragments WHERE size = ? AND fragment = ?",
                (size, fragment),
            ).fetchall()
        return [row["sentence_id"] for row in rows]

    def get_sentence(self, sentence_id):
        row = self.conn.execute(
            "SELECT src, tgt FROM sentences WHERE id = ?",
            (sentence_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Sentence id not found in index: {sentence_id}")
        return row["src"], row["tgt"]

    def close(self):
        if getattr(self, "conn", None) is not None:
            self._log.debug("Closing SQLite index connection for %s", self.index_path)
            self.conn.close()
            self.conn = None


class FragmentShotRetriever:
    def __init__(
        self,
        src_texts=None,
        tgt_texts=None,
        max_fragment_size=7,
        overlaps=False,
        mask_rules: Optional[MaskRulesInput] = None,
        index_path=None,
        logger=None,
    ):
        if max_fragment_size < 1:
            raise ValueError("max_fragment_size must be at least 1.")

        self._log = logger or _BASE_LOGGER.getChild("retriever")
        self.overlaps = overlaps
        self.max_fragment_size = max_fragment_size

        if index_path is not None and (src_texts is not None or tgt_texts is not None):
            raise ValueError("Use either src_texts/tgt_texts or index_path, not both.")

        if index_path is not None:
            self._backend = _SQLiteBackend(index_path, logger=self._log.getChild("sqlite"))
            self.max_fragment_size = min(self.max_fragment_size, self._backend.max_fragment_size)

            provided_mask_rules = _prepare_mask_rules(mask_rules) if mask_rules is not None else None
            if provided_mask_rules is not None:
                provided_fingerprint = _mask_rules_fingerprint(provided_mask_rules)
                if provided_fingerprint != self._backend.mask_rules_fingerprint:
                    self._log.error(
                        "Provided mask rules do not match index definition for %s", index_path
                    )
                    raise ValueError(
                        "Mask rules do not match the index definition. "
                        "Rebuild the index or pass the exact same mask rules."
                    )
                self.mask_rules = provided_mask_rules
            else:
                self.mask_rules = self._backend.mask_rules
            self._log.info(
                "Retriever initialized with SQLite index: path=%s max_fragment_size=%d overlaps=%s mask_rule_patterns=%d",
                index_path,
                self.max_fragment_size,
                self.overlaps,
                len(self.mask_rules),
            )
        else:
            self.mask_rules = _prepare_mask_rules(mask_rules)
            self._backend = _InMemoryBackend(
                max_fragment_size=self.max_fragment_size,
                mask_rules=self.mask_rules,
                logger=self._log.getChild("memory"),
            )
            self._log.info(
                "Retriever initialized in-memory: max_fragment_size=%d overlaps=%s mask_rule_patterns=%d",
                self.max_fragment_size,
                self.overlaps,
                len(self.mask_rules),
            )
            if src_texts is not None or tgt_texts is not None:
                if src_texts is None or tgt_texts is None:
                    raise ValueError("Both src_texts and tgt_texts must be provided together.")
                self.add_parallel_corpus(src_texts, tgt_texts)

    @property
    def src_texts(self):
        if self._backend.kind != "memory":
            raise AttributeError("src_texts is only available for in-memory retrievers.")
        return self._backend.src_texts

    @property
    def tgt_texts(self):
        if self._backend.kind != "memory":
            raise AttributeError("tgt_texts is only available for in-memory retrievers.")
        return self._backend.tgt_texts

    @property
    def corpus_fragments_map(self):
        if self._backend.kind != "memory":
            raise AttributeError("corpus_fragments_map is only available for in-memory retrievers.")
        return self._backend.corpus_fragments_map

    @classmethod
    def load_mask_rules(cls, mask_file):
        return _load_mask_rules_from_file(mask_file)

    @classmethod
    def _prepare_mask_rules(cls, mask_rules):
        return _prepare_mask_rules(mask_rules)

    def add_parallel_corpus(self, src_texts, tgt_texts):
        if self._backend.kind != "memory":
            raise ValueError("Cannot add corpus when using index_path. Rebuild the index with Indexer.")
        self._log.info("Adding corpus to in-memory retriever.")
        return self._backend.add_parallel_corpus(src_texts, tgt_texts)

    def add_corpus(self, src_texts, tgt_texts):
        return self.add_parallel_corpus(src_texts, tgt_texts)

    def _build_examples(self, sentence_ids, max_examples_per_shot):
        candidates = list(sentence_ids)
        random.shuffle(candidates)
        examples = []
        for sent_id in candidates[:max_examples_per_shot]:
            src, tgt = self._backend.get_sentence(sent_id)
            examples.append(
                FragmentExample(
                    src=src,
                    tgt=tgt,
                    src_masked=self._masked_text_if_diff(src),
                    tgt_masked=self._masked_text_if_diff(tgt),
                )
            )
        return examples

    def _masked_text_if_diff(self, text):
        if not self.mask_rules:
            return None
        normalized = _normalize_text(text)
        masked = " ".join(_tokenize_with_masks(text, self.mask_rules))
        return masked if masked != normalized else None

    def _mask_fragment_words(self, fragment_words):
        if not self.mask_rules:
            return list(fragment_words)
        return _tokenize_with_masks(" ".join(fragment_words), self.mask_rules)

    def search(self, text, num_shots=6, max_examples_per_shot=None):
        started_at = time.perf_counter()
        if not self._backend.has_corpus():
            raise ValueError("No corpus has been indexed yet. Call add_parallel_corpus or use index_path.")

        per_shot = max_examples_per_shot if max_examples_per_shot is not None else num_shots
        if per_shot < 1:
            raise ValueError("max_examples_per_shot must be at least 1.")

        self._log.info(
            "Search started: query_chars=%d num_shots=%d max_examples_per_shot=%d backend=%s",
            len(text),
            num_shots,
            per_shot,
            self._backend.kind,
        )

        shots = []
        query_tokens = _normalize_text(text).split()
        start_size = min(len(query_tokens), self.max_fragment_size)
        wi_marked = set()

        for size in range(start_size, 0, -1):
            query_fragments = _create_fragments(query_tokens, size)

            for f_idx, raw_fragment in enumerate(query_fragments):
                if ("#" in raw_fragment) or (f_idx in wi_marked and not self.overlaps):
                    continue

                masked_fragment = self._mask_fragment_words(raw_fragment)
                sent_ids = self._backend.get_sentence_ids_for_fragment(
                    size, raw_fragment, masked=False
                )
                match_type = "exact"
                if not sent_ids and self.mask_rules and masked_fragment:
                    sent_ids = self._backend.get_sentence_ids_for_fragment(
                        len(masked_fragment), masked_fragment, masked=True
                    )
                    if sent_ids:
                        match_type = "masked_fallback"

                if sent_ids:
                    shots.append(
                        FragmentShot(
                            index=f_idx,
                            fragment=" ".join(raw_fragment),
                            fragment_masked=" ".join(masked_fragment),
                            match_type=match_type,
                            examples=self._build_examples(sent_ids, per_shot),
                        )
                    )

                    for j in range(f_idx, f_idx + size):
                        query_tokens[j] = "#"
                        wi_marked.add(j)

        shots = sorted(shots, key=lambda x: x.index)
        result = FragmentSearchResult(
            shots=shots,
            num_words=len(query_tokens),
            unknown=[x for x in query_tokens if x != "#"],
        )
        self._log.info(
            "Search finished: shots=%d unknown_tokens=%d elapsed=%.3fs",
            len(result.shots),
            len(result.unknown),
            time.perf_counter() - started_at,
        )
        return result

    def search_batch(
        self,
        queries: Iterable[str],
        batch_size=1000,
        num_shots=6,
        max_examples_per_shot=None,
    ) -> Iterator[FragmentSearchResult]:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        self._log.info(
            "Batch search started: batch_size=%d num_shots=%d max_examples_per_shot=%s",
            batch_size,
            num_shots,
            "auto" if max_examples_per_shot is None else max_examples_per_shot,
        )
        batch = []
        yielded = 0
        for query in queries:
            batch.append(query)
            if len(batch) >= batch_size:
                for item in batch:
                    yielded += 1
                    yield self.search(
                        item,
                        num_shots=num_shots,
                        max_examples_per_shot=max_examples_per_shot,
                    )
                batch = []
                self._log.info("Batch search progress: yielded_results=%d", yielded)

        for item in batch:
            yielded += 1
            yield self.search(
                item,
                num_shots=num_shots,
                max_examples_per_shot=max_examples_per_shot,
            )
        self._log.info("Batch search finished: yielded_results=%d", yielded)

    def get_fragment_shots(self, text, num_shots=6):
        return self.search(text, num_shots=num_shots).to_legacy_dict()

    def close(self):
        if self._backend.kind == "sqlite":
            self._log.info("Closing retriever and SQLite resources.")
            self._backend.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class FragmentShotsRetriever(FragmentShotRetriever):
    """Backward-compatible alias for previous class name."""
