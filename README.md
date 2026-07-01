# 🧩 FragmentShot Retriever

A Python package for retrieving exemplary translations for text based on fragments from parallel corpora.  

## Features

- Fragment extraction from source and target texts
- Configurable maximum fragment size
- Option to enable or disable fragment overlaps
- Easy integration for retrieval workflows

## Installation

You can install this package locally:

```bash
pip install fragmentshot
```

Or clone the repo and install in editable mode:

```bash
git clone https://github.com/schtailmuel/fragmentshot.git
cd fragmentshot
pip install -e .
```

## Usage

```python
from fragmentshot import FragmentShotRetriever

src_texts = [
    "this is a sample source sentence.",
    "another example source sentence."
]

tgt_texts = [
    "dies ist ein Beispiel im Zieltext.",
    "noch ein Beispiel."
]

retriever = FragmentShotRetriever(max_fragment_size=5, overlaps=False)
retriever.add_parallel_corpus(src_texts, tgt_texts)

response = retriever.search("The source of this is unknown.")

for shot in response.shots:
    print(f"Fragment: {shot.fragment}")
    for example in shot.examples:
        print(f"  -> {example.src} | {example.tgt}")
```

### Build -> Save -> Load (Large Corpora)

```python
from fragmentshot import Indexer, FragmentShotRetriever

# Step 1: Build once (streams from files)
indexer = Indexer(max_fragment_size=6, overlaps=False, mask_rules={"NUM": r"[0-9]+"})
indexer.index_from_file(src_path="src.txt", tgt_path="tgt.txt", output_db="corpus.db")

# Step 2: Fast retrieval from the saved SQLite index
retriever = FragmentShotRetriever(
    index_path="corpus.db",
    max_fragment_size=6,  # Default for queries (can be overridden)
    # optional: omitted because rules are loaded from index metadata
    # mask_rules={"NUM": r"[0-9]+"},
)
result = retriever.search(
    "the 14. of december was nice", 
    max_examples_per_shot=5,
    max_fragment_size=8,  # Optional: override the default
)
```

### Masking

Masking is used as fallback. Retrieval first tries an exact fragment match, and if that fails it tries the masked fragment (for example replacing numbers with `[NUM]`). This keeps exact matches prioritized while preserving robust generalization.

```python
retriever = FragmentShotRetriever(
    max_fragment_size=6,
    mask_rules={
        "NUM": r"[0-9]+",
        "NAME": [r"Alice", r"Bob", r"Charlie"],
    },
)
retriever.add_parallel_corpus(src_texts, tgt_texts)
result = retriever.search("the 14. of december was nice")
# When fallback is used, examples can include:
# example.src_masked == "the [NUM] of december was nice"
```

### Batch Queries

```python
queries = ["Sentence one...", "Sentence two...", "Sentence three..."]

for result in retriever.search_batch(
    queries, 
    batch_size=1000, 
    max_examples_per_shot=5,
    max_fragment_size=6,  # Optional: override at query time
):
    print(result.shots)
```

### Retrieval Parameters

```python
result = retriever.search(
    "The source of this is unknown.",
    max_examples_per_shot=5,        # Limit examples per shot
    max_fragment_size=8,            # Override at query time (optional)
)
```

The `max_fragment_size` parameter can now be specified at query time, allowing you to experiment with different fragment sizes without rebuilding the index:

```python
retriever = FragmentShotRetriever(index_path="corpus.db")

# Use instance default
result = retriever.search("some text")

# Try shorter fragments (faster, broader matches)
result = retriever.search("some text", max_fragment_size=4)

# Try longer fragments (slower, more specific matches)
result = retriever.search("some text", max_fragment_size=8)
```

**Note:** Word boundaries are added automatically around each mask regex, so you do not need to include `\b` in your mask patterns.

This allows query fragment `the 14 of december was nice` to match corpus fragment `the 12 of december was nice` without rewriting either fragment text.
When using a saved SQLite index, mask rules are stored inside the index metadata. If you pass `mask_rules` at retrieval time, they must match the indexed definition.
Each shot includes `fragment_masked` and `match_type` (`exact` or `masked_fallback`) to show what was searched.
Each example may include `src_masked` and `tgt_masked` when masking changed the normalized text (those keys are omitted otherwise).

### CLI

Build an index once:

```bash
fragmentshot --src src.txt --tgt tgt.txt --build-index corpus.db --max-fragment-size 6
```

Build with masks:

```bash
fragmentshot --src src.txt --tgt tgt.txt --build-index corpus.db --max-fragment-size 6 --mask masks.json
```

Query from the index:

```bash
fragmentshot --index-path corpus.db --text "the 14. of december was nice" --max-fragment-size 6 --mask masks.json --log-level INFO
```

Legacy one-off mode (loads txt files directly):

```bash
fragmentshot --src src.txt --tgt tgt.txt --text "the 14. of december was nice" --max-fragment-size 6 --mask masks.json --log-level INFO
```

Logging is built in for index loading, index building progress, and search lifecycle. In library usage, configure Python logging for the `fragmentshot` logger to see these events.

Mask file format (`masks.json`):

```json
{
  "NUM": ["[0-9]+"],
  "NAME": ["Alice", "Bob", "Charlie"]
}
```

**Result**

```json
{
  "shots": [
    {
      "index": 1,
      "fragment": "source",
      "fragment_masked": "source",
      "match_type": "exact",
      "examples": [
        {
          "src": "this is a sample source sentence.",
          "tgt": "dies ist ein Beispiel im Zieltext."
        },
        {
          "src": "another example source sentence.",
          "tgt": "noch ein Beispiel."
        }
      ]
    },
    {
      "index": 3,
      "fragment": "this is",
      "fragment_masked": "this is",
      "match_type": "exact",
      "examples": [
        {
          "src": "this is a sample source sentence.",
          "tgt": "dies ist ein Beispiel im Zieltext."
        }
      ]
    }
  ],
  "num_words": 6,
  "unknown": [
    "The",
    "of",
    "unknown"
  ]
}
```

## Storage Model

The SQLite index uses a word-level inverted-index layout (schema v4):

- `word_index(variant, word) -> sentence_id` where `variant` is `raw` or `masked`
- `sentences(id) -> (src, tgt, src_tokens, src_tokens_masked)` where tokens are pre-tokenized and stored as JSON arrays

This design enables:
- **Fast indexing:** Only unique words are stored (~2-5M) instead of all n-grams (~42M)
- **Fast retrieval:** Word filtering + on-demand n-gram verification with batch token fetching
- **Query-time flexibility:** `max_fragment_size` can be adjusted per query without reindexing

**Legacy schema (v3):** Old indices use `fragments(variant, size, fragment) -> sentence_id` with precomputed n-grams. These are automatically detected and still supported for backward compatibility.

## Testing 

Run unit tests with:

```bash
python -m unittest discover tests
```

## License 

This project is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License.

Created by Samuel Frontull

## Citation
If you use this code in your research, please cite our paper (LREC 2026):

```bibtex
@inproceedings{frontull-etal-2026-every,
  title = {Every Word Presented in Context: Syntactic Coverage as Objective for Low-Resource Machine Translation with Large Language Models},
  author = {Frontull, Samuel and Ströhle, Thomas},
  booktitle = {Proceedings of the Fifteenth Language Resources and Evaluation Conference (LREC 2026)},
  month = {May},
  year = {2026},
  pages = {8824--8837},
  address = {Palma, Mallorca, Spain},
  publisher = {European Language Resources Association (ELRA)},
  editor = {Piperidis, Stelios and Bel, Núria and van den Heuvel, Henk and Ide, Nancy and Krek, Simon and Toral, Antonio},
  doi = {10.63317/5jpokiam9tjt},
  abstract = {Large Language Models (LLMs) have demonstrated strong capabilities in multilingual machine translation. However, they underperform for low-resource languages, indicating the need for more explicit instructional guidance. In this work, we introduce Fragment-Shot Prompting, a novel few-shot prompting method that aims to retrieve examples for every word occurring in the sentence to be translated, illustrating their use and meaning in context. We evaluate our method on translation between Italian, Ladin (Val Badia) and Ladin (Gherdëina) and compare its performance with zero-shot prompting, random few-shot prompting, as well as established lexical and semantic retrieval strategies. We conduct these experiments using state-of-the-art LLMs, including GPT-3.5, GPT-4o, o1-mini, LlaMA-3.3, and DeepSeek-R1. Our results demonstrate that LLMs can extract substantial value from limited data when translating from a low- to the high-resource language. However, this does not apply to translations into the low-resource languages, where the prompting method plays a much more important role. In particular, our method consistently delivers the best results and enables significant gains. Even though translation performance into Ladin remains limited with the available resources, our results highlight the importance of syntactic coverage for improving translation accuracy and ariant-specific adaptation in low-resource scenarios.}
}
```
