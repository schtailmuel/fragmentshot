# FragmentShot Retriever

A Python package for retrieving exemplary translations for text based on fragments from parallel corpora.  

## Features

- Fragment extraction from source and target texts
- Configurable maximum fragment size
- Option to enable or disable fragment overlaps
- Easy integration for retrieval workflows

## Installation

You can install this package locally:

```bash
pip install .
```

Or clone the repo and install in editable mode:

```bash
git clone https://github.com/schtailmuel/fragmentshot.git
cd fragmentshot
pip install -e .
```

## Usage 

```python 
from fragmentshot.retriever import FragmentShotsRetriever

src_texts = [
    "this is a sample source sentence.",
    "another example source sentence."
]

tgt_texts = [
    "dies ist ein Beispiel im Zieltext.",
    "noch ein Beispiel."
]

retriever = FragmentShotsRetriever(src_texts, tgt_texts, max_fragment_size=5, overlaps=False)

result = retriever.get_fragment_shots("The source of this is unknown.")
print(result)
```

## Testing 

Run unit tests with:

```bash
python -m unittest discover tests
```

## License 

This project is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License.

Created by Samuel Frontull
