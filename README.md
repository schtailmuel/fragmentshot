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

**Result**

```json
{
  "shots": [
    {
      "index": 1,
      "fragment": "source",
      "examples": [
        {
          "src_text": "this is a sample source sentence.",
          "tgt_text": "dies ist ein Beispiel im Zieltext."
        },
        {
          "src_text": "another example source sentence.",
          "tgt_text": "noch ein Beispiel."
        }
      ]
    },
    {
      "index": 3,
      "fragment": "this is",
      "examples": [
        {
          "src_text": "this is a sample source sentence.",
          "tgt_text": "dies ist ein Beispiel im Zieltext."
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

## Testing 

Run unit tests with:

```bash
python -m unittest discover tests
```

## License 

This project is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License.

Created by Samuel Frontull

## Citation
If you use this code in your research, please cite our paper (preprint):

```bibtex
@misc{frontull:stroehle:2025,
      title={Compensating for Data with Reasoning: Low-Resource Machine Translation with LLMs}, 
      author={Samuel Frontull and Thomas Ströhle},
      year={2025},
      eprint={2505.22293},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.22293}, 
}
```