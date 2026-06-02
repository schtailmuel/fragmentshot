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
