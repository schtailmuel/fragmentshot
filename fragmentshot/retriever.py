import re
import random

class FragmentShotsRetriever:

    def __init__(self, src_texts, tgt_texts, max_fragment_size=7, overlaps=False):

        if len(src_texts) != len(tgt_texts):
            raise ValueError(
                "Source and target files must have the same number of lines."
            )

        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.overlaps = overlaps
        self.max_fragment_size = max_fragment_size

        self.corpus_fragments_str = {}
        self.corpus_fragments_idx = {}
        self._init_corpus_fragments()

    def _init_corpus_fragments(self):

        src_tok = [self._remove_punctuation(s).split() for s in self.src_texts]

        for size in range(1, self.max_fragment_size + 1):

            self.corpus_fragments_str[size] = []
            self.corpus_fragments_idx[size] = []

            for i, src_sent in enumerate(src_tok):
                res = self._create_fragments(src_sent, size)
                for tok in res:
                    self.corpus_fragments_str[size].append(tok)
                    self.corpus_fragments_idx[size].append(i)

    def _remove_punctuation(self, text):
        text = re.sub(r"[ ]+", " ", text)
        text = re.sub(r"[.,!?:;]", "", text)
        return text

    def _create_fragments(self, xs, n):
        
        fragments = []
        
        for i in range(len(xs) - n + 1):
            fragments.append(xs[i : i + n])
        
        return fragments

    def get_fragment_shots(self, text, num_shots=6):
        """
        Retrieve fragments based on a text.
        """
        shots = []

        text_tokenized = self._remove_punctuation(text).split()
        start_size = min(len(text_tokenized), self.max_fragment_size)

        for size in range(start_size, 0, -1):

            src_fragments = self._create_fragments(text_tokenized, size)
            wi_marked = []

            for f_idx, fragment in enumerate(src_fragments):
                
                if ("#" in fragment) or (f_idx in wi_marked and not self.overlaps):
                    continue
                
                fragment_lower = [x.lower() for x in fragment]

                match_idxs = [
                    _idx
                    for _idx, x in enumerate(self.corpus_fragments_str[size])
                    if x == fragment or x == fragment_lower
                ]

                if match_idxs:
                    
                    sent_ids = [self.corpus_fragments_idx[size][i] for i in match_idxs]
                    random.shuffle(sent_ids)

                    examples = []

                    for sent_id in sent_ids:
                        
                        examples.append({
                            "src_text": self.src_texts[sent_id], 
                            "tgt_text": self.tgt_texts[sent_id]
                        })

                        if len(examples) >= num_shots:
                            break

                    shots.append(
                        {
                            "fragment": " ".join(fragment),
                            "examples": examples
                        }
                    )

                    for j in range(f_idx, f_idx + size):
                        text_tokenized[j] = "#"
                        wi_marked.append(j)

        return {
            "shots": shots,
            "num_words": len(text_tokenized),
            "unknown": [x for x in text_tokenized if x != "#"]
        }