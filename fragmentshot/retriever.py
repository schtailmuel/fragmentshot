import re
import random
from collections import defaultdict

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

        # Use dict of dicts for O(1) fragment lookup
        # Structure: {size: {fragment_tuple: [sentence_indices]}}
        self.corpus_fragments_map = {}
        self._init_corpus_fragments()

    def _init_corpus_fragments(self):
        """Initialize corpus fragments using parallel processing and hash-based indexing."""
        
        # Tokenize all sentences once
        src_tok = [self._remove_punctuation(s).split() for s in self.src_texts]
        
        # Process each fragment size
        for size in range(1, self.max_fragment_size + 1):
            self.corpus_fragments_map[size] = defaultdict(list)
            
            for i, src_sent in enumerate(src_tok):
                fragments = self._create_fragments(src_sent, size)
                for fragment in fragments:
                    # Store as tuple for hashing and lowercase for case-insensitive matching
                    fragment_tuple = tuple(word.lower() for word in fragment)
                    self.corpus_fragments_map[size][fragment_tuple].append(i)

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
        Retrieve fragments based on a text using O(1) hash lookup.
        """
        shots = []

        text_tokenized = self._remove_punctuation(text).split()
        start_size = min(len(text_tokenized), self.max_fragment_size)

        # Use set for O(1) lookup of marked indices
        wi_marked = set()

        for size in range(start_size, 0, -1):

            src_fragments = self._create_fragments(text_tokenized, size)

            for f_idx, fragment in enumerate(src_fragments):
                
                if ("#" in fragment) or (f_idx in wi_marked and not self.overlaps):
                    continue
                
                # Create hashable tuple for O(1) dictionary lookup
                fragment_tuple = tuple(word.lower() for word in fragment)

                sent_ids = self.corpus_fragments_map[size].get(fragment_tuple, [])

                if sent_ids:
                    # Create a copy to shuffle without modifying the original
                    sent_ids = list(sent_ids)
                    random.shuffle(sent_ids)

                    examples = []

                    for sent_id in sent_ids[:num_shots]:  # Slice to avoid unnecessary iteration
                        examples.append({
                            "src_text": self.src_texts[sent_id], 
                            "tgt_text": self.tgt_texts[sent_id]
                        })

                    shots.append(
                        {
                            "index": f_idx,
                            "fragment": " ".join(fragment),
                            "examples": examples
                        }
                    )

                    # Mark all positions covered by this fragment
                    for j in range(f_idx, f_idx + size):
                        text_tokenized[j] = "#"
                        wi_marked.add(j)

        shots = sorted(shots, key=lambda x: x["index"])

        return {
            "shots": shots,
            "num_words": len(text_tokenized),
            "unknown": [x for x in text_tokenized if x != "#"]
        }
