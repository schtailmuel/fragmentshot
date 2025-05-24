import unittest
from fragmentshot.retriever import FragmentShotsRetriever

class TestFragmentShotsRetriever(unittest.TestCase):

    @classmethod
    def setUp(self):
        self.src_texts = [
            "The cat sleeps on the couch",
            "A dog runs in the garden"
        ]
        self.tgt_texts = [
            "Il gatto dorme sul divano",
            "Un cane corre nel giardino"
        ]

    def test_constructor_with_valid_input(self):
        retriever = FragmentShotsRetriever(self.src_texts, self.tgt_texts)
        self.assertEqual(len(retriever.src_texts), 2)
        self.assertEqual(len(retriever.tgt_texts), 2)
        self.assertFalse(retriever.overlaps)
        self.assertEqual(retriever.max_fragment_size, 7)

    def test_constructor_raises_error_on_mismatched_input(self):
        with self.assertRaises(ValueError):
            FragmentShotsRetriever(self.src_texts, self.tgt_texts + ["extra"])

    def test_fragment_initialization(self):
        retriever = FragmentShotsRetriever(self.src_texts, self.tgt_texts, max_fragment_size=3)
        
        self.assertIn(3, retriever.corpus_fragments_str)
        self.assertIn(3, retriever.corpus_fragments_idx)
        self.assertTrue(len(retriever.corpus_fragments_str[3]) > 0)
        self.assertEqual(len(retriever.corpus_fragments_str[3]), len(retriever.corpus_fragments_idx[3]))

    def test_retrieve(self):
        input_sentence = "my dogs sleeps on the floow"
        retriever = FragmentShotsRetriever(self.src_texts, self.tgt_texts)
        results = retriever.get_fragment_shots(input_sentence)
        self.assertEqual(len(results['shots']), 1)

if __name__ == "__main__":
    unittest.main()
