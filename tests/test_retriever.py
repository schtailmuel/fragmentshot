import unittest
from fragmentshot.retriever import FragmentShotsRetriever

class TestFragmentShotsRetriever(unittest.TestCase):

    @classmethod
    def setUp(self):
        self.src_texts = [
            "this is a sample source sentence.",
            "another example source sentence."
        ]

        self.tgt_texts = [
            "dies ist ein Beispiel im Zieltext.",
            "noch ein Beispiel."
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
        input_sentence = "The source of this is unknown."
        retriever = FragmentShotsRetriever(self.src_texts, self.tgt_texts)
        results = retriever.get_fragment_shots(input_sentence)
        self.assertEqual(len(results['shots']), 2)
        self.assertEqual(results['shots'][0]['fragment'], "source")

if __name__ == "__main__":
    unittest.main()
