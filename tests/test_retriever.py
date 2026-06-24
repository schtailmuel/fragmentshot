import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from fragmentshot import FragmentShotRetriever as PublicFragmentShotRetriever
from fragmentshot.cli import main
from fragmentshot.retriever import (
    FragmentSearchResult,
    FragmentShotRetriever,
    FragmentShotsRetriever,
    Indexer,
)


class TestFragmentShotRetriever(unittest.TestCase):

    def setUp(self):
        self.src_texts = [
            "this is a sample source sentence.",
            "another example source sentence.",
        ]

        self.tgt_texts = [
            "dies ist ein Beispiel im Zieltext.",
            "noch ein Beispiel.",
        ]

    def test_constructor_supports_decoupled_indexing(self):
        retriever = FragmentShotRetriever(max_fragment_size=5, overlaps=False)
        added = retriever.add_parallel_corpus(self.src_texts, self.tgt_texts)
        self.assertEqual(added, 2)
        self.assertEqual(len(retriever.src_texts), 2)
        self.assertEqual(len(retriever.tgt_texts), 2)
        self.assertFalse(retriever.overlaps)
        self.assertEqual(retriever.max_fragment_size, 5)

    def test_public_import_is_available(self):
        retriever = PublicFragmentShotRetriever(max_fragment_size=5)
        self.assertEqual(retriever.max_fragment_size, 5)

    def test_add_parallel_corpus_raises_error_on_mismatched_input(self):
        retriever = FragmentShotRetriever()
        with self.assertRaises(ValueError):
            retriever.add_parallel_corpus(self.src_texts, self.tgt_texts + ["extra"])

    def test_fragment_initialization(self):
        retriever = FragmentShotRetriever(max_fragment_size=3)
        retriever.add_parallel_corpus(self.src_texts, self.tgt_texts)

        self.assertIn(3, retriever.corpus_fragments_map)
        self.assertTrue(len(retriever.corpus_fragments_map[3]) > 0)
        for key, value in retriever.corpus_fragments_map[3].items():
            self.assertIsInstance(key, tuple)
            self.assertIsInstance(value, list)

    def test_search_returns_structured_result(self):
        input_sentence = "The source of this is unknown."
        retriever = FragmentShotRetriever()
        retriever.add_parallel_corpus(self.src_texts, self.tgt_texts)
        result = retriever.search(input_sentence)

        self.assertIsInstance(result, FragmentSearchResult)
        self.assertEqual(len(result.shots), 2)
        self.assertEqual(result.shots[0].fragment, "source")
        self.assertIn(
            result.shots[0].examples[0].src,
            {
                "this is a sample source sentence.",
                "another example source sentence.",
            },
        )
        self.assertEqual(result.shots[0].examples[0].src_text, result.shots[0].examples[0].src)
        self.assertEqual(result.shots[0].match_type, "exact")
        self.assertEqual(result.shots[0].fragment_masked, "source")

    def test_mask_rules_match_corpus_and_query(self):
        src_texts = [
            "the 12. of december was nice",
            "this is unrelated",
        ]
        tgt_texts = ["ziel 1", "ziel 2"]

        retriever = FragmentShotRetriever(
            max_fragment_size=6,
            mask_rules={"NUM": r"[0-9]+"},
        )
        retriever.add_parallel_corpus(src_texts, tgt_texts)
        result = retriever.search("the 14. of december was nice", num_shots=1)

        self.assertEqual(result.unknown, [])
        self.assertEqual(len(result.shots), 1)
        self.assertEqual(result.shots[0].fragment, "the 14 of december was nice")
        self.assertEqual(result.shots[0].fragment_masked, "the [NUM] of december was nice")
        self.assertEqual(result.shots[0].match_type, "masked_fallback")
        self.assertEqual(
            result.shots[0].examples[0].src,
            "the 12. of december was nice",
        )
        self.assertEqual(
            result.shots[0].examples[0].src_masked,
            "the [NUM] of december was nice",
        )
        self.assertEqual(result.shots[0].examples[0].tgt_masked, "ziel [NUM]")

    def test_load_mask_rules_from_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            mask_file = Path(tmp_dir) / "masks.json"
            mask_file.write_text('{"NUM": ["[0-9]+"]}\n', encoding="utf-8")

            mask_rules = FragmentShotRetriever.load_mask_rules(mask_file)

        self.assertEqual(len(mask_rules), 1)
        self.assertEqual(mask_rules[0][0], "[NUM]")
        self.assertEqual(mask_rules[0][1].pattern, r"\b(?:[0-9]+)\b")
        self.assertIsNotNone(mask_rules[0][1].search("value 42"))
        self.assertIsNone(mask_rules[0][1].search("abc42def"))

    def test_load_mask_rules_from_file_with_regex_list(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            mask_file = Path(tmp_dir) / "masks.json"
            mask_file.write_text('{"NAME": ["Alice", "Bob"]}\n', encoding="utf-8")

            mask_rules = FragmentShotRetriever.load_mask_rules(mask_file)

        self.assertEqual(len(mask_rules), 2)
        self.assertEqual(mask_rules[0][0], "[NAME]")
        self.assertEqual(mask_rules[1][0], "[NAME]")
        self.assertIsNotNone(mask_rules[0][1].search("Alice"))
        self.assertIsNotNone(mask_rules[1][1].search("Bob"))

    def test_mask_patterns_are_wrapped_with_word_boundaries(self):
        retriever = FragmentShotRetriever(mask_rules={"NUM": r"[0-9]+"})

        replacement, pattern = retriever.mask_rules[0]
        self.assertEqual(replacement, "[NUM]")
        self.assertEqual(pattern.pattern, r"\b(?:[0-9]+)\b")
        self.assertIsNotNone(pattern.search("value 42"))
        self.assertIsNone(pattern.search("abc42def"))

    def test_load_mask_rules_raises_on_invalid_lines(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            mask_file = Path(tmp_dir) / "bad_masks.json"
            mask_file.write_text("invalid json\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                FragmentShotRetriever.load_mask_rules(mask_file)

    def test_load_mask_rules_accepts_single_string(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            mask_file = Path(tmp_dir) / "masks.json"
            mask_file.write_text('{"NUM": "[0-9]+"}\n', encoding="utf-8")

            mask_rules = FragmentShotRetriever.load_mask_rules(mask_file)

        self.assertEqual(len(mask_rules), 1)
        self.assertEqual(mask_rules[0][0], "[NUM]")

    def test_cli_supports_mask_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            src_file = tmp_path / "src.txt"
            tgt_file = tmp_path / "tgt.txt"
            mask_file = tmp_path / "mask.json"

            src_file.write_text("the 12. of december was nice\n", encoding="utf-8")
            tgt_file.write_text("ziel\n", encoding="utf-8")
            mask_file.write_text('{"NUM": ["[0-9]+"]}\n', encoding="utf-8")

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "--src", str(src_file),
                        "--tgt", str(tgt_file),
                        "--text", "the 14. of december was nice",
                        "--max-fragment-size", "6",
                        "--mask", str(mask_file),
                        "--num-shots", "1",
                    ]
                )

            self.assertEqual(exit_code, 0)
            parsed = json.loads(stdout.getvalue())
            self.assertEqual(parsed["shots"][0]["fragment"], "the 14 of december was nice")
            self.assertEqual(parsed["shots"][0]["fragment_masked"], "the [NUM] of december was nice")
            self.assertEqual(parsed["shots"][0]["match_type"], "masked_fallback")
            self.assertEqual(parsed["shots"][0]["examples"][0]["src"], "the 12. of december was nice")
            self.assertEqual(
                parsed["shots"][0]["examples"][0]["src_masked"],
                "the [NUM] of december was nice",
            )

    def test_cli_supports_multiple_regexes_for_single_key(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            src_file = tmp_path / "src.txt"
            tgt_file = tmp_path / "tgt.txt"
            mask_file = tmp_path / "mask.json"

            src_file.write_text("Alice was nice\n", encoding="utf-8")
            tgt_file.write_text("ziel\n", encoding="utf-8")
            mask_file.write_text('{"NAME": ["Alice", "Bob"]}\n', encoding="utf-8")

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "--src", str(src_file),
                        "--tgt", str(tgt_file),
                        "--text", "Bob was nice",
                        "--max-fragment-size", "3",
                        "--mask", str(mask_file),
                        "--num-shots", "1",
                    ]
                )

            self.assertEqual(exit_code, 0)
            parsed = json.loads(stdout.getvalue())
            self.assertEqual(parsed["shots"][0]["fragment"], "Bob was nice")

    def test_multiple_regexes_for_single_key_in_python_api(self):
        src_texts = ["Alice was nice"]
        tgt_texts = ["ziel"]

        retriever = FragmentShotRetriever(
            max_fragment_size=3,
            mask_rules={"NAME": [r"Alice", r"Bob"]},
        )
        retriever.add_parallel_corpus(src_texts, tgt_texts)
        result = retriever.search("Bob was nice", num_shots=1)

        self.assertEqual(len(result.shots), 1)
        self.assertEqual(result.shots[0].fragment, "Bob was nice")
        self.assertEqual(result.shots[0].fragment_masked, "[NAME] was nice")
        self.assertEqual(result.shots[0].match_type, "masked_fallback")
        self.assertEqual(result.unknown, [])

    def test_legacy_class_and_method_still_work(self):
        retriever = FragmentShotsRetriever(self.src_texts, self.tgt_texts)
        result = retriever.get_fragment_shots("The source of this is unknown.")
        self.assertEqual(result["shots"][0]["fragment"], "source")
        self.assertIn("src_text", result["shots"][0]["examples"][0])
        self.assertIn("tgt_text", result["shots"][0]["examples"][0])

    def test_sqlite_index_build_and_load(self):
        src_texts = [
            "the 12. of december was nice",
            "another source example",
        ]
        tgt_texts = ["ziel 1", "ziel 2"]

        with tempfile.TemporaryDirectory() as tmp_dir:
            index_path = Path(tmp_dir) / "corpus.db"
            indexer = Indexer(max_fragment_size=6, overlaps=False)
            count = indexer.index_from_iterables(src_texts, tgt_texts, index_path)
            self.assertEqual(count, 2)

            retriever = FragmentShotRetriever(index_path=index_path, max_fragment_size=6)
            result = retriever.search("the 12 of december was nice", max_examples_per_shot=1)

        self.assertEqual(len(result.shots), 1)
        self.assertEqual(result.shots[0].fragment, "the 12 of december was nice")
        self.assertEqual(len(result.shots[0].examples), 1)
        self.assertEqual(result.shots[0].examples[0].src, "the 12. of december was nice")

    def test_example_masked_fields_only_present_when_different(self):
        retriever = FragmentShotRetriever(max_fragment_size=3)
        retriever.add_parallel_corpus(["this is source"], ["this is target"])
        result = retriever.search("this is source", num_shots=1)

        example = result.shots[0].examples[0]
        self.assertIsNone(example.src_masked)
        self.assertIsNone(example.tgt_masked)

        payload = result.to_dict()
        self.assertNotIn("src_masked", payload["shots"][0]["examples"][0])
        self.assertNotIn("tgt_masked", payload["shots"][0]["examples"][0])

    def test_sqlite_index_with_masks_auto_loads_rules(self):
        src_texts = ["the 12. of december was nice"]
        tgt_texts = ["ziel 1"]

        with tempfile.TemporaryDirectory() as tmp_dir:
            index_path = Path(tmp_dir) / "corpus.db"
            indexer = Indexer(max_fragment_size=6, overlaps=False, mask_rules={"NUM": r"[0-9]+"})
            indexer.index_from_iterables(src_texts, tgt_texts, index_path)

            # No mask_rules passed at retrieval time: rules are loaded from index metadata.
            retriever = FragmentShotRetriever(index_path=index_path, max_fragment_size=6)
            result = retriever.search("the 14. of december was nice", max_examples_per_shot=1)

        self.assertEqual(len(result.shots), 1)
        self.assertEqual(result.shots[0].fragment, "the 14 of december was nice")
        self.assertEqual(result.shots[0].fragment_masked, "the [NUM] of december was nice")
        self.assertEqual(result.shots[0].match_type, "masked_fallback")
        self.assertEqual(result.shots[0].examples[0].src, "the 12. of december was nice")

    def test_masking_prefers_exact_before_fallback(self):
        src_texts = [
            "the 12. of december was nice",
            "the 14. of december was nice",
        ]
        tgt_texts = ["ziel 12", "ziel 14"]
        retriever = FragmentShotRetriever(
            max_fragment_size=6,
            mask_rules={"NUM": r"[0-9]+"},
        )
        retriever.add_parallel_corpus(src_texts, tgt_texts)
        result = retriever.search("the 14. of december was nice", num_shots=3)

        self.assertEqual(len(result.shots), 1)
        self.assertEqual(result.shots[0].match_type, "exact")
        self.assertEqual(result.shots[0].fragment_masked, "the [NUM] of december was nice")
        self.assertEqual(len(result.shots[0].examples), 1)
        self.assertEqual(result.shots[0].examples[0].src, "the 14. of december was nice")

    def test_sqlite_index_raises_on_mismatched_masks(self):
        src_texts = ["the 12. of december was nice"]
        tgt_texts = ["ziel 1"]

        with tempfile.TemporaryDirectory() as tmp_dir:
            index_path = Path(tmp_dir) / "corpus.db"
            indexer = Indexer(max_fragment_size=6, overlaps=False, mask_rules={"NUM": r"[0-9]+"})
            indexer.index_from_iterables(src_texts, tgt_texts, index_path)

            with self.assertRaises(ValueError):
                FragmentShotRetriever(
                    index_path=index_path,
                    max_fragment_size=6,
                    mask_rules={"NUM": r"[0-9]{2}"},
                )

    def test_search_batch_returns_generator_results(self):
        retriever = FragmentShotRetriever(max_fragment_size=3)
        retriever.add_parallel_corpus(["Alice was nice", "Bob was nice"], ["z1", "z2"])

        queries = ["Alice was nice", "Bob was nice", "Unknown tokens"]
        results = list(retriever.search_batch(queries, batch_size=2, max_examples_per_shot=1))

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].shots[0].fragment, "Alice was nice")
        self.assertEqual(results[1].shots[0].fragment, "Bob was nice")
        self.assertEqual(results[2].unknown, ["Unknown", "tokens"])

    def test_search_respects_max_examples_per_shot(self):
        retriever = FragmentShotRetriever(max_fragment_size=1)
        retriever.add_parallel_corpus(
            ["source a", "source b", "source c"],
            ["ziel a", "ziel b", "ziel c"],
        )
        result = retriever.search("source", max_examples_per_shot=2)
        self.assertEqual(len(result.shots[0].examples), 2)

    def test_cli_build_and_query_index(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            src_file = tmp_path / "src.txt"
            tgt_file = tmp_path / "tgt.txt"
            db_file = tmp_path / "corpus.db"

            src_file.write_text("the 12. of december was nice\n", encoding="utf-8")
            tgt_file.write_text("ziel\n", encoding="utf-8")

            build_stdout = io.StringIO()
            with redirect_stdout(build_stdout):
                build_code = main(
                    [
                        "--src",
                        str(src_file),
                        "--tgt",
                        str(tgt_file),
                        "--build-index",
                        str(db_file),
                        "--max-fragment-size",
                        "6",
                    ]
                )
            self.assertEqual(build_code, 0)
            self.assertTrue(db_file.exists())
            build_payload = json.loads(build_stdout.getvalue())
            self.assertEqual(build_payload["num_sentences"], 1)

            query_stdout = io.StringIO()
            with redirect_stdout(query_stdout):
                query_code = main(
                    [
                        "--index-path",
                        str(db_file),
                        "--text",
                        "the 12 of december was nice",
                        "--max-fragment-size",
                        "6",
                    ]
                )
            self.assertEqual(query_code, 0)
            parsed = json.loads(query_stdout.getvalue())
            self.assertEqual(parsed["shots"][0]["fragment"], "the 12 of december was nice")

if __name__ == "__main__":
    unittest.main()
