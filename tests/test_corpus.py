import tempfile
import unittest
from pathlib import Path

import numpy as np

from speedrun.corpus import load_corpus, save_corpus, sha256_file
from speedrun.prepare_corpus import _integer_sequence_ids


class CorpusContractTest(unittest.TestCase):
    def test_mmcif_sequence_ids_allow_unresolved_blanks(self) -> None:
        values = np.asarray(["", "2", "10", ".", "?"])
        normalized = _integer_sequence_ids(values)
        np.testing.assert_array_equal(normalized, [-1, 2, 10, -1, -1])

    def test_round_trip_without_pickle(self):
        sequence = np.arange(40, dtype=np.uint8) % 20
        coords = np.stack(
            [
                np.arange(40),
                np.zeros(40),
                np.zeros(40),
            ],
            axis=-1,
        ).astype(np.float32)
        structure = (sequence, coords, np.ones(40, dtype=np.bool_))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "corpus.npz"
            save_corpus(
                path,
                train_sequences=[sequence],
                probe_train=[structure],
                probe_eval=[structure],
            )
            corpus = load_corpus(path)
            self.assertEqual(len(corpus.train), 1)
            np.testing.assert_array_equal(corpus.train.sequence(0), sequence)

    def test_refuses_overwrite(self):
        sequence = np.arange(40, dtype=np.uint8) % 20
        coords = np.zeros((40, 3), dtype=np.float32)
        structure = (sequence, coords, np.ones(40, dtype=np.bool_))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "corpus.npz"
            save_corpus(
                path,
                train_sequences=[sequence],
                probe_train=[structure],
                probe_eval=[structure],
            )
            with self.assertRaises(FileExistsError):
                save_corpus(
                    path,
                    train_sequences=[sequence],
                    probe_train=[structure],
                    probe_eval=[structure],
                )

    def test_packed_corpus_bytes_are_deterministic(self):
        sequence = np.arange(40, dtype=np.uint8) % 20
        coords = np.stack(
            [np.arange(40), np.zeros(40), np.ones(40)], axis=-1
        ).astype(np.float32)
        structure = (sequence, coords, np.ones(40, dtype=np.bool_))
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first.npz"
            second = Path(temporary) / "second.npz"
            for output in (first, second):
                save_corpus(
                    output,
                    train_sequences=[sequence],
                    probe_train=[structure],
                    probe_eval=[structure],
                )
            self.assertEqual(sha256_file(first), sha256_file(second))


if __name__ == "__main__":
    unittest.main()
