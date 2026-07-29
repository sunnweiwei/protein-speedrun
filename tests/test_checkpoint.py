import tempfile
import unittest
from pathlib import Path

import torch

from speedrun.checkpoint import (
    load_model,
    load_submission_module,
    write_checkpoint,
)


MODEL_CODE = """
import torch
from torch import nn

class TinyEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config["vocab_size"], config["d_model"])

    def encode(self, tokens, padding_mask):
        return self.embedding(tokens)

def build_model(model_config):
    return TinyEncoder(model_config)
"""


class CheckpointContractTest(unittest.TestCase):
    def test_checkpoint_is_self_contained_and_detects_code_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "candidate_model.py"
            source.write_text(MODEL_CODE)
            config = {"vocab_size": 23, "d_model": 8}
            model = load_submission_module(source).build_model(config)
            checkpoint = root / "checkpoint"
            write_checkpoint(
                checkpoint,
                model=model,
                model_config=config,
                candidate_id="contract-test",
                seed=42,
                step=1,
                tokens_seen=100,
                training_seconds=1.0,
                objective="test",
                corpus_sha256="corpus",
                model_code_path=source,
            )
            source.unlink()
            restored, metadata = load_model(checkpoint, torch.device("cpu"))
            tokens = torch.zeros((1, 4), dtype=torch.long)
            embeddings = restored.encode(tokens, torch.ones_like(tokens).bool())
            self.assertEqual(tuple(embeddings.shape), (1, 4, 8))
            self.assertEqual(metadata["candidate_id"], "contract-test")

            (checkpoint / "model.py").write_text(MODEL_CODE + "\n# tampered\n")
            with self.assertRaises(ValueError):
                load_model(checkpoint, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
