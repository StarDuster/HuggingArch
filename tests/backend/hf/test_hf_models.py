import os
import sys
import unittest
from unittest import mock
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))


from dotenv import dotenv_values

from backend import analyze_model


def _dotenv_for_this_test() -> dict:
    """The repo `.env` entries the shell has not set (a Hugging Face token,
    say), for this test's duration only. Loading them into `os.environ` at
    import leaked every secret in `.env` (the production Supabase URL and
    service-role key among them) into every later test of the process —
    collection imports this module in every pytest worker."""
    values = dotenv_values(PROJECT_ROOT / ".env")
    return {k: v for k, v in values.items() if v is not None and k not in os.environ}


DEFAULT_MODELS = [
    "deepseek-ai/DeepSeek-R1",
    "Qwen/Qwen3-Coder-Next",
    "Qwen/Qwen3.5-35B-A3B",
    "meituan-longcat/LongCat-Flash-Thinking-2601",
    "moonshotai/Kimi-K2-Thinking",
    "zai-org/GLM-5",
]


def _get_models():
    env = os.getenv("HUGGINGARCH_TEST_MODELS", "").strip()
    if env:
        return [m.strip() for m in env.split(",") if m.strip()]
    return DEFAULT_MODELS


SEQ_LEN = int(os.getenv("HUGGINGARCH_TEST_SEQ_LEN", "64"))
BATCH_SIZE = int(os.getenv("HUGGINGARCH_TEST_BATCH", "1"))
DTYPE = os.getenv("HUGGINGARCH_TEST_DTYPE", "float16")


class TestHFModels(unittest.TestCase):
    def setUp(self):
        env = mock.patch.dict(os.environ, _dotenv_for_this_test())
        env.start()
        self.addCleanup(env.stop)

    def test_hf_models_smoke(self):
        models = _get_models()
        self.assertTrue(models, "No HuggingFace models configured for tests")

        for model_id in models:
            with self.subTest(model_id=model_id):
                result = analyze_model(
                    model_id=model_id,
                    batch_size=BATCH_SIZE,
                    seq_len=SEQ_LEN,
                    dtype=DTYPE,
                    trust_remote_code=True,
                )
                if not result.get("success", False):
                    raise AssertionError(result.get("error", "Analysis failed without error message"))

                summary = result.get("summary", {})
                self.assertGreater(summary.get("num_layers", 0), 0)
                self.assertGreater(summary.get("hidden_size", 0), 0)

                if "Qwen3-Coder-Next" in model_id:
                    attn_info = summary.get("attention_info", {})
                    self.assertTrue(attn_info.get("layer_types"), "Missing layer_types for hybrid attention")
                    self.assertTrue(attn_info.get("full_attention_interval"), "Missing full_attention_interval")


if __name__ == "__main__":
    unittest.main()
