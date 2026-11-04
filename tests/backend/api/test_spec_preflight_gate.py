import json
import sys
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))


from backend.api._spec_preflight import probe_repo
from backend.hf.task_gate import HubTask


class TestProbeRepoTaskGate(unittest.TestCase):
    """`probe_repo` is the last check before a generate run starts — an
    out-of-scope repo must be refused here, whichever shape it takes:
    no root config.json (the common case), or a root config.json on a
    task no spec can describe."""

    def _no_snapshot(self):
        return mock.patch(
            "backend.api._spec_preflight.load_snapshot_config", return_value=None
        )

    def _gate(self, task):
        return mock.patch(
            "backend.hf.task_gate.fetch_hub_task", return_value=task
        )

    @staticmethod
    def _diffusers_repo(task):
        return HubTask(page_tag=task, library="diffusers", resolved=False,
                       resolved_pipeline=None, registered=frozenset(),
                       architectures=(), files=frozenset())

    def test_wrong_task_without_config_refuses_by_task(self):
        from huggingface_hub.errors import EntryNotFoundError

        exc = EntryNotFoundError("no config")  # plain Exception, no response kwarg
        with self._no_snapshot(), \
             mock.patch("huggingface_hub.hf_hub_download", side_effect=exc), \
             self._gate(self._diffusers_repo("image-text-to-video")):
            config, refusal = probe_repo("MiniMaxAI/MiniMax-H3")
        self.assertIsNone(config)
        self.assertEqual(refusal["status"], "repo_unavailable")
        self.assertIn("Image-Text-to-Video model", refusal["message"])

    def test_wrong_task_with_config_refuses_before_run(self):
        # A diffusion repo can ship a root config.json — the download
        # succeeds, but the task still puts it out of scope.
        with self._no_snapshot(), \
             mock.patch("huggingface_hub.hf_hub_download", return_value="/tmp/cfg.json"), \
             mock.patch("builtins.open", mock.mock_open(
                 read_data=json.dumps({"model_type": "some-vae"}))), \
             self._gate(self._diffusers_repo("text-to-video")):
            config, refusal = probe_repo("org/video-model")
        self.assertIsNone(config)
        self.assertEqual(refusal["status"], "repo_unavailable")
        self.assertIn("Text-to-Video model", refusal["message"])

    def test_supported_task_with_config_proceeds(self):
        with self._no_snapshot(), \
             mock.patch("huggingface_hub.hf_hub_download", return_value="/tmp/cfg.json"), \
             mock.patch("builtins.open", mock.mock_open(
                 read_data=json.dumps({"model_type": "qwen3"}))), \
             self._gate(HubTask(
                 page_tag="text-generation", library="transformers",
                 resolved=True, resolved_pipeline="text-generation",
                 registered=frozenset(),
                 architectures=("Qwen3ForCausalLM",), files=frozenset())):
            config, refusal = probe_repo("Qwen/Qwen3-8B")
        self.assertEqual(config, {"model_type": "qwen3"})
        self.assertIsNone(refusal)


if __name__ == "__main__":
    unittest.main()
