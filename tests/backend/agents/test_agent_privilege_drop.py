"""The agent must never run as root — some CLIs refuse it (Claude Code:
`--dangerously-skip-permissions cannot be used with root/sudo`). The uid drop
to `HA_AGENT_USER` and environment scrubbing apply to host and BYOK runs.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from backend.agents.base import AgentBackend


class TestAgentPrivilegeDrop(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.get("HA_AGENT_USER")

    def tearDown(self):
        if self._saved is None:
            os.environ.pop("HA_AGENT_USER", None)
        else:
            os.environ["HA_AGENT_USER"] = self._saved

    def test_drops_for_both_trusted_and_untrusted_when_configured(self):
        os.environ["HA_AGENT_USER"] = "specagent"
        # untrusted (BYOK env provided)
        self.assertEqual(AgentBackend._agent_user({"n_experts_active": "v"}), "specagent")
        self.assertEqual(AgentBackend._priv_kwargs({"n_experts_active": "v"}), {"user": "specagent"})
        # trusted / host-admin (env=None) — this is the path that used to run
        # as root and crash the agent CLI.
        self.assertEqual(AgentBackend._agent_user(None), "specagent")
        self.assertEqual(AgentBackend._priv_kwargs(None), {"user": "specagent"})

    def test_no_drop_on_dev_when_unset(self):
        os.environ.pop("HA_AGENT_USER", None)
        self.assertIsNone(AgentBackend._agent_user(None))
        self.assertIsNone(AgentBackend._agent_user({"n_experts_active": "v"}))
        self.assertEqual(AgentBackend._priv_kwargs(None), {})


if __name__ == "__main__":
    unittest.main()
