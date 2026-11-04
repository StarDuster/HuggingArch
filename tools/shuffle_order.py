"""Run a test file's tests in a different order, on purpose.

A test that shares an object with the rest of its file is only as sound as
the file is order-independent: if one test leaves a mark on the shared
answer, the next test sees it, and whether that shows up depends on which
ran first. Running the file in several orders is how that gets caught.

`pytest-randomly` does this, and is what CI is expected to run under, but it
is not installed in this checkout's venv (`pytest -q` reports
`plugins: xdist, httpx, timeout, anyio`). So this plugin, which does the one
thing needed and nothing else:

    ./.venv/bin/python -m pytest tests/backend/rendering \
        -p tools.shuffle_order --shuffle-seed=7 -q

The seed is printed, so a failing order can be replayed exactly. Without
`--shuffle-seed` nothing is shuffled, so importing the plugin changes no
run that does not ask for an order.
"""
from __future__ import annotations

import random


def pytest_addoption(parser) -> None:
    parser.getgroup("shuffle-order").addoption(
        "--shuffle-seed", type=int, default=None, metavar="N",
        help="shuffle the collected tests with this seed (replayable)")


def pytest_collection_modifyitems(session, config, items) -> None:
    seed = config.getoption("--shuffle-seed")
    if seed is None:
        return
    random.Random(seed).shuffle(items)
    reporter = config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_line(f"shuffled {len(items)} tests with seed {seed}")
