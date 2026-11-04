"""Every model id the web app names on its own is a shipped model.

The app seeds a few model ids itself (the Compare page's default pair, the
Inspector's fallback examples). One of them outlived its spec
(`meituan-longcat/LongCat-Flash-Thinking`, replaced by its -2601 release),
and the Compare page opened on "No spec matches". A model id literal is any
`"<org>/<name>"` string in `web/src` whose org ships at least one snapshot;
each must name a snapshot.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tests.snapshots import SNAPSHOT_DIR

WEB_SRC = PROJECT_ROOT / "web" / "src"
_LITERAL = re.compile(r"""["'`]([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)["'`]""")


def test_every_model_id_literal_is_shipped():
    shipped = {p.stem.replace("__", "/", 1) for p in SNAPSHOT_DIR.glob("*.json")}
    orgs = {mid.split("/", 1)[0] for mid in shipped}
    assert orgs, "no snapshots checked out"
    stale = sorted(
        f"{path.relative_to(PROJECT_ROOT)}: {org}/{name}"
        for path in WEB_SRC.rglob("*.ts*")
        for org, name in _LITERAL.findall(path.read_text())
        if org in orgs and f"{org}/{name}" not in shipped)
    assert not stale, "model ids the app names but no snapshot ships:\n" + "\n".join(stale)
