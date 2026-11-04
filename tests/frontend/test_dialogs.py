"""The site asks its questions in its own dialog (`web/src/Dialog.tsx`), never
the browser's `confirm` / `prompt` / `alert` boxes."""
import re
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "web" / "src"
_NATIVE = re.compile(r"(?<![\w.])(?:window\.)?(confirm|prompt|alert)\(")


def test_no_native_dialogs():
    hits = [f"{path.relative_to(_SRC)}:{n}: {line.strip()}"
            for path in sorted(_SRC.rglob("*.ts*"))
            for n, line in enumerate(path.read_text().splitlines(), 1)
            if _NATIVE.search(line) and not line.lstrip().startswith(("//", "*"))]
    assert not hits, "use confirmDialog / promptDialog / alertDialog from Dialog.tsx:\n" + "\n".join(hits)
