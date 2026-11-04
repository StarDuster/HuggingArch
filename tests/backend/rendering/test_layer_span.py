"""How the architecture view writes a set of layer indices
(`_layout.layer_span`): runs as ranges, a long repeat of equal runs at one
stride as its first two and last with the stride."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from backend.rendering._layout import layer_span


def test_runs_read_as_ranges():
    assert layer_span([0, 1, 2, 5]) == "0–2, 5"
    assert layer_span([1]) == "1"


def test_a_long_repeat_reads_as_its_stride():
    assert layer_span(list(range(6, 75, 4))) == "6, 10, …, 74 (every 4)"
    readers = [i for i in range(3, 78) if (i - 2) % 4]
    assert layer_span(readers) == "3–5, 7–9, …, 75–77 (every 4)"


def test_a_short_repeat_is_listed():
    """Under five runs the ellipsis would hide one item or none."""
    assert layer_span([24, 28, 32, 36]) == "24, 28, 32, 36"


def test_a_repeat_breaks_where_its_stride_does():
    kv_readers = [*range(3, 8), *range(9, 14), *range(15, 20), *range(21, 24),
                  *range(25, 28), *range(29, 32), *range(33, 36), *range(37, 40)]
    assert layer_span(kv_readers) == (
        "3–7, 9–13, 15–19, 21–23, 25–27, …, 37–39 (every 4)")
