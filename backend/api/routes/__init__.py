"""Per-domain APIRouter modules. The package's `__init__` imports every
submodule for its side effect (registering routes against the shared app).

Adding a new endpoint:
  1. Pick the matching domain module below (or create a new one).
  2. `from .._app import app` and `@app.<verb>(...)` as usual.
  3. If creating a new module, add a `from . import <name>` line here.
"""
from . import analyze        # noqa: F401
from . import fusion         # noqa: F401
from . import gallery        # noqa: F401
from . import inference      # noqa: F401
from . import meta           # noqa: F401
from . import op_proposals   # noqa: F401
from . import op_reviews     # noqa: F401
from . import playground    # noqa: F401
from . import precision      # noqa: F401
from . import spec           # noqa: F401
from . import supported_models  # noqa: F401

__all__ = []
