"""Service-only Pydantic request models."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class AnalyticsEventRequest(BaseModel):
    """Body for POST /analytics/event.

    `endpoint` is 'analyze' or 'inference_estimate'. `metadata` is a
    free-form dict that gets stored as JSONB and queried by admin
    aggregation functions (gpu_sku, parallel.*, batch_size, etc.).
    """
    endpoint: str
    model_id: str
    duration_ms: Optional[int] = None
    metadata: Optional[dict] = None
