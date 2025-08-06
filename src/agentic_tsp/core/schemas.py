# In src/agentic_tsp/core/schemas.py

import uuid
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class MacroCall(BaseModel):
    """
    Defines the data structure for a self-contained task dispatched
    by the Orchestrator to a Worker.
    """
    task_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    algorithm: str
    distance_matrix: List[List[float]]
    parameters: Dict[str, Any]
    # Optional random seed for reproducible runs
    seed: Optional[int] = None

class WorkerResult(BaseModel):
    """
    Defines the data structure for the final result returned by a
    Worker to the Orchestrator.
    """
    task_id: uuid.UUID
    best_tour: List[int]
    tour_length: float