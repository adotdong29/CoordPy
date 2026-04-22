"""vision_mvp — CASR coordination layer for multi-agent teams.

The main public object is `CASRRouter`: feed it batched observations,
get back consensus estimates with O(log N) peak per-agent context.

Example
-------
>>> from vision_mvp import CASRRouter
>>> import numpy as np
>>> router = CASRRouter(n_agents=1000, state_dim=64, task_rank=10)
>>> obs = np.random.randn(1000, 64)
>>> estimates = router.step(obs)
>>> router.stats["peak_context_per_agent"]
10

See also
--------
- `FINAL_RESULTS.md` for measured scaling across N = 10 to 100 000.
- `PROOFS.md` for the formal theorems.
- `EXTENDED_MATH_[1-7].md` for the 72-framework theoretical survey.
"""

from .api import CASRRouter
from .core.hierarchical_router import HierarchicalRouter

# LLM-backed variants — require a local Ollama instance to be useful.
# Expose them from the top level so `from vision_mvp import LLMTeam` works.
from .core.llm_team import LLMTeam
from .core.llm_hierarchy import LLMHierarchy

# Composed routers built out during Waves 1–5.
from .core.composed_routers import AdversarialCASRRouter, DynamicCASRRouter

__all__ = [
    "CASRRouter",
    "HierarchicalRouter",
    "LLMTeam",
    "LLMHierarchy",
    "AdversarialCASRRouter",
    "DynamicCASRRouter",
]
__version__ = "0.3.0"
