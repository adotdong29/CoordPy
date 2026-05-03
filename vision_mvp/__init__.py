"""vision_mvp — Context Zero research programme top-level package.

**If you are looking for the shipped product**, you want CoordPy:

    from vision_mvp import coordpy
    report = coordpy.run(coordpy.RunSpec(profile="local_smoke",
                                     out_dir="/tmp/coordpy"))

**If you are looking at research substrate**, the original routing-layer
primitive is ``CASRRouter`` (Causal-Abstraction Scale-Renormalized
Routing), kept here as research-grade code. It is the substrate that
grounds CoordPy's bounded-context guarantees; it is not itself part of
the CoordPy SDK contract.

    >>> from vision_mvp import CASRRouter
    >>> import numpy as np
    >>> router = CASRRouter(n_agents=1000, state_dim=64, task_rank=10)
    >>> estimates = router.step(np.random.randn(1000, 64))
    >>> router.stats["peak_context_per_agent"]
    10

Orientation
-----------
- ``docs/START_HERE.md``   — one-pass orientation for new readers.
- ``vision_mvp.coordpy``     — stable product SDK surface.
- ``PROOFS.md``            — formal theorems.
- ``FINAL_RESULTS.md``     — measured scaling (N = 10 … 100 000).
- ``EXTENDED_MATH_[1-7]``  — 72-framework theoretical survey.
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
__version__ = "0.5.16"
