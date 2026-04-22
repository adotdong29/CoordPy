"""LLMTeam — a million-agent-ready team of real LLM agents.

The key insight that makes this tractable:

    Not every agent does an LLM call every round.
    The workspace admits only ⌈log₂ N⌉ agents per round to generate text.
    The remaining (N − k) agents carry an *embedding* that drifts through
    the shared manifold toward the group consensus — no LLM call required.

So a 1000-agent team needs ~10 LLM generate calls per round, not 1000,
making the demo feasible on a laptop in a few minutes.

Protocol per round:
  1. Compute salience = prediction error of each agent's embedding.
  2. Workspace selects top-k agents.
  3. Admitted agents call LLM to generate new text given current consensus.
  4. Their new texts are embedded in one batch call.
  5. Those embeddings update the streaming-PCA manifold.
  6. Consensus text = admitted agent whose embedding is closest to centroid.
  7. Non-admitted agents blend their embeddings toward the centroid.

To get a "final answer" from every agent at the end, we pick a sample
(default 50) to do one final LLM call each; the rest keep their
last-known text state.
"""

from __future__ import annotations
import math
import random
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .llm_client import LLMClient
from .learned_manifold import StreamingPCA
from .workspace import Workspace


def _archetype_prompt(persona: str, question: str,
                      private_chunk: str = "") -> str:
    parts = [persona.strip()]
    if private_chunk:
        parts.append(
            "You have access ONLY to this chunk of the source document:\n"
            f"---\n{private_chunk.strip()}\n---\n"
            "Report the CONCRETE FACTS you can see in your chunk that "
            "are relevant to the question — incident ids, vendor names, "
            "detection-to-mitigation times, runbook/doc issues, root "
            "causes. Be specific; the team will pool your facts."
        )
        parts.append(f"Overall team question: {question.strip()}")
        parts.append("Your observations (2–4 short sentences):")
    else:
        parts.append(f"Question: {question.strip()}")
        parts.append("Answer in one short sentence:")
    return "\n\n".join(parts)


def _context_prompt(persona: str, question: str, context: str,
                    private_chunk: str = "") -> str:
    parts = [persona.strip()]
    if private_chunk:
        parts.append(
            "You have access ONLY to this chunk of the source document:\n"
            f"---\n{private_chunk.strip()}\n---"
        )
    parts.append(f"Team consensus so far: {context.strip()}")
    parts.append(f"Question: {question.strip()}")
    if private_chunk:
        parts.append("State concrete facts from your chunk that support or "
                     "challenge the team consensus (2–3 short sentences):")
    else:
        parts.append("Answer in one short sentence, combining your chunk "
                     "with the team consensus:")
    return "\n\n".join(parts)


@dataclass
class LLMTeam:
    n_agents: int
    personas: list[str]                     # len == n_agents
    question: str
    model: str = "qwen2.5:0.5b"
    pca_lr: float = 0.3
    decay: float = 0.7
    blend_alpha: float = 0.25               # how hard non-admitted agents drift to centroid
    surprise_tau: float = 0.05
    workspace_epsilon: float = 0.1
    seed: int = 0
    n_parallel_llm: int = 1         # concurrency for LLM calls (1 = serial)
    # Optional per-agent private context — each agent sees only its own chunk.
    # When set, this enables GENUINE distributed collaboration: no single
    # agent sees the full corpus.
    per_agent_context: list[str] | None = None

    # Internal state (filled by initialize())
    client: LLMClient = field(default=None)          # type: ignore
    archetype_texts: dict = field(default_factory=dict)
    archetype_index: list[int] = field(default_factory=list)
    text_state: list[str] = field(default_factory=list)
    embeddings: np.ndarray = field(default=None)     # type: ignore
    prev_embeddings: np.ndarray = field(default=None)# type: ignore
    dim: int = 0
    pca: StreamingPCA = field(default=None)          # type: ignore
    workspace: Workspace = field(default=None)       # type: ignore
    reg_value: np.ndarray = field(default=None)      # type: ignore
    reg_weight: float = 0.0
    round_idx: int = 0

    # Accounting
    ctx_tokens_llm: int = 0
    ctx_tokens_broadcast: int = 0      # simulated tokens in embedding channel
    n_generate_calls: int = 0
    n_embed_calls: int = 0
    wall_llm: float = 0.0
    per_round_admitted: list[list[int]] = field(default_factory=list)
    per_round_consensus_text: list[str] = field(default_factory=list)

    def __post_init__(self):
        if len(self.personas) != self.n_agents:
            raise ValueError("personas must have length n_agents")
        if (self.per_agent_context is not None
                and len(self.per_agent_context) != self.n_agents):
            raise ValueError(
                "per_agent_context must have length n_agents when provided")
        if self.client is None:
            self.client = LLMClient(model=self.model)
        # Archetype = (persona, chunk) pair. When per_agent_context is set
        # every agent is typically its own archetype (distinct chunks).
        unique = []
        ix = {}
        for i, p in enumerate(self.personas):
            key = (p, self.per_agent_context[i]) if self.per_agent_context else p
            if key not in ix:
                ix[key] = len(unique)
                unique.append(key)
        self._unique_archetypes = unique       # list of (persona, chunk) or persona
        self.archetype_index = [
            ix[(self.personas[i], self.per_agent_context[i])
               if self.per_agent_context else self.personas[i]]
            for i in range(self.n_agents)
        ]
        self.workspace = Workspace(n_agents=self.n_agents,
                                   epsilon=self.workspace_epsilon)

    # ---- Initialization ----

    def initialize(self, progress_cb=None, max_tokens: int = 80) -> None:
        """Generate archetype answers once, then share across agents.

        Makes #unique-archetypes LLM calls. With `per_agent_context`, each
        agent is typically its own archetype (distinct chunks), so this
        scales as O(N) init — but subsequent rounds are still O(log N).
        """
        t0 = time.time()
        archetype_answers = []
        archetype_keys = list(self._unique_archetypes)
        for i, key in enumerate(archetype_keys):
            if progress_cb:
                progress_cb(f"  init archetype {i+1}/{len(archetype_keys)}")
            if self.per_agent_context is not None:
                persona, chunk = key
            else:
                persona, chunk = key, ""
            prompt = _archetype_prompt(persona, self.question, private_chunk=chunk)
            ans = self.client.generate(prompt, max_tokens=max_tokens,
                                       temperature=0.2)
            archetype_answers.append(ans)
            # Store by the full key so lookup works later
            self.archetype_texts[key] = ans
        self.wall_llm += time.time() - t0

        # Embed all archetypes in one batch
        t0 = time.time()
        archetype_embs = self.client.embed_batch(archetype_answers)
        self.wall_llm += time.time() - t0
        if not archetype_embs:
            raise RuntimeError("Empty embedding response — is Ollama reachable?")

        self.dim = len(archetype_embs[0])
        self.embeddings = np.zeros((self.n_agents, self.dim), dtype=np.float64)
        rng = np.random.default_rng(self.seed)
        for i in range(self.n_agents):
            base = np.array(archetype_embs[self.archetype_index[i]])
            perturb = 0.005 * rng.standard_normal(self.dim)
            self.embeddings[i] = base + perturb
        self.prev_embeddings = self.embeddings.copy()

        # Initial text = each agent's archetype answer (indexed per agent)
        self.text_state = [archetype_answers[self.archetype_index[i]]
                           for i in range(self.n_agents)]

        # PCA manifold
        m = max(2, math.ceil(math.log2(self.n_agents)))
        m = min(m, self.dim)
        self.pca = StreamingPCA.build(self.dim, m, lr=self.pca_lr, seed=self.seed)
        self.pca.update_batch(self.embeddings[:min(200, self.n_agents)])
        self.reg_value = np.zeros(m)

    # ---- Main loop ----

    def step(self, progress_cb=None) -> dict:
        """One round of CASR routing with selective LLM generation."""
        # 1. Salience = distance from current group centroid. Agents far
        # from consensus should be admitted to either bring diversity or
        # become convinced. This is a more useful signal than just EMA
        # residual because after the round-end blend, all embeddings match
        # their prev → raw residual is zero and admission becomes arbitrary.
        if self.reg_weight > 0:
            summary_prev = self.reg_value / self.reg_weight
            centroid_prev = self.pca.reconstruct(summary_prev)
        else:
            centroid_prev = self.embeddings.mean(axis=0)
        saliences = np.linalg.norm(
            self.embeddings - centroid_prev[None, :], axis=1)

        # 2. Workspace
        admitted = self.workspace.select(saliences,
                                         seed=self.seed + self.round_idx)
        admitted = [int(x) for x in admitted]
        self.per_round_admitted.append(admitted)

        # 3. Admitted agents do LLM generation (possibly in parallel).
        ctx = (self.per_round_consensus_text[-1]
               if self.per_round_consensus_text else "")
        t0 = time.time()

        def _gen_one(i: int) -> tuple[int, str]:
            chunk = self.per_agent_context[i] if self.per_agent_context else ""
            if ctx:
                prompt = _context_prompt(self.personas[i], self.question,
                                         ctx, private_chunk=chunk)
            else:
                prompt = _archetype_prompt(self.personas[i], self.question,
                                           private_chunk=chunk)
            out = self.client.generate(prompt, max_tokens=80, temperature=0.2)
            return i, out

        if self.n_parallel_llm > 1:
            if progress_cb:
                progress_cb(f"  r{self.round_idx}: dispatching "
                            f"{len(admitted)} agents in {self.n_parallel_llm}-way parallel…")
            with ThreadPoolExecutor(max_workers=self.n_parallel_llm) as ex:
                for i, out in ex.map(_gen_one, admitted):
                    self.text_state[i] = out
        else:
            for i in admitted:
                if progress_cb:
                    progress_cb(f"  r{self.round_idx}: agent {i} generating…")
                _, out = _gen_one(i)
                self.text_state[i] = out
        self.wall_llm += time.time() - t0

        # 4. Batch-embed admitted texts
        t0 = time.time()
        new_embs = self.client.embed_batch([self.text_state[i] for i in admitted])
        self.wall_llm += time.time() - t0
        for idx, i in enumerate(admitted):
            if idx < len(new_embs):
                self.embeddings[i] = np.array(new_embs[idx])

        # 5. PCA update on admitted batch mean
        admitted_embs = np.stack([self.embeddings[i] for i in admitted])
        self.pca.update(admitted_embs.mean(axis=0))

        # 6. Decay the register, then write surprising projections
        self.reg_value = self.reg_value * self.decay
        self.reg_weight = self.reg_weight * self.decay
        for idx, i in enumerate(admitted):
            if saliences[i] > self.surprise_tau:
                y = self.pca.project(self.embeddings[i])
                self.reg_value = self.reg_value + y
                self.reg_weight += 1.0
                # Accounting: wrote m+1 simulated tokens
                self.ctx_tokens_broadcast += self.pca.dim_manifold + 1

        # 7. Consensus: admitted agent whose embedding is nearest the register's reconstruction
        summary = self.reg_value / max(self.reg_weight, 1e-8)
        centroid = self.pca.reconstruct(summary)          # (dim,)
        cen_norm = float(np.linalg.norm(centroid)) + 1e-8
        dots = np.array([float(self.embeddings[i] @ centroid
                               / (np.linalg.norm(self.embeddings[i]) * cen_norm + 1e-8))
                         for i in admitted])
        winner_local = int(np.argmax(dots))
        consensus_text = self.text_state[admitted[winner_local]]
        self.per_round_consensus_text.append(consensus_text)

        # 8. Non-admitted agents: blend embedding toward centroid
        non_admitted = np.ones(self.n_agents, dtype=bool)
        non_admitted[admitted] = False
        alpha = self.blend_alpha
        self.embeddings[non_admitted] = (
            (1 - alpha) * self.embeddings[non_admitted]
            + alpha * centroid[None, :]
        )
        # Simulated token cost: manifold-dim per agent per read
        self.ctx_tokens_broadcast += (self.n_agents - len(admitted)) * self.pca.dim_manifold

        self.prev_embeddings = self.embeddings.copy()
        self.round_idx += 1

        # Accounting
        self.ctx_tokens_llm = (self.client.stats.prompt_tokens
                               + self.client.stats.output_tokens)
        self.n_generate_calls = self.client.stats.n_generate_calls
        self.n_embed_calls = self.client.stats.n_embed_calls

        return {
            "round": self.round_idx,
            "admitted": admitted,
            "consensus_text": consensus_text,
            "mean_salience": float(saliences.mean()),
            "max_salience": float(saliences.max()),
            "total_llm_tokens": self.client.stats.total_tokens(),
            "n_generate": self.client.stats.n_generate_calls,
            "ctx_tokens_broadcast": self.ctx_tokens_broadcast,
        }

    # ---- Final sampling ----

    def finalize_sample(self, sample_size: int = 50,
                        progress_cb=None) -> list[tuple[int, str]]:
        """Have a random sample of agents produce a final text answer.

        Not calling LLM for all N — we pick a representative subset and
        ask them to state their final opinion given the running consensus.
        """
        rng = random.Random(self.seed + 777)
        sample = rng.sample(range(self.n_agents), min(sample_size, self.n_agents))
        results = []
        ctx = (self.per_round_consensus_text[-1]
               if self.per_round_consensus_text else "")
        t0 = time.time()
        for count, i in enumerate(sample):
            if progress_cb:
                progress_cb(f"  final {count+1}/{len(sample)}: agent {i}")
            chunk = self.per_agent_context[i] if self.per_agent_context else ""
            if ctx:
                prompt = _context_prompt(self.personas[i], self.question,
                                         ctx, private_chunk=chunk)
            else:
                prompt = _archetype_prompt(self.personas[i], self.question,
                                           private_chunk=chunk)
            out = self.client.generate(prompt, max_tokens=60, temperature=0.1)
            self.text_state[i] = out
            results.append((i, out))
        self.wall_llm += time.time() - t0
        return results

    # ---- Post-hoc estimation for non-sampled agents ----

    def nearest_neighbor_texts(self) -> list[str]:
        """For each agent, return the text of the admitted agent whose
        embedding is closest to theirs (a cheap proxy for 'what they would
        say' without an extra LLM call).
        """
        # Admitted indices that have nonzero embeddings via LLM output
        admitted_set = set()
        for rnd_adm in self.per_round_admitted:
            admitted_set.update(rnd_adm)
        ref_indices = sorted(admitted_set)
        if not ref_indices:
            return list(self.text_state)
        ref_embs = np.stack([self.embeddings[i] for i in ref_indices])
        # Cosine similarity
        ref_norms = np.linalg.norm(ref_embs, axis=1) + 1e-8
        ag_norms = np.linalg.norm(self.embeddings, axis=1) + 1e-8
        sims = (self.embeddings @ ref_embs.T) / (ag_norms[:, None] * ref_norms[None, :])
        nearest = np.argmax(sims, axis=1)  # (N,)
        out = []
        for i in range(self.n_agents):
            ref = ref_indices[int(nearest[i])]
            out.append(self.text_state[ref])
        return out

    # ---- Synthesis — collapse top-k outputs into one structured review ----

    def synthesize(self, task_framing: str, max_tokens: int = 250,
                   top_k_from_admitted: int | None = None,
                   include_all_chunks: bool = False) -> str:
        """Aggregate agents' texts into a single final output.

        Two modes:

        1. **include_all_chunks=False (default, 'admitted' mode):**
           Pick top-k admitted agents closest to the consensus centroid;
           summarize their texts. Cheap and works for tasks where consensus
           has already converged (e.g. code review with one critical bug).

        2. **include_all_chunks=True ('broad' mode):**
           Include the initial chunk-specific text from *every* agent (or a
           deterministic down-sample if N is huge). This is essential for
           GENUINELY distributed tasks where each agent sees a unique chunk
           and cross-chunk patterns only emerge from seeing many chunks.
           Uses each agent's *first* text (the archetype output) so it
           represents that chunk's raw observation, not the converged view.

        `task_framing` is an instruction describing the output format.
        """
        # Decide which agents contribute
        if include_all_chunks:
            # Use all agents' *initial* outputs (pre-round, per-chunk observations)
            # If N is large, down-sample to a manageable number.
            max_chunks_in_synth = min(self.n_agents, 40)
            if self.n_agents > max_chunks_in_synth:
                step = self.n_agents // max_chunks_in_synth
                picked = list(range(0, self.n_agents, step))[:max_chunks_in_synth]
            else:
                picked = list(range(self.n_agents))
            # Use archetype_texts (the original chunk-based answers)
            pieces = []
            for rank, i in enumerate(picked):
                key = ((self.personas[i], self.per_agent_context[i])
                       if self.per_agent_context else self.personas[i])
                text = self.archetype_texts.get(key, self.text_state[i])
                pieces.append(f"Observation from team member {rank+1}: {text}")
        else:
            admitted_set = set()
            for rnd in self.per_round_admitted:
                admitted_set.update(rnd)
            admitted_list = sorted(admitted_set)
            if not admitted_list:
                return ""
            if self.reg_weight > 0:
                summary = self.reg_value / self.reg_weight
                centroid = self.pca.reconstruct(summary)
            else:
                centroid = self.embeddings[admitted_list].mean(axis=0)
            adm_embs = np.stack([self.embeddings[i] for i in admitted_list])
            adm_norms = np.linalg.norm(adm_embs, axis=1) + 1e-8
            cen_norm = float(np.linalg.norm(centroid)) + 1e-8
            sims = (adm_embs @ centroid) / (adm_norms * cen_norm)
            k = top_k_from_admitted or min(self.workspace.capacity(), len(admitted_list))
            k = min(k, len(admitted_list))
            order = np.argsort(-sims)[:k]
            picked = [admitted_list[int(i)] for i in order]
            pieces = [f"Reviewer {rank+1}: {self.text_state[i]}"
                      for rank, i in enumerate(picked)]

        bullet_texts = "\n".join(pieces)
        prompt = (
            f"{task_framing.strip()}\n\n"
            f"Question: {self.question.strip()}\n\n"
            f"{bullet_texts}\n\n"
            "Synthesis:"
        )
        t0 = time.time()
        out = self.client.generate(prompt, max_tokens=max_tokens,
                                   temperature=0.2)
        self.wall_llm += time.time() - t0
        return out

    # ---- Stats ----

    def stats(self) -> dict:
        return {
            "n_agents": self.n_agents,
            "rounds": self.round_idx,
            "workspace_size": self.workspace.capacity(),
            "llm_generate_calls": self.client.stats.n_generate_calls,
            "llm_embed_calls": self.client.stats.n_embed_calls,
            "llm_prompt_tokens": self.client.stats.prompt_tokens,
            "llm_output_tokens": self.client.stats.output_tokens,
            "llm_embed_tokens": self.client.stats.embed_tokens,
            "llm_total_tokens": self.client.stats.total_tokens(),
            "ctx_tokens_broadcast": self.ctx_tokens_broadcast,
            "total_tokens_equiv": (self.client.stats.total_tokens()
                                   + self.ctx_tokens_broadcast),
            "wall_llm_seconds": round(self.wall_llm, 2),
            "manifold_dim": self.pca.dim_manifold if self.pca else None,
        }
