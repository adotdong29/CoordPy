"""AgentNetwork — full orchestrator for hundreds of networked LLM agents.

This is where the pieces come together:
  - `AgentKeyIndex` for learned routing keys
  - `SparseRouter` for top-k MoE delivery
  - `HyperbolicAddressBook` (optional) for tree-structured address space
  - `TaskBoard` for shared DAG of subtasks
  - `SheafMonitor` (optional) for consistency diagnostics

Each agent is a `NetworkAgent` wrapper around an LLM client. In a round:

  1. Each agent polls its inbox (filled by the router this round).
  2. Each agent polls the task board for claimable ready tasks matching its key.
  3. Each agent acts (one LLM call max) — either replying to a message or
     completing a claimed task.
  4. Its output is posted to the bus; the router delivers it to top-k
     relevant agents + any explicit recipients + global coordinators.
  5. Keys update (positive signal when a reply is re-used by a successor).

Per-agent context per round is bounded by `capacity_per_round` (default 20
inbox items). Total LLM calls per round ≤ n_active_agents. So doubling
team size doesn't double anyone's context — it increases PARALLELISM, not
any individual's workload.
"""

from __future__ import annotations
import numpy as np
import time
import re
from dataclasses import dataclass, field
from typing import Callable

from .agent_keys import AgentKeyIndex, l2_normalize
from .sparse_router import SparseRouter
from .task_board import TaskBoard, Subtask
from .sheaf_monitor import SheafMonitor


@dataclass
class Message:
    sender_id: int
    content: str
    query_embedding: np.ndarray
    round: int
    recipients: list[int] = field(default_factory=list)
    reply_to: str | None = None
    subtask_id: str | None = None   # set when completing a subtask
    msg_id: str = ""


@dataclass
class NetworkAgent:
    """One agent in the network. Wraps an LLM client + per-agent state.

    The LLM callable gets:
        (persona, inbox_messages, task_context) -> reply_text
    It must return a string. In tests we use a mock callable.
    """
    agent_id: int
    persona: str
    role: str                              # coarse specialty, e.g. "data_cleaning"
    llm_call: Callable[[str, list[Message], str], str]
    embed_fn: Callable[[str], np.ndarray]  # text -> embedding vector
    _inbox: list[Message] = field(default_factory=list)
    _output_history: list[str] = field(default_factory=list)

    def receive(self, msg: Message) -> None:
        self._inbox.append(msg)

    def drain_inbox(self) -> list[Message]:
        msgs = list(self._inbox)
        self._inbox.clear()
        return msgs


@dataclass
class AgentNetwork:
    n_agents: int
    dim_keys: int = 64

    keys: AgentKeyIndex = field(default=None)      # type: ignore
    router: SparseRouter = field(default=None)      # type: ignore
    board: TaskBoard = field(default_factory=TaskBoard)
    sheaf: SheafMonitor | None = None
    agents: dict[int, NetworkAgent] = field(default_factory=dict)
    _round: int = 0
    _messages_log: list[Message] = field(default_factory=list)
    _msg_counter: int = 0

    def __post_init__(self):
        if self.keys is None:
            self.keys = AgentKeyIndex(n_agents=self.n_agents, dim=self.dim_keys)
        if self.router is None:
            self.router = SparseRouter(keys=self.keys, top_k=5,
                                        capacity_per_round=20)

    def register_agent(self, a: NetworkAgent,
                       key_init_text: str | None = None) -> None:
        self.agents[a.agent_id] = a
        if key_init_text:
            k = a.embed_fn(key_init_text)
            self.keys.set_key(a.agent_id, l2_normalize(k))

    # ---- Core loop ----

    def post(self, sender_id: int, content: str,
             recipient_hints: list[int] | None = None,
             reply_to: str | None = None,
             subtask_id: str | None = None) -> Message:
        q = self.agents[sender_id].embed_fn(content)
        q = l2_normalize(q)
        recipients = self.router.route(
            q, sender_id=sender_id, recipient_hints=recipient_hints or [])
        msg_id = f"m-{self._msg_counter}"
        self._msg_counter += 1
        msg = Message(
            sender_id=sender_id, content=content, query_embedding=q,
            round=self._round, recipients=recipients, reply_to=reply_to,
            subtask_id=subtask_id, msg_id=msg_id,
        )
        for rid in recipients:
            if rid in self.agents:
                self.agents[rid].receive(msg)
        self._messages_log.append(msg)
        return msg

    def run_round(self, progress_cb=None) -> dict:
        """One round of work: every agent gets ONE turn to act.

        Returns per-round stats.
        """
        self._round += 1
        self.router.begin_round()

        # Determine action order — active agents first (those with inbox or
        # that can claim a ready subtask matching their role).
        outputs_this_round: list[Message] = []

        for aid, agent in self.agents.items():
            if progress_cb:
                progress_cb(f"  round {self._round} agent {aid}")
            inbox = agent.drain_inbox()
            claimed_task = None

            # Priority 1: try to claim the BEST-matching ready subtask.
            # Productive work always beats chatter.
            ready = self.board.ready_tasks()
            key = self.keys.get_key(aid)
            key_norm = np.linalg.norm(key) + 1e-8
            best = None
            best_sim = 0.15   # threshold: even loose match is OK if nothing better
            for t in ready:
                if t.tag_embedding is None:
                    continue
                sim = float(np.dot(key, t.tag_embedding)
                            / (key_norm * np.linalg.norm(t.tag_embedding) + 1e-8))
                if sim > best_sim:
                    best_sim = sim
                    best = t
            if best is not None and self.board.claim(best.id, aid):
                claimed_task = best

            if claimed_task is None and not inbox:
                continue

            # Build task context string
            ctx_parts = []
            if inbox:
                ctx_parts.append("Inbox (messages addressed to you):")
                for m in inbox[:5]:
                    ctx_parts.append(f"  [from agent {m.sender_id}]: {m.content}")
            if claimed_task:
                ctx_parts.append(f"You claimed subtask {claimed_task.id}: {claimed_task.title}")
                ctx_parts.append(f"Description: {claimed_task.description}")
                deps_out = self.board.deps_outputs(claimed_task.id)
                if deps_out:
                    ctx_parts.append("Outputs of dependencies:")
                    for did, out in deps_out:
                        ctx_parts.append(f"  [{did}]: {out[:400]}")
            ctx = "\n".join(ctx_parts)

            # LLM call
            reply = agent.llm_call(agent.persona, inbox, ctx)
            agent._output_history.append(reply)

            if claimed_task:
                self.board.complete(claimed_task.id, aid, reply)
                # Broadcast completion
                content = f"Completed subtask {claimed_task.id}: {reply[:200]}"
                msg = self.post(aid, content, subtask_id=claimed_task.id)
                outputs_this_round.append(msg)
            elif inbox:
                # Reply addressed back to whoever sent us messages
                content = reply
                senders = list({m.sender_id for m in inbox[:3]})
                msg = self.post(aid, content, recipient_hints=senders,
                                reply_to=inbox[0].msg_id if inbox else None)
                outputs_this_round.append(msg)

            # Positive key update: nudge this agent's key toward the messages
            # it just replied to
            if inbox:
                for m in inbox[:3]:
                    self.keys.update_positive(aid, m.query_embedding)

        return {
            "round": self._round,
            "messages_posted": len(outputs_this_round),
            "board": self.board.summary(),
            "router": self.router.stats(),
        }

    def run_until_done(self, max_rounds: int = 20, progress_cb=None) -> dict:
        history = []
        for r in range(max_rounds):
            if self.board.all_done():
                break
            info = self.run_round(progress_cb=progress_cb)
            history.append(info)
        return {"rounds": self._round, "board": self.board.summary(),
                "history": history}

    # ---- Diagnostics ----

    def message_stats(self) -> dict:
        if not self._messages_log:
            return {}
        recipients_per_msg = [len(m.recipients) for m in self._messages_log]
        per_agent_inbox = {aid: 0 for aid in self.agents}
        for m in self._messages_log:
            for r in m.recipients:
                per_agent_inbox[r] = per_agent_inbox.get(r, 0) + 1
        return {
            "total_messages": len(self._messages_log),
            "mean_recipients_per_message": round(
                sum(recipients_per_msg) / len(recipients_per_msg), 2),
            "max_recipients_per_message": max(recipients_per_msg),
            "mean_inbox_size_per_agent": round(
                sum(per_agent_inbox.values()) / max(len(per_agent_inbox), 1), 2),
            "max_inbox_size_per_agent": max(per_agent_inbox.values()) if per_agent_inbox else 0,
        }
