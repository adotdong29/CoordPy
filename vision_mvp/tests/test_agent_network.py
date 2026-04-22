"""Tests for the Phase-10 agent-network machinery."""
from __future__ import annotations
import math
import unittest
import numpy as np
from dataclasses import dataclass, field

from vision_mvp.core.agent_keys import AgentKeyIndex, l2_normalize
from vision_mvp.core.sparse_router import SparseRouter
from vision_mvp.core.task_board import TaskBoard, Subtask
from vision_mvp.core.sheaf_monitor import SheafMonitor
from vision_mvp.core.hyperbolic import (
    lorentz_distance, origin, exp_map, project_to_hyperboloid,
    HyperbolicAddressBook,
)
from vision_mvp.core.agent_network import AgentNetwork, NetworkAgent, Message


# -------------------------------------------------------------------- keys

class TestAgentKeyIndex(unittest.TestCase):
    def test_keys_are_unit_norm(self):
        idx = AgentKeyIndex(n_agents=30, dim=16, seed=1)
        norms = np.linalg.norm(idx._keys, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_route_returns_top_k(self):
        idx = AgentKeyIndex(n_agents=40, dim=8, seed=2)
        q = np.random.default_rng(0).standard_normal(8)
        recipients = idx.route(q, top_k=5)
        self.assertLessEqual(len(recipients), 5)
        self.assertEqual(len(set(recipients)), len(recipients))

    def test_route_learns_toward_query(self):
        idx = AgentKeyIndex(n_agents=20, dim=4, seed=3)
        q = np.array([1., 0, 0, 0])
        # Initial similarity
        sims_before = idx._keys @ q
        # Pick one agent, nudge it positively many times
        for _ in range(30):
            idx.update_positive(agent_id=5, query=q)
        sim_after = idx._keys[5] @ q
        self.assertGreater(sim_after, sims_before[5])

    def test_cluster_members_partition_agents(self):
        idx = AgentKeyIndex(n_agents=50, dim=8, seed=4)
        all_members = []
        for c, members in idx._cluster_members.items():
            all_members.extend(members)
        self.assertEqual(sorted(all_members), sorted(range(50)))

    def test_exclude_parameter(self):
        idx = AgentKeyIndex(n_agents=30, dim=4, seed=5)
        q = np.array([1., 0, 0, 0])
        recipients = idx.route(q, top_k=10, exclude={3, 7, 11})
        for r in recipients:
            self.assertNotIn(r, {3, 7, 11})


# -------------------------------------------------------------------- router

class TestSparseRouter(unittest.TestCase):
    def test_capacity_limits_inbox(self):
        keys = AgentKeyIndex(n_agents=20, dim=4, seed=0)
        r = SparseRouter(keys=keys, top_k=3, capacity_per_round=2)
        r.begin_round()
        q = np.array([1., 0, 0, 0])
        # Send many messages; after capacity each agent hits cap
        counts = {}
        for _ in range(15):
            recipients = r.route(q)
            for a in recipients:
                counts[a] = counts.get(a, 0) + 1
        self.assertTrue(all(c <= 2 for c in counts.values()),
                        f"some agents exceeded capacity: {counts}")

    def test_global_agents_always_receive(self):
        keys = AgentKeyIndex(n_agents=30, dim=4, seed=0)
        r = SparseRouter(keys=keys, top_k=3, global_agents=[0, 1])
        r.begin_round()
        q = np.random.default_rng(0).standard_normal(4)
        recipients = r.route(q, sender_id=5)
        self.assertIn(0, recipients)
        self.assertIn(1, recipients)

    def test_recipient_hints_honored(self):
        keys = AgentKeyIndex(n_agents=30, dim=4, seed=0)
        r = SparseRouter(keys=keys, top_k=3)
        r.begin_round()
        q = np.random.default_rng(0).standard_normal(4)
        recipients = r.route(q, sender_id=5, recipient_hints=[12, 19])
        self.assertIn(12, recipients)
        self.assertIn(19, recipients)

    def test_sender_excluded(self):
        keys = AgentKeyIndex(n_agents=30, dim=4, seed=0)
        r = SparseRouter(keys=keys, top_k=10)
        r.begin_round()
        recipients = r.route(np.ones(4) / 2, sender_id=7)
        self.assertNotIn(7, recipients)


# -------------------------------------------------------------------- board

class TestTaskBoard(unittest.TestCase):
    def test_deps_control_readiness(self):
        b = TaskBoard()
        b.add(Subtask(id="a", title="A", description=""))
        b.add(Subtask(id="b", title="B", description="", deps=["a"]))
        self.assertEqual([t.id for t in b.ready_tasks()], ["a"])
        b.claim("a", agent_id=1)
        b.complete("a", agent_id=1, output="done-a")
        self.assertEqual([t.id for t in b.ready_tasks()], ["b"])

    def test_claim_is_mutually_exclusive(self):
        b = TaskBoard()
        b.add(Subtask(id="x", title="", description=""))
        self.assertTrue(b.claim("x", agent_id=1))
        self.assertFalse(b.claim("x", agent_id=2))

    def test_complete_only_by_assignee(self):
        b = TaskBoard()
        b.add(Subtask(id="x", title="", description=""))
        b.claim("x", agent_id=1)
        self.assertFalse(b.complete("x", agent_id=2, output="bad"))
        self.assertTrue(b.complete("x", agent_id=1, output="good"))

    def test_deps_outputs(self):
        b = TaskBoard()
        b.add(Subtask(id="a", title="", description=""))
        b.add(Subtask(id="b", title="", description="", deps=["a"]))
        b.claim("a", 1); b.complete("a", 1, "value-of-a")
        self.assertEqual(b.deps_outputs("b"), [("a", "value-of-a")])

    def test_all_done_detection(self):
        b = TaskBoard()
        b.add(Subtask(id="a", title="", description=""))
        self.assertFalse(b.all_done())
        b.claim("a", 1); b.complete("a", 1, "")
        self.assertTrue(b.all_done())


# -------------------------------------------------------------------- sheaf

class TestSheafMonitor(unittest.TestCase):
    def test_discord_zero_when_agreed(self):
        s = SheafMonitor(stalk_dim=3, interface_dim=2)
        s.add_edge(1, 2, [0, 1])
        beliefs = {1: np.array([1, 2, 3]), 2: np.array([1, 2, 99])}
        edges = s.edge_discord(beliefs)
        self.assertAlmostEqual(edges[0]["discord"], 0.0)

    def test_discord_positive_when_disagree(self):
        s = SheafMonitor(stalk_dim=3, interface_dim=2)
        s.add_edge(1, 2, [0, 1])
        beliefs = {1: np.array([1, 2, 3]), 2: np.array([4, 5, 3])}
        edges = s.edge_discord(beliefs)
        self.assertAlmostEqual(edges[0]["discord"], 9 + 9)  # (4-1)² + (5-2)² = 18

    def test_top_disagreements(self):
        s = SheafMonitor(stalk_dim=2, interface_dim=1)
        s.add_edge(1, 2, [0])
        s.add_edge(2, 3, [0])
        s.add_edge(3, 4, [0])
        beliefs = {
            1: np.array([0, 0]),
            2: np.array([1, 0]),   # disagrees with 1 by 1 on coord 0
            3: np.array([5, 0]),   # disagrees with 2 by 16
            4: np.array([5, 0]),   # agrees with 3
        }
        top = s.top_disagreements(beliefs, k=2)
        self.assertEqual((top[0]["u"], top[0]["v"]), (2, 3))  # biggest gap

    def test_cohomology_dims(self):
        s = SheafMonitor(stalk_dim=2, interface_dim=2)
        # triangle with identity restrictions
        s.add_edge(1, 2, [0, 1])
        s.add_edge(2, 3, [0, 1])
        s.add_edge(1, 3, [0, 1])
        dims = s.cohomology_dims([1, 2, 3])
        # rank of B for this sheaf on a triangle = 4  (since one row is dependent)
        # dim H^0 = 6 − 4 = 2 (agreement = one "average" belief, 2-dim)
        self.assertEqual(dims["dim_H0"], 2)


# -------------------------------------------------------------------- hyperbolic

class TestHyperbolic(unittest.TestCase):
    def test_distance_to_self_is_zero(self):
        o = origin(2)
        self.assertAlmostEqual(float(lorentz_distance(o, o)), 0.0, places=6)

    def test_exp_then_distance_matches_tangent_norm(self):
        o = origin(2)
        v = np.array([0.0, 0.5, 0.3])      # tangent at origin (first coord 0)
        p = exp_map(o, v)
        # distance equals Lorentzian norm of v
        expected = math.sqrt(0.5 ** 2 + 0.3 ** 2)
        self.assertAlmostEqual(float(lorentz_distance(o, p)), expected, places=4)

    def test_projection_enforces_constraint(self):
        x = np.array([0.5, 1.0, 0.5])   # NOT on hyperboloid
        px = project_to_hyperboloid(x)
        from vision_mvp.core.hyperbolic import minkowski_inner
        self.assertAlmostEqual(float(minkowski_inner(px, px)), -1.0, places=6)

    def test_address_book_tree(self):
        book = HyperbolicAddressBook(dim=2, branch_step=0.8)
        # tree: 0 root, children 1,2; 1 has children 3,4
        tree = {0: [1, 2], 1: [3, 4], 2: []}
        book.embed_tree(tree)
        # siblings 3,4 should be closer to each other than to cousin 2
        d_sibling = book.distance(3, 4)
        d_cousin = book.distance(3, 2)
        self.assertLess(d_sibling, d_cousin)


# -------------------------------------------------------------------- network

def _fake_embed(text: str, dim: int = 16, seed: int = 0):
    rng = np.random.default_rng(abs(hash(text)) % (2**31))
    return l2_normalize(rng.standard_normal(dim))


def _fake_llm(persona: str, inbox, ctx: str) -> str:
    return f"ack::{persona.split()[0]}::{len(inbox)}-inbox::{ctx[:30]}"


class TestAgentNetwork(unittest.TestCase):
    def _make_net(self, n=10, dim=16) -> AgentNetwork:
        net = AgentNetwork(n_agents=n, dim_keys=dim)
        for aid in range(n):
            a = NetworkAgent(
                agent_id=aid,
                persona=f"agent {aid} role{aid % 3}",
                role=f"role{aid % 3}",
                llm_call=_fake_llm,
                embed_fn=lambda t, dim=dim: _fake_embed(t, dim=dim),
            )
            net.register_agent(a, key_init_text=f"role{aid % 3}")
        return net

    def test_post_delivers_to_recipients(self):
        net = self._make_net(n=8)
        net.router.begin_round()
        msg = net.post(sender_id=0, content="hello team")
        # At least some recipients were selected
        self.assertGreater(len(msg.recipients), 0)
        # Each recipient has a non-empty inbox now
        for r in msg.recipients:
            self.assertGreaterEqual(len(net.agents[r]._inbox), 1)

    def test_run_round_respects_inbox_capacity(self):
        net = self._make_net(n=12)
        # Pre-stuff inbox via many posts from same agent
        net.router.begin_round()
        for _ in range(30):
            net.post(sender_id=0, content="spam")
        # No agent should have more than capacity_per_round in inbox
        cap = net.router.capacity_per_round
        for a in net.agents.values():
            self.assertLessEqual(len(a._inbox), cap)

    def test_task_completion_by_role_matched_agent(self):
        net = self._make_net(n=8, dim=16)
        # Add a subtask with a specific tag embedding matching role0 agents
        tag_emb = _fake_embed("role0", dim=16)
        net.board.add(Subtask(id="t1", title="Do the thing",
                              description="Do role0 things", tag_embedding=tag_emb))
        # Give role0 agent a key matching the tag
        for aid in (0, 3, 6):
            net.keys.set_key(aid, tag_emb)
        # Run a few rounds
        for _ in range(5):
            net.run_round()
            if net.board.done_count() >= 1:
                break
        self.assertGreaterEqual(net.board.done_count(), 1)

    def test_message_stats(self):
        net = self._make_net(n=10)
        net.router.begin_round()
        for i in range(5):
            net.post(sender_id=0, content=f"message {i}")
        s = net.message_stats()
        self.assertEqual(s["total_messages"], 5)
        self.assertGreater(s["mean_recipients_per_message"], 0)

    def test_peak_inbox_stays_bounded(self):
        """The point of the architecture: inbox doesn't blow up with N."""
        small = self._make_net(n=10)
        large = self._make_net(n=60)
        for net in (small, large):
            net.router.begin_round()
            for i in range(25):
                net.post(sender_id=i % net.n_agents, content=f"m{i}")
        small_stats = small.message_stats()
        large_stats = large.message_stats()
        # At N=60 the max inbox per agent should NOT be 6× that at N=10
        # (it's bounded by capacity + routing concentration)
        self.assertLess(
            large_stats["max_inbox_size_per_agent"],
            3 * small_stats["max_inbox_size_per_agent"] + 5,
        )


if __name__ == "__main__":
    unittest.main()
