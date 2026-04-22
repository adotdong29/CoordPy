"""Real-world code-review task with scoreable ground truth.

Each agent is a specialist reviewer (security, perf, readability, correctness,
API design, …) and looks at a piece of code. The team converges on the most
important issue.

The code snippets here are chosen to have one *clearly critical* issue plus
several minor issues, so we can score the team output unambiguously.
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class CodeReviewTask:
    code: str
    description: str
    critical_issue: str          # the primary bug the team should find
    critical_keywords: list[str] # substrings that indicate the critical issue was flagged
    minor_issues: list[str] = field(default_factory=list)

    def scores(self, text: str) -> dict[str, bool]:
        low = text.lower()
        return {
            "critical_found": any(kw.lower() in low for kw in self.critical_keywords),
            "n_minor_mentioned": sum(
                1 for m in self.minor_issues
                if any(w.lower() in low for w in m.split()[:3])
            ),
        }


# --- Curated tasks, each with a specific real-world bug ---

SQL_INJECTION = CodeReviewTask(
    code="""def get_user_session(user_id, db):
    query = "SELECT * FROM sessions WHERE user_id = " + str(user_id)
    rows = db.execute(query)
    if rows:
        return {"user": user_id, "data": rows[0]}
    return None
""",
    description=(
        "A Python helper that fetches a user's session from a database given a "
        "user_id. Review it for the most important problem a reviewer should "
        "block on."
    ),
    critical_issue="SQL injection — user_id is concatenated into the query string.",
    critical_keywords=[
        "sql injection", "parameterized", "injection", "parameterised",
        "bind", "prepared statement", "sanitiz", "escape",
    ],
    minor_issues=[
        "no error handling for db failure",
        "inconsistent return type dict vs None",
        "no session expiration check",
    ],
)

RACE_CONDITION = CodeReviewTask(
    code="""counter = 0

def increment_counter():
    global counter
    current = counter
    counter = current + 1
    return counter
""",
    description=(
        "A Python global counter used across threads to issue unique IDs. "
        "Review it for the most important problem."
    ),
    critical_issue="Race condition — read-modify-write is not atomic under threading.",
    critical_keywords=[
        "race condition", "thread safe", "not atomic", "atomic",
        "race", "concurrent", "thread-safe", "lock", "itertools.count",
        "mutex", "gil", "not safe",
    ],
    minor_issues=[
        "global state",
        "no docstring",
    ],
)

MEMORY_LEAK = CodeReviewTask(
    code="""class Cache:
    _cache = {}

    def get_or_compute(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]
""",
    description=(
        "A Python memoization cache used inside a long-running service. "
        "Review it for the most important problem."
    ),
    critical_issue=("Unbounded cache — grows without eviction, causing memory "
                    "leak under long-running workload."),
    critical_keywords=[
        "unbounded", "memory leak", "no eviction", "grow forever",
        "grows forever", "lru", "lru_cache", "ttl", "cache size",
        "maxsize", "unlimited", "memory", "eviction",
    ],
    minor_issues=[
        "class-level mutable default (_cache shared across instances)",
        "no thread-safety",
        "exception in fn poisons the cache",
    ],
)


DEFAULT_TASKS = [SQL_INJECTION, RACE_CONDITION, MEMORY_LEAK]


# Specialized reviewer personas — each comes from a real code-review angle.
REVIEWER_PERSONAS = [
    "You are a security engineer. You look for vulnerabilities like SQL injection, XSS, auth bypass, and unsafe deserialization. You flag the single most exploitable bug.",
    "You are a concurrency specialist. You look for race conditions, deadlocks, and thread-safety bugs in shared state.",
    "You are an SRE who maintains long-running services. You look for memory leaks, resource exhaustion, and issues that only show up after weeks of uptime.",
    "You are a database engineer. You review queries for performance, injection, correctness, and transaction issues.",
    "You are a Python correctness expert. You flag subtle bugs, wrong return types, and undefined behavior.",
    "You are a senior Python reviewer. You prioritize the highest-impact issue and state it in one sentence.",
    "You are a pragmatic tech lead. You want the single most important thing a reviewer would block the PR on.",
    "You are a defensive coder. You look for missing error handling and input validation.",
    "You are an API designer. You flag inconsistent return types, bad contracts, and footguns.",
    "You are a performance engineer. You look for O(N²) patterns and hot-path inefficiencies.",
    "You are a readability reviewer. You flag unclear names and missing documentation.",
    "You are a test engineer. You look for untestable code and missing edge cases.",
    "You are a junior developer eager to learn. You ask about anything that looks off.",
    "You are a strict linter. You only care about style, but you say it clearly.",
    "You are a production incident responder. You look for the code pattern most likely to page someone at 3 AM.",
    "You are a security auditor preparing a pentest report. You want the worst vulnerability.",
    "You are a thoughtful architect. You look for structural issues in how this code fits a larger system.",
    "You are a seasoned backend engineer skeptical of globals and shared state.",
    "You are a language lawyer who knows Python semantics down to the CPython reference.",
    "You are a platform engineer worried about scale and multi-tenancy.",
    "You are a code reviewer who prioritizes correctness over style.",
    "You are a DBA who has seen production SQL disasters firsthand.",
    "You are a cautious reviewer who would rather over-flag than miss a real bug.",
    "You are an engineering manager who wants the team to move fast but not break prod.",
    "You are a researcher who cares about soundness of algorithms and data structures.",
]


def assign_reviewer_personas(n: int) -> list[str]:
    return [REVIEWER_PERSONAS[i % len(REVIEWER_PERSONAS)] for i in range(n)]
