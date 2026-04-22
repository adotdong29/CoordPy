"""Sandboxed executable-plan DSL — safer-than-exec evaluator for agent-
generated programs.

Agents can ship *programs* (short arithmetic / control-flow scripts) instead
of natural-language paragraphs; receivers run the program in a sandbox under
their own state. This module implements a subset-Python interpreter over
`ast`:

  allowed:   arithmetic, comparisons, variable assignment, if, for (bounded),
             while (bounded), function calls from a whitelist, returns.
  forbidden: import, exec, eval, open, file I/O, attribute access, dunder,
             lambda, class, try/except, global/nonlocal, …

Execution is bounded: at most `max_steps` AST operations to prevent runaway
loops. Suitable for programmatic coordination where textual prose is too
lossy (I8 in the plan).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Callable


class PlanError(Exception):
    pass


_ALLOWED_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b if b != 0 else float("inf"),
    ast.FloorDiv: lambda a, b: a // b if b != 0 else 0,
    ast.Mod: lambda a, b: a % b if b != 0 else 0,
    ast.Pow: lambda a, b: a ** b,
}
_ALLOWED_UNARY = {
    ast.UAdd: lambda a: +a,
    ast.USub: lambda a: -a,
    ast.Not: lambda a: not a,
}
_ALLOWED_COMPARE = {
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
}
_ALLOWED_BOOLOP = {
    ast.And: all,
    ast.Or: any,
}


@dataclass
class PlanInterpreter:
    allowed_fns: dict[str, Callable] = field(default_factory=dict)
    max_steps: int = 10_000
    _steps: int = field(init=False, default=0)

    def _tick(self):
        self._steps += 1
        if self._steps > self.max_steps:
            raise PlanError("max_steps exceeded")

    def run(self, source: str, variables: dict | None = None) -> dict:
        self._steps = 0
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            raise PlanError(f"syntax error: {e}")
        env = dict(variables or {})
        self._exec_block(tree.body, env)
        return env

    def _exec_block(self, stmts, env: dict):
        for s in stmts:
            self._tick()
            self._exec_stmt(s, env)

    def _exec_stmt(self, node, env: dict):
        if isinstance(node, ast.Assign):
            value = self._eval(node.value, env)
            for tgt in node.targets:
                if not isinstance(tgt, ast.Name):
                    raise PlanError("only simple name assignment supported")
                env[tgt.id] = value
        elif isinstance(node, ast.AugAssign):
            if not isinstance(node.target, ast.Name):
                raise PlanError("only simple name AugAssign supported")
            cur = env.get(node.target.id)
            rhs = self._eval(node.value, env)
            op_fn = _ALLOWED_BINOPS.get(type(node.op))
            if op_fn is None:
                raise PlanError(f"disallowed op: {type(node.op).__name__}")
            env[node.target.id] = op_fn(cur, rhs)
        elif isinstance(node, ast.If):
            cond = self._eval(node.test, env)
            self._exec_block(node.body if cond else node.orelse, env)
        elif isinstance(node, ast.For):
            iterable = self._eval(node.iter, env)
            for item in iterable:
                if not isinstance(node.target, ast.Name):
                    raise PlanError("only simple-name for-loop target supported")
                env[node.target.id] = item
                self._exec_block(node.body, env)
        elif isinstance(node, ast.While):
            # Bounded while: max_steps clamps
            while self._eval(node.test, env):
                self._tick()
                self._exec_block(node.body, env)
        elif isinstance(node, ast.Expr):
            self._eval(node.value, env)
        else:
            raise PlanError(f"disallowed statement: {type(node).__name__}")

    def _eval(self, node, env: dict):
        self._tick()
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in env:
                raise PlanError(f"undefined name {node.id!r}")
            return env[node.id]
        if isinstance(node, ast.BinOp):
            op = _ALLOWED_BINOPS.get(type(node.op))
            if op is None:
                raise PlanError(f"disallowed binop {type(node.op).__name__}")
            return op(self._eval(node.left, env), self._eval(node.right, env))
        if isinstance(node, ast.UnaryOp):
            op = _ALLOWED_UNARY.get(type(node.op))
            if op is None:
                raise PlanError(f"disallowed unary {type(node.op).__name__}")
            return op(self._eval(node.operand, env))
        if isinstance(node, ast.Compare):
            lhs = self._eval(node.left, env)
            ok = True
            for op_node, comparator in zip(node.ops, node.comparators):
                cmp_fn = _ALLOWED_COMPARE.get(type(op_node))
                if cmp_fn is None:
                    raise PlanError(f"disallowed comparator {type(op_node).__name__}")
                rhs = self._eval(comparator, env)
                if not cmp_fn(lhs, rhs):
                    ok = False
                    break
                lhs = rhs
            return ok
        if isinstance(node, ast.BoolOp):
            combiner = _ALLOWED_BOOLOP.get(type(node.op))
            if combiner is None:
                raise PlanError(f"disallowed boolop {type(node.op).__name__}")
            return combiner(self._eval(v, env) for v in node.values)
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise PlanError("only whitelisted function calls supported")
            name = node.func.id
            if name not in self.allowed_fns:
                raise PlanError(f"function {name!r} not allowed")
            args = [self._eval(a, env) for a in node.args]
            kwargs = {k.arg: self._eval(k.value, env) for k in node.keywords}
            return self.allowed_fns[name](*args, **kwargs)
        if isinstance(node, ast.List):
            return [self._eval(e, env) for e in node.elts]
        if isinstance(node, ast.Tuple):
            return tuple(self._eval(e, env) for e in node.elts)
        if isinstance(node, ast.Dict):
            return {
                self._eval(k, env): self._eval(v, env)
                for k, v in zip(node.keys, node.values)
            }
        raise PlanError(f"disallowed expression: {type(node).__name__}")
