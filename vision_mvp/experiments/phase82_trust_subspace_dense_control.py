"""Phase 82 — Trust-subspace dense-control proxy (SDK v3.36, W35).

W35 is the next audited proxy step beyond W34.  W34 turns anchor
disagreement into safe abstention.  W35 asks whether a controller-
verified basis history can safely convert that abstention into a
correct reroute without weakening the trust boundary.

The mechanism is still capsule-layer only: it does NOT read hidden
states, transplant KV cache, or claim transformer-internal latent
control.  It builds a dense basis vector from W21 probe top_sets,
W33 EWMA trust, W34 live-attestation signatures, and host health.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from typing import Any, Sequence

from vision_mvp.coordpy.team_coord import (
    AttentionAwareBundleDecoder,
    BundleContradictionDisambiguator,
    OracleRegistration,
    ServiceGraphOracle,
    ChangeHistoryOracle,
    OnCallNotesOracle,
    CompromisedServiceGraphOracle,
    RelationalCompatibilityDisambiguator,
    TrustWeightedMultiOracleDisambiguator,
    TrustEWMATrackedMultiOracleOrchestrator,
    build_trust_ewma_registry,
    W33_DEFAULT_TRUST_THRESHOLD,
    W33_DEFAULT_TRUST_TRAJECTORY_WINDOW,
    W33_DEFAULT_EWMA_ALPHA,
    LiveAwareMultiAnchorOrchestrator,
    build_live_aware_registry,
    W34_DEFAULT_ANCHOR_QUORUM_MIN,
    W34_DEFAULT_HOST_DECAY_FACTOR,
    W34_BRANCH_MULTI_ANCHOR_NO_CONSENSUS,
    TrustSubspaceDenseControlOrchestrator,
    build_trivial_trust_subspace_registry,
    build_trust_subspace_dense_registry,
    W35_DEFAULT_BASIS_EWMA_ALPHA,
    W35_DEFAULT_PROJECTION_THRESHOLD,
    W35_DEFAULT_PROJECTION_MARGIN_MIN,
    W35_DEFAULT_BASIS_HISTORY_WINDOW,
    W35_DEFAULT_MIN_BASIS_OBSERVATIONS,
    W35_BRANCH_BASIS_HISTORY_REROUTED,
    W35_BRANCH_BASIS_HISTORY_ABSTAINED,
    _DecodedHandoff,
)
from vision_mvp.experiments.phase67_outside_information import (
    build_phase67_bank,
)


def _stable_schema_capsule():
    from vision_mvp.coordpy.team_coord import (
        build_incident_triage_schema_capsule)
    return build_incident_triage_schema_capsule()


def _scenario_to_per_round_handoffs(
        scenario: Any,
) -> Sequence[Sequence[_DecodedHandoff]]:
    round1: list[_DecodedHandoff] = []
    for src, emissions in scenario.round1_emissions.items():
        for kind, payload in emissions:
            round1.append(_DecodedHandoff(
                source_role=str(src),
                claim_kind=str(kind),
                payload=str(payload),
            ))
    round2: list[_DecodedHandoff] = []
    for src, emissions in scenario.round2_emissions.items():
        for kind, payload in emissions:
            round2.append(_DecodedHandoff(
                source_role=str(src),
                claim_kind=str(kind),
                payload=str(payload),
            ))
    return [round1, round2]


def _build_w21_disambiguator(
        *,
        T_decoder: int | None,
        registrations: tuple[OracleRegistration, ...],
        quorum_min: int = 2,
) -> TrustWeightedMultiOracleDisambiguator:
    inner = AttentionAwareBundleDecoder(T_decoder=T_decoder)
    inner_w18 = RelationalCompatibilityDisambiguator(inner=inner)
    inner_w19 = BundleContradictionDisambiguator(inner=inner_w18)
    return TrustWeightedMultiOracleDisambiguator(
        inner=inner_w19,
        oracle_registrations=registrations,
        enabled=True,
        quorum_min=int(quorum_min),
        min_trust_sum=0.0,
    )


def _build_w35_oracle_registrations_for_cell(
        *,
        bank: str,
        cell_idx: int,
        n_total: int,
) -> tuple[OracleRegistration, ...]:
    """Per-cell oracle regimes for R-82."""
    K1 = max(1, (3 * n_total) // 8)
    K2 = max(K1 + 1, (5 * n_total) // 8)

    if bank in ("trivial_w35", "no_anchor_disagreement"):
        return (
            OracleRegistration(
                oracle=ServiceGraphOracle(oracle_id="service_graph"),
                trust_prior=1.0, role_label="service_graph"),
            OracleRegistration(
                oracle=ChangeHistoryOracle(oracle_id="change_history"),
                trust_prior=1.0, role_label="change_history"),
            OracleRegistration(
                oracle=OnCallNotesOracle(oracle_id="oncall_notes"),
                trust_prior=1.0, role_label="oncall_notes"),
        )

    if bank in ("trust_subspace_shift", "frozen_basis"):
        if cell_idx < K1:
            sg = ServiceGraphOracle(oracle_id="service_graph")
            ch = ChangeHistoryOracle(oracle_id="change_history")
            oc = OnCallNotesOracle(oracle_id="oncall_notes")
        elif cell_idx < K2:
            sg = ServiceGraphOracle(oracle_id="service_graph")
            ch = ChangeHistoryOracle(oracle_id="change_history")
            oc = CompromisedServiceGraphOracle(oracle_id="oncall_notes")
        else:
            sg = CompromisedServiceGraphOracle(
                oracle_id="service_graph")
            ch = ChangeHistoryOracle(oracle_id="change_history")
            oc = CompromisedServiceGraphOracle(oracle_id="oncall_notes")
        return (
            OracleRegistration(oracle=sg, trust_prior=1.0,
                               role_label="service_graph"),
            OracleRegistration(oracle=ch, trust_prior=1.0,
                               role_label="change_history"),
            OracleRegistration(oracle=oc, trust_prior=1.0,
                               role_label="oncall_notes"),
        )

    if bank == "all_anchor_compromised":
        if cell_idx < K1:
            sg = ServiceGraphOracle(oracle_id="service_graph")
            ch = ChangeHistoryOracle(oracle_id="change_history")
            oc = OnCallNotesOracle(oracle_id="oncall_notes")
        elif cell_idx < K2:
            sg = ServiceGraphOracle(oracle_id="service_graph")
            ch = ChangeHistoryOracle(oracle_id="change_history")
            oc = CompromisedServiceGraphOracle(oracle_id="oncall_notes")
        else:
            sg = CompromisedServiceGraphOracle(
                oracle_id="service_graph")
            ch = CompromisedServiceGraphOracle(
                oracle_id="change_history")
            oc = CompromisedServiceGraphOracle(oracle_id="oncall_notes")
        return (
            OracleRegistration(oracle=sg, trust_prior=1.0,
                               role_label="service_graph"),
            OracleRegistration(oracle=ch, trust_prior=1.0,
                               role_label="change_history"),
            OracleRegistration(oracle=oc, trust_prior=1.0,
                               role_label="oncall_notes"),
        )

    raise ValueError(f"unknown phase82 bank {bank!r}")


def _gold_for_cell(scenario: Any) -> tuple[str, ...]:
    return tuple(sorted(scenario.gold_services_pair))


def _is_correct(answer: dict[str, Any], gold: tuple[str, ...]) -> bool:
    services = tuple(sorted(set(answer.get("services", ()))))
    return services == tuple(sorted(set(gold)))


def _is_ratified(answer: dict[str, Any]) -> bool:
    return bool(answer.get("services", ()))


@dataclasses.dataclass
class _Phase82Record:
    cell_idx: int
    expected: tuple[str, ...]
    correct_w21: bool
    correct_w33: bool
    correct_w34: bool
    correct_w35: bool
    ratified_w21: bool
    ratified_w33: bool
    ratified_w34: bool
    ratified_w35: bool
    w34_multi_anchor_branch: str
    w35_decoder_branch: str
    w35_projection_branch: str
    w35_selected_oracle_id: str
    w35_projection_top_set: tuple[str, ...]
    w34_visible: int
    w35_visible: int
    w35_overhead: int
    w35_structured_bits: int
    w35_cram_factor: float


def _make_w33(
        *,
        schema: Any,
        anchor_ids: tuple[str, ...],
        T_decoder: int | None,
        bank: str,
        n_eval: int,
        quorum_min: int,
) -> TrustEWMATrackedMultiOracleOrchestrator:
    registry = build_trust_ewma_registry(
        schema=schema,
        registered_oracle_ids=(
            "service_graph", "change_history", "oncall_notes"),
        anchor_oracle_ids=anchor_ids,
        trust_ewma_enabled=True,
        manifest_v3_disabled=False,
        trust_trajectory_window=W33_DEFAULT_TRUST_TRAJECTORY_WINDOW,
        trust_threshold=W33_DEFAULT_TRUST_THRESHOLD,
        ewma_alpha=W33_DEFAULT_EWMA_ALPHA,
    )
    placeholder = _build_w21_disambiguator(
        T_decoder=T_decoder,
        registrations=_build_w35_oracle_registrations_for_cell(
            bank=bank, cell_idx=0, n_total=n_eval),
        quorum_min=quorum_min,
    )
    return TrustEWMATrackedMultiOracleOrchestrator(
        inner=placeholder,
        registry=registry,
        enabled=True,
        require_w33_verification=True,
    )


def _make_w34(
        *,
        schema: Any,
        T_decoder: int | None,
        bank: str,
        n_eval: int,
        anchor_ids: tuple[str, ...],
        quorum_min: int,
) -> LiveAwareMultiAnchorOrchestrator:
    w33 = _make_w33(
        schema=schema,
        anchor_ids=anchor_ids,
        T_decoder=T_decoder,
        bank=bank,
        n_eval=n_eval,
        quorum_min=quorum_min,
    )
    w34_registry = build_live_aware_registry(
        schema=schema,
        inner_w33_registry=w33.registry,
        multi_anchor_quorum_min=W34_DEFAULT_ANCHOR_QUORUM_MIN,
        live_attestation_disabled=True,
        manifest_v4_disabled=False,
        host_decay_factor=W34_DEFAULT_HOST_DECAY_FACTOR,
        registered_hosts={},
    )
    return LiveAwareMultiAnchorOrchestrator(
        inner=w33,
        registry=w34_registry,
        enabled=True,
        require_w34_verification=True,
    )


def run_phase82(
        *,
        bank: str = "trust_subspace_shift",
        n_eval: int = 16,
        bank_seed: int = 11,
        bank_replicates: int = 2,
        T_decoder: int | None = None,
        anchor_oracle_ids: tuple[str, ...] = (
            "service_graph", "change_history"),
        quorum_min: int = 2,
        trust_subspace_enabled: bool = True,
        manifest_v5_disabled: bool = False,
        basis_history_window: int = W35_DEFAULT_BASIS_HISTORY_WINDOW,
        basis_ewma_alpha: float = W35_DEFAULT_BASIS_EWMA_ALPHA,
        projection_threshold: float = W35_DEFAULT_PROJECTION_THRESHOLD,
        projection_margin_min: float = W35_DEFAULT_PROJECTION_MARGIN_MIN,
        min_basis_observations: int = W35_DEFAULT_MIN_BASIS_OBSERVATIONS,
        abstain_on_unstable_consensus: bool = True,
) -> dict[str, Any]:
    schema = _stable_schema_capsule()
    n_repl = max(int(bank_replicates), (int(n_eval) + 3) // 4)
    scenarios = build_phase67_bank(
        bank="outside_resolves", n_replicates=n_repl,
        seed=bank_seed)
    if len(scenarios) > n_eval:
        scenarios = scenarios[:n_eval]
    n_eval_actual = min(n_eval, len(scenarios))

    w33 = _make_w33(
        schema=schema,
        anchor_ids=("service_graph",),
        T_decoder=T_decoder,
        bank=bank,
        n_eval=n_eval_actual,
        quorum_min=quorum_min,
    )
    w34 = _make_w34(
        schema=schema,
        T_decoder=T_decoder,
        bank=bank,
        n_eval=n_eval_actual,
        anchor_ids=anchor_oracle_ids,
        quorum_min=quorum_min,
    )
    w34_for_w35 = _make_w34(
        schema=schema,
        T_decoder=T_decoder,
        bank=bank,
        n_eval=n_eval_actual,
        anchor_ids=anchor_oracle_ids,
        quorum_min=quorum_min,
    )
    if not trust_subspace_enabled and manifest_v5_disabled:
        w35_registry = build_trivial_trust_subspace_registry(
            schema=schema,
            inner_w34_registry=w34_for_w35.registry,
            registered_oracle_ids=(
                "service_graph", "change_history", "oncall_notes"),
        )
    else:
        w35_registry = build_trust_subspace_dense_registry(
            schema=schema,
            inner_w34_registry=w34_for_w35.registry,
            registered_oracle_ids=(
                "service_graph", "change_history", "oncall_notes"),
            trust_subspace_enabled=trust_subspace_enabled,
            manifest_v5_disabled=manifest_v5_disabled,
            basis_history_window=basis_history_window,
            basis_ewma_alpha=basis_ewma_alpha,
            projection_threshold=projection_threshold,
            projection_margin_min=projection_margin_min,
            min_basis_observations=min_basis_observations,
            abstain_on_unstable_consensus=abstain_on_unstable_consensus,
        )
    w35 = TrustSubspaceDenseControlOrchestrator(
        inner=w34_for_w35,
        registry=w35_registry,
        enabled=True,
        require_w35_verification=True,
    )

    records: list[_Phase82Record] = []
    for cell_idx in range(n_eval_actual):
        scenario = scenarios[cell_idx]
        gold = _gold_for_cell(scenario)
        per_rounds = _scenario_to_per_round_handoffs(scenario)
        registrations = _build_w35_oracle_registrations_for_cell(
            bank=bank, cell_idx=cell_idx, n_total=n_eval_actual)

        w21 = _build_w21_disambiguator(
            T_decoder=T_decoder, registrations=registrations,
            quorum_min=quorum_min)
        w21_out = w21.decode_rounds(per_rounds)

        w33.inner = _build_w21_disambiguator(
            T_decoder=T_decoder, registrations=registrations,
            quorum_min=quorum_min)
        w33_out = w33.decode_rounds(per_rounds)

        w34.inner.inner = _build_w21_disambiguator(
            T_decoder=T_decoder, registrations=registrations,
            quorum_min=quorum_min)
        w34_out = w34.decode_rounds(per_rounds)
        w34_result = w34.last_result

        w35.inner.inner.inner = _build_w21_disambiguator(
            T_decoder=T_decoder, registrations=registrations,
            quorum_min=quorum_min)
        w35_out = w35.decode_rounds(per_rounds)
        w35_result = w35.last_result

        rat21 = _is_ratified(w21_out)
        rat33 = _is_ratified(w33_out)
        rat34 = _is_ratified(w34_out)
        rat35 = _is_ratified(w35_out)
        records.append(_Phase82Record(
            cell_idx=int(cell_idx),
            expected=tuple(gold),
            correct_w21=_is_correct(w21_out, gold) if rat21 else False,
            correct_w33=_is_correct(w33_out, gold) if rat33 else False,
            correct_w34=_is_correct(w34_out, gold) if rat34 else False,
            correct_w35=_is_correct(w35_out, gold) if rat35 else False,
            ratified_w21=bool(rat21),
            ratified_w33=bool(rat33),
            ratified_w34=bool(rat34),
            ratified_w35=bool(rat35),
            w34_multi_anchor_branch=(
                str(w34_result.multi_anchor_branch)
                if w34_result is not None else ""),
            w35_projection_branch=(
                str(w35_result.projection_branch)
                if w35_result is not None else ""),
            w35_decoder_branch=(
                str(w35_result.decoder_branch)
                if w35_result is not None else ""),
            w35_selected_oracle_id=(
                str(w35_result.selected_oracle_id)
                if w35_result is not None else ""),
            w35_projection_top_set=(
                tuple(w35_result.projection_top_set)
                if w35_result is not None else ()),
            w34_visible=int(w34_result.n_w34_visible_tokens
                            if w34_result is not None else 0),
            w35_visible=int(w35_result.n_w35_visible_tokens
                            if w35_result is not None else 0),
            w35_overhead=int(w35_result.n_w35_overhead_tokens
                             if w35_result is not None else 0),
            w35_structured_bits=int(w35_result.n_structured_bits
                                    if w35_result is not None else 0),
            w35_cram_factor=float(w35_result.cram_factor_w35
                                  if w35_result is not None else 0.0),
        ))

    def _rate(attr: str) -> float:
        return (sum(1 for r in records if getattr(r, attr)) / len(records)
                if records else 0.0)

    def _trust(correct_attr: str, rat_attr: str) -> float:
        n_rat = sum(1 for r in records if getattr(r, rat_attr))
        if n_rat == 0:
            return 1.0
        return sum(1 for r in records if getattr(r, correct_attr)) / n_rat

    total_w35_overhead = sum(r.w35_overhead for r in records)
    total_w35_bits = sum(r.w35_structured_bits for r in records)
    return {
        "bank": str(bank),
        "n_eval": len(records),
        "bank_seed": int(bank_seed),
        "anchor_oracle_ids": list(anchor_oracle_ids),
        "trust_subspace_enabled": bool(trust_subspace_enabled),
        "manifest_v5_disabled": bool(manifest_v5_disabled),
        "basis_ewma_alpha": round(float(basis_ewma_alpha), 4),
        "projection_threshold": round(float(projection_threshold), 4),
        "projection_margin_min": round(float(projection_margin_min), 4),
        "min_basis_observations": int(min_basis_observations),
        "correctness_ratified_rate_w21": round(_rate("correct_w21"), 4),
        "correctness_ratified_rate_w33": round(_rate("correct_w33"), 4),
        "correctness_ratified_rate_w34": round(_rate("correct_w34"), 4),
        "correctness_ratified_rate_w35": round(_rate("correct_w35"), 4),
        "trust_precision_w21": round(
            _trust("correct_w21", "ratified_w21"), 4),
        "trust_precision_w33": round(
            _trust("correct_w33", "ratified_w33"), 4),
        "trust_precision_w34": round(
            _trust("correct_w34", "ratified_w34"), 4),
        "trust_precision_w35": round(
            _trust("correct_w35", "ratified_w35"), 4),
        "delta_correctness_w35_w34": round(
            _rate("correct_w35") - _rate("correct_w34"), 4),
        "delta_trust_precision_w35_w34": round(
            _trust("correct_w35", "ratified_w35")
            - _trust("correct_w34", "ratified_w34"), 4),
        "n_w34_no_consensus": sum(
            1 for r in records
            if r.w34_multi_anchor_branch
            == W34_BRANCH_MULTI_ANCHOR_NO_CONSENSUS),
        "n_w35_basis_rerouted": sum(
            1 for r in records
            if r.w35_decoder_branch == W35_BRANCH_BASIS_HISTORY_REROUTED),
        "n_w35_basis_abstained": sum(
            1 for r in records
            if r.w35_decoder_branch == W35_BRANCH_BASIS_HISTORY_ABSTAINED),
        "mean_total_w34_visible_tokens": round(
            sum(r.w34_visible for r in records) / len(records), 4)
            if records else 0.0,
        "mean_total_w35_visible_tokens": round(
            sum(r.w35_visible for r in records) / len(records), 4)
            if records else 0.0,
        "mean_overhead_w35_per_cell": round(
            total_w35_overhead / len(records), 4) if records else 0.0,
        "max_overhead_w35_per_cell": max(
            (r.w35_overhead for r in records), default=0),
        "total_w35_structured_bits": int(total_w35_bits),
        "structured_state_transferred_per_visible_token": round(
            total_w35_bits / max(1, total_w35_overhead), 4),
        "mean_w35_cram_factor": round(
            sum(r.w35_cram_factor for r in records) / len(records), 4)
            if records else 0.0,
        "byte_equivalent_w35_w34": bool(
            all(r.w35_visible == r.w34_visible for r in records)
            and _rate("correct_w35") == _rate("correct_w34")
            and _trust("correct_w35", "ratified_w35")
            == _trust("correct_w34", "ratified_w34")),
        "records": [dataclasses.asdict(r) for r in records],
    }


def run_phase82_seed_sweep(
        *,
        bank: str = "trust_subspace_shift",
        n_eval: int = 16,
        seeds: Sequence[int] = (11, 17, 23, 29, 31),
        **kwargs: Any,
) -> dict[str, Any]:
    seed_results: list[dict[str, Any]] = []
    for seed in seeds:
        result = run_phase82(
            bank=bank, n_eval=n_eval, bank_seed=int(seed), **kwargs)
        short = {k: v for k, v in result.items() if k != "records"}
        short["seed"] = int(seed)
        seed_results.append(short)
    return {
        "bank": str(bank),
        "seeds": [int(s) for s in seeds],
        "min_delta_correctness_w35_w34": round(
            min(s["delta_correctness_w35_w34"] for s in seed_results)
            if seed_results else 0.0, 4),
        "max_delta_correctness_w35_w34": round(
            max(s["delta_correctness_w35_w34"] for s in seed_results)
            if seed_results else 0.0, 4),
        "min_trust_precision_w35": round(
            min(s["trust_precision_w35"] for s in seed_results)
            if seed_results else 1.0, 4),
        "max_overhead_w35_per_cell": max(
            (int(s["max_overhead_w35_per_cell"])
             for s in seed_results), default=0),
        "all_byte_equivalent_w35_w34": all(
            bool(s["byte_equivalent_w35_w34"]) for s in seed_results)
            if seed_results else False,
        "seed_results": seed_results,
    }


def _write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _cli() -> int:
    parser = argparse.ArgumentParser(
        prog="phase82",
        description="Phase 82 — W35 trust-subspace dense-control proxy.")
    parser.add_argument("--bank", default="trust_subspace_shift",
                        choices=("trivial_w35",
                                 "no_anchor_disagreement",
                                 "trust_subspace_shift",
                                 "frozen_basis",
                                 "all_anchor_compromised"))
    parser.add_argument("--n-eval", type=int, default=16)
    parser.add_argument("--bank-seed", type=int, default=11)
    parser.add_argument("--seed-sweep", action="store_true")
    parser.add_argument("--seeds", type=str, default="11,17,23,29,31")
    parser.add_argument("--basis-ewma-alpha", type=float,
                        default=W35_DEFAULT_BASIS_EWMA_ALPHA)
    parser.add_argument("--projection-threshold", type=float,
                        default=W35_DEFAULT_PROJECTION_THRESHOLD)
    parser.add_argument("--projection-margin-min", type=float,
                        default=W35_DEFAULT_PROJECTION_MARGIN_MIN)
    parser.add_argument("--min-basis-observations", type=int,
                        default=W35_DEFAULT_MIN_BASIS_OBSERVATIONS)
    parser.add_argument("--trust-subspace-disabled", action="store_true")
    parser.add_argument("--manifest-v5-disabled", action="store_true")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    seeds = tuple(int(s) for s in args.seeds.split(","))
    kwargs = dict(
        basis_ewma_alpha=float(args.basis_ewma_alpha),
        projection_threshold=float(args.projection_threshold),
        projection_margin_min=float(args.projection_margin_min),
        min_basis_observations=int(args.min_basis_observations),
        trust_subspace_enabled=not bool(args.trust_subspace_disabled),
        manifest_v5_disabled=bool(args.manifest_v5_disabled),
    )
    if args.bank == "trivial_w35":
        kwargs["trust_subspace_enabled"] = False
        kwargs["manifest_v5_disabled"] = True
    if args.bank == "frozen_basis":
        kwargs["basis_ewma_alpha"] = 0.0
    if args.seed_sweep:
        result = run_phase82_seed_sweep(
            bank=args.bank, n_eval=args.n_eval, seeds=seeds, **kwargs)
    else:
        result = run_phase82(
            bank=args.bank, n_eval=args.n_eval,
            bank_seed=args.bank_seed, **kwargs)
        result = {k: v for k, v in result.items() if k != "records"}
    if args.out:
        _write_json(args.out, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
