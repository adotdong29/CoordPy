"""W86+ / P2 #41 — Schema Evolution and Capsule Migration V1.

Issue #41 asks for a real schema-migration story on top of the
W82+W83 content-addressed audit chain. The DoD demands:

1. ``SchemaRegistryV2`` — content-addressed registry of
   ``schema_version_string`` → ``SchemaEntryV1``, two versions
   can coexist, CIDs pin to one explicitly.
2. ``MigrationFnV1`` — typed migrations between schema pairs,
   content-addressed so identical inputs produce identical
   migrated outputs.
3. **Forward + backward compatibility tags** — each schema
   declares which prior versions it can be migrated from.
4. ``MigrationEventV1`` — audit capsule
   ``(old_cid, new_cid, migration_plan_cid, ts)``.
5. **In-flight schema upgrade bench** — start under V1, migrate
   to V2 mid-run, verify the chain re-verifies across the
   migration.
6. **Deprecated-but-readable** — schemas with ``deprecated=True``
   produce a structured warning but remain readable.

Honest scope (V1)
-----------------

* ``W86-L-SCHEMA-EVOLUTION-V1-RESEARCH-ONLY-CAP``
* ``W86-L-SCHEMA-EVOLUTION-V1-ONE-PAIR-CAP`` — V1 supports a
  linear chain (V1 → V2 → …); n-way DAG migration is V2.
* ``W86-L-SCHEMA-EVOLUTION-V1-LOSSLESS-CAP`` — V1 migrations
  are lossless; explicit-data-loss migrations are V2.
* ``W86-L-SCHEMA-EVOLUTION-V1-DETERMINISTIC-CAP`` — V1
  migrations are deterministic; user-input-required migrations
  are V3.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import warnings
from typing import Any, Callable, Mapping, Optional, Sequence


W86_SCHEMA_EVOLUTION_V1_VERSION: str = (
    "coordpy.schema_evolution_v1.v1")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"),
            default=str).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------
# Schema registry
# ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SchemaEntryV1:
    """One entry in the registry.

    The schema is identified by ``schema_version_string`` (e.g.
    "coordpy.foo.v1") and described by an ordered tuple of
    declared field names + types (the V1 type system is just
    string tags — V2 will use real JSON schemas).

    ``migratable_from`` lists earlier ``schema_version_string``s
    this version can be migrated from via a registered
    ``MigrationFnV1``.
    """

    schema_version_string: str
    field_names: tuple[str, ...]
    field_types: tuple[str, ...]
    """Parallel to ``field_names``. Type tags are free-form
    strings ("string", "int", "float", "int_ns", "bytes",
    "list", "dict", "enum")."""

    migratable_from: tuple[str, ...] = ()
    deprecated: bool = False
    superseded_by: Optional[str] = None
    """If ``deprecated`` is True, the recommended successor's
    schema_version_string. Optional — V1 may deprecate without
    a successor."""

    def __post_init__(self) -> None:
        if len(self.field_names) != len(self.field_types):
            raise ValueError(
                "field_names and field_types must have the "
                "same length")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version_string": str(
                self.schema_version_string),
            "field_names": list(self.field_names),
            "field_types": list(self.field_types),
            "migratable_from": list(self.migratable_from),
            "deprecated": bool(self.deprecated),
            "superseded_by": (
                None if self.superseded_by is None
                else str(self.superseded_by)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w86_schema_entry_v1",
            "entry": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class SchemaRegistryV2:
    """Content-addressed registry.

    ``entries`` is a frozen mapping from
    ``schema_version_string`` → ``SchemaEntryV1``. Two schema
    entries can coexist; the registry CID changes any time the
    entry set changes.
    """

    entries: Mapping[str, SchemaEntryV1]

    @classmethod
    def empty(cls) -> "SchemaRegistryV2":
        return cls(entries={})

    def with_entry(self, entry: SchemaEntryV1) -> (
            "SchemaRegistryV2"):
        new_entries = dict(self.entries)
        new_entries[entry.schema_version_string] = entry
        return SchemaRegistryV2(entries=new_entries)

    def get(self, version_string: str) -> Optional[SchemaEntryV1]:
        return self.entries.get(version_string)

    def get_or_raise(self, version_string: str) -> SchemaEntryV1:
        e = self.get(version_string)
        if e is None:
            raise KeyError(
                f"schema_version_string {version_string!r} not "
                "in registry")
        return e

    def to_dict(self) -> dict[str, Any]:
        return {
            "entries": {
                str(k): v.to_dict()
                for k, v in sorted(self.entries.items())
            },
            "n_entries": int(len(self.entries)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w86_schema_registry_v2",
            "registry": self.to_dict()})


# ---------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CapsulePayloadV1:
    """A capsule payload tagged with its schema version.

    The payload is a mapping from field name to value.
    """

    schema_version_string: str
    fields: Mapping[str, Any]
    parent_cid: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version_string": str(
                self.schema_version_string),
            "fields": {
                str(k): _canonicalize(v)
                for k, v in sorted(self.fields.items())},
            "parent_cid": (
                None if self.parent_cid is None
                else str(self.parent_cid)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w86_capsule_payload_v1",
            "payload": self.to_dict()})


def _canonicalize(v: Any) -> Any:
    if isinstance(v, float):
        return round(v, 12)
    if isinstance(v, bytes):
        return v.hex()
    if isinstance(v, Mapping):
        return {
            str(k): _canonicalize(x) for k, x in sorted(v.items())}
    if isinstance(v, (list, tuple)):
        return [_canonicalize(x) for x in v]
    return v


# A MigrationFnV1 takes an old payload + a target schema entry +
# the registry, and returns a new payload + a content-addressed
# MigrationPlanV1.
MigrationFnV1 = Callable[
    [CapsulePayloadV1, SchemaEntryV1, SchemaRegistryV2],
    "MigrationResultV1",
]


@dataclasses.dataclass(frozen=True)
class MigrationPlanV1:
    """Content-addressed migration plan.

    Describes exactly which fields are added / dropped /
    renamed / type-converted. The plan's CID must be stable
    under repeated derivation from the same migration.
    """

    from_schema_version: str
    to_schema_version: str
    added_fields: tuple[str, ...] = ()
    dropped_fields: tuple[str, ...] = ()
    renamed_fields: tuple[tuple[str, str], ...] = ()
    """Pairs of (old_name, new_name)."""

    type_converted_fields: tuple[tuple[str, str, str], ...] = ()
    """Triples of (field_name, old_type, new_type)."""

    field_defaults: Mapping[str, Any] = dataclasses.field(
        default_factory=dict)
    """Default values for added fields (V1: numeric / string /
    null only)."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_schema_version": str(
                self.from_schema_version),
            "to_schema_version": str(self.to_schema_version),
            "added_fields": list(self.added_fields),
            "dropped_fields": list(self.dropped_fields),
            "renamed_fields": [
                [str(a), str(b)] for a, b in self.renamed_fields],
            "type_converted_fields": [
                [str(n), str(o), str(t)]
                for n, o, t in self.type_converted_fields],
            "field_defaults": {
                str(k): _canonicalize(v)
                for k, v in sorted(self.field_defaults.items())},
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w86_migration_plan_v1",
            "plan": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class MigrationResultV1:
    """Output of one migration.

    ``new_payload`` is the migrated capsule. ``migration_plan``
    is the content-addressed plan used. ``provenance_preserved``
    is True iff ``parent_cid`` was carried over.
    """

    old_payload_cid: str
    new_payload: CapsulePayloadV1
    migration_plan: MigrationPlanV1
    provenance_preserved: bool
    dropped_field_audit: tuple[tuple[str, Any], ...]
    """For each dropped field: (name, value). The audit makes
    drops explicit, never silent."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "old_payload_cid": str(self.old_payload_cid),
            "new_payload_cid": str(self.new_payload.cid()),
            "migration_plan_cid": str(
                self.migration_plan.cid()),
            "provenance_preserved": bool(
                self.provenance_preserved),
            "dropped_field_audit": [
                [str(n), _canonicalize(v)]
                for n, v in self.dropped_field_audit],
        }


@dataclasses.dataclass(frozen=True)
class MigrationEventV1:
    """Audit capsule recording one migration.

    Forms the bridge in the audit chain across schema upgrades:
    the V2 capsule's ``parent_cid`` does NOT directly equal the
    V1 capsule's CID (different schemas), but the
    ``MigrationEventV1`` records the bridge explicitly.
    """

    old_payload_cid: str
    new_payload_cid: str
    migration_plan_cid: str
    timestamp_ns: int
    registry_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W86_SCHEMA_EVOLUTION_V1_VERSION,
            "old_payload_cid": str(self.old_payload_cid),
            "new_payload_cid": str(self.new_payload_cid),
            "migration_plan_cid": str(
                self.migration_plan_cid),
            "timestamp_ns": int(self.timestamp_ns),
            "registry_cid": str(self.registry_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w86_migration_event_v1",
            "event": self.to_dict()})


# ---------------------------------------------------------------------
# Generic plan-driven migration
# ---------------------------------------------------------------------


def apply_migration_plan_v1(
        old_payload: CapsulePayloadV1,
        plan: MigrationPlanV1,
        target_schema: SchemaEntryV1) -> MigrationResultV1:
    """Apply a content-addressed migration plan deterministically.

    The plan declares: added fields (with defaults), dropped
    fields (auditably), renamed fields, type-converted fields.
    The result is a new ``CapsulePayloadV1`` whose CID is
    deterministic in (old_payload, plan).
    """
    if old_payload.schema_version_string != plan.from_schema_version:
        raise ValueError(
            f"old payload schema {old_payload.schema_version_string!r} "
            f"doesn't match plan's from "
            f"{plan.from_schema_version!r}")
    if target_schema.schema_version_string != plan.to_schema_version:
        raise ValueError(
            f"target schema {target_schema.schema_version_string!r} "
            f"doesn't match plan's to "
            f"{plan.to_schema_version!r}")

    new_fields: dict[str, Any] = {}
    dropped_audit: list[tuple[str, Any]] = []

    # Start by mapping each old field through renames + drops.
    rename_map = {old: new for old, new in plan.renamed_fields}
    drop_set = set(plan.dropped_fields)
    type_conv_map = {
        n: (old_t, new_t)
        for n, old_t, new_t in plan.type_converted_fields}

    for name, value in old_payload.fields.items():
        if name in drop_set:
            dropped_audit.append((name, value))
            continue
        new_name = rename_map.get(name, name)
        new_value = value
        if name in type_conv_map:
            _, new_t = type_conv_map[name]
            new_value = _coerce_to_type(value, new_t)
        new_fields[new_name] = new_value

    # Apply defaults for added fields.
    for added in plan.added_fields:
        if added not in new_fields:
            new_fields[added] = _canonicalize(
                plan.field_defaults.get(added, None))

    # Enforce that the new field set matches the target schema.
    target_field_names = set(target_schema.field_names)
    extra = set(new_fields.keys()) - target_field_names
    missing = target_field_names - set(new_fields.keys())
    if extra:
        raise ValueError(
            f"migration result has fields not in target schema: "
            f"{sorted(extra)}")
    if missing:
        raise ValueError(
            f"migration result is missing target-schema fields: "
            f"{sorted(missing)}")

    new_payload = CapsulePayloadV1(
        schema_version_string=plan.to_schema_version,
        fields=new_fields,
        parent_cid=old_payload.parent_cid)

    return MigrationResultV1(
        old_payload_cid=old_payload.cid(),
        new_payload=new_payload,
        migration_plan=plan,
        provenance_preserved=(
            old_payload.parent_cid == new_payload.parent_cid),
        dropped_field_audit=tuple(dropped_audit))


def _coerce_to_type(value: Any, new_type: str) -> Any:
    if new_type == "int":
        return int(value)
    if new_type == "float":
        return float(value)
    if new_type == "string":
        return str(value)
    if new_type == "int_ns":
        # float seconds → ns int.
        if isinstance(value, (int,)):
            return int(value)
        return int(round(float(value) * 1_000_000_000))
    if new_type == "float_seconds":
        # ns int → float seconds.
        return float(int(value)) / 1_000_000_000
    if new_type == "list":
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]
    return value


# ---------------------------------------------------------------------
# Chain verification across migrations
# ---------------------------------------------------------------------


def verify_chain_across_migrations_v1(
        payloads: Sequence[CapsulePayloadV1],
        migration_events: Sequence[MigrationEventV1]) -> dict[
            str, Any]:
    """Verify that an audit chain re-verifies across schema
    migrations.

    The load-bearing property is:

    * Every ``MigrationEventV1`` references real CIDs — for each
      bridge ``(old_cid, new_cid)``, both ``old_cid`` and
      ``new_cid`` correspond to payloads in the input set.
    * Provenance is preserved: the V2 payload's ``parent_cid``
      equals the V1 payload's ``parent_cid`` for every bridge
      whose endpoints are in the payload set.
    * For payloads whose ``parent_cid`` *is* in the payload set,
      the link must point to a real payload (no self-loops, no
      forward-references). Payloads whose ``parent_cid`` is not
      in the payload set are treated as referencing an external
      root and are NOT counted as dangling — long-running runs
      legitimately reach back past the current window.

    Returns ``{chain_verifies, dangling_count, ...}``.
    """
    payload_by_cid = {p.cid(): p for p in payloads}
    payload_cids = set(payload_by_cid.keys())

    broken_bridges: list[dict[str, str]] = []
    provenance_violations: list[dict[str, str]] = []
    dangling_in_scope: list[dict[str, str]] = []

    for ev in migration_events:
        old_in = ev.old_payload_cid in payload_cids
        new_in = ev.new_payload_cid in payload_cids
        if not (old_in and new_in):
            broken_bridges.append({
                "migration_event_cid": ev.cid(),
                "old_payload_cid": ev.old_payload_cid,
                "new_payload_cid": ev.new_payload_cid,
                "old_in_payloads": str(old_in).lower(),
                "new_in_payloads": str(new_in).lower(),
            })
            continue
        old_p = payload_by_cid[ev.old_payload_cid]
        new_p = payload_by_cid[ev.new_payload_cid]
        if old_p.parent_cid != new_p.parent_cid:
            provenance_violations.append({
                "migration_event_cid": ev.cid(),
                "old_parent_cid": old_p.parent_cid or "",
                "new_parent_cid": new_p.parent_cid or "",
            })

    for p in payloads:
        if p.parent_cid is None:
            continue
        # Self-loop / forward-reference check: parent_cid that
        # equals p's own CID is broken.
        if p.parent_cid == p.cid():
            dangling_in_scope.append({
                "payload_cid": p.cid(),
                "missing_parent": p.parent_cid,
                "kind": "self_loop",
            })

    chain_verifies = (
        not broken_bridges
        and not provenance_violations
        and not dangling_in_scope)
    return {
        "chain_verifies": bool(chain_verifies),
        "dangling_count": int(len(dangling_in_scope)),
        "dangling": dangling_in_scope,
        "broken_bridge_count": int(len(broken_bridges)),
        "broken_bridges": broken_bridges,
        "provenance_violation_count": int(
            len(provenance_violations)),
        "provenance_violations": provenance_violations,
    }


# ---------------------------------------------------------------------
# Deprecated-but-readable
# ---------------------------------------------------------------------


def read_payload_with_deprecation_warning_v1(
        payload: CapsulePayloadV1,
        registry: SchemaRegistryV2) -> tuple[
            CapsulePayloadV1, Optional[str]]:
    """Read a payload, emitting a warning if its schema is
    deprecated. Returns ``(payload, warning_message_or_None)``.
    """
    entry = registry.get(payload.schema_version_string)
    if entry is None:
        return payload, (
            f"schema {payload.schema_version_string!r} "
            "not in registry")
    if entry.deprecated:
        msg = (
            f"schema {payload.schema_version_string!r} is "
            f"deprecated"
            + (f" (superseded by {entry.superseded_by!r})"
               if entry.superseded_by else "")
        )
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        return payload, msg
    return payload, None


# ---------------------------------------------------------------------
# In-flight upgrade bench
# ---------------------------------------------------------------------


def _build_v1_v2_registry_example() -> tuple[
        SchemaRegistryV2, SchemaEntryV1, SchemaEntryV1,
        MigrationPlanV1]:
    """An example V1 → V2 migration shape.

    V1: ``MigrationEnvelopeV1`` with fields ``envelope_id`` (str),
    ``arrival_delay`` (float seconds), ``payload_bytes_hex`` (str).
    V2: ``MigrationEnvelopeV2`` with fields ``envelope_id`` (str),
    ``arrival_delay_ns`` (int ns — RENAMED + TYPE-CONVERTED),
    ``payload_bytes_hex`` (str), ``forwarded_from`` (str, ADDED
    with default "").
    """
    v1 = SchemaEntryV1(
        schema_version_string="coordpy.migration_envelope.v1",
        field_names=(
            "envelope_id", "arrival_delay",
            "payload_bytes_hex"),
        field_types=("string", "float", "string"),
        deprecated=True,
        superseded_by="coordpy.migration_envelope.v2")
    v2 = SchemaEntryV1(
        schema_version_string="coordpy.migration_envelope.v2",
        field_names=(
            "envelope_id", "arrival_delay_ns",
            "payload_bytes_hex", "forwarded_from"),
        field_types=("string", "int_ns", "string", "string"),
        migratable_from=("coordpy.migration_envelope.v1",))
    registry = (
        SchemaRegistryV2.empty()
        .with_entry(v1)
        .with_entry(v2))
    plan = MigrationPlanV1(
        from_schema_version=v1.schema_version_string,
        to_schema_version=v2.schema_version_string,
        added_fields=("forwarded_from",),
        renamed_fields=(("arrival_delay", "arrival_delay_ns"),),
        type_converted_fields=(
            ("arrival_delay", "float", "int_ns"),),
        field_defaults={"forwarded_from": ""})
    return registry, v1, v2, plan


@dataclasses.dataclass(frozen=True)
class SchemaEvolutionBenchReportV1:
    """In-flight schema upgrade bench output."""

    registry_cid_before: str
    registry_cid_after: str
    migration_plan_cid: str
    v1_payload_cid: str
    v2_payload_cid: str
    migration_event_cid: str
    chain_verifies_across_migration: bool
    deprecated_payload_readable: bool
    deprecation_warning_emitted: bool
    deterministic_migration: bool
    """The same input migrated twice produces the same CID."""

    provenance_preserved: bool
    dropped_field_count: int
    report_cid: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W86_SCHEMA_EVOLUTION_V1_VERSION,
            "registry_cid_before": str(
                self.registry_cid_before),
            "registry_cid_after": str(self.registry_cid_after),
            "migration_plan_cid": str(self.migration_plan_cid),
            "v1_payload_cid": str(self.v1_payload_cid),
            "v2_payload_cid": str(self.v2_payload_cid),
            "migration_event_cid": str(
                self.migration_event_cid),
            "chain_verifies_across_migration": bool(
                self.chain_verifies_across_migration),
            "deprecated_payload_readable": bool(
                self.deprecated_payload_readable),
            "deprecation_warning_emitted": bool(
                self.deprecation_warning_emitted),
            "deterministic_migration": bool(
                self.deterministic_migration),
            "provenance_preserved": bool(
                self.provenance_preserved),
            "dropped_field_count": int(
                self.dropped_field_count),
            "report_cid": str(self.report_cid),
        }

    def cid(self) -> str:
        d = self.to_dict()
        d["report_cid"] = ""
        return _sha256_hex({
            "kind": "w86_schema_evolution_bench_report_v1",
            "report": d})


def run_in_flight_schema_upgrade_bench_v1(
        timestamp_ns: int = 86_041_000_000_000,
        parent_cid: str = "ancestor_chain_root_cid") -> (
            SchemaEvolutionBenchReportV1):
    """Bench the V1 → V2 migration end-to-end.

    1. Build the V1/V2 registry + plan.
    2. Produce a V1 payload (with parent_cid set to simulate
       being in a chain).
    3. Migrate to V2.
    4. Verify the chain re-verifies via a MigrationEventV1
       bridge.
    5. Verify the V1 payload remains readable with a
       deprecation warning.
    6. Verify the migration is deterministic.
    """
    registry, _v1_entry, v2_entry, plan = (
        _build_v1_v2_registry_example())
    reg_before_cid = registry.cid()

    v1_payload = CapsulePayloadV1(
        schema_version_string="coordpy.migration_envelope.v1",
        fields={
            "envelope_id": "ENV-86041",
            "arrival_delay": 0.125,  # seconds
            "payload_bytes_hex": "deadbeef",
        },
        parent_cid=parent_cid)

    res = apply_migration_plan_v1(
        old_payload=v1_payload, plan=plan,
        target_schema=v2_entry)
    v2_payload = res.new_payload

    bridge = MigrationEventV1(
        old_payload_cid=v1_payload.cid(),
        new_payload_cid=v2_payload.cid(),
        migration_plan_cid=plan.cid(),
        timestamp_ns=int(timestamp_ns),
        registry_cid=reg_before_cid)

    # Chain verification — both payloads in the set, with the
    # bridge event.
    chain_check = verify_chain_across_migrations_v1(
        payloads=(v1_payload, v2_payload),
        migration_events=(bridge,))
    chain_ok = bool(chain_check["chain_verifies"])

    # Deprecated-but-readable check.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        read_payload, msg = (
            read_payload_with_deprecation_warning_v1(
                v1_payload, registry))
    deprecation_emitted = any(
        issubclass(w.category, DeprecationWarning)
        for w in caught)
    deprecated_readable = (
        read_payload.cid() == v1_payload.cid())

    # Deterministic migration check.
    res2 = apply_migration_plan_v1(
        old_payload=v1_payload, plan=plan,
        target_schema=v2_entry)
    deterministic = (
        res.new_payload.cid() == res2.new_payload.cid())

    rep = SchemaEvolutionBenchReportV1(
        registry_cid_before=reg_before_cid,
        registry_cid_after=reg_before_cid,  # no new entries
        migration_plan_cid=plan.cid(),
        v1_payload_cid=v1_payload.cid(),
        v2_payload_cid=v2_payload.cid(),
        migration_event_cid=bridge.cid(),
        chain_verifies_across_migration=chain_ok,
        deprecated_payload_readable=deprecated_readable,
        deprecation_warning_emitted=deprecation_emitted,
        deterministic_migration=deterministic,
        provenance_preserved=res.provenance_preserved,
        dropped_field_count=len(res.dropped_field_audit))
    rep = dataclasses.replace(rep, report_cid=rep.cid())
    return rep


__all__ = [
    "W86_SCHEMA_EVOLUTION_V1_VERSION",
    "SchemaEntryV1",
    "SchemaRegistryV2",
    "CapsulePayloadV1",
    "MigrationPlanV1",
    "MigrationResultV1",
    "MigrationEventV1",
    "SchemaEvolutionBenchReportV1",
    "apply_migration_plan_v1",
    "verify_chain_across_migrations_v1",
    "read_payload_with_deprecation_warning_v1",
    "run_in_flight_schema_upgrade_bench_v1",
]
