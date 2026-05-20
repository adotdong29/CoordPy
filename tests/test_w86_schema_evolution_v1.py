"""Tests for ``coordpy.schema_evolution_v1``."""

from __future__ import annotations

import warnings

import pytest

from coordpy.schema_evolution_v1 import (
    CapsulePayloadV1,
    MigrationEventV1,
    MigrationPlanV1,
    SchemaEntryV1,
    SchemaEvolutionBenchReportV1,
    SchemaRegistryV2,
    apply_migration_plan_v1,
    read_payload_with_deprecation_warning_v1,
    run_in_flight_schema_upgrade_bench_v1,
    verify_chain_across_migrations_v1,
)


def test_schema_entry_content_addressed():
    s1 = SchemaEntryV1(
        schema_version_string="coordpy.foo.v1",
        field_names=("a", "b"), field_types=("string", "int"))
    s2 = SchemaEntryV1(
        schema_version_string="coordpy.foo.v1",
        field_names=("a", "b"), field_types=("string", "int"))
    assert s1.cid() == s2.cid()
    s3 = SchemaEntryV1(
        schema_version_string="coordpy.foo.v1",
        field_names=("a", "c"), field_types=("string", "int"))
    assert s1.cid() != s3.cid()


def test_schema_entry_validates_field_lengths():
    with pytest.raises(ValueError):
        SchemaEntryV1(
            schema_version_string="x",
            field_names=("a",),
            field_types=("int", "string"))


def test_schema_registry_two_versions_coexist():
    s1 = SchemaEntryV1(
        schema_version_string="x.v1",
        field_names=("a",), field_types=("int",))
    s2 = SchemaEntryV1(
        schema_version_string="x.v2",
        field_names=("a", "b"),
        field_types=("int", "string"),
        migratable_from=("x.v1",))
    r = (
        SchemaRegistryV2.empty()
        .with_entry(s1)
        .with_entry(s2))
    assert r.get("x.v1") == s1
    assert r.get("x.v2") == s2
    assert r.cid() != SchemaRegistryV2.empty().cid()


def test_migration_plan_cid_stable():
    p1 = MigrationPlanV1(
        from_schema_version="x.v1",
        to_schema_version="x.v2",
        added_fields=("b",),
        field_defaults={"b": "default"})
    p2 = MigrationPlanV1(
        from_schema_version="x.v1",
        to_schema_version="x.v2",
        added_fields=("b",),
        field_defaults={"b": "default"})
    assert p1.cid() == p2.cid()


def test_migration_adds_default_for_added_field():
    s1 = SchemaEntryV1(
        schema_version_string="x.v1",
        field_names=("a",), field_types=("int",))
    s2 = SchemaEntryV1(
        schema_version_string="x.v2",
        field_names=("a", "b"),
        field_types=("int", "string"),
        migratable_from=("x.v1",))
    plan = MigrationPlanV1(
        from_schema_version="x.v1",
        to_schema_version="x.v2",
        added_fields=("b",),
        field_defaults={"b": "default_b"})
    p1 = CapsulePayloadV1(
        schema_version_string="x.v1", fields={"a": 7})
    res = apply_migration_plan_v1(p1, plan, s2)
    assert res.new_payload.fields["a"] == 7
    assert res.new_payload.fields["b"] == "default_b"


def test_migration_renames_field():
    s1 = SchemaEntryV1(
        schema_version_string="x.v1",
        field_names=("old_name",), field_types=("int",))
    s2 = SchemaEntryV1(
        schema_version_string="x.v2",
        field_names=("new_name",), field_types=("int",),
        migratable_from=("x.v1",))
    plan = MigrationPlanV1(
        from_schema_version="x.v1",
        to_schema_version="x.v2",
        renamed_fields=(("old_name", "new_name"),))
    p1 = CapsulePayloadV1(
        schema_version_string="x.v1",
        fields={"old_name": 42})
    res = apply_migration_plan_v1(p1, plan, s2)
    assert "old_name" not in res.new_payload.fields
    assert res.new_payload.fields["new_name"] == 42


def test_migration_type_converts_float_seconds_to_int_ns():
    s1 = SchemaEntryV1(
        schema_version_string="x.v1",
        field_names=("delay",), field_types=("float",))
    s2 = SchemaEntryV1(
        schema_version_string="x.v2",
        field_names=("delay_ns",), field_types=("int_ns",),
        migratable_from=("x.v1",))
    plan = MigrationPlanV1(
        from_schema_version="x.v1",
        to_schema_version="x.v2",
        renamed_fields=(("delay", "delay_ns"),),
        type_converted_fields=(
            ("delay", "float", "int_ns"),))
    p1 = CapsulePayloadV1(
        schema_version_string="x.v1",
        fields={"delay": 0.125})
    res = apply_migration_plan_v1(p1, plan, s2)
    assert res.new_payload.fields["delay_ns"] == 125_000_000


def test_migration_drops_field_with_audit():
    s1 = SchemaEntryV1(
        schema_version_string="x.v1",
        field_names=("a", "b"),
        field_types=("int", "string"))
    s2 = SchemaEntryV1(
        schema_version_string="x.v2",
        field_names=("a",), field_types=("int",),
        migratable_from=("x.v1",))
    plan = MigrationPlanV1(
        from_schema_version="x.v1",
        to_schema_version="x.v2",
        dropped_fields=("b",))
    p1 = CapsulePayloadV1(
        schema_version_string="x.v1",
        fields={"a": 1, "b": "secret"})
    res = apply_migration_plan_v1(p1, plan, s2)
    assert res.dropped_field_audit == (("b", "secret"),)
    assert "b" not in res.new_payload.fields


def test_migration_preserves_parent_cid():
    s1 = SchemaEntryV1(
        schema_version_string="x.v1",
        field_names=("a",), field_types=("int",))
    s2 = SchemaEntryV1(
        schema_version_string="x.v2",
        field_names=("a",), field_types=("int",),
        migratable_from=("x.v1",))
    plan = MigrationPlanV1(
        from_schema_version="x.v1",
        to_schema_version="x.v2")
    p1 = CapsulePayloadV1(
        schema_version_string="x.v1", fields={"a": 1},
        parent_cid="parent_cid_42")
    res = apply_migration_plan_v1(p1, plan, s2)
    assert res.new_payload.parent_cid == "parent_cid_42"
    assert res.provenance_preserved is True


def test_migration_rejects_wrong_source_schema():
    s2 = SchemaEntryV1(
        schema_version_string="x.v2",
        field_names=("a",), field_types=("int",))
    plan = MigrationPlanV1(
        from_schema_version="x.v1",
        to_schema_version="x.v2")
    p_wrong = CapsulePayloadV1(
        schema_version_string="y.v1", fields={"a": 1})
    with pytest.raises(ValueError):
        apply_migration_plan_v1(p_wrong, plan, s2)


def test_migration_rejects_extra_or_missing_target_fields():
    s2 = SchemaEntryV1(
        schema_version_string="x.v2",
        field_names=("a",), field_types=("int",),
        migratable_from=("x.v1",))
    plan = MigrationPlanV1(
        from_schema_version="x.v1",
        to_schema_version="x.v2",
        added_fields=("b",),  # but s2 doesn't have b
        field_defaults={"b": 99})
    p1 = CapsulePayloadV1(
        schema_version_string="x.v1", fields={"a": 1})
    with pytest.raises(ValueError):
        apply_migration_plan_v1(p1, plan, s2)


def test_chain_verifies_across_migration_bridge():
    s1 = SchemaEntryV1(
        schema_version_string="x.v1",
        field_names=("a",), field_types=("int",))
    s2 = SchemaEntryV1(
        schema_version_string="x.v2",
        field_names=("a",), field_types=("int",),
        migratable_from=("x.v1",))
    plan = MigrationPlanV1(
        from_schema_version="x.v1",
        to_schema_version="x.v2")
    p1 = CapsulePayloadV1(
        schema_version_string="x.v1", fields={"a": 1},
        parent_cid="ancestor_cid")
    res = apply_migration_plan_v1(p1, plan, s2)
    p2 = res.new_payload
    # Now make p3 whose parent is p1's CID (an event captured
    # under the V1 schema). After migration, p3 should still
    # verify because p2.parent_cid = p1.parent_cid = "ancestor".
    # The bridge event records the V1->V2 transition.
    bridge = MigrationEventV1(
        old_payload_cid=p1.cid(),
        new_payload_cid=p2.cid(),
        migration_plan_cid=plan.cid(),
        timestamp_ns=86_041_000_000_000,
        registry_cid="reg_cid")
    verdict = verify_chain_across_migrations_v1(
        payloads=(p1, p2), migration_events=(bridge,))
    assert verdict["chain_verifies"] is True


def test_deprecated_payload_emits_warning_but_still_readable():
    s1 = SchemaEntryV1(
        schema_version_string="x.v1",
        field_names=("a",), field_types=("int",),
        deprecated=True, superseded_by="x.v2")
    registry = SchemaRegistryV2.empty().with_entry(s1)
    p = CapsulePayloadV1(
        schema_version_string="x.v1", fields={"a": 1})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        read, msg = (
            read_payload_with_deprecation_warning_v1(
                p, registry))
    assert read.cid() == p.cid()
    assert msg is not None and "deprecated" in msg
    assert any(
        issubclass(w.category, DeprecationWarning)
        for w in caught)


def test_in_flight_schema_upgrade_bench_meets_all_dod():
    rep = run_in_flight_schema_upgrade_bench_v1()
    assert rep.chain_verifies_across_migration is True
    assert rep.deprecated_payload_readable is True
    assert rep.deprecation_warning_emitted is True
    assert rep.deterministic_migration is True
    assert rep.provenance_preserved is True


def test_bench_report_cid_is_deterministic():
    r1 = run_in_flight_schema_upgrade_bench_v1()
    r2 = run_in_flight_schema_upgrade_bench_v1()
    assert r1.report_cid == r2.report_cid


def test_broken_bridge_breaks_chain_verification():
    """A MigrationEvent that references a CID NOT in the
    payload set is a broken bridge → chain doesn't verify.
    """
    p = CapsulePayloadV1(
        schema_version_string="x.v1", fields={"a": 1})
    fake_bridge = MigrationEventV1(
        old_payload_cid="nonexistent_old",
        new_payload_cid=p.cid(),
        migration_plan_cid="plan_cid",
        timestamp_ns=0, registry_cid="reg_cid")
    verdict = verify_chain_across_migrations_v1(
        payloads=(p,), migration_events=(fake_bridge,))
    assert verdict["chain_verifies"] is False
    assert verdict["broken_bridge_count"] == 1


def test_provenance_violation_in_bridge_breaks_chain():
    """If a bridge connects two payloads with different
    parent_cids → provenance violation."""
    p1 = CapsulePayloadV1(
        schema_version_string="x.v1", fields={"a": 1},
        parent_cid="parent_A")
    p2 = CapsulePayloadV1(
        schema_version_string="x.v2", fields={"a": 1},
        parent_cid="parent_B")  # different — provenance broken
    bridge = MigrationEventV1(
        old_payload_cid=p1.cid(),
        new_payload_cid=p2.cid(),
        migration_plan_cid="plan_cid",
        timestamp_ns=0, registry_cid="reg_cid")
    verdict = verify_chain_across_migrations_v1(
        payloads=(p1, p2), migration_events=(bridge,))
    assert verdict["chain_verifies"] is False
    assert verdict["provenance_violation_count"] == 1
