"""Read-only scoring for selected Compose tasks.

This adapter never resets, steps, forwards, or changes a simulator. The caller
must sample it after every actual control. ``OrderedEvents`` tracks the RAIN
event-order rules.

VCN21 also observes the ten warmup controls at steps -9 through 0. The order
tracker begins at 0; the microwave runtime guard sees all warmup controls.
Contact and geometry helpers are compiled from selected pure function
definitions without importing or modifying their source modules.
"""
from __future__ import annotations

import ast
from collections import deque
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .composition_batch_order import OrderedEvents, parse_atom


ROOT = Path(__file__).resolve().parent
PROTOCOL = "selected_compose_pi05_exact_rain_order_and_vcn21_v3_v1"
VCN21_ENTRY = ROOT / "run_moka_microwave_compose_evaluator_entry.py"
GEOMETRY_ENTRY = ROOT / "build_composition_layout_revision.py"
ORDER_ENTRY = ROOT / "composition_batch_order.py"
# Source files included in the scoring protocol fingerprint.
DEPENDENCY_PATHS = (Path(__file__).resolve(), ORDER_ENTRY, VCN21_ENTRY,
                    GEOMETRY_ENTRY)
CONTACT_WINDOW = 2
MIN_NORMAL_FORCE_N = 1e-6
EXPECTED = {
    "VCN8_008": [("on", "chocolate_pudding_1", "akita_black_bowl_1"),
                 ("open", "wooden_cabinet_1_middle_region")],
    "VCN9_010": [("on", "cream_cheese_1", "flat_stove_1_cook_region"),
                 ("turnon", "flat_stove_1")],
    "VCN10_001": [("on", "cream_cheese_1", "flat_stove_1_cook_region"),
                  ("on", "plate_1", "main_table_stove_front_region")],
    "VCN19_020": [("open", "wooden_cabinet_1_top_region"),
                  ("on", "glazed_rim_porcelain_ramekin_1", "plate_1")],
    "VCN21_001": [("on", "moka_pot_2", "flat_stove_1_cook_region"),
                  ("close", "microwave_1")],
    "VCN35_004": [("in", "glazed_rim_porcelain_ramekin_1", "basket_1_contain_region"),
                  ("on", "alphabet_soup_1", "plate_2")],
    "UCOMP_040": [("in", "tomato_sauce_1", "basket_1_contain_region"),
                  ("on", "white_yellow_mug_1", "plate_1")],
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pure_functions(path: Path, names: tuple[str, ...], namespace: dict) -> dict:
    """Load exact trusted local function ASTs, with no module-level execution."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    if sorted(node.name for node in functions) != sorted(names):
        raise RuntimeError(f"Expected unique pure helper definitions in {path}")
    module = ast.Module(body=functions, type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def _helpers() -> dict:
    ns = {"np": np,
          "MIN_DIRECT_CONTACT_NORMAL_FORCE_N": MIN_NORMAL_FORCE_N}
    _pure_functions(VCN21_ENTRY,
                    ("contact_certifies", "descendants", "contact_sets",
                     "direct_contacts", "overlap"), ns)
    _pure_functions(GEOMETRY_ENTRY, ("bounds",), ns)
    return ns


def _vcn21_sweep(bundle: Path, index: int, digest: str,
                 sweep_path: str | Path | None) -> tuple[dict, list[Path]]:
    replay_path = bundle / "FIXTURE_REPLAY.json"
    replay_digest = _sha(replay_path)
    matches = [row for row in json.loads(replay_path.read_text())["rows"]
               if row["state_sha256"] == digest
               and int(row["init_state_index"]) == int(index)]
    if len(matches) != 1:
        raise RuntimeError("VCN21 requires one exact initial-state/replay match")
    provenance_path = bundle / "VCN21_V3_SUPPLEMENTAL_PROVENANCE_MANIFEST.json"
    provenance = json.loads(provenance_path.read_text())
    artifacts = {row["name"]: row for row in provenance["artifacts"]}
    source = artifacts["benchmark_microwave_sweep_evidence"]
    sweep_path = Path(sweep_path) if sweep_path else bundle / source["path"]
    if _sha(sweep_path) != source["sha256"]:
        raise RuntimeError("VCN21 sweep differs from immutable v3 provenance")
    document = json.loads(sweep_path.read_text())
    if document.get("fixture_replay_sha256") != replay_digest:
        raise RuntimeError("VCN21 sweep/replay digest mismatch")
    matches = [row for row in document["states"]
               if int(row.get("init_state_index", -1)) == int(index)
               and row.get("state_sha256") == digest
               and row.get("fixture_replay_sha256") == replay_digest]
    if len(matches) != 1:
        raise RuntimeError("VCN21 requires one content-addressed sweep row")
    row = matches[0]
    if row.get("passes") is not True or row.get("moving_door_body") != "microwave_1_microdoorroot":
        raise RuntimeError("VCN21 initial sweep was not admissible")
    swept = np.asarray(row["swept_moving_subtree_aabb"], dtype=float)
    if swept.shape != (2, 3) or not np.all(np.isfinite(swept)) or not np.all(swept[1] >= swept[0]):
        raise RuntimeError("VCN21 invalid swept door bounds")
    info = {"init_state_index": int(index), "state_sha256": digest,
            "fixture_replay_sha256": replay_digest,
            "sweep_sha256": _sha(sweep_path),
            "historical_v3_entry_sha256": artifacts["v3_evaluator_entrypoint"]["sha256"],
            "current_helper_source_sha256": _sha(VCN21_ENTRY),
            "historical_full_entry_byte_identical": _sha(VCN21_ENTRY) == artifacts["v3_evaluator_entrypoint"]["sha256"],
            "helper_loading_scope": "Only contact_certifies/descendants/contact_sets/direct_contacts/overlap function ASTs; full evaluator is not imported",
            "moving_door_body": row["moving_door_body"],
            "swept_moving_subtree_aabb": swept.tolist()}
    return info, [replay_path, provenance_path, sweep_path, VCN21_ENTRY, GEOMETRY_ENTRY]


class SelectedComposeScoring:
    """Serialisable observation-only exact RAIN composition success adapter."""

    def __init__(self, bundle: str | Path, init_state_index: int,
                 state_sha256: str, sweep_path: str | Path | None = None,
                 warmup_steps: int = 10):
        self.bundle = Path(bundle).resolve()
        rules_path = self.bundle / "eval_rules.yaml"
        self.rules = yaml.safe_load(rules_path.read_text())
        self.task_id = self.rules.get("legacy_task_id", self.rules["task_id"])
        if self.task_id not in EXPECTED:
            raise ValueError(f"Task outside supported Compose scope: {self.task_id}")
        self.atoms = list(self.rules.get("ordered_event_atoms") or [])
        self.parsed = [parse_atom(a) for a in self.atoms]
        if self.parsed != EXPECTED[self.task_id]:
            raise RuntimeError("Selected composition atoms do not match frozen task contract")
        if ([parse_atom(a) for a in self.rules.get("required_goal_atoms", [])] != self.parsed
                or self.rules.get("strict_event_order") is not True
                or self.rules.get("order_sensitive") is not True
                or self.rules.get("forbidden_goal_atoms")
                or self.rules.get("final_tc_gate", False) is not False):
            raise RuntimeError("Unsupported selected composition scoring rule")
        self.ordered = OrderedEvents(self.atoms)
        self.warmup_steps = int(warmup_steps)
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be nonnegative")
        self.special = self.task_id == "VCN21_001"
        self.sweep_info = None
        dependencies = [Path(__file__).resolve(), ORDER_ENTRY, rules_path]
        if self.special:
            required = {
                "final_success_requires_all_bddl_goals": True,
                "microwave_close_direct_contact_required": True,
                "direct_contact_window_control_steps": CONTACT_WINDOW,
                "runtime_contract_version": "vcn21_runtime_contact_clearance_v3",
                "direct_contact_requires_active_constraint": True,
                "minimum_direct_contact_normal_force_n": MIN_NORMAL_FORCE_N,
                "success_gate_participates_before_termination": True,
            }
            if any(self.rules.get(k) != v for k, v in required.items()):
                raise RuntimeError("VCN21 strict-v3 runtime contract missing")
            if self.rules.get("runtime_moka_door_sweep_clearance_required_at") != ["close_rising", "final_bddl_success"]:
                raise RuntimeError("VCN21 clearance timing contract changed")
            self.sweep_info, extra = _vcn21_sweep(
                self.bundle, init_state_index, state_sha256, sweep_path)
            dependencies += extra
        self.dependency_paths = tuple(dict.fromkeys(dependencies))
        self.dependency_hashes = {str(p): _sha(p) for p in self.dependency_paths}
        self.records: list[dict] = []
        self.close_events: list[dict] = []
        self.placement_events: list[dict] = []
        self.final_gate_checks: list[dict] = []
        self.window: deque[dict] = deque(maxlen=CONTACT_WINDOW)
        self.previous_special: tuple[bool, bool] | None = None
        self.last_step: int | None = None
        self.native_final = False
        self.final_clearance: dict | None = None
        self.init_state_index = int(init_state_index)
        self.state_sha256 = str(state_sha256)
        self._helper = None
        self._contact_model = None
        self._door_geoms: set[int] | None = None
        self._gripper_geoms: set[int] | None = None

    def snapshot(self, env, step: int) -> dict:
        """Observe only: no simulator reset/step/forward or array mutation."""
        values = [bool(env.env._eval_predicate(atom)) for atom in self.parsed]
        fact = {"step": int(step), "native_event_values": values,
                "native_bddl_success": bool(env.check_success())}
        if self.special:
            if self._helper is None:
                self._helper = _helpers()
            if self._contact_model is not env.sim.model:
                self._door_geoms, self._gripper_geoms = self._helper["contact_sets"](env)
                self._contact_model = env.sim.model
            fact["direct_microdoor_contact_pairs"] = self._helper["direct_contacts"](
                env, self._door_geoms, self._gripper_geoms)
            bounds = self._helper["bounds"](env, "moka_pot_2")
            fact["moka_collision_aabb"] = [np.asarray(v).tolist() for v in bounds]
        return fact

    def _clearance(self, fact: dict, semantic_time: str) -> dict:
        aabb = np.asarray(fact["moka_collision_aabb"], dtype=float)
        swept = np.asarray(self.sweep_info["swept_moving_subtree_aabb"], dtype=float)
        if aabb.shape != (2, 3) or not np.all(np.isfinite(aabb)) or not np.all(aabb[1] >= aabb[0]):
            raise ValueError("Invalid observed moka collision AABB")
        # Exact same strict-positive-axis AABB intersection as original overlap.
        intersects = bool(np.all(np.minimum(aabb[1], swept[1]) - np.maximum(aabb[0], swept[0]) > 0.0))
        return {"actual_env_step_including_warmup": fact["step"] + self.warmup_steps,
                "post_warmup_control_step": fact["step"],
                "semantic_time": semantic_time, "moka_collision_aabb": aabb.tolist(),
                "door_swept_aabb": swept.tolist(), "overlap": intersects,
                "runtime_door_clearance_pass": not intersects}

    def update(self, fact_or_step: dict | int, fact: dict | None = None) -> None:
        """Consume one snapshot; accepts update(fact) or update(step, fact)."""
        if fact is None:
            fact = deepcopy(fact_or_step)
        else:
            fact = deepcopy(fact)
            if "step" in fact and int(fact["step"]) != int(fact_or_step):
                raise ValueError("Snapshot/update step mismatch")
            fact["step"] = int(fact_or_step)
        if not isinstance(fact, dict):
            raise TypeError("update requires a raw snapshot dictionary")
        step = fact["step"]
        if not isinstance(step, int) or isinstance(step, bool):
            raise TypeError("Step must be an actual integer control index")
        if self.last_step is None:
            expected = 1 - self.warmup_steps if self.special and self.warmup_steps else 0
            if step != expected:
                raise RuntimeError(f"Missing initial control audit: expected {expected}, got {step}")
        elif step != self.last_step + 1:
            raise RuntimeError("Every actual control must be observed exactly once, contiguously")
        values = fact["native_event_values"]
        if len(values) != len(self.atoms) or not all(type(v) is bool for v in values):
            raise ValueError("Raw native event values must be exact booleans")
        if type(fact["native_bddl_success"]) is not bool:
            raise ValueError("Raw final BDDL outcome must be boolean")
        # Every in-scope BDDL has exactly the two canonical conjunction atoms.
        # Keep both independent readings, and fail closed on disagreement.
        if fact["native_bddl_success"] != all(values):
            raise RuntimeError("Native final BDDL disagrees with its two canonical predicates")
        if self.special:
            pairs = fact["direct_microdoor_contact_pairs"]
            certifying = []
            for pair in pairs:
                force = float(pair["normal_force_n"])
                if not np.isfinite(force):
                    raise ValueError("Nonfinite direct-contact force")
                certifies = bool(int(pair["efc_address"]) >= 0 and force > MIN_NORMAL_FORCE_N)
                if bool(pair["certifying_physical_contact"]) != certifies:
                    raise RuntimeError("Raw contact certification disagrees with original v3 definition")
                if bool(pair["solver_constraint_active"]) != (int(pair["efc_address"]) >= 0):
                    raise RuntimeError("Raw contact active-constraint flag disagrees with solver address")
                if certifies:
                    certifying.append(pair)
            # step 0 is the tenth real warmup control, not an extra fake step.
            self.window.append({
                "actual_env_step_including_warmup": step + self.warmup_steps,
                "post_warmup_control_step": step,
                "contact_pairs_present": bool(pairs),
                "certifying_physical_contact": bool(certifying), "pairs": deepcopy(pairs)})
            p_now, c_now = values
            if self.previous_special is not None:
                if p_now and self.previous_special[0] is False and not self.placement_events:
                    self.placement_events.append(self._clearance(fact, "first_on_rising"))
                if c_now and self.previous_special[1] is False:
                    sampled = deepcopy(list(self.window))
                    self.close_events.append({
                        "actual_env_step_including_warmup": step + self.warmup_steps,
                        "post_warmup_control_step": step,
                        "contact_window_control_steps": CONTACT_WINDOW,
                        "contact_window": sampled,
                        "direct_microdoor_contact_pairs_in_window": any(r["contact_pairs_present"] for r in sampled),
                        "direct_microdoor_physical_contact_in_window": any(r["certifying_physical_contact"] for r in sampled),
                        "moka_at_close_rising": self._clearance(fact, "close_rising")})
            self.previous_special = (p_now, c_now)
            self.final_clearance = self._clearance(fact, "post_return_final")
            if fact["native_bddl_success"]:
                self.final_gate_checks.append(self._clearance(fact, "current_final_bddl"))
        if step >= 0:
            self.ordered.update(step, values)
        self.native_final = fact["native_bddl_success"]
        self.last_step = step
        self.records.append(fact)

    @property
    def violation(self) -> str | None:
        return self.ordered.violation

    @property
    def complete(self) -> bool:
        result = bool(self.ordered.complete and self.native_final)
        if not self.special:
            return result
        first = self.close_events[0] if self.close_events else None
        return bool(result and first and self.placement_events
                    and first["direct_microdoor_physical_contact_in_window"]
                    and first["moka_at_close_rising"]["runtime_door_clearance_pass"]
                    and self.final_clearance["runtime_door_clearance_pass"])

    def as_dict(self) -> dict[str, Any]:
        data = {"protocol": PROTOCOL, "task_id": self.task_id,
                "init_state_index": self.init_state_index, "initial_state_sha256": self.state_sha256,
                "composition_order": self.ordered.as_dict(),
                "final_bddl_success": self.native_final,
                "all_control_steps_observed": max(self.last_step or 0, 0),
                "warmup_control_steps_observed": sum(r["step"] <= 0 for r in self.records) if self.special else 0,
                "success": self.complete, "complete": self.complete, "final_tc_gate": False,
                "dependency_hashes": self.dependency_hashes,
                "records": deepcopy(self.records)}
        if self.special:
            first = self.close_events[0] if self.close_events else None
            contact = bool(first and first["direct_microdoor_physical_contact_in_window"])
            close_clear = bool(first and first["moka_at_close_rising"]["runtime_door_clearance_pass"])
            placement_clear = bool(self.placement_events and close_clear and self.final_clearance
                                   and self.final_clearance["runtime_door_clearance_pass"])
            data.update(
                microwave_close_direct_contact_required=True,
                microwave_close_direct_contact_window_control_steps=CONTACT_WINDOW,
                microwave_close_minimum_certifying_normal_force_n=MIN_NORMAL_FORCE_N,
                microwave_close_rising_events=deepcopy(self.close_events),
                microwave_close_direct_contact_pass=contact,
                moka_placement_runtime_door_clearance_events=deepcopy(self.placement_events),
                moka_at_close_runtime_door_clearance_pass=close_clear,
                moka_final_runtime_door_clearance=deepcopy(self.final_clearance),
                moka_final_runtime_door_clearance_checks=deepcopy(self.final_gate_checks),
                moka_placement_runtime_door_clearance_pass=placement_clear,
                vcn21_success_gate_participated_before_termination=True,
                microwave_sweep_lookup=deepcopy(self.sweep_info),
                placement_clearance_is_logged_not_an_extra_gate=True)
        return data


def make_scoring_adapter(bundle: str | Path, init_state_index: int, state_sha256: str,
                         sweep_path: str | Path | None = None,
                         warmup_steps: int = 10) -> SelectedComposeScoring:
    return SelectedComposeScoring(bundle, init_state_index, state_sha256, sweep_path, warmup_steps)


def replay_scoring(bundle: str | Path, records: list[dict], init_state_index: int,
                   state_sha256: str, sweep_path: str | Path | None = None,
                   warmup_steps: int = 10) -> dict:
    """Reconstruct every reported success/order/contact decision without a sim."""
    observer = make_scoring_adapter(bundle, init_state_index, state_sha256, sweep_path, warmup_steps)
    for fact in records:
        observer.update(fact)
    return observer.as_dict()
