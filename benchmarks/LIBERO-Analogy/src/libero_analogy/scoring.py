"""Frozen baseline environment preparation and scoring; portability imports only."""
from pathlib import Path
import numpy as np
from ._support import custom_eval as ce
from ._support.composition_batch_order import OrderedEvents, parse_atom
from ._support.cream_cheese_bowl_layout import apply_fixtures, position_error, state_hash
from .observations import policy_observation
from .environment import make_env, DUMMY_ACTION
class Scoring:
    def __init__(self, task, env, index):
        self.task, self.env = task, env
        self.rules = task['rules']
        self.tracker = ce.init_eval_tracker(ce.compile_eval_rules(self.rules))
        self.ordered = None
        self.special = None
        self.probe = None
        self.physical = None
        self.drainer = None
        self.records = []
        self.native = False
        tid = task.get('legacy_task_id', task['task_id'])
        if tid == 'VCN21_001':
            from ._support.selected_compose_pi05_scoring import make_scoring_adapter
            self.special = make_scoring_adapter(task['bundle_path'], index, state_hash(task['initial_states'][index]))
        elif self.rules.get('drainer_completion_mode'):
            from ._support.bowl_drainer_ordered_placement_observer import DrainerContactProbe, OrderedPlacementObserver
            atoms = self.rules['ordered_event_atoms']
            parsed = [parse_atom(a) for a in atoms]
            self.drainer_regions = [a[2] for a in parsed]
            self.probe = DrainerContactProbe(env, [a[1] for a in parsed])
            self.drainer = OrderedPlacementObserver(atoms, support_hold_control_steps=5)
        elif self.rules.get('ordered_event_atoms'):
            self.ordered = OrderedEvents(self.rules['ordered_event_atoms'])
        if tid == 'WTRAYR_004':
            from ._support.wooden_tray_strict_placement_observer import WoodenTrayContactProbe, StrictTrayPlacementObserver
            self.probe = WoodenTrayContactProbe(env, 'akita_black_bowl_1')
            self.physical = StrictTrayPlacementObserver(5)
        elif tid == 'GRACK_002':
            from ._support.wine_rack_object_support import RackPlacementProbe, RackPlacementObserver
            self.probe = RackPlacementProbe(env, 'ketchup_1')
            self.physical = RackPlacementObserver('ketchup_1', 5)
        elif self.rules.get('basket_completion_mode'):
            from ._support.basket_released_support_observer import validate_contract, BasketContactProbe, BasketSupportObserver
            destinations = validate_contract(task['meta'], self.rules)
            self.probe = BasketContactProbe(env, destinations)
            self.physical = BasketSupportObserver(destinations)

    def observe(self, step):
        if self.special:
            fact = self.special.snapshot(self.env, step)
            self.special.update(fact)
            if step < 0: return
        # Resolve every predicate explicitly first: unknown predicates must raise, not become false.
        for raw in self.rules.get('required_goal_atoms', []) + self.rules.get('forbidden_goal_atoms', []):
            self.env.env._eval_predicate(parse_atom(raw))
        ce.update_eval_tracker(self.env, self.tracker, step_idx=step)
        self.native = bool(self.env.check_success())
        fact = {'step': step, 'required': list(self.tracker['required_atom_final']),
                'forbidden': [bool(self.env.env._eval_predicate(parse_atom(a))) for a in self.rules.get('forbidden_goal_atoms', [])],
                'native_goal': self.native}
        if self.ordered:
            self.ordered.update(step, [bool(self.env.env._eval_predicate(parse_atom(a))) for a in self.ordered.atoms])
        if self.drainer:
            snap = self.probe.snapshot()
            objects = snap['objects']
            selected = [o['native_predicates'][region] for o, region in zip(objects, self.drainer_regions)]
            opposite = [o['native_predicates'][self.drainer_regions[1-i]] for i,o in enumerate(objects)]
            self.drainer.update(step, selected, opposite, [o['native_base_contact'] for o in objects],
                                [o['any_gripper_contact'] for o in objects])
            fact['physical'] = snap
        elif self.physical:
            snap = self.probe.snapshot()
            self.physical.update(step, snap)
            fact['physical'] = snap
        self.records.append(fact)

    @property
    def violation(self):
        return bool(ce.custom_eval_failed(self.tracker) or
                    (self.ordered and self.ordered.violation) or
                    (self.special and self.special.violation) or
                    (self.drainer and self.drainer.violation('released_supported')))

    def success(self, final=False):
        if self.violation: return False
        if self.special: return bool(self.special.complete)
        if self.drainer: return bool(self.drainer.complete('released_supported') and self.native)
        if self.physical and not self.physical.complete: return False
        if self.task.get('legacy_task_id', self.task['task_id']) in {'WTRAYR_004', 'GRACK_002'}: return bool(self.physical.complete)
        if self.ordered: return bool(self.ordered.complete and self.native)
        if self.tracker['custom']:
            if self.rules.get('continue_after_success'):
                return bool(final and ce.custom_eval_success(self.tracker))
            now, forbidden = ce.custom_eval_now(self.env, self.tracker)
            return bool(now and not forbidden)
        return self.native

    def evidence(self):
        result = {'tracker':self.tracker, 'control_observations':self.records, 'native_final':self.native}
        if self.ordered: result['ordered_events'] = self.ordered.as_dict()
        if self.special: result['special_composition'] = self.special.as_dict()
        if self.drainer: result['drainer'] = self.drainer.as_dict()
        if self.physical: result['physical'] = self.physical.as_dict()
        return result

def prepared_env(task, episode):
    index = episode % len(task['initial_states'])
    seed = 7 + (task['order']-1)*100 + episode
    np.random.seed(seed)
    env = make_env(task['bddl_path'], seed, task['max_steps'])
    try:
        env.reset()
        replay = None
        if task['replays']:
            digest = state_hash(task['initial_states'][index])
            matches = [r for r in task['replays'] if r['state_sha256']==digest and int(r['init_state_index'])==index]
            assert len(matches)==1
            replay = matches[0]
            apply_fixtures(env, replay['fixture_model_poses'])
            apply_fixtures(env, replay['fixture_model_poses'])
        obs = env.set_init_state(task['initial_states'][index])
        scorer = Scoring(task, env, index) if task.get('legacy_task_id', task['task_id'])=='VCN21_001' else None
        for step in range(-9, 1):
            obs, _, _, _ = env.step(DUMMY_ACTION)
            if scorer is not None: scorer.observe(step)
        if scorer is None:
            scorer = Scoring(task, env, index)
            scorer.observe(0)
        error = position_error(env, replay['settled_body_positions']) if replay else None
        if error is not None: assert error <= 1e-8, f'Fixture replay position error {error}'
        assert not scorer.success(final=True), 'Goal already true initially'
        assert not scorer.violation, 'Forbidden/ordered event true initially'
        payload, frame = policy_observation(obs, task['instruction'])
        assert payload['observation/image'].shape == payload['observation/wrist_image'].shape == (224,224,3)
        evidence = {'seed':seed, 'init_state_index':index, 'initial_state_sha256':state_hash(task['initial_states'][index]),
                    'fixture_replay_calls':2 if replay else 0, 'initial_position_max_abs_error_m':error,
                    'observation_keys':sorted(payload), 'image_shape':[224,224,3], 'warmup_steps':10}
        return env, obs, scorer, evidence
    except BaseException:
        env.close()
        raise
