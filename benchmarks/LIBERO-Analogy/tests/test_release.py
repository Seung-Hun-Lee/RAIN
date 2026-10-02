import json
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from libero_analogy.tasks import load_index, validate, benchmark_root
from libero_analogy.policy import OpenPIClient, validate_actions
from libero_analogy._support.composition_batch_order import OrderedEvents, self_test
from libero_analogy._support import custom_eval


def test_frozen_manifest_and_tasks():
    assert validate()['tasks'] == 60
    rows = load_index()
    assert sum(r['init_state_count'] == 50 for r in rows) == 24
    assert sum(r['init_state_count'] == 5 for r in rows) == 36
    adapt = next(r for r in rows if r['task_id'] == 'Adapt_017')
    assert 'left compartment' in adapt['instruction']
    bddl = (benchmark_root() / adapt['bundle'] / 'task.bddl').read_text()
    assert 'bowl_drainer_1_right_region' in bddl


def test_historical_order_observer():
    self_test()


def test_forbidden_ever_is_sticky():
    state = {'on': False, 'open': False}
    env = SimpleNamespace(env=SimpleNamespace(_eval_predicate=lambda atom: state[atom[0]]))
    tracker = custom_eval.init_eval_tracker(custom_eval.compile_eval_rules(dict(
        custom_eval_needed=True, category='TD', required_goal_atoms=['on(a,b)'],
        forbidden_goal_atoms=['open(drawer)'], overshoot_policy='fail_on_forbidden_ever')))
    custom_eval.update_eval_tracker(env, tracker, 0)
    state.update(on=True, open=True)
    custom_eval.update_eval_tracker(env, tracker, 1)
    state['open'] = False
    custom_eval.update_eval_tracker(env, tracker, 2)
    assert custom_eval.custom_eval_failed(tracker)
    assert not custom_eval.custom_eval_success(tracker)


@pytest.mark.parametrize('reply', [{}, {'actions': [0] * 7}, {'actions': np.zeros((5, 8))}, {'actions': np.full((1, 7), np.nan)}])
def test_invalid_actions_rejected(reply):
    with pytest.raises(ValueError):
        validate_actions(reply)


def test_variable_action_chunk_lengths():
    for size in (1, 5, 10, 50):
        assert validate_actions({'actions': np.zeros((size, 7))}).shape == (size, 7)


def test_official_wire_protocol_without_private_metadata():
    """Real local websocket + the official openpi_client msgpack codec."""
    from libero_analogy._vendor.openpi import msgpack_numpy
    from websockets.sync.server import serve
    received = []

    def handler(socket):
        socket.send(msgpack_numpy.packb({}))
        payload = msgpack_numpy.unpackb(socket.recv())
        received.append(payload)
        socket.send(msgpack_numpy.packb({'actions': np.zeros((6, 7), np.float32)}))

    with serve(handler, '127.0.0.1', 0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        client = OpenPIClient(port=server.socket.getsockname()[1])
        payload = {'observation/image': np.zeros((224, 224, 3), np.uint8),
                   'observation/wrist_image': np.ones((224, 224, 3), np.uint8),
                   'observation/state': np.zeros(8, np.float32), 'prompt': 'test'}
        try:
            assert validate_actions(client.infer(payload)).shape == (6, 7)
        finally:
            client.close()
            server.shutdown()
        thread.join(timeout=5)
    assert received[0]['observation/wrist_image'].sum() == 224 * 224 * 3
    assert set(received[0]) == set(payload)


def test_vendored_codec_byte_parity_with_official_installation():
    original = pytest.importorskip('openpi_client.msgpack_numpy')
    from libero_analogy._vendor.openpi import msgpack_numpy as vendored
    payload = {'observation/image': np.arange(224 * 224 * 3, dtype=np.uint8).reshape(224, 224, 3),
               'observation/state': np.arange(8, dtype=np.float32), 'prompt': 'current task',
               'scalar': np.float32(.25)}
    assert vendored.packb(payload) == original.packb(payload)
    assert np.array_equal(original.unpackb(vendored.packb(payload))['observation/image'], payload['observation/image'])


def test_vendored_image_resize_parity_with_official_installation():
    original = pytest.importorskip('openpi_client.image_tools')
    from libero_analogy._vendor.openpi import image_tools as vendored
    image = np.arange(256 * 256 * 3, dtype=np.uint8).reshape(256, 256, 3)
    assert np.array_equal(vendored.resize_with_pad(image, 224, 224), original.resize_with_pad(image, 224, 224))


def test_vcn21_canonical_name_and_frozen_sweep():
    from libero_analogy._support.selected_compose_pi05_scoring import make_scoring_adapter
    root = benchmark_root() / 'tasks/Compose/Compose_005'
    replay = json.loads((root / 'FIXTURE_REPLAY.json').read_text())['rows'][0]
    scorer = make_scoring_adapter(root, 0, replay['state_sha256'])
    assert scorer.task_id == 'VCN21_001'
    assert scorer.special and not scorer.complete


def vcn_adapter():
    from libero_analogy._support.selected_compose_pi05_scoring import make_scoring_adapter
    root = benchmark_root() / 'tasks/Compose/Compose_005'
    state = json.loads((root / 'FIXTURE_REPLAY.json').read_text())['rows'][0]['state_sha256']
    scorer = make_scoring_adapter(root, 0, state)
    for step in range(-9, 1):
        scorer.update(vcn_fact(step))
    return scorer


def vcn_fact(step, values=(False, False), force=None, aabb=None):
    pairs = [] if force is None else [dict(contact_index=0, gripper_geom='finger', door_geom='door',
                distance_m=-.0001, efc_address=0, solver_constraint_active=True, normal_force_n=force,
                minimum_certifying_normal_force_n=1e-6, certifying_physical_contact=force > 1e-6)]
    return dict(step=step, native_event_values=list(values), native_bddl_success=all(values),
                direct_microdoor_contact_pairs=pairs, moka_collision_aabb=aabb or [[2., 2., 2.], [2.1, 2.1, 2.1]])


@pytest.mark.parametrize('force,expected', [(0, False), (1e-6, False), (1.0001e-6, True)])
def test_microwave_positive_force_boundary(force, expected):
    scorer = vcn_adapter()
    scorer.update(vcn_fact(1, (True, False)))
    scorer.update(vcn_fact(2, (True, True), force=force))
    assert scorer.complete == expected


def test_microwave_sweep_clearance_is_required():
    scorer = vcn_adapter()
    scorer.update(vcn_fact(1, (True, False)))
    scorer.update(vcn_fact(2, (True, True), force=1., aabb=scorer.sweep_info['swept_moving_subtree_aabb']))
    assert not scorer.complete


def test_microwave_first_uncontacted_close_cannot_be_rescued():
    scorer = vcn_adapter()
    scorer.update(vcn_fact(1, (True, False)))
    scorer.update(vcn_fact(2, (True, True)))
    scorer.update(vcn_fact(3, (True, False)))
    scorer.update(vcn_fact(4, (True, True), force=1.))
    assert not scorer.complete
