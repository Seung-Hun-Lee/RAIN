"""Shared simulator construction and process-local fixture registration."""
DUMMY_ACTION = [0.0] * 6 + [-1.0]


def install_support(legacy_id):
    if legacy_id in {"NAFR3_001", "NAFR3_002", "DSET_001"}:
        from ._support.novel_feedback_fixture_geometry import install_feedback_fixture_geometry
        install_feedback_fixture_geometry()
    if legacy_id in {"BDRSWAP_001", "BDRMIN_002", "CTR_103"}:
        from ._support.bowl_drainer_sections_geometry import install_floor_drainer_support
        # RAIN mask bindings are unrelated to environment construction.
        install_floor_drainer_support()
    if legacy_id == "WTRAYR_004":
        from ._support.wooden_tray_object_choices_support import install_wooden_tray_object_choices_support
        install_wooden_tray_object_choices_support()
    if legacy_id == "GRACK_002":
        from ._support.wine_rack_object_support import install_rack_adapt_geometry
        install_rack_adapt_geometry()


def make_env(bddl_path, seed, max_steps):
    from libero.libero.envs import OffScreenRenderEnv
    env = OffScreenRenderEnv(bddl_file_name=str(bddl_path), camera_heights=256, camera_widths=256,
                            horizon=max(1000, int(max_steps) + 256), ignore_done=True)
    env.seed(seed)
    return env
