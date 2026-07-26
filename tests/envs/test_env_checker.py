from __future__ import annotations

from gymnasium.utils.env_checker import check_env

from homotopy_path_learning.envs import BezierPhamEnv

from .conftest import FakeBackend


def test_bezier_pham_env_passes_gymnasium_checker(smoke_spec) -> None:
    env = BezierPhamEnv(system_spec=smoke_spec, backend=FakeBackend())

    check_env(env, skip_render_check=True)
