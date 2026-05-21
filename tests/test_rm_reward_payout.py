"""RM 보상이 baseline.update_reward_machine_reward → rm_reward_total → 스텝 델타로 지급되는지 검증."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from pokemonred_puffer.data.tm_hm import TmHmMoves
from pokemonred_puffer.rewards import baseline as baseline_mod
from pokemonred_puffer.rewards.baseline import BaselineRewardEnv
from pokemonred_puffer.rewards.reward_machine import (
    CUTTABLE_TILES,
    SURF_TILE_IN_FRONT,
    RewardMachine,
    RewardMachineState,
)


def _default_reward_config() -> dict:
    return {
        "rm_enabled": True,
        "rm_clawback_intermediate_on_abort": True,
        "rm_clawback_fraction": 1.0,
        "rm_intermediate": 0.25,
        "rm_cut_success": 1.5,
        "rm_surf_success": 1.0,
        "rm_flash_success": 0.7,
        "step_penalty": 0.0001,
        "unnecessary_hm_usage_penalty": 0.0,
        "hm_supervision_proactive": False,
    }


class FakeEvents:
    def get_event(self, _name: str) -> bool:
        return False


def _make_harness(reward_config: dict | None = None) -> SimpleNamespace:
    """PyBoy 없이 BaselineRewardEnv RM 지급 메서드만 호출 가능한 최소 env."""
    cfg = reward_config or _default_reward_config()
    env = SimpleNamespace(
        reward_config=cfg,
        reward_machine=RewardMachine(),
        events=FakeEvents(),
        auto_flash=False,
        valid_cut_coords={},
        invalid_cut_coords={},
        valid_surf_coords={},
        invalid_surf_coords={},
        valid_flash_coords={},
        invalid_flash_coords={},
        use_surf=0,
        seen_start_menu=0,
        seen_pokemon_menu=0,
        step_count=0,
        rm_reward_total=0.0,
        rm_last_step_delta=0.0,
        rm_transition_count=0,
        rm_success_count=0,
        rm_cut_success_count=0,
        rm_surf_success_count=0,
        rm_flash_success_count=0,
        rm_intermediate_paid_count=0,
        rm_reward_from_success=0.0,
        rm_reward_from_intermediate=0.0,
        rm_reward_intermediate_net=0.0,
        rm_clawback_total=0.0,
        rm_clawback_count=0,
        _rm_attempt_intermediate_pending=0.0,
        last_rm_transition_key="",
        hm_supervision_target=baseline_mod.HMTarget.NONE,
        hm_supervision_latch_target=baseline_mod.HMTarget.NONE,
        hm_supervision_latch_steps_remaining=0,
        _prev_invalid_cut_count=0,
        _prev_invalid_surf_count=0,
        _prev_invalid_flash_count=0,
        _prev_valid_cut_count=0,
        _prev_valid_surf_count=0,
        _surf_hook_success_count=0,
        _prev_surf_hook_success_count=0,
        _prev_valid_flash_count=0,
        _rm_valid_cut_delta=0,
        _rm_valid_surf_delta=0,
        _rm_valid_flash_delta=0,
        unnecessary_hm_penalty_total=0.0,
        invalid_action_total=0.0,
        step_penalty_total=0.0,
        total_reward=0.0,
        progress_reward={},
        missing_cut_reported=False,
    )
    env.check_if_party_has_hm = MagicMock(return_value=True)
    env.get_items_in_bag = MagicMock(return_value=[])
    env.get_tile_in_front_of_player = MagicMock(return_value=0)
    env.get_map_pal_offset = MagicMock(return_value=0)
    env.get_rm_flash_cycle_start = MagicMock(
        return_value=int(env.reward_machine.flash_cycle_start_count)
    )
    env.get_adjacent_water_count = MagicMock(return_value=0)
    env.is_start_menu_illegal_navigation_active = MagicMock(return_value=False)
    env.teach_hm = MagicMock()
    for _name in (
        "ensure_cut_for_reward_machine",
        "update_reward_machine_reward",
        "get_game_state_reward",
        "_rm_reward_for_transition_key",
        "_apply_rm_clawback",
        "_record_rm_transition_stats",
    ):
        _meth = getattr(BaselineRewardEnv, _name)

        def _bind(m=_meth, e=env):
            return m(e)

        if _name == "_rm_reward_for_transition_key":

            def _bind_key(key, m=_meth, e=env):
                return m(e, key)

            setattr(env, _name, _bind_key)
        elif _name == "_apply_rm_clawback":

            def _bind_claw(key, m=_meth, e=env):
                return m(e, key)

            setattr(env, _name, _bind_claw)
        elif _name == "_record_rm_transition_stats":

            def _bind_stats(key, amt, m=_meth, e=env):
                return m(e, key, amt)

            setattr(env, _name, _bind_stats)
        else:
            setattr(env, _name, _bind)
    return env


def _run_before_progress(env: SimpleNamespace) -> None:
    BaselineRewardEnv._before_progress_reward(env)


def _run_update_reward(env: SimpleNamespace) -> float:
    env.progress_reward = BaselineRewardEnv.get_game_state_reward(env)
    new_total = sum(env.progress_reward.values())
    step = new_total - env.total_reward
    env.total_reward = new_total
    return step


def _set_cut_success_snapshot(env: SimpleNamespace, *, valid_count: int = 1) -> None:
    """한 스텝에 컷 완료된 최종 스냅샷 (나무 제거, valid_cut 증가)."""
    env.valid_cut_coords = {("x", 0, 0): 1} if valid_count else {}
    env._prev_valid_cut_count = max(0, valid_count - 1)
    env.get_tile_in_front_of_player = MagicMock(return_value=0x00)


def _set_surf_success_snapshot(env: SimpleNamespace, *, valid_count: int = 1) -> None:
    env.valid_surf_coords = {("x", 0, 0): 1} if valid_count else {}
    env._prev_valid_surf_count = max(0, valid_count - 1)
    env._surf_hook_success_count = valid_count
    env._prev_surf_hook_success_count = max(0, valid_count - 1)
    env.use_surf = 1
    env.get_tile_in_front_of_player = MagicMock(return_value=0x00)


class TestRmRewardPayout:
    def test_one_step_cut_idle_to_success_pays_config_amount(self):
        env = _make_harness()
        _set_cut_success_snapshot(env, valid_count=1)
        _run_before_progress(env)
        # 한 번의 update에서 IDLE→SUCCESS→DONE 체인 가능; 마지막 키는 rm_cut_done.
        assert env.rm_cut_success_count == 1
        assert env.rm_reward_total == pytest.approx(1.5)
        assert env.rm_last_step_delta == pytest.approx(1.5)

    def test_step_reward_delta_includes_rm_not_only_penalty(self):
        env = _make_harness()
        _set_cut_success_snapshot(env)
        _run_before_progress(env)
        step_r = _run_update_reward(env)
        assert env.progress_reward["rm_reward"] == pytest.approx(1.5)
        assert step_r == pytest.approx(1.5 - 0.0001)

    def test_multi_step_cut_chain_pays_intermediate_plus_success(self):
        env = _make_harness()
        cut_tile = next(iter(CUTTABLE_TILES))
        env.get_tile_in_front_of_player = MagicMock(return_value=cut_tile)

        _run_before_progress(env)
        # rm_cut_detected는 _RM_NO_REWARD_KEYS → 0, 상태만 DETECTED.
        assert env.rm_reward_total == pytest.approx(0.0)
        assert env.reward_machine.state == RewardMachineState.CUT_DETECTED

        env.get_current_menu_item = MagicMock(return_value=1)
        env.seen_start_menu = 1
        _run_before_progress(env)
        assert env.rm_reward_total == pytest.approx(0.25)
        assert env.reward_machine.state == RewardMachineState.CUT_MENU_OPEN

        env.seen_pokemon_menu = 1
        env.seen_field_move_menu = 0
        _run_before_progress(env)
        assert env.rm_reward_total == pytest.approx(0.50)
        assert env.reward_machine.state == RewardMachineState.CUT_PARTY_MENU

        env.seen_field_move_menu = 1
        _run_before_progress(env)
        assert env.rm_reward_total == pytest.approx(0.75)

        _set_cut_success_snapshot(env, valid_count=1)
        env._rm_valid_cut_delta = 1
        BaselineRewardEnv.update_reward_machine_reward(env)
        assert env.rm_reward_total == pytest.approx(0.75 + 1.5)
        assert env.rm_cut_success_count == 1

    def test_cut_abort_claws_back_intermediate(self):
        env = _make_harness()
        cut_tile = next(iter(CUTTABLE_TILES))
        env.get_tile_in_front_of_player = MagicMock(return_value=cut_tile)
        _run_before_progress(env)
        assert env.reward_machine.state == RewardMachineState.CUT_DETECTED

        env.get_current_menu_item = MagicMock(return_value=1)
        env.seen_start_menu = 1
        _run_before_progress(env)
        assert env.rm_reward_total == pytest.approx(0.25)
        assert env._rm_attempt_intermediate_pending == pytest.approx(0.25)
        assert env.reward_machine.state == RewardMachineState.CUT_MENU_OPEN

        # DETECTED로 돌아가진 않지만, 타일 이탈 시 DETECTED에서만 abort 가능 → 리셋 후 시뮬.
        env.reward_machine.state = RewardMachineState.CUT_DETECTED
        env.get_tile_in_front_of_player = MagicMock(return_value=0x00)
        env._rm_valid_cut_delta = 0
        _run_before_progress(env)
        assert env.last_rm_transition_key == "rm_cut_aborted"
        assert env.rm_reward_total == pytest.approx(0.0)
        assert env.rm_clawback_total == pytest.approx(0.25)

    def test_cut_abort_partial_clawback_fraction(self):
        cfg = _default_reward_config()
        cfg["rm_clawback_fraction"] = 0.95
        env = _make_harness(cfg)
        cut_tile = next(iter(CUTTABLE_TILES))
        env.get_tile_in_front_of_player = MagicMock(return_value=cut_tile)
        _run_before_progress(env)

        env.get_current_menu_item = MagicMock(return_value=1)
        env.seen_start_menu = 1
        _run_before_progress(env)
        assert env._rm_attempt_intermediate_pending == pytest.approx(0.25)

        env.reward_machine.state = RewardMachineState.CUT_DETECTED
        env.get_tile_in_front_of_player = MagicMock(return_value=0x00)
        env._rm_valid_cut_delta = 0
        _run_before_progress(env)
        assert env.last_rm_transition_key == "rm_cut_aborted"
        # 0.25 지급 후 95% 회수 → 순 0.0125
        assert env.rm_reward_total == pytest.approx(0.25 * 0.05)
        assert env.rm_clawback_total == pytest.approx(0.25 * 0.95)
        assert env.rm_reward_from_intermediate == pytest.approx(0.25)
        assert env.rm_reward_intermediate_net == pytest.approx(0.25 * 0.05)

    def test_surf_one_step_idle_success(self):
        env = _make_harness()
        _set_surf_success_snapshot(env)
        _run_before_progress(env)
        assert env.rm_surf_success_count == 1
        assert env.rm_reward_total == pytest.approx(1.0)

    def test_rm_disabled_pays_nothing(self):
        cfg = _default_reward_config()
        cfg["rm_enabled"] = False
        env = _make_harness(cfg)
        _set_cut_success_snapshot(env)
        _run_before_progress(env)
        assert env.rm_reward_total == 0.0
        assert env.rm_last_step_delta == 0.0

    def test_detected_success_without_valid_delta_aborts_not_pays(self):
        """valid_cut 증분 없이 타일만 비면 SUCCESS 대신 ABORT (옛 버그 회귀 방지)."""
        env = _make_harness()
        cut_tile = next(iter(CUTTABLE_TILES))
        env.get_tile_in_front_of_player = MagicMock(return_value=cut_tile)
        _run_before_progress(env)
        assert env.reward_machine.state == RewardMachineState.CUT_DETECTED

        env.get_tile_in_front_of_player = MagicMock(return_value=0x00)
        env._rm_valid_cut_delta = 0
        env._prev_valid_cut_count = len(env.valid_cut_coords)
        _run_before_progress(env)
        assert env.last_rm_transition_key == "rm_cut_aborted"
        assert env.rm_cut_success_count == 0
        assert env.rm_reward_total == pytest.approx(0.0)

    def test_detected_to_success_when_valid_delta_same_step(self):
        env = _make_harness()
        cut_tile = next(iter(CUTTABLE_TILES))
        env.get_tile_in_front_of_player = MagicMock(return_value=cut_tile)
        _run_before_progress(env)
        assert env.reward_machine.state == RewardMachineState.CUT_DETECTED

        env.valid_cut_coords = {("a", 1, 1): 1}
        env._rm_valid_cut_delta = 1
        env._prev_valid_cut_count = 0
        env.get_tile_in_front_of_player = MagicMock(return_value=0x00)
        _run_before_progress(env)
        assert env.rm_cut_success_count == 1
        assert env.rm_reward_total == pytest.approx(1.5)
        assert env.rm_last_step_delta == pytest.approx(1.5)

    def test_no_double_cut_success_in_same_chain(self):
        """MON_SELECTED→SUCCESS→DONE→IDLE 후 같은 스냅샷에서 rm_cut_success 이중 지급 없음."""
        env = _make_harness()
        cut_tile = next(iter(CUTTABLE_TILES))
        env.get_tile_in_front_of_player = MagicMock(return_value=cut_tile)
        _run_before_progress(env)
        env.get_current_menu_item = MagicMock(return_value=1)
        env.seen_start_menu = 1
        _run_before_progress(env)
        env.seen_pokemon_menu = 1
        env.seen_field_move_menu = 0
        _run_before_progress(env)
        env.seen_field_move_menu = 1
        _run_before_progress(env)
        _set_cut_success_snapshot(env, valid_count=1)
        env._rm_valid_cut_delta = 1
        BaselineRewardEnv.update_reward_machine_reward(env)
        assert env.rm_cut_success_count == 1
        assert env.rm_reward_total == pytest.approx(0.75 + 1.5)
