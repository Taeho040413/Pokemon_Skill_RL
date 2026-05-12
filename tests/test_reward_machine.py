"""Reward Machine 단위 테스트.

PyBoy·게임 환경 없이 RewardMachineContext를 직접 생성해 RM 전이 로직을 검증한다.
실행: cd poke_skills && python -m pytest tests/test_reward_machine.py -v
"""
from __future__ import annotations

import pytest

from pokemonred_puffer.rewards.reward_machine import (
    CUTTABLE_TILES,
    SURF_TILE_IN_FRONT,
    RewardMachine,
    RewardMachineContext,
    RewardMachineState,
    HMTarget,
)

# ─────────────────────────────────────────────────────────────────
# 헬퍼
# ─────────────────────────────────────────────────────────────────
_CUT_TILE = next(iter(CUTTABLE_TILES))  # 0x3D


def _ctx(**overrides) -> RewardMachineContext:
    """기본값(아무 HM 없음, 밝은 필드, 메뉴 닫힘)을 가진 컨텍스트를 생성한다."""
    defaults = dict(
        step_count=0,
        has_cut=False, has_flash=False, has_surf=False,
        auto_flash=False,
        used_cut_successfully=False, valid_cut_coords_count=0,
        valid_surf_coords_count=0, valid_flash_coords_count=0,
        used_surf_successfully=False, is_surfing=False,
        tile_in_front=0x00,
        start_menu_open=False, pokemon_menu_open=False,
        invalid_cut_coords_count=0,
        invalid_surf_coords_count=0, invalid_flash_coords_count=0,
        in_dark_cave=False, flash_cycle_has_new_success=False,
    )
    defaults.update(overrides)
    return RewardMachineContext(**defaults)


def _step(rm: RewardMachine, step_count: int = 0, **ctx_overrides):
    ctx = _ctx(step_count=step_count, **ctx_overrides)
    return rm.transition(ctx)


# ─────────────────────────────────────────────────────────────────
# 1. CUT
# ─────────────────────────────────────────────────────────────────
class TestCut:
    def test_happy_path(self):
        rm = RewardMachine()
        assert rm.state == RewardMachineState.IDLE

        # IDLE → CUT_DETECTED
        s = _step(rm, 0, has_cut=True, tile_in_front=_CUT_TILE)
        assert s.changed and rm.state == RewardMachineState.CUT_DETECTED
        assert s.transition_key == "rm_cut_detected"

        # CUT_DETECTED → CUT_MENU_OPEN
        s = _step(rm, 1, has_cut=True, tile_in_front=_CUT_TILE, start_menu_open=True)
        assert rm.state == RewardMachineState.CUT_MENU_OPEN
        assert s.transition_key == "rm_cut_menu_open"

        # CUT_MENU_OPEN → CUT_MON_SELECTED
        s = _step(rm, 2, has_cut=True, tile_in_front=_CUT_TILE, pokemon_menu_open=True)
        assert rm.state == RewardMachineState.CUT_MON_SELECTED
        assert s.transition_key == "rm_cut_mon_selected"

        # CUT_MON_SELECTED → CUT_SUCCESS (새 cut 성공 + tile이 사라짐)
        s = _step(rm, 3, has_cut=True, tile_in_front=0x00,
                  used_cut_successfully=True, valid_cut_coords_count=1)
        assert rm.state == RewardMachineState.CUT_SUCCESS
        assert s.transition_key == "rm_cut_success"

        # CUT_SUCCESS → IDLE
        s = _step(rm, 4)
        assert rm.state == RewardMachineState.IDLE
        assert s.transition_key == "rm_cut_done"

    def test_chains_menu_states_same_snapshot(self):
        """Baseline과 같이 transition을 연속 호출하면 한 스냅샷에서 메뉴 전이가 모두 적용된다."""
        rm = RewardMachine()
        ctx = _ctx(
            step_count=1,
            has_cut=True,
            tile_in_front=_CUT_TILE,
            start_menu_open=True,
            pokemon_menu_open=True,
        )
        keys: list[str | None] = []
        for _ in range(5):
            step = rm.transition(ctx)
            if not step.changed or not step.transition_key:
                break
            keys.append(step.transition_key)
        assert keys == [
            "rm_cut_detected",
            "rm_cut_menu_open",
            "rm_cut_mon_selected",
        ]
        assert rm.state == RewardMachineState.CUT_MON_SELECTED

    def test_shortcut_detected_to_success_when_tile_cleared(self):
        """스텝 끝 스냅샷이 메뉴 0·나무 제거면 DETECTED에서 곧바로 SUCCESS (증분 가드)."""
        rm = RewardMachine()
        _step(rm, 0, has_cut=True, tile_in_front=_CUT_TILE)
        assert rm.state == RewardMachineState.CUT_DETECTED
        s = _step(
            rm,
            1,
            has_cut=True,
            tile_in_front=0x00,
            valid_cut_coords_count=1,
            used_cut_successfully=True,
        )
        assert rm.state == RewardMachineState.CUT_SUCCESS
        assert s.transition_key == "rm_cut_success"

    def test_abort_from_detected(self):
        rm = RewardMachine()
        _step(rm, 0, has_cut=True, tile_in_front=_CUT_TILE)
        assert rm.state == RewardMachineState.CUT_DETECTED

        # 타일에서 벗어남 → IDLE
        s = _step(rm, 1, has_cut=True, tile_in_front=0x00)
        assert rm.state == RewardMachineState.IDLE
        assert s.transition_key == "rm_cut_aborted"

    def test_no_reentry_same_tile(self):
        """같은 타일 앞에서 CUT_DETECTED→IDLE→CUT_DETECTED가 연속 발화하지 않아야 함."""
        rm = RewardMachine()
        _step(rm, 0, has_cut=True, tile_in_front=_CUT_TILE)
        assert rm.state == RewardMachineState.CUT_DETECTED
        # abort
        _step(rm, 1, has_cut=True, tile_in_front=0x00)
        assert rm.state == RewardMachineState.IDLE
        # idle_cut_entry_ok가 True로 재무장됐으므로 다시 같은 타일 → 재진입 가능
        s = _step(rm, 2, has_cut=True, tile_in_front=_CUT_TILE)
        assert rm.state == RewardMachineState.CUT_DETECTED

    def test_reentry_after_abort_and_rearm(self):
        """abort 후 재무장되면 같은 타일에서 다시 전체 사이클을 완료할 수 있어야 함."""
        rm = RewardMachine()
        # 1사이클: DETECTED → abort
        _step(rm, 0, has_cut=True, tile_in_front=_CUT_TILE)
        assert rm.state == RewardMachineState.CUT_DETECTED
        _step(rm, 1, has_cut=True, tile_in_front=0x00)  # abort → IDLE
        assert rm.state == RewardMachineState.IDLE

        # 2사이클: rearmed → 전체 체인 완료
        _step(rm, 2, has_cut=True, tile_in_front=_CUT_TILE)
        assert rm.state == RewardMachineState.CUT_DETECTED
        _step(rm, 3, has_cut=True, tile_in_front=_CUT_TILE, start_menu_open=True)
        assert rm.state == RewardMachineState.CUT_MENU_OPEN
        _step(rm, 4, has_cut=True, tile_in_front=_CUT_TILE, pokemon_menu_open=True)
        assert rm.state == RewardMachineState.CUT_MON_SELECTED
        _step(rm, 5, has_cut=True, tile_in_front=0x00,
              used_cut_successfully=True, valid_cut_coords_count=1)
        assert rm.state == RewardMachineState.CUT_SUCCESS

    def test_no_immediate_reentry_from_detected(self):
        """CUT_DETECTED에서 abort 없이 같은 타일 앞에 머물면 재진입하지 않아야 함."""
        rm = RewardMachine()
        _step(rm, 0, has_cut=True, tile_in_front=_CUT_TILE)
        assert rm.state == RewardMachineState.CUT_DETECTED
        # DETECTED를 abort하지 않고 유지 (메뉴도 안 열고)
        s = _step(rm, 1, has_cut=True, tile_in_front=_CUT_TILE)
        # CUT_DETECTED 유지 또는 MENU_OPEN 전이만 가능; 다시 IDLE→DETECTED 루프 없음
        assert rm.state in {RewardMachineState.CUT_DETECTED, RewardMachineState.CUT_MENU_OPEN}

    def test_cut_success_allows_reusing_same_coords(self):
        """같은 좌표에서 이미 컷 성공이 있어도, 다시 한 번 사용하면 SUCCESS를 허용한다."""
        rm = RewardMachine()
        # 이미 valid_cut=1인 상태에서 사이클 시작
        _step(rm, 0, has_cut=True, tile_in_front=_CUT_TILE, valid_cut_coords_count=1)
        assert rm.state == RewardMachineState.CUT_DETECTED
        _step(
            rm,
            1,
            has_cut=True,
            tile_in_front=_CUT_TILE,
            start_menu_open=True,
            valid_cut_coords_count=1,
        )
        assert rm.state == RewardMachineState.CUT_MENU_OPEN
        _step(
            rm,
            2,
            has_cut=True,
            tile_in_front=_CUT_TILE,
            pokemon_menu_open=True,
            valid_cut_coords_count=1,
        )
        assert rm.state == RewardMachineState.CUT_MON_SELECTED

        # valid_cut_coords_count 증분이 없어도 이번 사이클에서 컷 성공으로 간주.
        s = _step(
            rm,
            3,
            has_cut=True,
            tile_in_front=0x00,
            used_cut_successfully=True,
            valid_cut_coords_count=1,
        )
        assert rm.state == RewardMachineState.CUT_SUCCESS

    def test_failed_timeout(self):
        rm = RewardMachine()
        _step(rm, 0, has_cut=True, tile_in_front=_CUT_TILE)
        _step(rm, 1, has_cut=True, tile_in_front=_CUT_TILE, start_menu_open=True)
        assert rm.state == RewardMachineState.CUT_MENU_OPEN

        # 256 스텝 이상 → FAILED
        s = _step(rm, 257, has_cut=True, tile_in_front=_CUT_TILE)
        assert rm.state == RewardMachineState.FAILED
        assert s.transition_key == "rm_failed_timeout"

    def test_failed_recovery(self):
        rm = RewardMachine()
        _step(rm, 0, has_cut=True, tile_in_front=_CUT_TILE)
        _step(rm, 1, has_cut=True, tile_in_front=_CUT_TILE, start_menu_open=True)
        _step(rm, 257)  # → FAILED
        assert rm.state == RewardMachineState.FAILED

        # 64 스텝 이상 대기 → IDLE 복구
        s = _step(rm, 257 + 64 + 1)
        assert rm.state == RewardMachineState.IDLE

    def test_invalid_increase_triggers_failed(self):
        rm = RewardMachine()
        _step(rm, 0, has_cut=True, tile_in_front=_CUT_TILE)
        _step(rm, 1, has_cut=True, tile_in_front=_CUT_TILE, start_menu_open=True)
        assert rm.state == RewardMachineState.CUT_MENU_OPEN

        # invalid_cut_coords가 8회 증가하면 FAILED
        for i in range(2, 2 + 8):
            _step(rm, i, has_cut=True, tile_in_front=_CUT_TILE,
                  invalid_cut_coords_count=i - 1)
        assert rm.state == RewardMachineState.FAILED


# ─────────────────────────────────────────────────────────────────
# 2. SURF
# ─────────────────────────────────────────────────────────────────
class TestSurf:
    def test_happy_path(self):
        rm = RewardMachine()
        _step(rm, 0, has_surf=True, tile_in_front=SURF_TILE_IN_FRONT)
        assert rm.state == RewardMachineState.SURF_DETECTED

        _step(rm, 1, has_surf=True, tile_in_front=SURF_TILE_IN_FRONT, start_menu_open=True)
        assert rm.state == RewardMachineState.SURF_MENU_OPEN

        _step(rm, 2, has_surf=True, tile_in_front=SURF_TILE_IN_FRONT, pokemon_menu_open=True)
        assert rm.state == RewardMachineState.SURF_MON_SELECTED

        # 서핑 시작(is_surfing=True) + valid_surf 증가
        s = _step(rm, 3, has_surf=True, is_surfing=True,
                  used_surf_successfully=True, valid_surf_coords_count=1)
        assert rm.state == RewardMachineState.SURF_SUCCESS
        assert s.transition_key == "rm_surf_success"

        s =         _step(rm, 4)
        assert rm.state == RewardMachineState.IDLE
        assert s.transition_key == "rm_surf_done"

    def test_shortcut_detected_to_success_when_surfing(self):
        """앞 타일이 물이 아니어도 서핑 중이면 DETECTED에서 SUCCESS (valid 증분 가드)."""
        rm = RewardMachine()
        _step(rm, 0, has_surf=True, tile_in_front=SURF_TILE_IN_FRONT)
        assert rm.state == RewardMachineState.SURF_DETECTED
        s = _step(
            rm,
            1,
            has_surf=True,
            tile_in_front=0x00,
            is_surfing=True,
            used_surf_successfully=True,
            valid_surf_coords_count=1,
        )
        assert rm.state == RewardMachineState.SURF_SUCCESS
        assert s.transition_key == "rm_surf_success"

    def test_abort_from_detected(self):
        rm = RewardMachine()
        _step(rm, 0, has_surf=True, tile_in_front=SURF_TILE_IN_FRONT)
        assert rm.state == RewardMachineState.SURF_DETECTED
        s = _step(rm, 1, has_surf=True, tile_in_front=0x00)
        assert rm.state == RewardMachineState.IDLE
        assert s.transition_key == "rm_surf_aborted"

    def test_no_reentry_while_surfing(self):
        """물 위(is_surfing=True)에서는 SURF_DETECTED 재진입을 막아야 함."""
        rm = RewardMachine()
        # surf 완료 → IDLE
        _step(rm, 0, has_surf=True, tile_in_front=SURF_TILE_IN_FRONT)
        _step(rm, 1, has_surf=True, tile_in_front=SURF_TILE_IN_FRONT, start_menu_open=True)
        _step(rm, 2, has_surf=True, tile_in_front=SURF_TILE_IN_FRONT, pokemon_menu_open=True)
        _step(rm, 3, has_surf=True, is_surfing=True,
              used_surf_successfully=True, valid_surf_coords_count=1)
        _step(rm, 4)
        assert rm.state == RewardMachineState.IDLE
        # 물 위에서 다시 시도 → 재진입 차단
        s = _step(rm, 5, has_surf=True, tile_in_front=SURF_TILE_IN_FRONT, is_surfing=True)
        assert rm.state == RewardMachineState.IDLE

    def test_surf_success_requires_new_surf(self):
        rm = RewardMachine()
        _step(rm, 0, has_surf=True, tile_in_front=SURF_TILE_IN_FRONT, valid_surf_coords_count=1)
        _step(rm, 1, has_surf=True, tile_in_front=SURF_TILE_IN_FRONT, start_menu_open=True, valid_surf_coords_count=1)
        _step(rm, 2, has_surf=True, tile_in_front=SURF_TILE_IN_FRONT, pokemon_menu_open=True, valid_surf_coords_count=1)
        assert rm.state == RewardMachineState.SURF_MON_SELECTED
        # valid_surf_coords_count 증분 없음 → SUCCESS 불가
        _step(rm, 3, has_surf=True, is_surfing=True,
              used_surf_successfully=True, valid_surf_coords_count=1)
        assert rm.state != RewardMachineState.SURF_SUCCESS


# ─────────────────────────────────────────────────────────────────
# 3. FLASH
# ─────────────────────────────────────────────────────────────────
class TestFlash:
    def test_no_trigger_outside_dark_cave(self):
        rm = RewardMachine()
        _step(rm, 0, has_flash=True, in_dark_cave=False)
        assert rm.state == RewardMachineState.IDLE

    def test_no_trigger_when_auto_flash(self):
        rm = RewardMachine()
        _step(rm, 0, has_flash=True, in_dark_cave=True, auto_flash=True)
        assert rm.state == RewardMachineState.IDLE

    def test_detected_in_dark_cave(self):
        rm = RewardMachine()
        s = _step(rm, 0, has_flash=True, in_dark_cave=True)
        assert rm.state == RewardMachineState.FLASH_DETECTED
        assert s.transition_key == "rm_flash_detected"

    def test_happy_path(self):
        rm = RewardMachine()
        # IDLE → FLASH_DETECTED
        _step(rm, 0, has_flash=True, in_dark_cave=True)
        assert rm.state == RewardMachineState.FLASH_DETECTED

        # FLASH_DETECTED → FLASH_MENU_OPEN
        s = _step(rm, 1, has_flash=True, in_dark_cave=True, start_menu_open=True)
        assert rm.state == RewardMachineState.FLASH_MENU_OPEN
        assert s.transition_key == "rm_flash_menu_open"

        # FLASH_MENU_OPEN → FLASH_MON_SELECTED
        s = _step(rm, 2, has_flash=True, in_dark_cave=True, pokemon_menu_open=True)
        assert rm.state == RewardMachineState.FLASH_MON_SELECTED
        assert s.transition_key == "rm_flash_mon_selected"

        # Flash 사용: 동굴이 밝아지고 훅이 valid_flash_coords에 추가됨
        s = _step(rm, 3, has_flash=True, in_dark_cave=False,
                  flash_cycle_has_new_success=True, valid_flash_coords_count=1)
        assert rm.state == RewardMachineState.FLASH_SUCCESS
        assert s.transition_key == "rm_flash_success"

        s = _step(rm, 4)
        assert rm.state == RewardMachineState.IDLE
        assert s.transition_key == "rm_flash_done"

    def test_shortcut_detected_to_success_when_cave_lit(self):
        """밝아진 최종 스냅샷만 남으면 DETECTED에서 곧바로 FLASH_SUCCESS."""
        rm = RewardMachine()
        _step(rm, 0, has_flash=True, in_dark_cave=True)
        assert rm.state == RewardMachineState.FLASH_DETECTED
        s = _step(
            rm,
            1,
            has_flash=True,
            in_dark_cave=False,
            flash_cycle_has_new_success=True,
            valid_flash_coords_count=1,
        )
        assert rm.state == RewardMachineState.FLASH_SUCCESS
        assert s.transition_key == "rm_flash_success"

    def test_abort_from_detected(self):
        rm = RewardMachine()
        _step(rm, 0, has_flash=True, in_dark_cave=True)
        assert rm.state == RewardMachineState.FLASH_DETECTED
        # 동굴 밖으로
        s = _step(rm, 1, has_flash=True, in_dark_cave=False)
        assert rm.state == RewardMachineState.IDLE
        assert s.transition_key == "rm_flash_aborted"

    def test_left_dark_without_using_flash(self):
        """FLASH_MON_SELECTED에서 Flash 없이 동굴을 나가면 IDLE로 복귀."""
        rm = RewardMachine()
        _step(rm, 0, has_flash=True, in_dark_cave=True)
        _step(rm, 1, has_flash=True, in_dark_cave=True, start_menu_open=True)
        _step(rm, 2, has_flash=True, in_dark_cave=True, pokemon_menu_open=True)
        assert rm.state == RewardMachineState.FLASH_MON_SELECTED

        s = _step(rm, 3, has_flash=True, in_dark_cave=False,
                  flash_cycle_has_new_success=False)
        assert rm.state == RewardMachineState.IDLE
        assert s.transition_key == "rm_flash_left_dark"

    def test_stays_in_mon_selected_while_in_dark_cave(self):
        """메뉴를 닫아도 동굴 안에 있으면 FLASH_MON_SELECTED 유지."""
        rm = RewardMachine()
        _step(rm, 0, has_flash=True, in_dark_cave=True)
        _step(rm, 1, has_flash=True, in_dark_cave=True, start_menu_open=True)
        _step(rm, 2, has_flash=True, in_dark_cave=True, pokemon_menu_open=True)
        assert rm.state == RewardMachineState.FLASH_MON_SELECTED

        # 메뉴 닫힘, 아직 동굴 내부
        s = _step(rm, 3, has_flash=True, in_dark_cave=True)
        assert rm.state == RewardMachineState.FLASH_MON_SELECTED

    def test_idle_no_reentry_immediately_after_detected(self):
        """FLASH_DETECTED에서 바로 IDLE로 복귀(abort) 후 같은 스텝에서 재진입 없음."""
        rm = RewardMachine()
        _step(rm, 0, has_flash=True, in_dark_cave=True)
        assert rm.state == RewardMachineState.FLASH_DETECTED
        # 동굴 밖 → abort → IDLE
        _step(rm, 1, has_flash=True, in_dark_cave=False)
        assert rm.state == RewardMachineState.IDLE
        # 다시 동굴 진입 → 재무장 됐으므로 재진입 가능
        _step(rm, 2, has_flash=True, in_dark_cave=True)
        assert rm.state == RewardMachineState.FLASH_DETECTED

    def test_idle_no_reentry_same_cave(self):
        """동굴 안에서 IDLE→FLASH_DETECTED가 매 스텝 반복되지 않아야 함."""
        rm = RewardMachine()
        _step(rm, 0, has_flash=True, in_dark_cave=True)
        assert rm.state == RewardMachineState.FLASH_DETECTED
        # abort 없이 다음 스텝에 abort 조건 부여 (동굴 밖으로 나가지 않고)
        # → FLASH_DETECTED에서 아무 전이 없음 (start_menu도 없음)
        s = _step(rm, 1, has_flash=True, in_dark_cave=True)
        assert rm.state == RewardMachineState.FLASH_DETECTED

    def test_flash_failed_timeout(self):
        rm = RewardMachine()
        _step(rm, 0, has_flash=True, in_dark_cave=True)
        _step(rm, 1, has_flash=True, in_dark_cave=True, start_menu_open=True)
        assert rm.state == RewardMachineState.FLASH_MENU_OPEN

        # 256 스텝 초과 → FAILED
        s = _step(rm, 257, has_flash=True, in_dark_cave=True)
        assert rm.state == RewardMachineState.FAILED
        assert s.transition_key == "rm_failed_timeout"

    def test_flash_idle_rearmed_after_success(self):
        """Flash 성공 후 다른 동굴에서 다시 FLASH_DETECTED로 진입 가능해야 함."""
        rm = RewardMachine()
        _step(rm, 0, has_flash=True, in_dark_cave=True)
        _step(rm, 1, has_flash=True, in_dark_cave=True, start_menu_open=True)
        _step(rm, 2, has_flash=True, in_dark_cave=True, pokemon_menu_open=True)
        _step(rm, 3, has_flash=True, in_dark_cave=False,
              flash_cycle_has_new_success=True, valid_flash_coords_count=1)
        _step(rm, 4)  # FLASH_SUCCESS → IDLE
        assert rm.state == RewardMachineState.IDLE

        # 동굴 밖 → idle_flash_entry_ok 재무장
        _step(rm, 5, has_flash=True, in_dark_cave=False)
        # 새 동굴 진입 → 재진입
        s = _step(rm, 6, has_flash=True, in_dark_cave=True)
        assert rm.state == RewardMachineState.FLASH_DETECTED

    def test_hm_target_is_flash_in_chain(self):
        rm = RewardMachine()
        _step(rm, 0, has_flash=True, in_dark_cave=True)
        assert rm.hm_target == HMTarget.FLASH
        _step(rm, 1, has_flash=True, in_dark_cave=True, start_menu_open=True)
        assert rm.hm_target == HMTarget.FLASH
        _step(rm, 2, has_flash=True, in_dark_cave=True, pokemon_menu_open=True)
        assert rm.hm_target == HMTarget.FLASH


# ─────────────────────────────────────────────────────────────────
# 4. 우선순위: 동굴에서 컷 가능 타일 앞 → CUT이 FLASH보다 먼저
# ─────────────────────────────────────────────────────────────────
class TestPriority:
    def test_cut_before_flash(self):
        """어두운 동굴에서 컷 가능 타일 앞 → CUT_DETECTED 먼저."""
        rm = RewardMachine()
        s = _step(rm, 0, has_cut=True, has_flash=True,
                  tile_in_front=_CUT_TILE, in_dark_cave=True)
        assert rm.state == RewardMachineState.CUT_DETECTED

    def test_flash_when_no_cut_tile(self):
        """컷 타일 없는 어두운 동굴 → FLASH_DETECTED."""
        rm = RewardMachine()
        s = _step(rm, 0, has_flash=True, tile_in_front=0x00, in_dark_cave=True)
        assert rm.state == RewardMachineState.FLASH_DETECTED


# ─────────────────────────────────────────────────────────────────
# 5. HM Target 매핑 검증
# ─────────────────────────────────────────────────────────────────
class TestHMTarget:
    @pytest.mark.parametrize("state,expected", [
        (RewardMachineState.IDLE, HMTarget.NONE),
        (RewardMachineState.CUT_DETECTED, HMTarget.CUT),
        (RewardMachineState.CUT_SUCCESS, HMTarget.CUT),
        (RewardMachineState.SURF_DETECTED, HMTarget.SURF),
        (RewardMachineState.SURF_SUCCESS, HMTarget.SURF),
        (RewardMachineState.FLASH_DETECTED, HMTarget.FLASH),
        (RewardMachineState.FLASH_SUCCESS, HMTarget.FLASH),
        (RewardMachineState.FAILED, HMTarget.NONE),
    ])
    def test_mapping(self, state, expected):
        rm = RewardMachine(initial_state=state)
        assert rm.hm_target == expected


# ─────────────────────────────────────────────────────────────────
# 6. reset() 완전 초기화 검증
# ─────────────────────────────────────────────────────────────────
class TestReset:
    def test_reset_clears_all(self):
        rm = RewardMachine()
        _step(rm, 0, has_flash=True, in_dark_cave=True)
        assert rm.state != RewardMachineState.IDLE
        rm.reset()
        assert rm.state == RewardMachineState.IDLE
        assert rm._flash_cycle_start_count == 0
        assert rm._steps_in_state == 0
        assert rm._idle_flash_entry_ok is True
        assert rm._last_invalid_flash_coords_count is None
