from omegaconf import DictConfig, OmegaConf

from pokemonred_puffer.data.items import Items
from pokemonred_puffer.data.tm_hm import CUT_SPECIES_IDS, TmHmMoves
from pokemonred_puffer.environment import RedGymEnv
from pokemonred_puffer.rewards.reward_machine import (
    CUTTABLE_TILES,
    HMTarget,
    POKEFLUTE_TILE_IN_FRONT,
    RewardMachine,
    RewardMachineContext,
    RewardMachineState,
    SURF_TILE_IN_FRONT,
)

# HM 사용 “성공” 전이만 rm_success.
_RM_SUCCESS_KEYS = frozenset(
    {
        "rm_cut_success",
        "rm_surf_success",
        "rm_pokeflute_success",
        "rm_flash_success",
    }
)

# 탐지/정리/실패 복구/중단 등 보상 없음. rm_transition은 MENU_OPEN · MON_SELECTED 등 중간 단계만.
_RM_NO_REWARD_KEYS = frozenset(
    {
        "rm_cut_detected",
        "rm_surf_detected",
        "rm_pokeflute_detected",
        "rm_flash_detected",
        "rm_cut_done",
        "rm_surf_done",
        "rm_pokeflute_done",
        "rm_flash_done",
        "rm_failed_timeout",
        # *_DETECTED → IDLE 탈출 전이: 에이전트가 트리거 타일에서 벗어날 때 발생.
        "rm_cut_aborted",
        "rm_surf_aborted",
        "rm_pokeflute_aborted",
        "rm_flash_aborted",
        "rm_flash_left_dark",
    }
)

_HM_SUPERVISION_TRANSITION_TARGETS: dict[str, HMTarget] = {
    "rm_cut_detected": HMTarget.CUT,
    "rm_cut_menu_open": HMTarget.CUT,
    "rm_cut_mon_selected": HMTarget.CUT,
    "rm_cut_success": HMTarget.CUT,
    "rm_surf_detected": HMTarget.SURF,
    "rm_surf_menu_open": HMTarget.SURF,
    "rm_surf_mon_selected": HMTarget.SURF,
    "rm_surf_success": HMTarget.SURF,
    "rm_flash_detected": HMTarget.FLASH,
    "rm_flash_menu_open": HMTarget.FLASH,
    "rm_flash_mon_selected": HMTarget.FLASH,
    "rm_flash_success": HMTarget.FLASH,
    "rm_pokeflute_detected": HMTarget.POKEFLUTE,
    "rm_pokeflute_bag_open": HMTarget.POKEFLUTE,
    "rm_pokeflute_success": HMTarget.POKEFLUTE,
}
_HM_PERSISTENT_LATCH_STEPS = 8


def get_hm_supervision_target(
    final_target: HMTarget, transition_keys: list[str]
) -> HMTarget:
    for transition_key in transition_keys:
        target = _HM_SUPERVISION_TRANSITION_TARGETS.get(transition_key)
        if target is not None:
            return target
    return final_target


def count_new_invalid_hm_uses(
    *,
    prev_invalid_cut_count: int,
    current_invalid_cut_count: int,
    prev_invalid_surf_count: int,
    current_invalid_surf_count: int,
    prev_invalid_flash_count: int,
    current_invalid_flash_count: int,
) -> int:
    d_cut = max(0, current_invalid_cut_count - prev_invalid_cut_count)
    d_surf = max(0, current_invalid_surf_count - prev_invalid_surf_count)
    d_flash = max(0, current_invalid_flash_count - prev_invalid_flash_count)
    return d_cut + d_surf + d_flash


def get_hm_needed_target(
    final_target: HMTarget,
    context: RewardMachineContext,
    adjacent_water_count: int,
) -> HMTarget:
    if final_target != HMTarget.NONE:
        return final_target
    if context.tile_in_front in CUTTABLE_TILES and context.can_use_cut:
        return HMTarget.CUT
    if context.can_use_surf and (context.is_surfing or context.tile_in_front == SURF_TILE_IN_FRONT):
        return HMTarget.SURF
    if context.can_use_surf and adjacent_water_count > 0:
        return HMTarget.SURF
    if context.tile_in_front == POKEFLUTE_TILE_IN_FRONT and context.can_use_pokeflute:
        return HMTarget.POKEFLUTE
    if context.in_dark_cave and context.can_use_flash:
        return HMTarget.FLASH
    return HMTarget.NONE


def should_clear_persistent_hm_supervision(
    latched_target: HMTarget,
    transition_keys: list[str],
    context: RewardMachineContext,
) -> bool:
    if latched_target == HMTarget.NONE:
        return True
    if "rm_failed_timeout" in transition_keys:
        return True
    if latched_target == HMTarget.CUT:
        return True
    if latched_target == HMTarget.POKEFLUTE:
        return True
    if latched_target == HMTarget.FLASH:
        return (not context.in_dark_cave) or (not context.can_use_flash)
    if latched_target == HMTarget.SURF:
        return (not context.can_use_surf) or ("rm_surf_aborted" in transition_keys)
    return True


def get_persistent_hm_supervision_target(
    final_target: HMTarget,
    transition_keys: list[str],
    context: RewardMachineContext,
    adjacent_water_count: int,
    previous_target: HMTarget,
    previous_steps_remaining: int,
) -> tuple[HMTarget, HMTarget, int]:
    current_target = get_hm_needed_target(final_target, context, adjacent_water_count)
    if current_target != HMTarget.NONE:
        return current_target, current_target, _HM_PERSISTENT_LATCH_STEPS

    if previous_target == HMTarget.NONE or previous_steps_remaining <= 0:
        return HMTarget.NONE, HMTarget.NONE, 0

    if should_clear_persistent_hm_supervision(previous_target, transition_keys, context):
        return HMTarget.NONE, HMTarget.NONE, 0

    next_steps_remaining = max(previous_steps_remaining - 1, 0)
    if next_steps_remaining <= 0:
        return HMTarget.NONE, HMTarget.NONE, 0
    return previous_target, previous_target, next_steps_remaining


class BaselineRewardEnv(RedGymEnv):
    def get_rm_flash_cycle_start(self) -> int:
        return int(self.reward_machine.flash_cycle_start_count)

    def __init__(self, env_config: DictConfig, reward_config: DictConfig):
        self.reward_machine = RewardMachine()
        self.rm_reward_total = 0.0
        self.step_penalty_total = 0.0
        self.rm_transition_count = 0
        self.rm_success_count = 0
        self.rm_cut_success_count = 0
        self.rm_surf_success_count = 0
        self.rm_pokeflute_success_count = 0
        self.rm_flash_success_count = 0
        self.rm_intermediate_paid_count = 0
        self.rm_reward_from_success = 0.0
        self.rm_reward_from_intermediate = 0.0
        self.rm_last_step_delta = 0.0
        self.last_rm_transition_key = ""
        self.hm_supervision_target = HMTarget.NONE
        self.hm_supervision_latch_target = HMTarget.NONE
        self.hm_supervision_latch_steps_remaining = 0
        self.missing_cut_reported = False
        self.unnecessary_hm_penalty_total = 0.0
        self._prev_invalid_cut_count = 0
        self._prev_invalid_surf_count = 0
        self._prev_invalid_flash_count = 0
        super().__init__(env_config)
        self.reward_config = OmegaConf.to_object(reward_config)

    def reset(self, *args, **kwargs):
        self.reward_machine.reset()
        self.rm_reward_total = 0.0
        self.step_penalty_total = 0.0
        self.rm_transition_count = 0
        self.rm_success_count = 0
        self.rm_cut_success_count = 0
        self.rm_surf_success_count = 0
        self.rm_pokeflute_success_count = 0
        self.rm_flash_success_count = 0
        self.rm_intermediate_paid_count = 0
        self.rm_reward_from_success = 0.0
        self.rm_reward_from_intermediate = 0.0
        self.rm_last_step_delta = 0.0
        self.last_rm_transition_key = ""
        self.hm_supervision_target = HMTarget.NONE
        self.hm_supervision_latch_target = HMTarget.NONE
        self.hm_supervision_latch_steps_remaining = 0
        self.missing_cut_reported = False
        self.unnecessary_hm_penalty_total = 0.0
        ret = super().reset(*args, **kwargs)
        self._prev_invalid_cut_count = len(self.invalid_cut_coords)
        self._prev_invalid_surf_count = len(self.invalid_surf_coords)
        self._prev_invalid_flash_count = len(self.invalid_flash_coords)
        return ret

    def _before_progress_reward(self) -> None:
        """스텝당 RM 전이는 여기서만 실행. `get_game_state_reward()`는 전이 없이 누적값만 읽는다."""
        # Cut/Surf를 실제로 시도했지만 효과 없음(맨땅, 물 아님 등): PyBoy 훅이 invalid_*에 기록.
        hm_pen = float(self.reward_config.get("unnecessary_hm_usage_penalty", 0.0))
        if hm_pen != 0.0:
            total_new = count_new_invalid_hm_uses(
                prev_invalid_cut_count=self._prev_invalid_cut_count,
                current_invalid_cut_count=len(self.invalid_cut_coords),
                prev_invalid_surf_count=self._prev_invalid_surf_count,
                current_invalid_surf_count=len(self.invalid_surf_coords),
                prev_invalid_flash_count=self._prev_invalid_flash_count,
                current_invalid_flash_count=len(self.invalid_flash_coords),
            )
            if total_new > 0:
                self.unnecessary_hm_penalty_total += -abs(hm_pen) * total_new

        self.update_reward_machine_reward()
        # step_penalty는 스텝마다 누적. rm_reward와 동일하게 cumulative로 관리해
        # 에피소드 말미 로그에서 episode 총 패널티가 보이도록 한다.
        penalty = float(self.reward_config.get("step_penalty", 0.0))
        self.step_penalty_total += -abs(penalty)

        self._prev_invalid_cut_count = len(self.invalid_cut_coords)
        self._prev_invalid_surf_count = len(self.invalid_surf_coords)
        self._prev_invalid_flash_count = len(self.invalid_flash_coords)

    def get_game_state_reward(self) -> dict[str, float]:
        return {
            "rm_reward": self.rm_reward_total,
            "step_penalty": self.step_penalty_total,
            "unnecessary_hm_penalty": self.unnecessary_hm_penalty_total,
        }

    def _rm_reward_for_transition_key(self, key: str) -> float:
        # HM별 보상을 config에서 개별 조회. 키가 없으면 5.0 기본값.
        if key in _RM_SUCCESS_KEYS:
            return float(self.reward_config.get(key, 5.0))
        if key in _RM_NO_REWARD_KEYS:
            return 0.0
        return float(self.reward_config.get("rm_transition", 0.0))

    def update_reward_machine_reward(self) -> float:
        self.rm_last_step_delta = 0.0
        if not self.reward_config.get("rm_enabled", True):
            self.hm_supervision_target = self.reward_machine.hm_target
            return 0.0

        self.ensure_cut_for_reward_machine()
        context = RewardMachineContext.from_env(self)
        transition_keys_this_step: list[str] = []
        # 한 PyBoy step(에이전트 스텝) 안에서 메뉴→HM까지 모두 진행되면, 같은 스냅샷으로
        # 여러 RM 전이가 연쇄되어야 한다. transition()은 1회 1전이만 하므로, 메뉴 플래그가
        # 다음 스텝 맨 앞에 0으로 초기화되면 체인이 끊긴다.
        _MAX_RM_CHAIN = 32
        for _ in range(_MAX_RM_CHAIN):
            step = self.reward_machine.transition(context)
            if not step.changed or not step.transition_key:
                break
            transition_keys_this_step.append(step.transition_key)

            amt = self._rm_reward_for_transition_key(step.transition_key)
            self.rm_reward_total += amt
            self.rm_last_step_delta += amt
            self.rm_transition_count += 1
            self.last_rm_transition_key = step.transition_key
            if step.transition_key in _RM_SUCCESS_KEYS:
                self.rm_success_count += 1
                self.rm_reward_from_success += amt
                if step.transition_key == "rm_cut_success":
                    self.rm_cut_success_count += 1
                elif step.transition_key == "rm_surf_success":
                    self.rm_surf_success_count += 1
                elif step.transition_key == "rm_pokeflute_success":
                    self.rm_pokeflute_success_count += 1
                elif step.transition_key == "rm_flash_success":
                    self.rm_flash_success_count += 1
            elif amt > 0.0:
                self.rm_intermediate_paid_count += 1
                self.rm_reward_from_intermediate += amt

        transition_target = get_hm_supervision_target(
            self.reward_machine.hm_target, transition_keys_this_step
        )
        (
            self.hm_supervision_target,
            self.hm_supervision_latch_target,
            self.hm_supervision_latch_steps_remaining,
        ) = get_persistent_hm_supervision_target(
            transition_target,
            transition_keys_this_step,
            context,
            self.get_adjacent_water_count(),
            self.hm_supervision_latch_target,
            self.hm_supervision_latch_steps_remaining,
        )
        return self.rm_reward_total

    def ensure_cut_for_reward_machine(self) -> None:
        if self.reward_machine.state != RewardMachineState.CUT_DETECTED:
            return
        if self.check_if_party_has_hm(TmHmMoves.CUT.value):
            return

        if Items.HM_01 in self.get_items_in_bag():
            self.teach_hm(TmHmMoves.CUT.value, 30, CUT_SPECIES_IDS)
            self.missing_cut_reported = False
            return

        if not self.missing_cut_reported:
            print("cut 없음")
            self.missing_cut_reported = True


class ObjectRewardRequiredEventsMapIds(BaselineRewardEnv):
    """이벤트/맵 보상 확장 지점. 현재는 Baseline과 동일한 RM·step_penalty dict."""


class ObjectRewardRequiredEventsMapIdsFieldMoves(ObjectRewardRequiredEventsMapIds):
    """필드무브 RM 전용 엔트리 이름."""
