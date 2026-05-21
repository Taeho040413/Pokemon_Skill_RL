"""Baseline env rewards + Reward Machine payout.

RM 보상 3단계:
  1. 무음(0): detected/start_menu/done/aborted — 상태만, PPO 보상 없음
  2. 중간: menu_open / party_menu / mon_selected — `rm_intermediate` (키별 override 가능)
  3. 성공: rm_*_success — config의 rm_cut_success / rm_surf_success / rm_flash_success

HM aux CE 라벨: reward_machine.hm_target 우선, 없으면 latch(옵션).
`hm_supervision_proactive`는 기본 끔 (assist + rm_state obs와 중복).
"""
from __future__ import annotations

from omegaconf import DictConfig, OmegaConf

from pokemonred_puffer.data.items import Items
from pokemonred_puffer.data.tm_hm import CUT_SPECIES_IDS, TmHmMoves
from pokemonred_puffer.environment import RedGymEnv
from pokemonred_puffer.rewards.reward_machine import (
    CUTTABLE_TILES,
    HMTarget,
    RewardMachine,
    RewardMachineContext,
    RewardMachineState,
    SURF_TILE_IN_FRONT,
)

# ── RM payout tiers ───────────────────────────────────────────────────────────

_RM_SUCCESS_KEYS = frozenset(
    {"rm_cut_success", "rm_surf_success", "rm_flash_success"}
)

# 전이는 기록되지만 PPO 보상 0
_RM_SILENT_KEYS = frozenset(
    {
        "rm_cut_detected",
        "rm_cut_start_menu",
        "rm_surf_detected",
        "rm_surf_start_menu",
        "rm_flash_detected",
        "rm_flash_start_menu",
        "rm_cut_done",
        "rm_surf_done",
        "rm_flash_done",
        "rm_failed_timeout",
        "rm_cut_aborted",
        "rm_surf_aborted",
        "rm_flash_aborted",
        "rm_flash_left_dark",
    }
)

# 메뉴 체인 중간 단계 (config `rm_intermediate` 또는 키별 override)
_RM_INTERMEDIATE_KEYS = frozenset(
    {
        "rm_cut_menu_open",
        "rm_cut_pokemon_row",
        "rm_cut_party_menu",
        "rm_cut_mon_selected",
        "rm_surf_menu_open",
        "rm_surf_pokemon_row",
        "rm_surf_party_menu",
        "rm_surf_mon_selected",
        "rm_flash_menu_open",
        "rm_flash_pokemon_row",
        "rm_flash_party_menu",
        "rm_flash_mon_selected",
    }
)

_RM_CLAWBACK_KEYS = frozenset(
    {
        "rm_cut_aborted",
        "rm_surf_aborted",
        "rm_flash_aborted",
        "rm_failed_timeout",
        "rm_flash_left_dark",
    }
)

_DEFAULT_HM_LATCH_STEPS = 8


def _hm_target_from_transition_key(key: str) -> HMTarget | None:
    if key in _RM_SILENT_KEYS:
        return None
    if key.startswith("rm_cut_"):
        return HMTarget.CUT
    if key.startswith("rm_surf_"):
        return HMTarget.SURF
    if key.startswith("rm_flash_"):
        return HMTarget.FLASH
    return None


def _set_rm_step_deltas(env) -> None:
    """RM context용 cut/surf/flash 스텝 델타 (BaselineRewardEnv·테스트 mock 공용)."""
    env._rm_valid_cut_delta = max(0, len(env.valid_cut_coords) - env._prev_valid_cut_count)
    env._rm_valid_surf_delta = max(
        0,
        int(getattr(env, "_surf_hook_success_count", 0))
        - int(getattr(env, "_prev_surf_hook_success_count", 0)),
    )
    env._rm_valid_flash_delta = max(
        0, len(env.valid_flash_coords) - env._prev_valid_flash_count
    )


def get_hm_supervision_target(
    final_target: HMTarget, transition_keys: list[str]
) -> HMTarget:
    for transition_key in transition_keys:
        target = _hm_target_from_transition_key(transition_key)
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


def compute_hm_opportunity_flags(context: RewardMachineContext) -> tuple[int, int, int]:
    """RM IDLE→*_DETECTED 와 동일: 지금 이 스텝에서 어떤 HM이 의미 있는지 (0/1)."""
    cut = int(context.tile_in_front in CUTTABLE_TILES and context.can_use_cut)
    surf = int(
        context.can_use_surf and not context.is_surfing and context.surf_detect_ok
    )
    flash = int(context.in_dark_cave and context.can_use_flash)
    return cut, surf, flash


def get_hm_needed_target(
    final_target: HMTarget,
    context: RewardMachineContext,
    adjacent_water_count: int = 0,  # noqa: ARG001 — API 호환
    *,
    proactive_supervision: bool = False,
) -> HMTarget:
    if final_target != HMTarget.NONE:
        return final_target
    if not proactive_supervision:
        return HMTarget.NONE
    cut_ok, surf_ok, flash_ok = compute_hm_opportunity_flags(context)
    if cut_ok:
        return HMTarget.CUT
    if surf_ok:
        return HMTarget.SURF
    if flash_ok:
        return HMTarget.FLASH
    return HMTarget.NONE


def should_clear_hm_supervision_latch(
    latched_target: HMTarget,
    transition_keys: list[str],
    context: RewardMachineContext,
) -> bool:
    if latched_target == HMTarget.NONE:
        return True
    if "rm_failed_timeout" in transition_keys:
        return True
    if latched_target == HMTarget.CUT:
        return (context.tile_in_front not in CUTTABLE_TILES) or not context.can_use_cut
    if latched_target == HMTarget.FLASH:
        return (not context.in_dark_cave) or (not context.can_use_flash)
    if latched_target == HMTarget.SURF:
        return (
            (not context.can_use_surf)
            or ("rm_surf_aborted" in transition_keys)
            or not context.surf_water_context_ok
        )
    return True


def resolve_hm_supervision_target(
    final_target: HMTarget,
    transition_keys: list[str],
    context: RewardMachineContext,
    adjacent_water_count: int,  # noqa: ARG001 — API 호환
    previous_target: HMTarget,
    previous_steps_remaining: int,
    *,
    proactive_supervision: bool = False,
    latch_steps: int = _DEFAULT_HM_LATCH_STEPS,
) -> tuple[HMTarget, HMTarget, int]:
    """이번 스텝 HM aux / stats용 타깃. proactive·latch는 config로만 켠다."""
    step_target = get_hm_supervision_target(final_target, transition_keys)
    if step_target != HMTarget.NONE:
        return step_target, step_target, latch_steps

    proactive = get_hm_needed_target(
        HMTarget.NONE, context, proactive_supervision=proactive_supervision
    )
    if proactive != HMTarget.NONE:
        return proactive, proactive, latch_steps

    if previous_target == HMTarget.NONE or previous_steps_remaining <= 0:
        return HMTarget.NONE, HMTarget.NONE, 0

    if should_clear_hm_supervision_latch(previous_target, transition_keys, context):
        return HMTarget.NONE, HMTarget.NONE, 0

    next_steps = max(previous_steps_remaining - 1, 0)
    if next_steps <= 0:
        return HMTarget.NONE, HMTarget.NONE, 0
    return previous_target, previous_target, next_steps


# 레거시 테스트·import 호환
get_persistent_hm_supervision_target = resolve_hm_supervision_target
should_clear_persistent_hm_supervision = should_clear_hm_supervision_latch


def _init_rm_reward_state(env) -> None:
    env.rm_reward_total = 0.0
    env.step_penalty_total = 0.0
    env.rm_transition_count = 0
    env.rm_success_count = 0
    env.rm_cut_success_count = 0
    env.rm_surf_detected_count = 0
    env.rm_surf_menu_open_count = 0
    env.rm_surf_mon_selected_count = 0
    env.rm_surf_aborted_count = 0
    env.rm_surf_success_count = 0
    env.rm_flash_success_count = 0
    env.rm_intermediate_paid_count = 0
    env.rm_reward_from_success = 0.0
    env.rm_reward_from_intermediate = 0.0
    env.rm_reward_intermediate_net = 0.0
    env.rm_clawback_total = 0.0
    env.rm_clawback_count = 0
    env._rm_attempt_intermediate_pending = 0.0
    env.rm_last_step_delta = 0.0
    env.last_rm_transition_key = ""
    env.hm_supervision_target = HMTarget.NONE
    env.hm_supervision_latch_target = HMTarget.NONE
    env.hm_supervision_latch_steps_remaining = 0
    env.missing_cut_reported = False
    env.unnecessary_hm_penalty_total = 0.0
    env.invalid_action_total = 0.0


class BaselineRewardEnv(RedGymEnv):
    def get_rm_flash_cycle_start(self) -> int:
        return int(self.reward_machine.flash_cycle_start_count)

    def refresh_hm_aux_label_for_obs(self) -> None:
        """HM aux CE: RM hm_target, 없으면 supervision latch."""
        rm_target = HMTarget(self.get_reward_machine_hm_target_id())
        if rm_target != HMTarget.NONE:
            self._hm_aux_label_for_obs = int(rm_target)
            return
        latched = getattr(self, "hm_supervision_latch_target", HMTarget.NONE)
        if latched != HMTarget.NONE and self.hm_supervision_latch_steps_remaining > 0:
            self._hm_aux_label_for_obs = int(latched)
            return
        self._hm_aux_label_for_obs = int(HMTarget.NONE)

    def __init__(self, env_config: DictConfig, reward_config: DictConfig):
        self.reward_machine = RewardMachine()
        _init_rm_reward_state(self)
        self._prev_invalid_cut_count = 0
        self._prev_invalid_surf_count = 0
        self._prev_invalid_flash_count = 0
        self._prev_valid_cut_count = 0
        self._prev_valid_surf_count = 0
        self._prev_surf_hook_success_count = 0
        self._prev_valid_flash_count = 0
        self._rm_valid_cut_delta = 0
        self._rm_valid_surf_delta = 0
        self._rm_valid_flash_delta = 0
        super().__init__(env_config)
        self.reward_config = OmegaConf.to_object(reward_config)

    def reset(self, *args, **kwargs):
        self.reward_machine.reset()
        _init_rm_reward_state(self)
        ret = super().reset(*args, **kwargs)
        self._prev_invalid_cut_count = len(self.invalid_cut_coords)
        self._prev_invalid_surf_count = len(self.invalid_surf_coords)
        self._prev_invalid_flash_count = len(self.invalid_flash_coords)
        self._prev_valid_cut_count = len(self.valid_cut_coords)
        self._prev_valid_surf_count = len(self.valid_surf_coords)
        self._prev_surf_hook_success_count = int(getattr(self, "_surf_hook_success_count", 0))
        self._prev_valid_flash_count = len(self.valid_flash_coords)
        self._rm_valid_cut_delta = 0
        self._rm_valid_surf_delta = 0
        self._rm_valid_flash_delta = 0
        return ret

    def _before_progress_reward(self) -> None:
        """스텝당 RM 전이는 여기서만 실행. `get_game_state_reward()`는 전이 없이 누적값만 읽는다."""
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

        _set_rm_step_deltas(self)
        self.update_reward_machine_reward()
        penalty = float(self.reward_config.get("step_penalty", 0.0))
        self.step_penalty_total += -abs(penalty)

        self._prev_invalid_cut_count = len(self.invalid_cut_coords)
        self._prev_invalid_surf_count = len(self.invalid_surf_coords)
        self._prev_invalid_flash_count = len(self.invalid_flash_coords)
        self._prev_valid_cut_count = len(self.valid_cut_coords)
        self._prev_valid_surf_count = len(self.valid_surf_coords)
        self._prev_surf_hook_success_count = int(getattr(self, "_surf_hook_success_count", 0))
        self._prev_valid_flash_count = len(self.valid_flash_coords)

    def get_game_state_reward(self) -> dict[str, float]:
        return {
            "rm_reward": self.rm_reward_total,
            "step_penalty": self.step_penalty_total,
            "unnecessary_hm_penalty": self.unnecessary_hm_penalty_total,
            "invalid_action": self.invalid_action_total,
        }

    def _rm_reward_for_transition_key(self, key: str) -> float:
        if key in _RM_SILENT_KEYS:
            return 0.0
        if key in _RM_SUCCESS_KEYS:
            return float(self.reward_config.get(key, 5.0))
        if key in _RM_INTERMEDIATE_KEYS:
            default = float(self.reward_config.get("rm_intermediate", 0.0))
            return float(self.reward_config.get(key, default))
        return 0.0

    def _apply_rm_clawback(self, key: str) -> bool:
        """중단 시 pending 중간 보상 회수. True면 이번 전이는 보상 처리 생략."""
        if key not in _RM_CLAWBACK_KEYS:
            return False
        pending = self._rm_attempt_intermediate_pending
        self._rm_attempt_intermediate_pending = 0.0
        if not bool(self.reward_config.get("rm_clawback_intermediate_on_abort", False)):
            return True
        if pending <= 0.0:
            return True
        fraction = float(self.reward_config.get("rm_clawback_fraction", 1.0))
        fraction = max(0.0, min(1.0, fraction))
        clawback = pending * fraction
        if clawback > 0.0:
            self.rm_reward_total -= clawback
            self.rm_last_step_delta -= clawback
            self.rm_clawback_total += clawback
            self.rm_reward_intermediate_net -= clawback
            self.rm_clawback_count += 1
        return True

    def _record_rm_transition_stats(self, key: str, amt: float) -> None:
        if key == "rm_surf_detected":
            self.rm_surf_detected_count += 1
        elif key in ("rm_surf_menu_open", "rm_surf_pokemon_row", "rm_surf_party_menu"):
            self.rm_surf_menu_open_count += 1
        elif key == "rm_surf_mon_selected":
            self.rm_surf_mon_selected_count += 1
        elif key == "rm_surf_aborted":
            self.rm_surf_aborted_count += 1

        if key in _RM_SUCCESS_KEYS:
            self._rm_attempt_intermediate_pending = 0.0
            self.rm_success_count += 1
            self.rm_reward_from_success += amt
            if key == "rm_cut_success":
                self.rm_cut_success_count += 1
            elif key == "rm_surf_success":
                self.rm_surf_success_count += 1
            elif key == "rm_flash_success":
                self.rm_flash_success_count += 1
        elif amt > 0.0:
            self._rm_attempt_intermediate_pending += amt
            self.rm_intermediate_paid_count += 1
            self.rm_reward_from_intermediate += amt
            self.rm_reward_intermediate_net += amt

    def update_reward_machine_reward(self) -> float:
        self.rm_last_step_delta = 0.0
        if not self.reward_config.get("rm_enabled", True):
            self.hm_supervision_target = self.reward_machine.hm_target
            return 0.0

        self.ensure_cut_for_reward_machine()
        transition_keys_this_step: list[str] = []
        max_chain = max(1, int(self.reward_config.get("rm_max_transitions_per_step", 3)))
        latch_steps = int(
            self.reward_config.get("hm_supervision_latch_steps", _DEFAULT_HM_LATCH_STEPS)
        )

        context = RewardMachineContext.from_env(self)
        for _ in range(max_chain):
            step = self.reward_machine.transition(context)
            if not step.changed or not step.transition_key:
                break
            key = step.transition_key
            transition_keys_this_step.append(key)
            self.rm_transition_count += 1
            self.last_rm_transition_key = key

            if self._apply_rm_clawback(key):
                context = RewardMachineContext.from_env(self)
                continue

            amt = self._rm_reward_for_transition_key(key)
            self.rm_reward_total += amt
            self.rm_last_step_delta += amt
            self._record_rm_transition_stats(key, amt)
            # 루프 시작마다 from_env 하지 않고, 전이 직후에만 갱신.
            context = RewardMachineContext.from_env(self)

        (
            self.hm_supervision_target,
            self.hm_supervision_latch_target,
            self.hm_supervision_latch_steps_remaining,
        ) = resolve_hm_supervision_target(
            self.reward_machine.hm_target,
            transition_keys_this_step,
            context,
            self.get_adjacent_water_count(),
            self.hm_supervision_latch_target,
            self.hm_supervision_latch_steps_remaining,
            proactive_supervision=bool(
                self.reward_config.get("hm_supervision_proactive", False)
            ),
            latch_steps=latch_steps,
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


class ObjectRewardRequiredEventsMapIdsFieldMoves(BaselineRewardEnv):
    """train 기본 reward 클래스 (레거시 이름).

    PPO 보상 dict: rm_reward, step_penalty, unnecessary_hm_penalty, invalid_action.
    스토리 진행(required_events)은 swarm/sqlite 쪽; dense event/map shaping은 없음.
    """
