from omegaconf import DictConfig, OmegaConf

from pokemonred_puffer.data.items import Items
from pokemonred_puffer.data.tm_hm import CUT_SPECIES_IDS, TmHmMoves
from pokemonred_puffer.environment import RedGymEnv
from pokemonred_puffer.rewards.reward_machine import (
    RewardMachine,
    RewardMachineContext,
    RewardMachineState,
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
        "rm_cut_done",
        "rm_surf_done",
        "rm_pokeflute_done",
        "rm_failed_timeout",
        # *_DETECTED → IDLE 탈출 전이: 에이전트가 트리거 타일에서 벗어날 때 발생.
        "rm_cut_aborted",
        "rm_surf_aborted",
        "rm_pokeflute_aborted",
    }
)


class BaselineRewardEnv(RedGymEnv):
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
        self.missing_cut_reported = False
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
        self.missing_cut_reported = False
        return super().reset(*args, **kwargs)

    def _before_progress_reward(self) -> None:
        """스텝당 RM 전이는 여기서만 실행. `get_game_state_reward()`는 전이 없이 누적값만 읽는다."""
        self.update_reward_machine_reward()
        # step_penalty는 스텝마다 누적. rm_reward와 동일하게 cumulative로 관리해
        # 에피소드 말미 로그에서 episode 총 패널티가 보이도록 한다.
        penalty = float(self.reward_config.get("step_penalty", 0.0))
        self.step_penalty_total += -abs(penalty)

    def get_game_state_reward(self) -> dict[str, float]:
        return {
            "rm_reward": self.rm_reward_total,
            "step_penalty": self.step_penalty_total,
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
            return 0.0

        self.ensure_cut_for_reward_machine()
        context = RewardMachineContext.from_env(self)
        step = self.reward_machine.transition(context)

        if step.changed and step.transition_key:
            amt = self._rm_reward_for_transition_key(step.transition_key)
            self.rm_reward_total += amt
            self.rm_last_step_delta = amt
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

        return self.rm_reward_total

    def get_step_penalty_reward(self) -> float:
        """Apply a constant negative reward each step."""
        penalty = float(self.reward_config.get("step_penalty", 0.0))
        return -abs(penalty)

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
