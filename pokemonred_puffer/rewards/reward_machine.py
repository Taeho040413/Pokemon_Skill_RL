from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, Iterable, Protocol

from pokemonred_puffer.data.events import EventFlags
from pokemonred_puffer.data.items import Items
from pokemonred_puffer.data.tm_hm import TmHmMoves

# wTileInFrontOfPlayer — 필드 HM “지금 이 타일 앞에서 시도해야 하는가” (cut_hook / surf 등과 동일 기준).
CUTTABLE_TILES = frozenset({0x3D, 0x50})
SURF_TILE_IN_FRONT = 0x14
POKEFLUTE_TILE_IN_FRONT = 0x43


class HMTarget(IntEnum):
    CUT = 0
    SURF = 1
    FLASH = 2
    POKEFLUTE = 3
    NONE = 4


class RewardMachineState(IntEnum):
    IDLE = 0

    # Cut
    CUT_DETECTED = 1
    CUT_MENU_OPEN = 2
    CUT_MON_SELECTED = 3
    CUT_SUCCESS = 4

    # Surf
    SURF_DETECTED = 5
    SURF_MENU_OPEN = 6
    SURF_MON_SELECTED = 7
    SURF_SUCCESS = 8

    # Flash
    FLASH_DETECTED = 9
    FLASH_MENU_OPEN = 10
    FLASH_MON_SELECTED = 11
    FLASH_SUCCESS = 12

    # Pokeflute
    POKEFLUTE_DETECTED = 13
    POKEFLUTE_BAG_OPEN = 14
    POKEFLUTE_SUCCESS = 15

    FAILED = 16  # timeout


class RewardMachineEnv(Protocol):
    events: EventFlags
    auto_flash: bool
    valid_cut_coords: dict
    invalid_cut_coords: dict
    valid_pokeflute_coords: dict
    invalid_pokeflute_coords: dict
    valid_surf_coords: dict
    invalid_surf_coords: dict
    use_surf: int
    seen_start_menu: int
    seen_pokemon_menu: int
    seen_bag_menu: int
    step_count: int

    def check_if_party_has_hm(self, hm: int) -> bool: ...

    def get_items_in_bag(self) -> Iterable[Items]: ...

    def get_tile_in_front_of_player(self) -> int: ...


@dataclass(frozen=True)
class RewardMachineContext:
    step_count: int
    beat_brock: bool
    beat_misty: bool
    got_hm01: bool
    beat_lt_surge: bool
    got_hm05: bool
    beat_rocket_hideout_giovanni: bool
    got_pokeflute: bool
    beat_route12_snorlax: bool
    beat_route16_snorlax: bool
    got_hm03: bool
    beat_koga: bool
    has_cut: bool
    has_flash: bool
    has_surf: bool
    has_pokeflute: bool
    auto_flash: bool
    used_cut_successfully: bool
    # 컷 성공 횟수: RM이 이번 사이클 시작 시점과 비교해 새로운 성공인지 판단한다.
    # (에피소드 누적값이므로 증분만 의미 있음.)
    valid_cut_coords_count: int
    # 서핑 성공 횟수: per-cycle count 가드에 사용. is_surfing은 물 위 모든 스텝에서 True라서
    # 단독으로 쓰면 매 스텝 SUCCESS가 발화한다.
    valid_surf_coords_count: int
    used_pokeflute_successfully: bool
    # 포케플루트 성공 횟수: RewardMachine이 이번 RM 사이클 시작 시점과 비교해
    # "새로운" 사용이 있었는지 판단하는 데 쓴다. (에피소드 누적값이므로 증분만 의미 있음.)
    valid_pokeflute_coords_count: int
    used_surf_successfully: bool
    # 서핑 중이면 앞 타일이 물/0x14가 아닐 때가 많아 재무장만으로는 루프가 남음 → IDLE→SURF 차단에 사용.
    is_surfing: bool
    tile_in_front: int
    start_menu_open: bool
    pokemon_menu_open: bool
    bag_menu_open: bool
    invalid_cut_coords_count: int
    invalid_pokeflute_coords_count: int
    invalid_surf_coords_count: int

    @classmethod
    def from_env(cls, env: RewardMachineEnv) -> RewardMachineContext:
        items = set(env.get_items_in_bag())
        events = env.events

        return cls(
            step_count=env.step_count,
            beat_brock=events.get_event("EVENT_BEAT_BROCK"),
            beat_misty=events.get_event("EVENT_BEAT_MISTY"),
            got_hm01=events.get_event("EVENT_GOT_HM01"),
            beat_lt_surge=events.get_event("EVENT_BEAT_LT_SURGE"),
            got_hm05=Items.HM_05 in items,
            beat_rocket_hideout_giovanni=events.get_event(
                "EVENT_BEAT_ROCKET_HIDEOUT_GIOVANNI"
            ),
            got_pokeflute=events.get_event("EVENT_GOT_POKE_FLUTE"),
            beat_route12_snorlax=events.get_event("EVENT_BEAT_ROUTE12_SNORLAX"),
            beat_route16_snorlax=events.get_event("EVENT_BEAT_ROUTE16_SNORLAX"),
            got_hm03=events.get_event("EVENT_GOT_HM03"),
            beat_koga=events.get_event("EVENT_BEAT_KOGA"),
            has_cut=env.check_if_party_has_hm(TmHmMoves.CUT.value),
            has_flash=env.check_if_party_has_hm(TmHmMoves.FLASH.value),
            has_surf=env.check_if_party_has_hm(TmHmMoves.SURF.value),
            has_pokeflute=Items.POKE_FLUTE in items,
            auto_flash=env.auto_flash,
            used_cut_successfully=bool(env.valid_cut_coords),
            valid_cut_coords_count=len(env.valid_cut_coords),
            valid_surf_coords_count=len(env.valid_surf_coords),
            used_pokeflute_successfully=bool(env.valid_pokeflute_coords),
            valid_pokeflute_coords_count=len(env.valid_pokeflute_coords),
            used_surf_successfully=bool(env.valid_surf_coords) or bool(env.use_surf),
            is_surfing=bool(env.use_surf),
            tile_in_front=env.get_tile_in_front_of_player(),
            start_menu_open=bool(env.seen_start_menu),
            pokemon_menu_open=bool(env.seen_pokemon_menu),
            bag_menu_open=bool(env.seen_bag_menu),
            invalid_cut_coords_count=len(env.invalid_cut_coords),
            invalid_pokeflute_coords_count=len(env.invalid_pokeflute_coords),
            invalid_surf_coords_count=len(env.invalid_surf_coords),
        )

    @property
    def can_use_cut(self) -> bool:
        # baseline/reward-machine 보조 로직은 HM을 "가르치기"로 상태를 만들 수 있어
        # EVENT_GOT_HM01 / beat_misty 같은 진척 플래그에 의존하면
        # CUT_DETECTED에서 전이가 막힐 수 있습니다.
        # 따라서 학습/전이에는 실제로 CUT을 쓸 수 있는지(has_cut)만 봅니다.
        return self.has_cut

    @property
    def can_use_flash(self) -> bool:
        return self.has_flash and not self.auto_flash

    @property
    def can_use_pokeflute(self) -> bool:
        return self.has_pokeflute

    @property
    def can_use_surf(self) -> bool:
        return self.has_surf


@dataclass(frozen=True)
class RewardMachineTransition:
    source: RewardMachineState
    target: RewardMachineState
    reward_key: str
    condition: Callable[[RewardMachineContext], bool]


@dataclass(frozen=True)
class RewardMachineStep:
    previous_state: RewardMachineState
    current_state: RewardMachineState
    transition_key: str | None

    @property
    def changed(self) -> bool:
        return self.previous_state != self.current_state


# CUT / SURF / POKEFLUTE만 전이가 연결되어 있습니다. FLASH_* 상태·HMTarget·FAILED
# 타임아웃 집합은 HM 보조 신호·확장용으로 두었으며, README의 FLASH 체인과 동일하게
# 쓰려면 환경에서 flash 훅·좌표( cut/surf 와 동일 패턴)를 노출한 뒤 전이를 추가해야 합니다.
REWARD_MACHINE_TRANSITIONS: tuple[RewardMachineTransition, ...] = (
    # ── CUT ──────────────────────────────────────────────────────────────────
    RewardMachineTransition(
        RewardMachineState.IDLE,
        RewardMachineState.CUT_DETECTED,
        "rm_cut_detected",
        lambda ctx: ctx.tile_in_front in CUTTABLE_TILES and ctx.can_use_cut,
    ),
    # Skip 전이: RM이 DETECTED / MENU_OPEN 상태에서도 실제 컷이 발생하면 바로 SUCCESS로.
    # per-cycle count 가드는 _next_transition에서 처리한다(abort보다 반드시 먼저 검사).
    RewardMachineTransition(
        RewardMachineState.CUT_DETECTED,
        RewardMachineState.CUT_SUCCESS,
        "rm_cut_success",
        lambda ctx: ctx.used_cut_successfully and ctx.can_use_cut
        and ctx.tile_in_front not in CUTTABLE_TILES,
    ),
    RewardMachineTransition(
        RewardMachineState.CUT_MENU_OPEN,
        RewardMachineState.CUT_SUCCESS,
        "rm_cut_success",
        lambda ctx: ctx.used_cut_successfully and ctx.can_use_cut
        and ctx.tile_in_front not in CUTTABLE_TILES,
    ),
    # 정상 순서 전이
    RewardMachineTransition(
        RewardMachineState.CUT_DETECTED,
        RewardMachineState.CUT_MENU_OPEN,
        "rm_cut_menu_open",
        lambda ctx: ctx.tile_in_front in CUTTABLE_TILES
        and ctx.start_menu_open
        and ctx.can_use_cut,
    ),
    RewardMachineTransition(
        RewardMachineState.CUT_MENU_OPEN,
        RewardMachineState.CUT_MON_SELECTED,
        "rm_cut_mon_selected",
        lambda ctx: ctx.tile_in_front in CUTTABLE_TILES
        and ctx.pokemon_menu_open
        and ctx.can_use_cut,
    ),
    RewardMachineTransition(
        RewardMachineState.CUT_MON_SELECTED,
        RewardMachineState.CUT_SUCCESS,
        "rm_cut_success",
        lambda ctx: (
            ctx.used_cut_successfully
            and ctx.can_use_cut
            and ctx.tile_in_front not in CUTTABLE_TILES
        ),
    ),
    RewardMachineTransition(
        RewardMachineState.CUT_SUCCESS,
        RewardMachineState.IDLE,
        "rm_cut_done",
        lambda ctx: True,
    ),
    # Abort: skip보다 뒤에 두어야 "컷 성공 + 타일 변경" 시 SUCCESS가 abort보다 먼저 발화한다.
    RewardMachineTransition(
        RewardMachineState.CUT_DETECTED,
        RewardMachineState.IDLE,
        "rm_cut_aborted",
        lambda ctx: ctx.tile_in_front not in CUTTABLE_TILES or not ctx.can_use_cut,
    ),

    # ── SURF ─────────────────────────────────────────────────────────────────
    RewardMachineTransition(
        RewardMachineState.IDLE,
        RewardMachineState.SURF_DETECTED,
        "rm_surf_detected",
        lambda ctx: (
            ctx.tile_in_front == SURF_TILE_IN_FRONT
            and ctx.can_use_surf
            and not ctx.is_surfing
        ),
    ),
    # Skip 전이: 서핑이 시작됐으면 어느 중간 상태에서든 즉시 SUCCESS로.
    RewardMachineTransition(
        RewardMachineState.SURF_DETECTED,
        RewardMachineState.SURF_SUCCESS,
        "rm_surf_success",
        lambda ctx: (
            ctx.can_use_surf
            and ctx.is_surfing
            and ctx.tile_in_front != SURF_TILE_IN_FRONT
        ),
    ),
    RewardMachineTransition(
        RewardMachineState.SURF_MENU_OPEN,
        RewardMachineState.SURF_SUCCESS,
        "rm_surf_success",
        lambda ctx: (
            ctx.can_use_surf
            and ctx.is_surfing
            and ctx.tile_in_front != SURF_TILE_IN_FRONT
        ),
    ),
    # 정상 순서 전이
    RewardMachineTransition(
        RewardMachineState.SURF_DETECTED,
        RewardMachineState.SURF_MENU_OPEN,
        "rm_surf_menu_open",
        lambda ctx: ctx.tile_in_front == SURF_TILE_IN_FRONT
        and ctx.start_menu_open
        and ctx.can_use_surf,
    ),
    RewardMachineTransition(
        RewardMachineState.SURF_MENU_OPEN,
        RewardMachineState.SURF_MON_SELECTED,
        "rm_surf_mon_selected",
        lambda ctx: ctx.tile_in_front == SURF_TILE_IN_FRONT
        and ctx.pokemon_menu_open
        and ctx.can_use_surf,
    ),
    RewardMachineTransition(
        RewardMachineState.SURF_MON_SELECTED,
        RewardMachineState.SURF_SUCCESS,
        "rm_surf_success",
        lambda ctx: (
            ctx.can_use_surf
            and ctx.is_surfing
            and ctx.tile_in_front != SURF_TILE_IN_FRONT
        ),
    ),
    RewardMachineTransition(
        RewardMachineState.SURF_SUCCESS,
        RewardMachineState.IDLE,
        "rm_surf_done",
        lambda ctx: True,
    ),
    # Abort: skip보다 뒤에 위치.
    RewardMachineTransition(
        RewardMachineState.SURF_DETECTED,
        RewardMachineState.IDLE,
        "rm_surf_aborted",
        lambda ctx: ctx.tile_in_front != SURF_TILE_IN_FRONT or not ctx.can_use_surf,
    ),

    # ── POKEFLUTE ────────────────────────────────────────────────────────────
    RewardMachineTransition(
        RewardMachineState.IDLE,
        RewardMachineState.POKEFLUTE_DETECTED,
        "rm_pokeflute_detected",
        lambda ctx: ctx.tile_in_front == POKEFLUTE_TILE_IN_FRONT and ctx.can_use_pokeflute,
    ),
    # Skip 전이: BAG_OPEN을 거치지 않아도 실제 플루트 사용이 감지되면 바로 SUCCESS로.
    RewardMachineTransition(
        RewardMachineState.POKEFLUTE_DETECTED,
        RewardMachineState.POKEFLUTE_SUCCESS,
        "rm_pokeflute_success",
        lambda ctx: (
            ctx.used_pokeflute_successfully
            and ctx.can_use_pokeflute
            and ctx.tile_in_front != POKEFLUTE_TILE_IN_FRONT
        ),
    ),
    # 정상 순서 전이
    RewardMachineTransition(
        RewardMachineState.POKEFLUTE_DETECTED,
        RewardMachineState.POKEFLUTE_BAG_OPEN,
        "rm_pokeflute_bag_open",
        lambda ctx: ctx.tile_in_front == POKEFLUTE_TILE_IN_FRONT
        and ctx.bag_menu_open
        and ctx.can_use_pokeflute,
    ),
    RewardMachineTransition(
        RewardMachineState.POKEFLUTE_BAG_OPEN,
        RewardMachineState.POKEFLUTE_SUCCESS,
        "rm_pokeflute_success",
        lambda ctx: (
            ctx.used_pokeflute_successfully
            and ctx.can_use_pokeflute
            and ctx.tile_in_front != POKEFLUTE_TILE_IN_FRONT
        ),
    ),
    RewardMachineTransition(
        RewardMachineState.POKEFLUTE_SUCCESS,
        RewardMachineState.IDLE,
        "rm_pokeflute_done",
        lambda ctx: True,
    ),
    # Abort: skip보다 뒤에 위치.
    RewardMachineTransition(
        RewardMachineState.POKEFLUTE_DETECTED,
        RewardMachineState.IDLE,
        "rm_pokeflute_aborted",
        lambda ctx: (
            ctx.tile_in_front != POKEFLUTE_TILE_IN_FRONT or not ctx.can_use_pokeflute
        ),
    ),
)


HM_TARGET_BY_STATE: dict[RewardMachineState, HMTarget] = {
    RewardMachineState.IDLE: HMTarget.NONE,

    RewardMachineState.CUT_DETECTED: HMTarget.CUT,
    RewardMachineState.CUT_MENU_OPEN: HMTarget.CUT,
    RewardMachineState.CUT_MON_SELECTED: HMTarget.CUT,
    RewardMachineState.CUT_SUCCESS: HMTarget.CUT,

    RewardMachineState.SURF_DETECTED: HMTarget.SURF,
    RewardMachineState.SURF_MENU_OPEN: HMTarget.SURF,
    RewardMachineState.SURF_MON_SELECTED: HMTarget.SURF,
    RewardMachineState.SURF_SUCCESS: HMTarget.SURF,

    RewardMachineState.FLASH_DETECTED: HMTarget.FLASH,
    RewardMachineState.FLASH_MENU_OPEN: HMTarget.FLASH,
    RewardMachineState.FLASH_MON_SELECTED: HMTarget.FLASH,
    RewardMachineState.FLASH_SUCCESS: HMTarget.FLASH,

    RewardMachineState.POKEFLUTE_DETECTED: HMTarget.POKEFLUTE,
    RewardMachineState.POKEFLUTE_BAG_OPEN: HMTarget.POKEFLUTE,
    RewardMachineState.POKEFLUTE_SUCCESS: HMTarget.POKEFLUTE,

    RewardMachineState.FAILED: HMTarget.NONE,
}


class RewardMachine:
    def __init__(self, initial_state: RewardMachineState = RewardMachineState.IDLE):
        self.state = initial_state
        self.failed_after_steps = 256
        # FAILED 상태에 너무 오래 머물면 다시 IDLE로 복구합니다.
        self.failed_recovery_steps = 64
        # Menu/selection 상태에서 "잘못된 HM 시도"가 누적되면
        # timeout보다 먼저 FAILED로 빠지게 한다(이건 튜닝 포인트).
        self.failed_after_invalid_increases = 8
        self._last_step_count: int | None = None
        self._steps_in_state = 0
        self._invalid_increase_counter = 0

        self._last_invalid_cut_coords_count: int | None = None
        self._last_invalid_surf_coords_count: int | None = None
        self._last_invalid_pokeflute_coords_count: int | None = None

        # IDLE→*_DETECTED는 “상승 에지”만: 같은 타일 앞에 서 있는 동안 매 스텝 재진입 방지.
        # 조건에서 벗어났다가(다른 타일/방향) 다시 맞으면 True로 재무장.
        self._idle_cut_entry_ok = True
        self._idle_surf_entry_ok = True
        self._idle_pokeflute_entry_ok = True

        # CUT / POKEFLUTE 사이클 시작 시점의 valid_*_coords_count 스냅샷.
        # *_MON_SELECTED / *_BAG_OPEN → SUCCESS 조건:
        #   이 사이클에서 실제로 새 성공이 있어야 함 (에피소드 누적 True 방지).
        # used_*_successfully는 에피소드 전체에서 True로 유지되므로
        # tile 조건만으로 게이팅하면 메뉴 열림 중 tile이 일시 변경될 때 즉시 SUCCESS 발화 버그.
        self._cut_cycle_start_count: int = 0
        self._surf_cycle_start_count: int = 0
        self._pokeflute_cycle_start_count: int = 0


    @property
    def steps_in_state(self) -> int:
        return self._steps_in_state

    @property
    def state_id(self) -> int:
        return int(self.state)

    @property
    def hm_target(self) -> HMTarget:
        return HM_TARGET_BY_STATE[self.state]

    def reset(self) -> None:
        self.state = RewardMachineState.IDLE
        self._last_step_count = None
        self._steps_in_state = 0
        self._invalid_increase_counter = 0
        self._last_invalid_cut_coords_count = None
        self._last_invalid_surf_coords_count = None
        self._last_invalid_pokeflute_coords_count = None
        self._idle_cut_entry_ok = True
        self._idle_surf_entry_ok = True
        self._idle_pokeflute_entry_ok = True
        self._cut_cycle_start_count = 0
        self._surf_cycle_start_count = 0
        self._pokeflute_cycle_start_count = 0

    def _rearm_idle_detect_entry(self, context: RewardMachineContext) -> None:
        if context.tile_in_front not in CUTTABLE_TILES or not context.can_use_cut:
            self._idle_cut_entry_ok = True
        # 물 위에서는 앞 타일이 0x14가 아닌 프레임이 잦아 `tile != SURF`만으로 재무장하면
        # SURF_SUCCESS→IDLE 직후 다시 SURF_DETECTED로 들어가는 루프가 생김.
        if context.is_surfing:
            self._idle_surf_entry_ok = False
        elif context.tile_in_front != SURF_TILE_IN_FRONT or not context.can_use_surf:
            self._idle_surf_entry_ok = True
        if (
            context.tile_in_front != POKEFLUTE_TILE_IN_FRONT
            or not context.can_use_pokeflute
        ):
            self._idle_pokeflute_entry_ok = True

    def transition(self, context: RewardMachineContext) -> RewardMachineStep:
        previous_state = self.state
        self._update_state_duration(context.step_count)
        self._rearm_idle_detect_entry(context)
        self._update_invalid_increase_counter(context)

        # (a) FAILED 복구
        if self.state == RewardMachineState.FAILED and self.steps_in_state >= self.failed_recovery_steps:
            self.state = RewardMachineState.IDLE
            self._steps_in_state = 0
            self._invalid_increase_counter = 0
            self._idle_cut_entry_ok = True
            self._idle_surf_entry_ok = True
            self._idle_pokeflute_entry_ok = True
            self._cut_cycle_start_count = 0
            self._surf_cycle_start_count = 0
            self._pokeflute_cycle_start_count = 0
            return RewardMachineStep(previous_state, self.state, None)

        if (
            self.state
            in {
                RewardMachineState.CUT_MENU_OPEN,
                RewardMachineState.CUT_MON_SELECTED,
                RewardMachineState.SURF_MENU_OPEN,
                RewardMachineState.SURF_MON_SELECTED,
                RewardMachineState.FLASH_MENU_OPEN,
                RewardMachineState.FLASH_MON_SELECTED,
                RewardMachineState.POKEFLUTE_BAG_OPEN,
            }
            and (
                self.steps_in_state >= self.failed_after_steps
                or self._invalid_increase_counter >= self.failed_after_invalid_increases
            )
        ):
            self.state = RewardMachineState.FAILED
            # FAILED로 들어간 직후에는 복구 카운트가 즉시 발동하지 않게 초기화합니다.
            self._steps_in_state = 0
            self._invalid_increase_counter = 0
            # FAILED는 HM supervision/타겟에서 NONE으로 매핑되므로
            # 학습을 더 오염시키지 않기 위해 transition_key만 제공.
            return RewardMachineStep(previous_state, self.state, "rm_failed_timeout")

        transition = self._next_transition(context)
        if transition is None:
            return RewardMachineStep(previous_state, self.state, None)

        self.state = transition.target
        if self.state != previous_state:
            self._steps_in_state = 0
            self._invalid_increase_counter = 0
            if previous_state == RewardMachineState.IDLE:
                if self.state == RewardMachineState.CUT_DETECTED:
                    self._idle_cut_entry_ok = False
                    self._cut_cycle_start_count = context.valid_cut_coords_count
                elif self.state == RewardMachineState.SURF_DETECTED:
                    self._idle_surf_entry_ok = False
                    self._surf_cycle_start_count = context.valid_surf_coords_count
                elif self.state == RewardMachineState.POKEFLUTE_DETECTED:
                    self._idle_pokeflute_entry_ok = False
                    # 이 사이클에서 새로운 flute 사용이 있는지 판단하기 위해 현재 count를 기록.
                    self._pokeflute_cycle_start_count = context.valid_pokeflute_coords_count
        return RewardMachineStep(previous_state, self.state, transition.reward_key)

    def _update_state_duration(self, step_count: int) -> None:
        if self._last_step_count is None:
            self._last_step_count = step_count
            self._steps_in_state = 0
            return

        if step_count != self._last_step_count:
            self._steps_in_state += step_count - self._last_step_count
            self._last_step_count = step_count

    def _update_invalid_increase_counter(self, context: RewardMachineContext) -> None:
        # First observation initializes baselines.
        if (
            self._last_invalid_cut_coords_count is None
            or self._last_invalid_surf_coords_count is None
            or self._last_invalid_pokeflute_coords_count is None
        ):
            self._last_invalid_cut_coords_count = context.invalid_cut_coords_count
            self._last_invalid_surf_coords_count = context.invalid_surf_coords_count
            self._last_invalid_pokeflute_coords_count = context.invalid_pokeflute_coords_count
            self._invalid_increase_counter = 0
            return

        cut_delta = (
            context.invalid_cut_coords_count - self._last_invalid_cut_coords_count
        )
        surf_delta = (
            context.invalid_surf_coords_count - self._last_invalid_surf_coords_count
        )
        pokeflute_delta = (
            context.invalid_pokeflute_coords_count - self._last_invalid_pokeflute_coords_count
        )

        # Update baselines every step; we only count deltas while the RM is
        # stuck in the corresponding menu/selection states.
        self._last_invalid_cut_coords_count = context.invalid_cut_coords_count
        self._last_invalid_surf_coords_count = context.invalid_surf_coords_count
        self._last_invalid_pokeflute_coords_count = context.invalid_pokeflute_coords_count

        # If we've already "succeeded" for the current HM stage, don't count invalids.
        if self.state in {RewardMachineState.CUT_MENU_OPEN, RewardMachineState.CUT_MON_SELECTED}:
            if context.used_cut_successfully:
                self._invalid_increase_counter = 0
            elif cut_delta > 0:
                self._invalid_increase_counter += int(cut_delta)
        elif self.state in {RewardMachineState.SURF_MENU_OPEN, RewardMachineState.SURF_MON_SELECTED}:
            if context.used_surf_successfully:
                self._invalid_increase_counter = 0
            elif surf_delta > 0:
                self._invalid_increase_counter += int(surf_delta)
        elif self.state in {RewardMachineState.POKEFLUTE_BAG_OPEN}:
            if context.used_pokeflute_successfully:
                self._invalid_increase_counter = 0
            elif pokeflute_delta > 0:
                self._invalid_increase_counter += int(pokeflute_delta)

    def _next_transition(
        self, context: RewardMachineContext
    ) -> RewardMachineTransition | None:
        for transition in REWARD_MACHINE_TRANSITIONS:
            if transition.source != self.state:
                continue
            if self.state == RewardMachineState.IDLE:
                if (
                    transition.target == RewardMachineState.CUT_DETECTED
                    and not self._idle_cut_entry_ok
                ):
                    continue
                if (
                    transition.target == RewardMachineState.SURF_DETECTED
                    and not self._idle_surf_entry_ok
                ):
                    continue
                if (
                    transition.target == RewardMachineState.POKEFLUTE_DETECTED
                    and not self._idle_pokeflute_entry_ok
                ):
                    continue
            # → CUT_SUCCESS: 어느 상태에서든 이 사이클에서 실제 새 컷이 있어야 함.
            # skip 전이(DETECTED/MENU_OPEN→SUCCESS)도 포함해 동일 가드 적용.
            if (
                transition.target == RewardMachineState.CUT_SUCCESS
                and context.valid_cut_coords_count <= self._cut_cycle_start_count
            ):
                continue
            # → SURF_SUCCESS: is_surfing은 물 위 매 스텝 True라서 단독으로 쓰면 폭주.
            # valid_surf_coords_count 증분(실제 surf_hook 발화)이 있어야만 SUCCESS 허용.
            if (
                transition.target == RewardMachineState.SURF_SUCCESS
                and context.valid_surf_coords_count <= self._surf_cycle_start_count
            ):
                continue
            # → POKEFLUTE_SUCCESS: 이 사이클에서 실제로 새 flute 사용이 있어야 함.
            # skip 전이(DETECTED→SUCCESS)도 포함.
            if (
                transition.target == RewardMachineState.POKEFLUTE_SUCCESS
                and context.valid_pokeflute_coords_count <= self._pokeflute_cycle_start_count
            ):
                continue
            if transition.condition(context):
                return transition
        return None
