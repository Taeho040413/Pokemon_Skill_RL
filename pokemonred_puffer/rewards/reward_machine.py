from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, Iterable, Protocol

from pokemonred_puffer.data.events import EventFlags
from pokemonred_puffer.data.items import Items
from pokemonred_puffer.data.tm_hm import TmHmMoves

# tile_in_front = env.get_tile_in_front_of_player() → WRAM wTileInFrontOfPlayer (cut_hook과 동일).
CUTTABLE_TILES = frozenset({0x3D, 0x50})
SURF_TILE_IN_FRONT = 0x14
# cut_if_next / surf_if_attempt: 스타트 메뉴에서 포켓몬 줄 = wCurrentMenuItem 1.
START_MENU_POKEMON_CURSOR = 1
# pokered: 어두운 동굴 맵에서 wMapPalOffset == 6 (환경의 auto_flash와 동일 기준).
DARK_CAVE_MAP_PAL_OFFSET = 6


def _surf_menu_progress_ok(ctx: RewardMachineContext) -> bool:
    """메뉴 체인 진행: Cut은 앞 타일이 나무인 동안만, Surf는 파티에 기술만 있으면 됨."""
    return ctx.can_use_surf


def _surf_abort_from_detected(ctx: RewardMachineContext) -> bool:
    if not ctx.can_use_surf:
        return True
    if ctx.start_menu_open:
        return False
    return not ctx.surf_water_context_ok


def _surf_abort_from_menu_state(ctx: RewardMachineContext) -> bool:
    if ctx.is_surfing:
        return False
    if not ctx.can_use_surf:
        return True
    if ctx.start_menu_open or ctx.pokemon_menu_open or ctx.field_move_menu_open:
        return False
    return not ctx.surf_water_context_ok


class HMTarget(IntEnum):
    CUT = 0
    SURF = 1
    FLASH = 2
    NONE = 3


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

    # wCurrentMenuItem·필드기술 메뉴 훅에 맞춘 매크로 하위 상태.
    CUT_START_MENU = 13
    CUT_PARTY_MENU = 14
    SURF_START_MENU = 15
    SURF_PARTY_MENU = 16
    FLASH_START_MENU = 17
    FLASH_PARTY_MENU = 18

    FAILED = 19  # timeout


def hm_supervision_label_from_rm_state(state_id: int) -> int:
    """obs에 hm_aux_label이 없을 때(레거시) HM 보조 CE용 라벨을 rm_state로부터 유도한다."""
    try:
        s = RewardMachineState(state_id)
    except ValueError:
        return int(HMTarget.NONE)
    if s in (
        RewardMachineState.CUT_DETECTED,
        RewardMachineState.CUT_START_MENU,
        RewardMachineState.CUT_MENU_OPEN,
        RewardMachineState.CUT_PARTY_MENU,
        RewardMachineState.CUT_MON_SELECTED,
    ):
        return int(HMTarget.CUT)
    if s in (
        RewardMachineState.SURF_DETECTED,
        RewardMachineState.SURF_START_MENU,
        RewardMachineState.SURF_MENU_OPEN,
        RewardMachineState.SURF_PARTY_MENU,
        RewardMachineState.SURF_MON_SELECTED,
    ):
        return int(HMTarget.SURF)
    if s in (
        RewardMachineState.FLASH_DETECTED,
        RewardMachineState.FLASH_START_MENU,
        RewardMachineState.FLASH_MENU_OPEN,
        RewardMachineState.FLASH_PARTY_MENU,
        RewardMachineState.FLASH_MON_SELECTED,
    ):
        return int(HMTarget.FLASH)
    return int(HMTarget.NONE)


class RewardMachineEnv(Protocol):
    events: EventFlags
    auto_flash: bool
    valid_cut_coords: dict
    invalid_cut_coords: dict
    valid_surf_coords: dict
    invalid_surf_coords: dict
    valid_flash_coords: dict
    invalid_flash_coords: dict
    use_surf: int
    seen_start_menu: int
    seen_pokemon_menu: int
    step_count: int

    def check_if_party_has_hm(self, hm: int) -> bool: ...

    def get_items_in_bag(self) -> Iterable[Items]: ...

    def get_tile_in_front_of_player(self) -> int: ...

    def get_map_pal_offset(self) -> int: ...

    def get_rm_flash_cycle_start(self) -> int: ...

    def player_faces_adjacent_water(self) -> bool: ...


@dataclass(frozen=True)
class RewardMachineContext:
    step_count: int
    beat_brock: bool
    beat_misty: bool
    got_hm01: bool
    beat_lt_surge: bool
    got_hm05: bool
    beat_rocket_hideout_giovanni: bool
    beat_route12_snorlax: bool
    beat_route16_snorlax: bool
    got_hm03: bool
    beat_koga: bool
    has_cut: bool
    has_flash: bool
    has_surf: bool
    auto_flash: bool
    used_cut_successfully: bool
    valid_cut_coords_count: int
    # 이번 에이전트 스텝에서 cut_hook 등으로 새로 추가된 성공 횟수 (한 스텝에 메뉴+컷 완료 시 RM용).
    valid_cut_coords_delta: int
    # 서핑: per-cycle 가드는 훅 발화 횟수 기준 (valid_surf_coords는 고유 좌표만이라 재서핑 시 len 불변).
    valid_surf_coords_count: int
    valid_surf_coords_delta: int
    surf_hook_success_count: int
    surf_hook_success_delta: int
    valid_flash_coords_count: int
    valid_flash_coords_delta: int
    used_surf_successfully: bool
    # 서핑 중이면 앞 타일이 물/0x14가 아닐 때가 많아 재무장만으로는 루프가 남음 → IDLE→SURF 차단에 사용.
    is_surfing: bool
    tile_in_front: int
    # wTileMap 기준 바라보는 방향에 물(0x14)이 있음 (정면 wTileInFrontOfPlayer와 다를 수 있음).
    faces_adjacent_water: bool
    start_menu_open: bool
    pokemon_menu_open: bool
    field_move_menu_open: bool
    current_menu_item: int
    invalid_cut_coords_count: int
    invalid_surf_coords_count: int
    invalid_flash_coords_count: int
    # 어두운 동굴 여부 (Flash 필요 맵). Flash 성공 직후에는 False가 된다.
    in_dark_cave: bool
    # 이번 FLASH 사이클(DETECTED 진입 시점) 이후 새 Flash 성공 훅이 있었는지.
    flash_cycle_has_new_success: bool

    @classmethod
    def from_env(cls, env: RewardMachineEnv) -> RewardMachineContext:
        items = set(env.get_items_in_bag())
        events = env.events
        _flash_start = int(env.get_rm_flash_cycle_start())

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
            beat_route12_snorlax=events.get_event("EVENT_BEAT_ROUTE12_SNORLAX"),
            beat_route16_snorlax=events.get_event("EVENT_BEAT_ROUTE16_SNORLAX"),
            got_hm03=events.get_event("EVENT_GOT_HM03"),
            beat_koga=events.get_event("EVENT_BEAT_KOGA"),
            has_cut=env.check_if_party_has_hm(TmHmMoves.CUT.value),
            has_flash=env.check_if_party_has_hm(TmHmMoves.FLASH.value),
            has_surf=env.check_if_party_has_hm(TmHmMoves.SURF.value),
            auto_flash=env.auto_flash,
            used_cut_successfully=bool(env.valid_cut_coords),
            valid_cut_coords_count=len(env.valid_cut_coords),
            valid_cut_coords_delta=int(getattr(env, "_rm_valid_cut_delta", 0)),
            valid_surf_coords_count=len(env.valid_surf_coords),
            valid_surf_coords_delta=int(getattr(env, "_rm_valid_surf_delta", 0)),
            surf_hook_success_count=int(getattr(env, "_surf_hook_success_count", 0)),
            surf_hook_success_delta=int(getattr(env, "_rm_valid_surf_delta", 0)),
            valid_flash_coords_count=len(env.valid_flash_coords),
            valid_flash_coords_delta=int(getattr(env, "_rm_valid_flash_delta", 0)),
            used_surf_successfully=bool(env.valid_surf_coords) or bool(env.use_surf),
            is_surfing=bool(env.use_surf),
            tile_in_front=env.get_tile_in_front_of_player(),
            faces_adjacent_water=bool(
                getattr(env, "player_faces_adjacent_water", lambda: False)()
            ),
            start_menu_open=(
                bool(env._start_menu_open)
                if hasattr(env, "_start_menu_open")
                else bool(getattr(env, "seen_start_menu", 0))
            ),
            pokemon_menu_open=bool(env.seen_pokemon_menu),
            field_move_menu_open=bool(getattr(env, "seen_field_move_menu", 0)),
            current_menu_item=int(getattr(env, "get_current_menu_item", lambda: 0)()),
            invalid_cut_coords_count=len(env.invalid_cut_coords),
            invalid_surf_coords_count=len(env.invalid_surf_coords),
            invalid_flash_coords_count=len(env.invalid_flash_coords),
            # 화면 팔레트가 기본(0)이 아니면 어두운 화면으로 간주한다.
            in_dark_cave=env.get_map_pal_offset() != 0,
            flash_cycle_has_new_success=len(env.valid_flash_coords) > _flash_start,
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
    def can_use_surf(self) -> bool:
        return self.has_surf

    @property
    def surf_detect_ok(self) -> bool:
        """IDLE→SURF_DETECTED: 정면 물 타일만 (옆 물만으로는 감지·라벨 오탐 방지)."""
        return self.tile_in_front == SURF_TILE_IN_FRONT

    @property
    def surf_water_context_ok(self) -> bool:
        """메뉴 전이·abort: 정면 물 또는 facing 방향 인접 물."""
        return self.surf_detect_ok or self.faces_adjacent_water


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


REWARD_MACHINE_TRANSITIONS: tuple[RewardMachineTransition, ...] = (
    # ── CUT ──────────────────────────────────────────────────────────────────
    # 한 PyBoy 스텝(에이전트 action 1회) 안에서 메뉴→컷까지 끝나면 최종 스냅샷만 남아
    # 앞 타일이 이미 non-cuttable → IDLE→CUT_DETECTED 불가. 증분으로 SUCCESS 직행.
    RewardMachineTransition(
        RewardMachineState.IDLE,
        RewardMachineState.CUT_SUCCESS,
        "rm_cut_success",
        lambda ctx: (
            ctx.can_use_cut
            and ctx.valid_cut_coords_delta > 0
            and ctx.tile_in_front not in CUTTABLE_TILES
        ),
    ),
    RewardMachineTransition(
        RewardMachineState.IDLE,
        RewardMachineState.CUT_DETECTED,
        "rm_cut_detected",
        lambda ctx: ctx.tile_in_front in CUTTABLE_TILES and ctx.can_use_cut,
    ),
    # 정상 순서 전이
    RewardMachineTransition(
        RewardMachineState.CUT_DETECTED,
        RewardMachineState.CUT_START_MENU,
        "rm_cut_start_menu",
        lambda ctx: (
            ctx.tile_in_front in CUTTABLE_TILES
            and ctx.can_use_cut
            and ctx.start_menu_open
            and ctx.current_menu_item != START_MENU_POKEMON_CURSOR
        ),
    ),
    RewardMachineTransition(
        RewardMachineState.CUT_DETECTED,
        RewardMachineState.CUT_MENU_OPEN,
        "rm_cut_menu_open",
        lambda ctx: (
            ctx.tile_in_front in CUTTABLE_TILES
            and ctx.can_use_cut
            and ctx.start_menu_open
            and ctx.current_menu_item == START_MENU_POKEMON_CURSOR
        ),
    ),
    RewardMachineTransition(
        RewardMachineState.CUT_START_MENU,
        RewardMachineState.CUT_MENU_OPEN,
        "rm_cut_pokemon_row",
        lambda ctx: (
            ctx.tile_in_front in CUTTABLE_TILES
            and ctx.can_use_cut
            and ctx.start_menu_open
            and ctx.current_menu_item == START_MENU_POKEMON_CURSOR
        ),
    ),
    RewardMachineTransition(
        RewardMachineState.CUT_START_MENU,
        RewardMachineState.IDLE,
        "rm_cut_aborted",
        lambda ctx: ctx.tile_in_front not in CUTTABLE_TILES or not ctx.can_use_cut,
    ),
    # auto_use_cut 등: 한 스텝 끝에 메뉴가 닫히고 나무도 제거된 스냅샷만 남으면
    # CUT_DETECTED → ABORT만 반복되어 RM 보상이 0이 된다. valid_cut 증분으로 가드.
    RewardMachineTransition(
        RewardMachineState.CUT_DETECTED,
        RewardMachineState.CUT_SUCCESS,
        "rm_cut_success",
        lambda ctx: ctx.can_use_cut and ctx.tile_in_front not in CUTTABLE_TILES,
    ),
    # 한 스텝 스냅샷에 포켓몬·필드기술 메뉴 플래그가 동시에 있으면 PARTY를 건너뛴다.
    RewardMachineTransition(
        RewardMachineState.CUT_MENU_OPEN,
        RewardMachineState.CUT_MON_SELECTED,
        "rm_cut_mon_selected",
        lambda ctx: (
            ctx.tile_in_front in CUTTABLE_TILES
            and ctx.can_use_cut
            and ctx.pokemon_menu_open
            and ctx.field_move_menu_open
        ),
    ),
    RewardMachineTransition(
        RewardMachineState.CUT_MENU_OPEN,
        RewardMachineState.CUT_PARTY_MENU,
        "rm_cut_party_menu",
        lambda ctx: (
            ctx.tile_in_front in CUTTABLE_TILES
            and ctx.can_use_cut
            and ctx.pokemon_menu_open
            and not ctx.field_move_menu_open
        ),
    ),
    RewardMachineTransition(
        RewardMachineState.CUT_PARTY_MENU,
        RewardMachineState.CUT_MON_SELECTED,
        "rm_cut_mon_selected",
        lambda ctx: (
            ctx.tile_in_front in CUTTABLE_TILES
            and ctx.can_use_cut
            and ctx.field_move_menu_open
        ),
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
    # Abort
    RewardMachineTransition(
        RewardMachineState.CUT_DETECTED,
        RewardMachineState.IDLE,
        "rm_cut_aborted",
        lambda ctx: ctx.tile_in_front not in CUTTABLE_TILES or not ctx.can_use_cut,
    ),

    # ── SURF ─────────────────────────────────────────────────────────────────
    # (1) IDLE/DETECTED/MENU 등에서 is_surfing·훅으로 SUCCESS
    # (2) 에이전트가 메뉴를 연 경우 START_MENU → … → SUCCESS 체인
    RewardMachineTransition(
        RewardMachineState.IDLE,
        RewardMachineState.SURF_SUCCESS,
        "rm_surf_success",
        lambda ctx: (
            ctx.can_use_surf
            and ctx.surf_hook_success_delta > 0
            and ctx.is_surfing
        ),
    ),
    RewardMachineTransition(
        RewardMachineState.IDLE,
        RewardMachineState.SURF_DETECTED,
        "rm_surf_detected",
        lambda ctx: (
            ctx.surf_detect_ok
            and ctx.can_use_surf
            and not ctx.is_surfing
        ),
    ),
    # 정상 순서 전이
    RewardMachineTransition(
        RewardMachineState.SURF_DETECTED,
        RewardMachineState.SURF_START_MENU,
        "rm_surf_start_menu",
        lambda ctx: (
            _surf_menu_progress_ok(ctx)
            and ctx.start_menu_open
            and ctx.current_menu_item != START_MENU_POKEMON_CURSOR
        ),
    ),
    RewardMachineTransition(
        RewardMachineState.SURF_DETECTED,
        RewardMachineState.SURF_MENU_OPEN,
        "rm_surf_menu_open",
        lambda ctx: (
            _surf_menu_progress_ok(ctx)
            and (
                (ctx.start_menu_open and ctx.current_menu_item == START_MENU_POKEMON_CURSOR)
                or ctx.pokemon_menu_open
            )
        ),
    ),
    RewardMachineTransition(
        RewardMachineState.SURF_START_MENU,
        RewardMachineState.SURF_PARTY_MENU,
        "rm_surf_party_menu",
        lambda ctx: (
            _surf_menu_progress_ok(ctx)
            and ctx.pokemon_menu_open
            and not ctx.field_move_menu_open
        ),
    ),
    RewardMachineTransition(
        RewardMachineState.SURF_START_MENU,
        RewardMachineState.SURF_MENU_OPEN,
        "rm_surf_pokemon_row",
        lambda ctx: (
            _surf_menu_progress_ok(ctx)
            and (
                (ctx.start_menu_open and ctx.current_menu_item == START_MENU_POKEMON_CURSOR)
                or ctx.pokemon_menu_open
            )
        ),
    ),
    RewardMachineTransition(
        RewardMachineState.SURF_START_MENU,
        RewardMachineState.IDLE,
        "rm_surf_aborted",
        _surf_abort_from_detected,
    ),
    RewardMachineTransition(
        RewardMachineState.SURF_DETECTED,
        RewardMachineState.SURF_SUCCESS,
        "rm_surf_success",
        lambda ctx: ctx.can_use_surf and ctx.is_surfing,
    ),
    RewardMachineTransition(
        RewardMachineState.SURF_MENU_OPEN,
        RewardMachineState.SURF_SUCCESS,
        "rm_surf_success",
        lambda ctx: ctx.can_use_surf and ctx.is_surfing,
    ),
    RewardMachineTransition(
        RewardMachineState.SURF_PARTY_MENU,
        RewardMachineState.SURF_SUCCESS,
        "rm_surf_success",
        lambda ctx: ctx.can_use_surf and ctx.is_surfing,
    ),
    RewardMachineTransition(
        RewardMachineState.SURF_MENU_OPEN,
        RewardMachineState.IDLE,
        "rm_surf_aborted",
        _surf_abort_from_menu_state,
    ),
    RewardMachineTransition(
        RewardMachineState.SURF_PARTY_MENU,
        RewardMachineState.IDLE,
        "rm_surf_aborted",
        _surf_abort_from_menu_state,
    ),
    RewardMachineTransition(
        RewardMachineState.SURF_MENU_OPEN,
        RewardMachineState.SURF_MON_SELECTED,
        "rm_surf_mon_selected",
        lambda ctx: (
            _surf_menu_progress_ok(ctx)
            and ctx.pokemon_menu_open
            and ctx.field_move_menu_open
        ),
    ),
    RewardMachineTransition(
        RewardMachineState.SURF_MENU_OPEN,
        RewardMachineState.SURF_PARTY_MENU,
        "rm_surf_party_menu",
        lambda ctx: (
            _surf_menu_progress_ok(ctx)
            and ctx.pokemon_menu_open
            and not ctx.field_move_menu_open
        ),
    ),
    RewardMachineTransition(
        RewardMachineState.SURF_PARTY_MENU,
        RewardMachineState.SURF_MON_SELECTED,
        "rm_surf_mon_selected",
        lambda ctx: _surf_menu_progress_ok(ctx) and ctx.field_move_menu_open,
    ),
    RewardMachineTransition(
        RewardMachineState.SURF_MON_SELECTED,
        RewardMachineState.IDLE,
        "rm_surf_aborted",
        _surf_abort_from_menu_state,
    ),
    RewardMachineTransition(
        RewardMachineState.SURF_MON_SELECTED,
        RewardMachineState.SURF_SUCCESS,
        "rm_surf_success",
        # 물 위에서는 앞 타일이 0x14인 프레임이 대부분이라 tile 조건을 두면 성공 전이가
        # 거의 안 난다. valid_surf_coords_count 증분(_next_transition)으로 한 번만 지급.
        lambda ctx: ctx.can_use_surf and ctx.is_surfing,
    ),
    RewardMachineTransition(
        RewardMachineState.SURF_SUCCESS,
        RewardMachineState.IDLE,
        "rm_surf_done",
        lambda ctx: True,
    ),
    # Abort
    RewardMachineTransition(
        RewardMachineState.SURF_DETECTED,
        RewardMachineState.IDLE,
        "rm_surf_aborted",
        _surf_abort_from_detected,
    ),

    # ── FLASH (어두운 동굴 wMapPalOffset==6; 훅: StartMenu_Pokemon.flash) ─────
    # IDLE 순서: CUT → SURF → FLASH
    RewardMachineTransition(
        RewardMachineState.IDLE,
        RewardMachineState.FLASH_DETECTED,
        "rm_flash_detected",
        lambda ctx: ctx.in_dark_cave and ctx.can_use_flash,
    ),
    RewardMachineTransition(
        RewardMachineState.FLASH_DETECTED,
        RewardMachineState.FLASH_START_MENU,
        "rm_flash_start_menu",
        lambda ctx: (
            ctx.in_dark_cave
            and ctx.can_use_flash
            and ctx.start_menu_open
            and ctx.current_menu_item != START_MENU_POKEMON_CURSOR
        ),
    ),
    RewardMachineTransition(
        RewardMachineState.FLASH_DETECTED,
        RewardMachineState.FLASH_MENU_OPEN,
        "rm_flash_menu_open",
        lambda ctx: (
            ctx.in_dark_cave
            and ctx.can_use_flash
            and ctx.start_menu_open
            and ctx.current_menu_item == START_MENU_POKEMON_CURSOR
        ),
    ),
    RewardMachineTransition(
        RewardMachineState.FLASH_START_MENU,
        RewardMachineState.FLASH_MENU_OPEN,
        "rm_flash_pokemon_row",
        lambda ctx: (
            ctx.in_dark_cave
            and ctx.can_use_flash
            and ctx.start_menu_open
            and ctx.current_menu_item == START_MENU_POKEMON_CURSOR
        ),
    ),
    RewardMachineTransition(
        RewardMachineState.FLASH_START_MENU,
        RewardMachineState.IDLE,
        "rm_flash_aborted",
        lambda ctx: not ctx.in_dark_cave or not ctx.can_use_flash,
    ),
    # auto_flash off: 플래시 연출 후 밝은 스냅샷만 남으면 MON 경로 없이 성공 처리.
    # rm_flash_aborted(밝음)와 경합 → ABORT보다 먼저 평가되게 둔다.
    RewardMachineTransition(
        RewardMachineState.FLASH_DETECTED,
        RewardMachineState.FLASH_SUCCESS,
        "rm_flash_success",
        lambda ctx: (
            ctx.can_use_flash
            and not ctx.in_dark_cave
            and ctx.flash_cycle_has_new_success
        ),
    ),
    RewardMachineTransition(
        RewardMachineState.FLASH_MENU_OPEN,
        RewardMachineState.FLASH_MON_SELECTED,
        "rm_flash_mon_selected",
        lambda ctx: (
            ctx.in_dark_cave
            and ctx.can_use_flash
            and ctx.pokemon_menu_open
            and ctx.field_move_menu_open
        ),
    ),
    RewardMachineTransition(
        RewardMachineState.FLASH_MENU_OPEN,
        RewardMachineState.FLASH_PARTY_MENU,
        "rm_flash_party_menu",
        lambda ctx: (
            ctx.in_dark_cave
            and ctx.can_use_flash
            and ctx.pokemon_menu_open
            and not ctx.field_move_menu_open
        ),
    ),
    RewardMachineTransition(
        RewardMachineState.FLASH_PARTY_MENU,
        RewardMachineState.FLASH_MON_SELECTED,
        "rm_flash_mon_selected",
        lambda ctx: (
            ctx.in_dark_cave and ctx.can_use_flash and ctx.field_move_menu_open
        ),
    ),
    RewardMachineTransition(
        RewardMachineState.FLASH_MON_SELECTED,
        RewardMachineState.FLASH_SUCCESS,
        "rm_flash_success",
        lambda ctx: (
            ctx.can_use_flash
            and not ctx.in_dark_cave
            and ctx.flash_cycle_has_new_success
        ),
    ),
    RewardMachineTransition(
        RewardMachineState.FLASH_MON_SELECTED,
        RewardMachineState.IDLE,
        "rm_flash_left_dark",
        lambda ctx: not ctx.in_dark_cave and not ctx.flash_cycle_has_new_success,
    ),
    RewardMachineTransition(
        RewardMachineState.FLASH_SUCCESS,
        RewardMachineState.IDLE,
        "rm_flash_done",
        lambda ctx: True,
    ),
    RewardMachineTransition(
        RewardMachineState.FLASH_DETECTED,
        RewardMachineState.IDLE,
        "rm_flash_aborted",
        lambda ctx: not ctx.in_dark_cave or not ctx.can_use_flash,
    ),
)


HM_TARGET_BY_STATE: dict[RewardMachineState, HMTarget] = {
    RewardMachineState.IDLE: HMTarget.NONE,

    RewardMachineState.CUT_DETECTED: HMTarget.CUT,
    RewardMachineState.CUT_START_MENU: HMTarget.CUT,
    RewardMachineState.CUT_MENU_OPEN: HMTarget.CUT,
    RewardMachineState.CUT_PARTY_MENU: HMTarget.CUT,
    RewardMachineState.CUT_MON_SELECTED: HMTarget.CUT,
    RewardMachineState.CUT_SUCCESS: HMTarget.CUT,

    RewardMachineState.SURF_DETECTED: HMTarget.SURF,
    RewardMachineState.SURF_START_MENU: HMTarget.SURF,
    RewardMachineState.SURF_MENU_OPEN: HMTarget.SURF,
    RewardMachineState.SURF_PARTY_MENU: HMTarget.SURF,
    RewardMachineState.SURF_MON_SELECTED: HMTarget.SURF,
    RewardMachineState.SURF_SUCCESS: HMTarget.SURF,

    RewardMachineState.FLASH_DETECTED: HMTarget.FLASH,
    RewardMachineState.FLASH_START_MENU: HMTarget.FLASH,
    RewardMachineState.FLASH_MENU_OPEN: HMTarget.FLASH,
    RewardMachineState.FLASH_PARTY_MENU: HMTarget.FLASH,
    RewardMachineState.FLASH_MON_SELECTED: HMTarget.FLASH,
    RewardMachineState.FLASH_SUCCESS: HMTarget.FLASH,

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
        self._last_invalid_flash_coords_count: int | None = None

        # IDLE→*_DETECTED는 “상승 에지”만: 같은 타일 앞에 서 있는 동안 매 스텝 재진입 방지.
        # 조건에서 벗어났다가(다른 타일/방향) 다시 맞으면 True로 재무장.
        self._idle_cut_entry_ok = True
        self._idle_surf_entry_ok = True
        self._idle_flash_entry_ok = True

        # HM 사이클 시작 시점의 valid_*_coords_count 스냅샷 (CUT/SURF/FLASH).
        # *_MON_SELECTED / *_BAG_OPEN → SUCCESS 조건:
        #   이 사이클에서 실제로 새 성공이 있어야 함 (에피소드 누적 True 방지).
        # used_*_successfully는 에피소드 전체에서 True로 유지되므로
        # tile 조건만으로 게이팅하면 메뉴 열림 중 tile이 일시 변경될 때 즉시 SUCCESS 발화 버그.
        self._cut_cycle_start_count: int = 0
        self._surf_cycle_start_count: int = 0
        self._flash_cycle_start_count: int = 0


    @property
    def steps_in_state(self) -> int:
        return self._steps_in_state

    @property
    def state_id(self) -> int:
        return int(self.state)

    @property
    def hm_target(self) -> HMTarget:
        return HM_TARGET_BY_STATE[self.state]

    @property
    def flash_cycle_start_count(self) -> int:
        return self._flash_cycle_start_count

    def reset(self) -> None:
        self.state = RewardMachineState.IDLE
        self._last_step_count = None
        self._steps_in_state = 0
        self._invalid_increase_counter = 0
        self._last_invalid_cut_coords_count = None
        self._last_invalid_surf_coords_count = None
        self._last_invalid_flash_coords_count = None
        self._idle_cut_entry_ok = True
        self._idle_surf_entry_ok = True
        self._idle_flash_entry_ok = True
        self._cut_cycle_start_count = 0
        self._surf_cycle_start_count = 0
        self._flash_cycle_start_count = 0

    def _rearm_idle_detect_entry(self, context: RewardMachineContext) -> None:
        if context.tile_in_front not in CUTTABLE_TILES or not context.can_use_cut:
            self._idle_cut_entry_ok = True
        if context.is_surfing:
            self._idle_surf_entry_ok = False
        elif not context.surf_detect_ok or not context.can_use_surf:
            self._idle_surf_entry_ok = True
        if not context.in_dark_cave or not context.can_use_flash:
            self._idle_flash_entry_ok = True

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
            self._idle_flash_entry_ok = True
            self._cut_cycle_start_count = 0
            self._surf_cycle_start_count = 0
            self._flash_cycle_start_count = 0
            return RewardMachineStep(previous_state, self.state, None)

        if (
            self.state
            in {
                RewardMachineState.CUT_START_MENU,
                RewardMachineState.CUT_MENU_OPEN,
                RewardMachineState.CUT_PARTY_MENU,
                RewardMachineState.CUT_MON_SELECTED,
                RewardMachineState.SURF_START_MENU,
                RewardMachineState.SURF_MENU_OPEN,
                RewardMachineState.SURF_PARTY_MENU,
                RewardMachineState.SURF_MON_SELECTED,
                RewardMachineState.FLASH_START_MENU,
                RewardMachineState.FLASH_MENU_OPEN,
                RewardMachineState.FLASH_PARTY_MENU,
                RewardMachineState.FLASH_MON_SELECTED,
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
            if self.state == RewardMachineState.CUT_SUCCESS:
                # SUCCESS→DONE→IDLE 직후 같은 스냅샷에서 IDLE→SUCCESS 재지급 방지.
                self._cut_cycle_start_count = context.valid_cut_coords_count
            elif self.state == RewardMachineState.SURF_SUCCESS:
                self._surf_cycle_start_count = context.surf_hook_success_count
            elif self.state == RewardMachineState.FLASH_SUCCESS:
                self._flash_cycle_start_count = context.valid_flash_coords_count
            elif previous_state == RewardMachineState.IDLE:
                if self.state == RewardMachineState.CUT_DETECTED:
                    self._idle_cut_entry_ok = False
                    self._cut_cycle_start_count = context.valid_cut_coords_count
                elif self.state == RewardMachineState.SURF_DETECTED:
                    self._idle_surf_entry_ok = False
                    self._surf_cycle_start_count = context.surf_hook_success_count
                elif self.state == RewardMachineState.FLASH_DETECTED:
                    self._idle_flash_entry_ok = False
                    self._flash_cycle_start_count = context.valid_flash_coords_count
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
            or self._last_invalid_flash_coords_count is None
        ):
            self._last_invalid_cut_coords_count = context.invalid_cut_coords_count
            self._last_invalid_surf_coords_count = context.invalid_surf_coords_count
            self._last_invalid_flash_coords_count = context.invalid_flash_coords_count
            self._invalid_increase_counter = 0
            return

        cut_delta = (
            context.invalid_cut_coords_count - self._last_invalid_cut_coords_count
        )
        surf_delta = (
            context.invalid_surf_coords_count - self._last_invalid_surf_coords_count
        )
        flash_delta = (
            context.invalid_flash_coords_count - self._last_invalid_flash_coords_count
        )

        # Update baselines every step; we only count deltas while the RM is
        # stuck in the corresponding menu/selection states.
        self._last_invalid_cut_coords_count = context.invalid_cut_coords_count
        self._last_invalid_surf_coords_count = context.invalid_surf_coords_count
        self._last_invalid_flash_coords_count = context.invalid_flash_coords_count

        # If we've already "succeeded" for the current HM stage, don't count invalids.
        if self.state in {
            RewardMachineState.CUT_START_MENU,
            RewardMachineState.CUT_MENU_OPEN,
            RewardMachineState.CUT_PARTY_MENU,
            RewardMachineState.CUT_MON_SELECTED,
        }:
            if context.used_cut_successfully:
                self._invalid_increase_counter = 0
            elif cut_delta > 0:
                self._invalid_increase_counter += int(cut_delta)
        elif self.state in {
            RewardMachineState.SURF_START_MENU,
            RewardMachineState.SURF_MENU_OPEN,
            RewardMachineState.SURF_PARTY_MENU,
            RewardMachineState.SURF_MON_SELECTED,
        }:
            if context.used_surf_successfully:
                self._invalid_increase_counter = 0
            elif surf_delta > 0:
                self._invalid_increase_counter += int(surf_delta)
        elif self.state in {
            RewardMachineState.FLASH_START_MENU,
            RewardMachineState.FLASH_MENU_OPEN,
            RewardMachineState.FLASH_PARTY_MENU,
            RewardMachineState.FLASH_MON_SELECTED,
        }:
            if context.flash_cycle_has_new_success:
                self._invalid_increase_counter = 0
            elif flash_delta > 0:
                self._invalid_increase_counter += int(flash_delta)

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
                    transition.target == RewardMachineState.FLASH_DETECTED
                    and not self._idle_flash_entry_ok
                ):
                    continue
            # → SURF_SUCCESS: IDLE에서 is_surfing만으로는 매 스텝 지급 → 훅·사이클 가드.
            if (
                transition.target == RewardMachineState.SURF_SUCCESS
                and transition.source == RewardMachineState.IDLE
                and context.surf_hook_success_count <= self._surf_cycle_start_count
            ):
                continue
            if (
                transition.target == RewardMachineState.SURF_SUCCESS
                and transition.source
                in {
                    RewardMachineState.SURF_DETECTED,
                    RewardMachineState.SURF_MENU_OPEN,
                    RewardMachineState.SURF_PARTY_MENU,
                    RewardMachineState.SURF_MON_SELECTED,
                }
                and context.surf_hook_success_count <= self._surf_cycle_start_count
                and context.surf_hook_success_delta <= 0
            ):
                continue
            if (
                transition.target == RewardMachineState.CUT_SUCCESS
                and transition.source == RewardMachineState.IDLE
                and context.valid_cut_coords_count <= self._cut_cycle_start_count
            ):
                continue
            if (
                transition.target == RewardMachineState.CUT_SUCCESS
                and transition.source == RewardMachineState.CUT_DETECTED
                and context.valid_cut_coords_count <= self._cut_cycle_start_count
                and context.valid_cut_coords_delta <= 0
            ):
                continue
            if transition.condition(context):
                return transition
        return None
