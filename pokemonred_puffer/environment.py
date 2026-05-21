import io
import os
import random
from abc import abstractmethod
from ctypes import c_uint8
from collections import deque
from multiprocessing import Lock, shared_memory
from pathlib import Path
from typing import Any, Iterable, Optional

import mediapy as media
import numpy as np
import numpy.typing as npt
from gymnasium import Env, spaces
from omegaconf import DictConfig, ListConfig, OmegaConf
from pyboy import PyBoy
from pyboy.utils import WindowEvent

from pokemonred_puffer.data.elevators import NEXT_ELEVATORS
from pokemonred_puffer.data.events import (
    EVENT_FLAGS_START,
    EVENTS_FLAGS_LENGTH,
    MUSEUM_TICKET,
    REQUIRED_EVENTS,
    EventFlags,
)
from pokemonred_puffer.data.field_moves import FieldMoves
from pokemonred_puffer.data.flags import Flags
from pokemonred_puffer.data.items import (
    HM_ITEMS,
    KEY_ITEMS,
    MAX_ITEM_CAPACITY,
    REQUIRED_ITEMS,
    USEFUL_ITEMS,
    Items,
)
from pokemonred_puffer.data.map import (
    MAP_ID_COMPLETION_EVENTS,
    MapIds,
)
from pokemonred_puffer.data.missable_objects import MissableFlags
from pokemonred_puffer.data.party import PartyMons
from pokemonred_puffer.data.strength_puzzles import STRENGTH_SOLUTIONS
from pokemonred_puffer.data.tilesets import Tilesets
from pokemonred_puffer.data.tm_hm import CUT_SPECIES_IDS, TmHmMoves
from pokemonred_puffer.global_map import GLOBAL_MAP_SHAPE, local_to_global
from pokemonred_puffer.rewards.reward_machine import (
    CUTTABLE_TILES,
    DARK_CAVE_MAP_PAL_OFFSET,
    HMTarget,
    RewardMachineState,
    START_MENU_POKEMON_CURSOR,
    SURF_TILE_IN_FRONT,
)

# wJoyIgnore 해제 루프 상한 (구 1000 → SPS·정지 방지).
_JOY_IGNORE_DISMISS_MAX = 32

PIXEL_VALUES = np.array([0, 85, 153, 255], dtype=np.uint8)
# wTileMap 상 플레이어 기준 (cut_if_next 등과 동일). 8방 단일 타일 ID (순서: 상·하·좌·우·좌상·우상·좌하·우하).
NEAR_TILE_PLAYER_ROW = 8
NEAR_TILE_PLAYER_COL = 8
NEAR_TILE_MEMORY_DIM = 8
MENU_FLAGS_DIM = 5  # start, pokemon, bag, stats, illegal_nav (per-step hooks + latch)
UI_LOCK_DIM = 3  # joy_ignore, font_loaded, in_battle
PARTY_HM_CAP_SHAPE = (6, 3)  # 슬롯별 cut/surf/flash 기술 보유 (0/1)
MAX_PARTY_SIZE = 6
MAX_ENEMY_PARTY_SIZE = 6


VALID_ACTIONS = [
    WindowEvent.PRESS_ARROW_DOWN,
    WindowEvent.PRESS_ARROW_LEFT,
    WindowEvent.PRESS_ARROW_RIGHT,
    WindowEvent.PRESS_ARROW_UP,
    WindowEvent.PRESS_BUTTON_A,
    WindowEvent.PRESS_BUTTON_B,
    WindowEvent.PRESS_BUTTON_START,
]

VALID_ACTIONS_STR = ["down", "left", "right", "up", "a", "b", "start"]

VALID_RELEASE_ACTIONS = [
    WindowEvent.RELEASE_ARROW_DOWN,
    WindowEvent.RELEASE_ARROW_LEFT,
    WindowEvent.RELEASE_ARROW_RIGHT,
    WindowEvent.RELEASE_ARROW_UP,
    WindowEvent.RELEASE_BUTTON_A,
    WindowEvent.RELEASE_BUTTON_B,
    WindowEvent.RELEASE_BUTTON_START,
]

ACTION_SPACE = spaces.Discrete(len(VALID_ACTIONS))

# x, y, map_n
SEAFOAM_SURF_SPOTS = {
    (23, 5, 162),
    (7, 11, 162),
    (7, 3, 162),
    (15, 7, 161),
    (23, 9, 161),
    (25, 16, 162),
}


# TODO: Make global map usage a configuration parameter
class RedGymEnv(Env):
    env_id = shared_memory.SharedMemory(create=True, size=4)
    lock = Lock()

    def __init__(self, env_config: DictConfig):
        self.video_dir = Path(env_config.video_dir)
        self.headless = env_config.headless
        self.state_dir = Path(env_config.state_dir)
        self.init_state = env_config.init_state
        self.init_state_name = self.init_state
        self.init_state_path = self.state_dir / f"{self.init_state_name}.state"
        self.action_freq = env_config.action_freq
        self.max_steps = env_config.max_steps
        self.save_video = env_config.save_video
        self.fast_video = env_config.fast_video
        self.video_tail_steps = int(
            OmegaConf.select(env_config, "video_tail_steps", default=2000)
        )
        if self.fast_video:
            self.fps = 60
        else:
            self.fps = 6
        self.n_record = env_config.n_record
        self.perfect_ivs = env_config.perfect_ivs
        self.reduce_res = env_config.reduce_res
        self.gb_path = env_config.gb_path
        self.log_frequency = env_config.log_frequency
        self.two_bit = env_config.two_bit
        self.auto_flash = env_config.auto_flash
        # A mapping of event to completion rate across
        # all environments in a run
        self.required_rate = 1.0
        self.required_tolerance = env_config.required_tolerance
        if isinstance(env_config.disable_wild_encounters, bool):
            self.disable_wild_encounters = env_config.disable_wild_encounters
            self.disable_wild_encounters_maps = set([])
        elif isinstance(env_config.disable_wild_encounters, ListConfig):
            self.disable_wild_encounters = len(env_config.disable_wild_encounters) > 0
            self.disable_wild_encounters_maps = {
                MapIds[item].name for item in env_config.disable_wild_encounters
            }
        else:
            raise ValueError("Disable wild enounters must be a boolean or a list of MapIds")

        self.disable_ai_actions = env_config.disable_ai_actions
        self.auto_teach_cut = env_config.auto_teach_cut
        self.auto_teach_surf = env_config.auto_teach_surf
        self.auto_teach_strength = env_config.auto_teach_strength
        self.auto_use_cut = env_config.auto_use_cut
        self.auto_use_strength = env_config.auto_use_strength
        self.auto_use_surf = env_config.auto_use_surf
        self.auto_solve_strength_puzzles = env_config.auto_solve_strength_puzzles
        self.auto_remove_all_nonuseful_items = env_config.auto_remove_all_nonuseful_items
        self.auto_next_elevator_floor = env_config.auto_next_elevator_floor
        self.insert_saffron_guard_drinks = env_config.insert_saffron_guard_drinks
        self.infinite_money = env_config.infinite_money
        self.infinite_health = env_config.infinite_health
        self.use_global_map = env_config.use_global_map
        # False면 screen/hm_screen은 0으로만 넣고 픽셀 전처리 생략 (RM·타일·메뉴 obs로 학습).
        self.include_screen_obs = bool(
            OmegaConf.select(env_config, "include_screen_obs", default=True)
        )
        self.save_state = env_config.save_state
        self.animate_scripts = env_config.animate_scripts
        self.exploration_inc = env_config.exploration_inc
        self.exploration_max = env_config.exploration_max
        self.max_steps_scaling = env_config.max_steps_scaling
        self.map_id_scalefactor = env_config.map_id_scalefactor
        self.action_space = ACTION_SPACE

        # Obs space-related. TODO: avoid hardcoding?
        self.global_map_shape = GLOBAL_MAP_SHAPE
        if self.reduce_res:
            self.screen_output_shape = (72, 80, 1)
        else:
            self.screen_output_shape = (144, 160, 1)
        if self.two_bit:
            self.screen_output_shape = (
                self.screen_output_shape[0],
                self.screen_output_shape[1] // 4,
                1,
            )
            self.global_map_shape = (self.global_map_shape[0], self.global_map_shape[1] // 4, 1)
        self.coords_pad = 12
        self.enc_freqs = 8

        self.all_runs = []

        # Set this in SOME subclasses
        self.metadata = {"render.modes": []}
        self.reward_range = (0, 15000)

        self.essential_map_locations = {
            v: i for i, v in enumerate([40, 0, 12, 1, 13, 51, 2, 54, 14, 59, 60, 61, 15, 3, 65])
        }

        # 관측: 방향·가방·파티·RM·주변 타일·메뉴·맵 ID (+ 옵션 screen).
        obs_dict = {
            "screen": spaces.Box(low=0, high=255, shape=self.screen_output_shape, dtype=np.uint8),
            # Discrete은 맞지만 pufferlib에서 Discrete 처리가 느려 Box로 둔다.
            "direction": spaces.Box(low=0, high=4, shape=(1,), dtype=np.uint8),
            "map_id": spaces.Box(low=0, high=0xF7, shape=(1,), dtype=np.uint8),
            "bag_items": spaces.Box(
                low=0, high=max(Items._value2member_map_.keys()), shape=(20,), dtype=np.uint8
            ),
            "bag_quantity": spaces.Box(low=0, high=100, shape=(20,), dtype=np.uint8),
            "party_count": spaces.Box(low=0, high=6, shape=(1,), dtype=np.uint8),
            "species": spaces.Box(low=0, high=0xBE, shape=(6,), dtype=np.uint8),
            "hp": spaces.Box(low=0, high=714, shape=(6,), dtype=np.uint32),
            "status": spaces.Box(low=0, high=7, shape=(6,), dtype=np.uint8),
            "type1": spaces.Box(low=0, high=0x1A, shape=(6,), dtype=np.uint8),
            "type2": spaces.Box(low=0, high=0x1A, shape=(6,), dtype=np.uint8),
            "level": spaces.Box(low=0, high=100, shape=(6,), dtype=np.uint8),
            "maxHP": spaces.Box(low=0, high=714, shape=(6,), dtype=np.uint32),
            "moves": spaces.Box(low=0, high=0xA4, shape=(6, 4), dtype=np.uint8),
            "rm_state": spaces.Box(
                low=0, high=len(RewardMachineState) - 1, shape=(1,), dtype=np.uint8
            ),
            # HM aux CE 전용. 액션 직전 래치(스텝 시작 시점) — 메뉴/타일 obs와 동시에 주면 hm_accuracy가 과장됨.
            "hm_aux_label": spaces.Box(
                low=0, high=int(HMTarget.NONE), shape=(1,), dtype=np.uint8
            ),
            "near_tile": spaces.Box(
                low=0, high=255, shape=(NEAR_TILE_MEMORY_DIM,), dtype=np.uint8
            ),
            # 이번 에이전트 스텝에서 훅으로 관측된 메뉴 (스텝 시작 시 0으로 리셋 후 run_action).
            "menu_flags": spaces.Box(low=0, high=1, shape=(MENU_FLAGS_DIM,), dtype=np.uint8),
            # 파티 6슬롯 × (cut, surf, flash) 보유 여부 — HM 타워 상황 인식용.
            "party_hm_cap": spaces.Box(
                low=0, high=1, shape=PARTY_HM_CAP_SHAPE, dtype=np.uint8
            ),
            # wTileInFrontOfPlayer — UsedCut/RM CUT·SURF 판정과 동일 (wTileMap 파생 아님).
            "tile_in_front": spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8),
            # wCurrentMenuItem — 스타트/파티/필드기술 커서 (메뉴 닫힘 시 0 근처).
            "current_menu_item": spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8),
            # wJoyIgnore / wFontLoaded / wIsInBattle — 대화·텍스트·전투 중 입력 잠금.
            "ui_lock": spaces.Box(low=0, high=1, shape=(UI_LOCK_DIM,), dtype=np.uint8),
        }

        if self.use_global_map:
            obs_dict["global_map"] = spaces.Box(
                low=0, high=255, shape=self.global_map_shape, dtype=np.uint8
            )
        self.observation_space = spaces.Dict(obs_dict)
        if not self.include_screen_obs:
            self._zero_screen_obs = np.zeros(self.screen_output_shape, dtype=np.uint8)

        self.pyboy = PyBoy(
            str(env_config.gb_path),
            debug=False,
            no_input=False,
            window="null" if self.headless else "SDL2",
            log_level="CRITICAL",
            symbols=os.path.join(os.path.dirname(__file__), "pokered.sym"),
            sound_emulated=False,
        )
        self.register_hooks()
        if not self.headless:
            self.pyboy.set_emulation_speed(6)
        self.screen = self.pyboy.screen

        self.first = True

        with RedGymEnv.lock:
            env_id = (
                (int(RedGymEnv.env_id.buf[0]) << 24)
                + (int(RedGymEnv.env_id.buf[1]) << 16)
                + (int(RedGymEnv.env_id.buf[2]) << 8)
                + (int(RedGymEnv.env_id.buf[3]))
            )
            self.env_id = env_id
            env_id += 1
            RedGymEnv.env_id.buf[0] = (env_id >> 24) & 0xFF
            RedGymEnv.env_id.buf[1] = (env_id >> 16) & 0xFF
            RedGymEnv.env_id.buf[2] = (env_id >> 8) & 0xFF
            RedGymEnv.env_id.buf[3] = (env_id) & 0xFF

        self._episode_video_writer = None
        if self.save_video and self.n_record:
            self.save_video = self.env_id < self.n_record
        if self.save_video:
            self.video_dir.mkdir(parents=True, exist_ok=True)
        self.init_mem()
        self._cache_wram_addresses()

    def _cache_wram_addresses(self) -> None:
        """symbol_lookup는 초기화 시 1회만. 스텝마다 반복 호출하면 SPS가 크게 떨어진다."""
        lu = self.pyboy.symbol_lookup

        def addr(name: str) -> int:
            return lu(name)[1]

        self._ram = {
            "wBagItems": addr("wBagItems"),
            "wNumBagItems": addr("wNumBagItems"),
            "wMapPalOffset": addr("wMapPalOffset"),
            "wPlayerMoney": addr("wPlayerMoney"),
            "wRepelRemainingSteps": addr("wRepelRemainingSteps"),
            "wPartyCount": addr("wPartyCount"),
            "wPartyMons": addr("wPartyMons"),
            "wTileMap": addr("wTileMap"),
            "wTileInFrontOfPlayer": addr("wTileInFrontOfPlayer"),
            "wCurrentMenuItem": addr("wCurrentMenuItem"),
            "wSpritePlayerStateData1FacingDirection": addr(
                "wSpritePlayerStateData1FacingDirection"
            ),
        }
        self._flags_start = addr("wStatusFlags1")
        self._flags_end = addr("wElite4Flags") + 1
        self._event_flags_start = EVENT_FLAGS_START
        self._event_flags_end = EVENT_FLAGS_START + EVENTS_FLAGS_LENGTH
        self._missable_start = 0xD5A6
        self._missable_end = 0xD5A6 + 32

    def _refresh_game_state_objects(self) -> None:
        """EventFlags/Flags/Party 재생성 대신 WRAM만 갱신."""
        self.events.asbytes = (c_uint8 * 320)(
            *self.pyboy.memory[self._event_flags_start : self._event_flags_end]
        )
        self.missables.asbytes = (c_uint8 * 32)(
            *self.pyboy.memory[self._missable_start : self._missable_end]
        )
        self.flags.asbytes = (c_uint8 * 13)(
            *self.pyboy.memory[self._flags_start : self._flags_end]
        )
        self.party.refresh(self.pyboy)

    def register_hooks(self):
        self.pyboy.hook_register(None, "DisplayStartMenu", self.start_menu_hook, None)
        self.pyboy.hook_register(None, "RedisplayStartMenu", self.start_menu_hook, None)
        self.pyboy.hook_register(None, "CloseStartMenu", self.close_start_menu_hook, None)
        self.pyboy.hook_register(None, "StartMenu_Item", self.item_menu_hook, None)
        self.pyboy.hook_register(None, "StartMenu_Pokemon", self.pokemon_menu_hook, None)
        for _lbl in (
            "StartMenu_Pokedex",
            "StartMenu_TrainerInfo",
            "StartMenu_SaveReset",
            "StartMenu_Option",
        ):
            self.pyboy.hook_register(None, _lbl, self.start_menu_non_pokemon_branch_hook, None)
        self.pyboy.hook_register(None, "StartMenu_Pokemon.choseStats", self.chose_stats_hook, None)
        self.pyboy.hook_register(
            None, "DisplayFieldMoveMonMenu", self.field_move_menu_hook, None
        )
        self.pyboy.hook_register(None, "StartMenu_Item.choseItem", self.chose_item_hook, None)
        self.pyboy.hook_register(None, "DisplayTextID.spriteHandling", self.sprite_hook, None)
        self.pyboy.hook_register(
            None, "CheckForHiddenObject.foundMatchingObject", self.hidden_object_hook, None
        )
        self.pyboy.hook_register(None, "HandleBlackOut", self.blackout_hook, None)
        self.pyboy.hook_register(None, "SetLastBlackoutMap.done", self.blackout_update_hook, None)
        if not self.auto_use_cut:
            self.pyboy.hook_register(None, "UsedCut.nothingToCut", self.cut_hook, context=False)
            self.pyboy.hook_register(None, "UsedCut.canCut", self.cut_hook, context=True)
        if not self.auto_use_surf:
            self.pyboy.hook_register(None, "SurfingAttemptFailed", self.surf_hook, context=False)
            self.pyboy.hook_register(None, "ItemUseSurfboard.surf", self.surf_hook, context=True)
        if not self.auto_flash:
            self.pyboy.hook_register(
                None, "StartMenu_Pokemon.flash", self.flash_hook, None
            )

        if self.disable_wild_encounters:
            self.setup_disable_wild_encounters()
        self.pyboy.hook_register(None, "AnimateHealingMachine", self.pokecenter_heal_hook, None)
        # self.pyboy.hook_register(None, "OverworldLoopLessDelay", self.overworld_loop_hook, None)
        self.pyboy.hook_register(None, "CheckWarpsNoCollisionLoop", self.update_warps_hook, None)
        signBank, signAddr = self.pyboy.symbol_lookup("IsSpriteOrSignInFrontOfPlayer.retry")
        self.pyboy.hook_register(
            signBank,
            signAddr - 1,
            self.sign_hook,
            None,
        )
        self.pyboy.hook_register(None, "ItemUseBall.loop", self.use_ball_hook, None)
        self.reset_count = 0

    def setup_disable_wild_encounters(self):
        bank, addr = self.pyboy.symbol_lookup("TryDoWildEncounter.gotWildEncounterType")
        self.pyboy.hook_register(
            bank,
            addr + 8,
            self.disable_wild_encounter_hook,
            None,
        )

    def setup_enable_wild_encounters(self):
        bank, addr = self.pyboy.symbol_lookup("TryDoWildEncounter.gotWildEncounterType")
        self.pyboy.hook_deregister(bank, addr + 8)

    def update_state(self, state: bytes):
        self.reset(seed=random.randint(0, 10), options={"state": state})

    def reset(self, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None):
        # restart game, skipping credits
        options = options or {}

        if self.save_video:
            self._close_episode_video()

        infos = {}
        self.explore_map_dim = 384
        if self.first or options.get("state", None) is not None:
            # We only init seen hidden objs once cause they can only be found once!
            if options.get("state", None) is not None:
                self.pyboy.load_state(io.BytesIO(options["state"]))
            else:
                with open(self.init_state_path, "rb") as f:
                    self.pyboy.load_state(f)

                self.events = EventFlags(self.pyboy)
                self.missables = MissableFlags(self.pyboy)
                self.flags = Flags(self.pyboy)
                self.required_events = self.get_required_events()
                self.required_items = self.get_required_items()
                self.base_event_flags = sum(
                    self.read_m(i).bit_count()
                    for i in range(EVENT_FLAGS_START, EVENT_FLAGS_START + EVENTS_FLAGS_LENGTH)
                )

                if self.save_state:
                    state = io.BytesIO()
                    self.pyboy.save_state(state)
                    state.seek(0)
                    infos |= {
                        "state": {
                            tuple(
                                sorted(list(self.required_events) + list(self.required_items))
                            ): state.read()
                        },
                        "required_count": len(self.required_events) + len(self.required_items),
                        "env_id": self.env_id,
                    }
            # lazy random seed setting
            # if not seed:
            #     seed = random.randint(0, 4096)
            #  self.pyboy.tick(seed, render=False)
        self.reset_count += 1

        if not hasattr(self, "party"):
            self.events = EventFlags(self.pyboy)
            self.missables = MissableFlags(self.pyboy)
            self.flags = Flags(self.pyboy)
            self.party = PartyMons(self.pyboy)
        else:
            self._refresh_game_state_objects()
        self.required_events = self.get_required_events()
        self.required_items = self.get_required_items()
        self.seen_pokemon = np.zeros(152, dtype=np.uint8)
        self.caught_pokemon = np.zeros(152, dtype=np.uint8)
        self.obtained_move_ids = np.zeros(0xA5, dtype=np.uint8)
        self.pokecenters = np.zeros(252, dtype=np.uint8)

        self.recent_screens = deque()
        self.recent_actions = deque()
        self.a_press = set()
        self.explore_map *= 0
        self.reward_explore_map *= 0
        self.cut_explore_map *= 0
        self.reset_mem()

        self.update_pokedex()
        self.update_tm_hm_obtained_move_ids()
        self.party_size = self.read_m("wPartyCount")
        self.taught_cut = self.check_if_party_has_hm(TmHmMoves.CUT.value)
        self.taught_surf = self.check_if_party_has_hm(TmHmMoves.SURF.value)
        self.taught_strength = self.check_if_party_has_hm(TmHmMoves.STRENGTH.value)
        self.levels_satisfied = False
        self.base_explore = 0
        self.max_opponent_level = 0
        self.max_level_rew = 0
        self.max_level_sum = 0
        self.last_health = 1
        self.total_heal_health = 0
        self.died_count = 0
        self.step_count = 0
        self.blackout_check = 0
        self.blackout_count = 0
        self.use_surf = 0

        self.current_event_flags_set = {}

        self.action_hist = np.zeros(len(VALID_ACTIONS))

        self.max_map_progress = 0
        self.progress_reward = self.get_game_state_reward()
        # update_reward()와 동일: 누적 보상 dict 전체 합을 기준점으로 둔다.
        self.total_reward = sum(self.progress_reward.values())

        self.first = False
        self._hm_aux_label_for_obs = int(HMTarget.NONE)

        return self._get_obs(), infos

    def init_mem(self):
        # Maybe I should preallocate a giant matrix for all map ids
        # All map ids have the same size, right?
        self.seen_coords: dict[int, dict[tuple[int, int, int], int]] = {}
        self.explore_map = np.zeros(GLOBAL_MAP_SHAPE, dtype=np.float32)
        self.reward_explore_map = np.zeros(GLOBAL_MAP_SHAPE, dtype=np.float32)
        self.cut_explore_map = np.zeros(GLOBAL_MAP_SHAPE, dtype=np.float32)
        self.seen_map_ids = np.zeros(256)
        self.seen_npcs = {}
        self.seen_warps = {}
        self.valid_cut_coords = {}
        self.invalid_cut_coords = {}
        self.cut_tiles = {}

        self.valid_surf_coords = {}
        self.invalid_surf_coords = {}
        # RM 가드용: 동일 좌표 재서핑도 훅 발화마다 증가 (len(valid_surf_coords)는 고유 좌표만).
        self._surf_hook_success_count = 0

        self.valid_flash_coords = {}
        self.invalid_flash_coords = {}

        self.seen_hidden_objs = {}
        self.seen_signs = {}

        # 스타트 메뉴가 현재 열려 있는지(지속 상태). 훅으로 켜고 CloseStartMenu에서 끈다.
        self._start_menu_open = False
        # 스타트 메뉴에서 ITEM/도감 등 포켓몬 외 하위 메뉴로 진입했을 때 True (배틀 제외).
        # RM용 seen_* 플래그와 달리 에피소드 동안 유지되며, CloseStartMenu·메인 메뉴 복귀 훅으로 해제된다.
        self._start_menu_illegal_navigation = False

        self.seen_start_menu = 0
        self.seen_pokemon_menu = 0
        self.seen_stats_menu = 0
        self.seen_bag_menu = 0
        self.seen_action_bag_menu = 0
        self.seen_field_move_menu = 0
        self.pokecenter_heal = 0
        self.use_ball_count = 0

    def reset_mem(self):
        self._start_menu_open = False
        self._start_menu_illegal_navigation = False
        self.seen_start_menu = 0
        self.seen_pokemon_menu = 0
        self.seen_stats_menu = 0
        self.seen_bag_menu = 0
        self.seen_action_bag_menu = 0
        self.seen_field_move_menu = 0
        self.pokecenter_heal = 0
        self.use_ball_count = 0
        # 에피소드마다 필드무브 시도/성공 추적을 비워 훅·통계가 리셋 후 깨끗이 쌓이게 함.
        # tile_in_front / RM / cut_hook → wTileInFrontOfPlayer. near_tile → wTileMap 8방.
        self.cut_tiles = {}
        self.surf_tiles = {}
        self.valid_cut_coords = {}
        self.invalid_cut_coords = {}
        self.valid_surf_coords = {}
        self.invalid_surf_coords = {}
        self._surf_hook_success_count = 0
        self.valid_flash_coords = {}
        self.invalid_flash_coords = {}

    def render(self) -> npt.NDArray[np.uint8]:
        return self.screen.ndarray[:, :, 1]

    def screen_obs(self):
        """화면 (+ 옵션 global_map). include_screen_obs=False면 0 텐서만 반환."""
        if not self.include_screen_obs:
            out: dict[str, npt.NDArray[np.uint8]] = {
                "screen": self._zero_screen_obs,
            }
            if self.use_global_map:
                out["global_map"] = np.zeros(self.global_map_shape, dtype=np.uint8)
            return out

        game_pixels_render = np.expand_dims(self.screen.ndarray[:, :, 1], axis=-1)

        if self.reduce_res:
            game_pixels_render = game_pixels_render[::2, ::2, :]

        global_map = None
        if self.use_global_map:
            global_map = np.expand_dims(
                255 * self.explore_map,
                axis=-1,
            ).astype(np.uint8)

        if self.two_bit:
            game_pixels_render = (
                (
                    np.digitize(
                        game_pixels_render.reshape((-1, 4)), PIXEL_VALUES, right=True
                    ).astype(np.uint8)
                    << np.array([6, 4, 2, 0], dtype=np.uint8)
                )
                .sum(axis=1, dtype=np.uint8)
                .reshape((-1, game_pixels_render.shape[1] // 4, 1))
            )
            if self.use_global_map and global_map is not None:
                global_map = (
                    (
                        np.digitize(
                            global_map.reshape((-1, 4)),
                            np.array([0, 64, 128, 255], dtype=np.uint8),
                            right=True,
                        ).astype(np.uint8)
                        << np.array([6, 4, 2, 0], dtype=np.uint8)
                    )
                    .sum(axis=1, dtype=np.uint8)
                    .reshape(self.global_map_shape)
                )

        out: dict[str, npt.NDArray[np.uint8]] = {
            "screen": game_pixels_render,
        }
        if self.use_global_map and global_map is not None:
            out["global_map"] = global_map
        return out

    def _get_obs(self):
        # player_x, player_y, map_n = self.get_game_coords()
        w_bag = self._ram["wBagItems"]
        bag = np.array(self.pyboy.memory[w_bag : w_bag + 40], dtype=np.uint8)
        numBagItems = self.read_m("wNumBagItems")
        # item ids start at 1 so using 0 as the nothing value is okay
        bag[2 * numBagItems :] = 0

        return (
            self.screen_obs()
            | {
                "direction": np.array(
                    [self.read_m("wSpritePlayerStateData1FacingDirection") // 4], dtype=np.uint8
                ),
                "map_id": np.array([self.read_m(0xD35E)], dtype=np.uint8),
                "bag_items": bag[::2].copy(),
                "bag_quantity": bag[1::2].copy(),
                **self._party_slot_obs(),
                "hp": np.array([self.party[i].HP for i in range(6)], dtype=np.uint32),
                "status": np.array([self.party[i].Status for i in range(6)], dtype=np.uint8),
                "type1": np.array([self.party[i].Type1 for i in range(6)], dtype=np.uint8),
                "type2": np.array([self.party[i].Type2 for i in range(6)], dtype=np.uint8),
                "level": np.array([self.party[i].Level for i in range(6)], dtype=np.uint8),
                "maxHP": np.array([self.party[i].MaxHP for i in range(6)], dtype=np.uint32),
                "rm_state": np.array([self.get_reward_machine_state_id()], dtype=np.uint8),
                "hm_aux_label": np.array(
                    [int(getattr(self, "_hm_aux_label_for_obs", int(HMTarget.NONE)))],
                    dtype=np.uint8,
                ),
                "near_tile": self.get_near_tile_memory_8(),
                "menu_flags": self._menu_flags_obs(),
                "tile_in_front": np.array(
                    [self.get_tile_in_front_of_player()], dtype=np.uint8
                ),
                "current_menu_item": np.array(
                    [self.get_current_menu_item()], dtype=np.uint8
                ),
                "ui_lock": self._ui_lock_obs(),
                "party_hm_cap": self.get_party_hm_cap_obs(),
            }
        )

    def set_perfect_iv_dvs(self):
        party_size = self.read_m("wPartyCount")
        for i in range(party_size):
            _, addr = self.pyboy.symbol_lookup(f"wPartyMon{i+1}Species")
            self.pyboy.memory[addr + 17 : addr + 17 + 12] = 0xFF

    def check_if_party_has_hm(self, hm: int) -> bool:
        party_size = self._clamp_party_count()
        for i in range(party_size):
            if hm in self.party[i].Moves:
                return True
        return False

    def step(self, action):
        # HM aux 라벨: BaselineRewardEnv.step 에서 refresh (기회 타일 vs RM 메뉴 단계).
        if not hasattr(self, "refresh_hm_aux_label_for_obs"):
            self._hm_aux_label_for_obs = self.get_hm_supervision_target_id()

        w_map_pal = self._ram["wMapPalOffset"]
        if self.auto_flash and self.pyboy.memory[w_map_pal] == DARK_CAVE_MAP_PAL_OFFSET:
            self.pyboy.memory[w_map_pal] = 0

        if self.auto_remove_all_nonuseful_items:
            self.remove_all_nonuseful_items()

        w_player_money = self._ram["wPlayerMoney"]
        if (
            self.infinite_money
            and int.from_bytes(
                self.pyboy.memory[w_player_money : w_player_money + 3], "little"
            )
            < 10000
        ):
            self.pyboy.memory[w_player_money : w_player_money + 3] = int(10000).to_bytes(
                3, "little"
            )

        if (
            self.disable_wild_encounters
            and MapIds(self.blackout_check).name not in self.disable_wild_encounters_maps
        ):
            self.pyboy.memory[self._ram["wRepelRemainingSteps"]] = 0xFF

        # update the a press before we use it so we dont trigger the font loaded early return
        if VALID_ACTIONS[action] == WindowEvent.PRESS_BUTTON_A:
            self.update_a_press()
        self.run_action_on_emulator(action)
        self._refresh_game_state_objects()
        self.update_health()
        self.update_pokedex()
        self.update_tm_hm_obtained_move_ids()
        self.party_size = self.read_m("wPartyCount")
        self.update_max_op_level()
        # RM·보상은 update_reward() 안에서 돈다. use_surf는 그보다 먼저 갱신해야
        # RewardMachineContext.is_surfing이 같은 스텝의 wWalkBikeSurfState와 일치한다.
        self.use_surf = 1 if self.read_m("wWalkBikeSurfState") == 0x2 else 0
        new_reward = self.update_reward()
        if hasattr(self, "refresh_hm_aux_label_for_obs"):
            self.refresh_hm_aux_label_for_obs()
        self.last_health = self.read_hp_fraction()
        self.update_map_progress()
        if self.perfect_ivs:
            self.set_perfect_iv_dvs()
        self.taught_cut = self.check_if_party_has_hm(TmHmMoves.CUT.value)
        self.taught_surf = self.check_if_party_has_hm(TmHmMoves.SURF.value)
        self.taught_strength = self.check_if_party_has_hm(TmHmMoves.STRENGTH.value)
        self.pokecenters[self.read_m("wLastBlackoutMap")] = 1
        if self.infinite_health:
            self.reverse_damage()

        info = {}

        required_events = self.get_required_events()
        required_items = self.get_required_items()
        new_required_events = required_events - self.required_events
        new_required_items = required_items - self.required_items
        if self.save_state and (new_required_events or new_required_items):
            state = io.BytesIO()
            self.pyboy.save_state(state)
            state.seek(0)
            info["state"] = {
                tuple(sorted(list(required_events) + list(required_items))): state.read()
            }
            info["required_count"] = len(required_events) + len(required_items)
            info["env_id"] = self.env_id
            info = info | self.agent_stats(action)
        elif (
            self.step_count != 0
            and self.log_frequency
            and self.step_count % self.log_frequency == 0
        ):
            info = info | self.agent_stats(action)
        self.required_events = required_events
        self.required_items = required_items

        obs = self._get_obs()

        self.step_count += 1

        # cut mon check
        reset = False
        if not self.party_has_cut_capable_mon():
            reset = True
            self.first = True

        # only check periodically since this is expensive
        # we have a tolerance cause some events may be really hard to get
        if (new_required_events or new_required_items) and self.required_tolerance is not None:
            # calculate the current required completion percentage
            required_completion = len(required_events) + len(required_items)
            reset = (required_completion - self.required_rate) > self.required_tolerance

        if self.step_count >= self.get_max_steps():
            reset = True
            self.first = True

        if self.save_video:
            ms = self.get_max_steps()
            tail = max(1, self.video_tail_steps)
            start_at = max(1, ms - tail + 1)
            if self._episode_video_writer is None and self.step_count >= start_at:
                self.start_episode_video()
            if self._episode_video_writer is not None:
                self.add_video_frame()
            if reset:
                self._close_episode_video()

        return obs, new_reward, reset, False, info

    def run_action_on_emulator(self, action, *, skip_agent_input: bool = False):
        if not skip_agent_input:
            self.action_hist[action] += 1
        # 메뉴 플래그를 스텝 시작마다 초기화: 훅이 틱 도중에 1로 설정하므로
        # "이번 스텝에서 해당 메뉴가 실제로 열렸는가"를 정확히 반영한다.
        # 에피소드 내 누적(sticky)이 되면 RM 메뉴 전이가 무조건 통과돼 리워드 해킹으로 이어진다.
        self.seen_start_menu = 0
        self.seen_pokemon_menu = 0
        self.seen_bag_menu = 0
        self.seen_stats_menu = 0
        self.seen_action_bag_menu = 0
        self.seen_field_move_menu = 0

        if not skip_agent_input and not self.disable_ai_actions:
            self.pyboy.send_input(VALID_ACTIONS[action])
            self.pyboy.send_input(VALID_RELEASE_ACTIONS[action], delay=8)
            self.pyboy.tick(self.action_freq - 1, render=False)
        else:
            self.pyboy.tick(self.action_freq, render=False)

        # TODO: Split this function up. update_seen_coords should not be here!
        self.update_seen_coords()

        # DO NOT DELETE. Some animations require dialog navigation
        for _ in range(_JOY_IGNORE_DISMISS_MAX):
            if not self.read_m("wJoyIgnore"):
                break
            self.pyboy.button("a", 8)
            self.pyboy.tick(self.action_freq, render=False)

        if self.events.get_event("EVENT_GOT_HM01"):
            if self.auto_use_cut:
                self.cut_if_next()

        if self.events.get_event("EVENT_GOT_HM03"):
            if self.auto_use_surf:
                self.surf_if_attempt(VALID_ACTIONS[action])

        if self.events.get_event("EVENT_GOT_HM04"):
            if self.auto_solve_strength_puzzles:
                self.solve_strength_puzzle()
            if not self.check_if_party_has_hm(TmHmMoves.STRENGTH.value) and self.auto_use_strength:
                self.use_strength()

        if self.get_game_coords() == (18, 4, 7) and self.skip_safari_zone:
            self.skip_safari_zone_atn()

        if self.auto_next_elevator_floor:
            self.next_elevator_floor()

        if self.insert_saffron_guard_drinks:
            self.insert_guard_drinks()

        # One last tick just in case
        self.pyboy.tick(1, render=True)

    def party_has_cut_capable_mon(self) -> bool:
        """CUT RM·리셋 가드용. RewardMachineContext.can_use_cut(has_cut)과 동일 기준 우선.

        예전에는 CUT 배울 수 있는 *종*만 검사해서, 영상/테스트용으로 파티를 줄이면
        CUT을 이미 쓰고 있어도 매 스텝 reset → rm_* 카운트가 0으로만 보였다.
        """
        party_size = self.read_m("wPartyCount")
        if party_size == 0:
            return False
        if self.check_if_party_has_hm(TmHmMoves.CUT.value):
            return True
        if Items.HM_01 not in self.get_items_in_bag():
            return False
        for i in range(party_size):
            if self.party[i].Species in CUT_SPECIES_IDS:
                return True
        return False

    def teach_hm(self, tmhm: int, pp: int, pokemon_species_ids):
        # find bulba and replace tackle (first skill) with cut
        party_size = self.read_m("wPartyCount")
        for i in range(party_size):
            # PRET 1-indexes
            # https://github.com/pret/pokered/blob/d38cf5281a902b4bd167a46a7c9fd9db436484a7/constants/pokemon_constants.asm
            if self.party[i].Species in pokemon_species_ids:
                _, move_addr = self.pyboy.symbol_lookup(f"wPartyMon{i+1}Moves")
                _, pp_addr = self.pyboy.symbol_lookup(f"wPartyMon{i+1}PP")
                for slot in range(4):
                    if self.party[i].Moves[slot] not in {
                        TmHmMoves.CUT.value,
                        TmHmMoves.FLY.value,
                        TmHmMoves.SURF.value,
                        TmHmMoves.STRENGTH.value,
                        TmHmMoves.FLASH.value,
                    }:
                        self.pyboy.memory[move_addr + slot] = tmhm
                        self.pyboy.memory[pp_addr + slot] = pp
                        # fill up pp: 30/30
                        break
                        break

    def cut_if_next(self):
        # https://github.com/pret/pokered/blob/d38cf5281a902b4bd167a46a7c9fd9db436484a7/constants/tileset_constants.asm#L11C8-L11C11
        in_erika_gym = self.read_m("wCurMapTileset") == Tilesets.GYM.value
        in_overworld = self.read_m("wCurMapTileset") == Tilesets.OVERWORLD.value
        if self.read_m(0xD057) == 0 and (in_erika_gym or in_overworld):
            _, wTileMap = self.pyboy.symbol_lookup("wTileMap")
            tileMap = self.pyboy.memory[wTileMap : wTileMap + 20 * 18]
            tileMap = np.array(tileMap, dtype=np.uint8)
            tileMap = np.reshape(tileMap, (18, 20))
            y, x = 8, 8
            up, down, left, right = (
                tileMap[y - 2 : y, x : x + 2],  # up
                tileMap[y + 2 : y + 4, x : x + 2],  # down
                tileMap[y : y + 2, x - 2 : x],  # left
                tileMap[y : y + 2, x + 2 : x + 4],  # right
            )

            # Gym trees apparently get the same tile map as outside bushes
            # GYM = 7
            if (in_overworld and 0x3D in up) or (in_erika_gym and 0x50 in up):
                self.pyboy.button("UP", delay=8)
                self.pyboy.tick(self.action_freq, render=True)
            elif (in_overworld and 0x3D in down) or (in_erika_gym and 0x50 in down):
                self.pyboy.button("DOWN", delay=8)
                self.pyboy.tick(self.action_freq, render=True)
            elif (in_overworld and 0x3D in left) or (in_erika_gym and 0x50 in left):
                self.pyboy.button("LEFT", delay=8)
                self.pyboy.tick(self.action_freq, render=True)
            elif (in_overworld and 0x3D in right) or (in_erika_gym and 0x50 in right):
                self.pyboy.button("RIGHT", delay=8)
                self.pyboy.tick(self.action_freq, render=True)
            else:
                return

            # open start menu
            self.pyboy.button("START", delay=8)
            self.pyboy.tick(self.action_freq, self.animate_scripts)
            # scroll to pokemon
            # 1 is the item index for pokemon
            for _ in range(24):
                if self.pyboy.memory[self.pyboy.symbol_lookup("wCurrentMenuItem")[1]] == 1:
                    break
                self.pyboy.button("DOWN", delay=8)
                self.pyboy.tick(self.action_freq, render=self.animate_scripts)
            self.pyboy.button("A", delay=8)
            self.pyboy.tick(self.action_freq, self.animate_scripts)

            # find pokemon with cut
            # We run this over all pokemon so we dont end up in an infinite for loop
            for _ in range(7):
                self.pyboy.button("DOWN", delay=8)
                self.pyboy.tick(self.action_freq, self.animate_scripts)
                party_mon = self.pyboy.memory[self.pyboy.symbol_lookup("wCurrentMenuItem")[1]]
                _, addr = self.pyboy.symbol_lookup(f"wPartyMon{party_mon%6+1}Moves")
                if 0xF in self.pyboy.memory[addr : addr + 4]:
                    break

            # Enter submenu
            self.pyboy.button("A", delay=8)
            self.pyboy.tick(4 * self.action_freq, self.animate_scripts)

            # Scroll until the field move is found
            _, wFieldMoves = self.pyboy.symbol_lookup("wFieldMoves")
            field_moves = self.pyboy.memory[wFieldMoves : wFieldMoves + 4]

            for _ in range(10):
                current_item = self.read_m("wCurrentMenuItem")
                if current_item < 4 and FieldMoves.CUT.value == field_moves[current_item]:
                    break
                self.pyboy.button("DOWN", delay=8)
                self.pyboy.tick(self.action_freq, self.animate_scripts)

            # press a bunch of times
            for _ in range(5):
                self.pyboy.button("A", delay=8)
                self.pyboy.tick(4 * self.action_freq, self.animate_scripts)

    def _dismiss_joy_ignore(self) -> None:
        for _ in range(_JOY_IGNORE_DISMISS_MAX):
            if not self.read_m("wJoyIgnore"):
                break
            self.pyboy.button("a", 8)
            self.pyboy.tick(self.action_freq, self.animate_scripts)

    def _execute_surf_menu_sequence(self) -> None:
        """Start → 포켓몬 → Surf 필드기술 (`auto_use_surf` 전용)."""
        self._dismiss_joy_ignore()
        self.pyboy.send_input(WindowEvent.PRESS_BUTTON_START)
        self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START, delay=8)
        self.pyboy.tick(self.action_freq, self.animate_scripts)
        self._dismiss_joy_ignore()

        for _ in range(24):
            if self.read_m("wCurrentMenuItem") == START_MENU_POKEMON_CURSOR:
                break
            self.pyboy.send_input(WindowEvent.PRESS_ARROW_DOWN)
            self.pyboy.send_input(WindowEvent.RELEASE_ARROW_DOWN, delay=8)
            self.pyboy.tick(self.action_freq, self.animate_scripts)

        self.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
        self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A, delay=8)
        self.pyboy.tick(self.action_freq, self.animate_scripts)

        for _ in range(7):
            self.pyboy.send_input(WindowEvent.PRESS_ARROW_DOWN)
            self.pyboy.send_input(WindowEvent.RELEASE_ARROW_DOWN, delay=8)
            self.pyboy.tick(self.action_freq, self.animate_scripts)
            party_mon = self.read_m("wCurrentMenuItem")
            _, addr = self.pyboy.symbol_lookup(f"wPartyMon{party_mon % 6 + 1}Moves")
            if TmHmMoves.SURF.value in self.pyboy.memory[addr : addr + 4]:
                break

        self.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
        self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A, delay=8)
        self.pyboy.tick(4 * self.action_freq, self.animate_scripts)

        _, wFieldMoves = self.pyboy.symbol_lookup("wFieldMoves")
        field_moves = self.pyboy.memory[wFieldMoves : wFieldMoves + 4]
        for _ in range(10):
            current_item = self.read_m("wCurrentMenuItem")
            if current_item < 4 and field_moves[current_item] in (
                FieldMoves.SURF.value,
                FieldMoves.SURF_2.value,
            ):
                break
            self.pyboy.send_input(WindowEvent.PRESS_ARROW_DOWN)
            self.pyboy.send_input(WindowEvent.RELEASE_ARROW_DOWN, delay=8)
            self.pyboy.tick(self.action_freq, self.animate_scripts)

        for _ in range(5):
            self._dismiss_joy_ignore()
            self.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
            self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A, delay=8)
            self.pyboy.tick(4 * self.action_freq, self.animate_scripts)

    def surf_if_attempt(self, action: WindowEvent):
        if (
            self.read_m("wIsInBattle") == 0
            and self.read_m("wWalkBikeSurfState") != 2
            and self.check_if_party_has_hm(TmHmMoves.SURF.value)
            and action
            in [
                WindowEvent.PRESS_ARROW_DOWN,
                WindowEvent.PRESS_ARROW_LEFT,
                WindowEvent.PRESS_ARROW_RIGHT,
                WindowEvent.PRESS_ARROW_UP,
            ]
            and self._action_points_to_adjacent_water(action)
        ):
            self._execute_surf_menu_sequence()

    def solve_strength_puzzle(self):
        in_cavern = self.read_m("wCurMapTileset") == Tilesets.CAVERN.value
        if self.read_m(0xD057) == 0 and in_cavern:
            for sprite_id in range(1, self.read_m("wNumSprites") + 1):
                picture_id = self.read_m(f"wSprite{sprite_id:02}StateData1PictureID")
                mapY = self.read_m(f"wSprite{sprite_id:02}StateData2MapY")
                mapX = self.read_m(f"wSprite{sprite_id:02}StateData2MapX")
                if solution := STRENGTH_SOLUTIONS.get(
                    (picture_id, mapY, mapX) + self.get_game_coords(), None
                ):
                    missable, steps = solution
                    if missable and self.missables.get_missable(missable):
                        break
                    if not self.disable_wild_encounters:
                        self.setup_disable_wild_encounters()
                    # Activate strength
                    self.flags.set_bit("BIT_STRENGTH_ACTIVE", 1)
                    # Perform solution
                    current_repel_steps = self.read_m("wRepelRemainingSteps")
                    for step in steps:
                        self.pyboy.memory[self.pyboy.symbol_lookup("wRepelRemainingSteps")[1]] = (
                            0xFF
                        )
                        match step:
                            case str(button):
                                self.pyboy.button(button, 8)
                                self.pyboy.tick(self.action_freq * 2, self.animate_scripts)
                            case (str(button), int(button_freq), int(action_freq)):
                                self.pyboy.button(button, button_freq)
                                self.pyboy.tick(action_freq, self.animate_scripts)
                            case _:
                                raise
                        while self.read_m("wJoyIgnore"):
                            self.pyboy.tick(self.action_freq, render=False)
                    self.pyboy.memory[self.pyboy.symbol_lookup("wRepelRemainingSteps")[1]] = (
                        current_repel_steps
                    )
                    if not self.disable_wild_encounters:
                        self.setup_enable_wild_encounters()
                    break

    def use_strength(self):
        self.flags.set_bit("BIT_STRENGTH_ACTIVE", 1)

    def next_elevator_floor(self):
        curMapId = MapIds(self.read_m("wCurMap"))
        if curMapId in (MapIds.SILPH_CO_ELEVATOR, MapIds.CELADON_MART_ELEVATOR):
            for _ in range(5):
                self.pyboy.button("up", 8)
                self.pyboy.tick(self.action_freq, render=self.animate_scripts)
            # walk right
            for _ in range(5):
                self.pyboy.button("right", 8)
                self.pyboy.tick(self.action_freq, render=self.animate_scripts)
        elif (
            curMapId == MapIds.ROCKET_HIDEOUT_ELEVATOR
            and Items.LIFT_KEY.name in self.required_items
        ):
            for _ in range(5):
                self.pyboy.button("left", 8)
                self.pyboy.tick(self.action_freq, render=self.animate_scripts)
        else:
            return

        self.pyboy.button("up", 8)
        self.pyboy.tick(self.action_freq, render=self.animate_scripts)
        self.pyboy.button("a", 8)
        self.pyboy.tick(5 * self.action_freq, render=self.animate_scripts)
        for _ in range(NEXT_ELEVATORS.get(MapIds(self.read_m("wWarpedFromWhichMap")), 0)):
            self.pyboy.button("down", 8)
            self.pyboy.tick(self.action_freq, render=self.animate_scripts)

        self.pyboy.button("a", 8)
        self.pyboy.tick(20 * self.action_freq, render=self.animate_scripts)
        # now leave elevator
        if curMapId in (MapIds.SILPH_CO_ELEVATOR, MapIds.CELADON_MART_ELEVATOR):
            for _ in range(5):
                self.pyboy.button("down", 8)
                self.pyboy.tick(self.action_freq, render=self.animate_scripts)
            self.pyboy.button("left", 8)
            self.pyboy.tick(self.action_freq, render=self.animate_scripts)
            self.pyboy.button("down", 8)
            self.pyboy.tick(self.action_freq, render=self.animate_scripts)
        elif (
            curMapId == MapIds.ROCKET_HIDEOUT_ELEVATOR
            and Items.LIFT_KEY.name in self.required_items
        ):
            self.pyboy.button("right", 8)
            self.pyboy.tick(self.action_freq, render=self.animate_scripts)
            self.pyboy.button("up", 8)
            self.pyboy.tick(self.action_freq, render=self.animate_scripts)

    def insert_guard_drinks(self):
        if not self.flags.get_bit("BIT_GAVE_SAFFRON_GUARDS_DRINK") and MapIds(
            self.read_m("wCurMap")
        ) in [
            MapIds.CELADON_MART_1F,
            MapIds.CELADON_MART_2F,
            MapIds.CELADON_MART_3F,
            MapIds.CELADON_MART_4F,
            MapIds.CELADON_MART_5F,
            MapIds.CELADON_MART_ELEVATOR,
            MapIds.CELADON_MART_ROOF,
        ]:
            _, wBagItems = self.pyboy.symbol_lookup("wBagItems")
            _, wNumBagItems = self.pyboy.symbol_lookup("wNumBagItems")
            numBagItems = self.read_m(wNumBagItems)
            bag = np.array(self.pyboy.memory[wBagItems : wBagItems + 40], dtype=np.uint8)
            if numBagItems < 20 and not {
                Items.LEMONADE.value,
                Items.FRESH_WATER.value,
                Items.SODA_POP.value,
            }.intersection(bag[::2]):
                bag[numBagItems * 2] = Items.LEMONADE.value
                bag[numBagItems * 2 + 1] = 1
                numBagItems += 1
                bag[numBagItems * 2 :] = 0xFF
                self.pyboy.memory[wBagItems : wBagItems + 40] = bag
                self.pyboy.memory[wNumBagItems] = numBagItems

    def sign_hook(self, *args, **kwargs):
        sign_id = self.read_m("hSpriteIndexOrTextID")
        map_id = self.read_m("wCurMap")
        # self.seen_signs[(map_id, sign_id)] = 1.0 if self.scale_map_id(map_id) else 0.0
        self.seen_signs[(map_id, sign_id)] = 1.0

    def hidden_object_hook(self, *args, **kwargs):
        hidden_object_id = self.pyboy.memory[self.pyboy.symbol_lookup("wHiddenObjectIndex")[1]]
        map_id = self.pyboy.memory[self.pyboy.symbol_lookup("wCurMap")[1]]
        # self.seen_hidden_objs[(map_id, hidden_object_id)] = (
        #     1.0 if self.scale_map_id(map_id) else 0.0
        # )
        self.seen_hidden_objs[(map_id, hidden_object_id)] = 1.0

    def sprite_hook(self, *args, **kwargs):
        sprite_id = self.pyboy.memory[self.pyboy.symbol_lookup("hSpriteIndexOrTextID")[1]]
        map_id = self.pyboy.memory[self.pyboy.symbol_lookup("wCurMap")[1]]
        # self.seen_npcs[(map_id, sprite_id)] = 1.0 if self.scale_map_id(map_id) else 0.0
        self.seen_npcs[(map_id, sprite_id)] = 1.0

    def start_menu_hook(self, *args, **kwargs):
        if self.read_m("wIsInBattle") == 0:
            self._start_menu_open = True
            self.seen_start_menu = 1
            # 메인 일시정지 메뉴(처음 열기 또는 하위 메뉴에서 복귀)에서는 포켓몬 외 분기 패널티 구간 해제.
            self._start_menu_illegal_navigation = False

    def close_start_menu_hook(self, *args, **kwargs):
        self._start_menu_open = False
        self._start_menu_illegal_navigation = False

    def is_start_menu_illegal_navigation_active(self) -> bool:
        """포켓몬 외 스타트 메뉴 분기(가방 등)에 있는 동안이며 필드에서만 패널티 대상."""
        return bool(self._start_menu_illegal_navigation) and self.read_m("wIsInBattle") == 0

    def start_menu_non_pokemon_branch_hook(self, *args, **kwargs):
        if self.read_m("wIsInBattle") == 0:
            self._start_menu_open = True
            self._start_menu_illegal_navigation = True

    def item_menu_hook(self, *args, **kwargs):
        self.seen_bag_menu = 1
        if self.read_m("wIsInBattle") == 0:
            self._start_menu_open = True
            self._start_menu_illegal_navigation = True

    def pokemon_menu_hook(self, *args, **kwargs):
        if self.read_m("wIsInBattle") == 0:
            self._start_menu_open = True
            self.seen_pokemon_menu = 1
            self._start_menu_illegal_navigation = False

    def field_move_menu_hook(self, *args, **kwargs):
        if self.read_m("wIsInBattle") == 0:
            self.seen_field_move_menu = 1

    def chose_stats_hook(self, *args, **kwargs):
        if self.read_m("wIsInBattle") == 0:
            self._start_menu_open = True
            self.seen_stats_menu = 1

    def chose_item_hook(self, *args, **kwargs):
        # if self.read_m("wIsInBattle") == 0:
        self.seen_action_bag_menu = 1

    def blackout_hook(self, *args, **kwargs):
        self.blackout_count += 1

    def blackout_update_hook(self, *args, **kwargs):
        self.blackout_check = self.read_m("wLastBlackoutMap")
        if (
            self.disable_wild_encounters
            and MapIds(self.blackout_check).name in self.disable_wild_encounters_maps
        ):
            self.pyboy.memory[self.pyboy.symbol_lookup("wRepelRemainingSteps")[1]] = 0x01

    def pokecenter_heal_hook(self, *args, **kwargs):
        self.pokecenter_heal = 1

    def overworld_loop_hook(self, *args, **kwargs):
        self.user_control = True

    def update_warps_hook(self, *args, **kwargs):
        # current map id, destiation map id, warp id
        key = (
            self.read_m("wCurMap"),
            self.read_m("hWarpDestinationMap"),
            self.read_m("wDestinationWarpID"),
        )
        if key[-1] != 0xFF:
            self.seen_warps[key] = 1

    def cut_hook(self, context: bool):
        player_direction = self.pyboy.memory[
            self.pyboy.symbol_lookup("wSpritePlayerStateData1FacingDirection")[1]
        ]
        x, y, map_id = self.get_game_coords()  # x, y, map_id
        if player_direction == 0:  # down
            coords = (x, y + 1, map_id)
        if player_direction == 4:
            coords = (x, y - 1, map_id)
        if player_direction == 8:
            coords = (x - 1, y, map_id)
        if player_direction == 0xC:
            coords = (x + 1, y, map_id)

        wTileInFrontOfPlayer = self.get_tile_in_front_of_player()
        if context:
            if wTileInFrontOfPlayer in CUTTABLE_TILES:
                self.valid_cut_coords[coords] = 1
            else:
                self.invalid_cut_coords[coords] = 1
        else:
            self.invalid_cut_coords[coords] = 1

        self.cut_tiles[wTileInFrontOfPlayer] = 1
        self.cut_explore_map[local_to_global(y, x, map_id)] = 1

    def surf_hook(self, context: bool, *args, **kwargs):
        player_direction = self.pyboy.memory[
            self.pyboy.symbol_lookup("wSpritePlayerStateData1FacingDirection")[1]
        ]
        x, y, map_id = self.get_game_coords()  # x, y, map_id
        if player_direction == 0:  # down
            coords = (x, y + 1, map_id)
        elif player_direction == 4:  # up
            coords = (x, y - 1, map_id)
        elif player_direction == 8:  # left
            coords = (x - 1, y, map_id)
        elif player_direction == 0xC:  # right
            coords = (x + 1, y, map_id)
        else:
            coords = (x, y, map_id)
        if context:
            self.valid_surf_coords[coords] = 1
            self._surf_hook_success_count += 1
        else:
            self.invalid_surf_coords[coords] = 1

    def flash_hook(self, *args, **kwargs):
        """Record Flash usage.

        Dark-cave uses count as valid attempts for the reward machine cycle.
        Bright-area uses count as invalid HM usage so `unnecessary_hm_usage_penalty`
        can discourage wasting Flash outside caves.
        """
        player_direction = self.pyboy.memory[
            self.pyboy.symbol_lookup("wSpritePlayerStateData1FacingDirection")[1]
        ]
        _, wMapPalOffset = self.pyboy.symbol_lookup("wMapPalOffset")
        in_dark_cave = self.pyboy.memory[wMapPalOffset] == DARK_CAVE_MAP_PAL_OFFSET
        x, y, map_id = self.get_game_coords()
        if player_direction == 0:
            coords = (x, y + 1, map_id)
        if player_direction == 4:
            coords = (x, y - 1, map_id)
        if player_direction == 8:
            coords = (x - 1, y, map_id)
        if player_direction == 0xC:
            coords = (x + 1, y, map_id)
        if in_dark_cave:
            self.valid_flash_coords[coords] = 1
        else:
            self.invalid_flash_coords[coords] = 1

    def use_ball_hook(self, *args, **kwargs):
        self.use_ball_count += 1

    def disable_wild_encounter_hook(self, *args, **kwargs):
        if (
            self.disable_wild_encounters
            and MapIds(self.blackout_check).name not in self.disable_wild_encounters_maps
        ):
            self.pyboy.memory[self.pyboy.symbol_lookup("wRepelRemainingSteps")[1]] = 0xFF
            self.pyboy.memory[self.pyboy.symbol_lookup("wCurEnemyLevel")[1]] = 0x01

    def _clamp_party_count(self, count: int | None = None) -> int:
        if count is None:
            count = self.read_m("wPartyCount")
        return min(max(int(count), 0), MAX_PARTY_SIZE)

    def agent_stats(self, action):
        party_count = self._clamp_party_count()
        levels = [self.read_m(f"wPartyMon{i+1}Level") for i in range(party_count)]
        badges = int(self.read_m("wObtainedBadges")) & 0xFF

        _, wBagItems = self.pyboy.symbol_lookup("wBagItems")
        bag = np.array(self.pyboy.memory[wBagItems : wBagItems + 40], dtype=np.uint8)
        numBagItems = min(max(self.read_m("wNumBagItems"), 0), MAX_ITEM_CAPACITY)
        # item ids start at 1 so using 0 as the nothing value is okay
        bag[2 * numBagItems :] = 0
        bag_item_ids = bag[::2]

        exploration_sum = max(
            sum(sum(self.seen_coords.get(tileset.value, {}).values()) for tileset in Tilesets), 1
        )

        return {
            "env_ids": int(self.env_id),
            "stats": {
                "step": self.step_count + self.reset_count * self.max_steps,
                "max_map_progress": self.max_map_progress,
                "last_action": action,
                "party_count": party_count,
                "levels": levels,
                "levels_sum": sum(levels),
                "ptypes": self.read_party(),
                "hp": self.read_hp_fraction(),
                "coord": sum(sum(tileset.values()) for tileset in self.seen_coords.values()),
                "warps": len(self.seen_warps),
                "a_press": len(self.a_press),
                "map_id": np.sum(self.seen_map_ids),
                "npc": sum(self.seen_npcs.values()),
                "hidden_obj": sum(self.seen_hidden_objs.values()),
                "sign": sum(self.seen_signs.values()),
                "deaths": self.died_count,
                "badge": self.get_badges(),
                "healr": self.total_heal_health,
                "action_hist": self.action_hist,
                "caught_pokemon": int(sum(self.caught_pokemon)),
                "seen_pokemon": int(sum(self.seen_pokemon)),
                "obtained_move_ids": int(sum(self.obtained_move_ids)),
                "opponent_level": self.max_opponent_level,
                "taught_cut": int(self.check_if_party_has_hm(TmHmMoves.CUT.value)),
                "taught_surf": int(self.check_if_party_has_hm(TmHmMoves.SURF.value)),
                "taught_strength": int(self.check_if_party_has_hm(TmHmMoves.STRENGTH.value)),
                "cut_tiles": len(self.cut_tiles),
                # 성공한 컷 훅 횟수(고유 좌표 수).
                "cut_count": len(self.valid_cut_coords),
                "valid_cut_coords": len(self.valid_cut_coords),
                "invalid_cut_coords": len(self.invalid_cut_coords),
                "valid_surf_coords": len(self.valid_surf_coords),
                "invalid_surf_coords": len(self.invalid_surf_coords),
                "valid_flash_coords": len(self.valid_flash_coords),
                "invalid_flash_coords": len(self.invalid_flash_coords),
                "rm_state": self.get_reward_machine_state_id(),
                "hm_target": self.get_reward_machine_hm_target_id(),
                "hm_aux_label": int(getattr(self, "_hm_aux_label_for_obs", int(HMTarget.NONE))),
                "hm_supervision_target": self.get_hm_supervision_target_id(),
                "rm_transition_count": getattr(self, "rm_transition_count", 0),
                "rm_reward_total": getattr(self, "rm_reward_total", 0.0),
                "rm_success_count": getattr(self, "rm_success_count", 0),
                "rm_cut_success_count": getattr(self, "rm_cut_success_count", 0),
                "rm_surf_detected_count": getattr(self, "rm_surf_detected_count", 0),
                "rm_surf_menu_open_count": getattr(self, "rm_surf_menu_open_count", 0),
                "rm_surf_mon_selected_count": getattr(self, "rm_surf_mon_selected_count", 0),
                "rm_surf_aborted_count": getattr(self, "rm_surf_aborted_count", 0),
                "rm_surf_success_count": getattr(self, "rm_surf_success_count", 0),
                "rm_flash_success_count": getattr(self, "rm_flash_success_count", 0),
                "rm_intermediate_paid_count": getattr(
                    self, "rm_intermediate_paid_count", 0
                ),
                "rm_reward_from_success": getattr(self, "rm_reward_from_success", 0.0),
                "rm_transition_reward": getattr(self, "rm_reward_from_intermediate", 0.0),
                "rm_transition_reward_net": getattr(
                    self, "rm_reward_intermediate_net", 0.0
                ),
                "rm_clawback_total": getattr(self, "rm_clawback_total", 0.0),
                "rm_clawback_count": getattr(self, "rm_clawback_count", 0),
                "rm_step_delta": getattr(self, "rm_last_step_delta", 0.0),
                "last_rm_transition": getattr(self, "last_rm_transition_key", ""),
                "menu": {
                    "start_menu": self.seen_start_menu,
                    "pokemon_menu": self.seen_pokemon_menu,
                    "stats_menu": self.seen_stats_menu,
                    "bag_menu": self.seen_bag_menu,
                    "action_bag_menu": self.seen_action_bag_menu,
                },
                "blackout_check": self.blackout_check,
                "item_count": self.read_m(0xD31D),
                "reset_count": self.reset_count,
                "blackout_count": self.blackout_count,
                "pokecenter": np.sum(self.pokecenters),
                "pokecenter_heal": self.pokecenter_heal,
                "in_battle": self.read_m("wIsInBattle") > 0,
                "event": self.progress_reward.get("event", 0),
                "max_steps": self.get_max_steps(),
                # redundant but this is so we don't interfere with the swarm logic
                "required_count": len(self.required_events) + len(self.required_items),
                "use_ball_count": self.use_ball_count,
            }
            | {
                "exploration": {
                    tileset.name.lower(): sum(self.seen_coords.get(tileset.value, {}).values())
                    / exploration_sum
                    for tileset in Tilesets
                }
            }
            | {f"badge_{i+1}": bool(badges & (1 << i)) for i in range(8)},
            "events": {event: self.events.get_event(event) for event in REQUIRED_EVENTS}
            | {
                "rival3": int(self.read_m(0xD665) == 4),
                "saffron_guard": self.flags.get_bit("BIT_GAVE_SAFFRON_GUARDS_DRINK"),
            },
            "required_items": {item.name: item.value in bag_item_ids for item in REQUIRED_ITEMS},
            "useful_items": {item.name: item.value in bag_item_ids for item in USEFUL_ITEMS},
            # update_reward() 직후에만 호출됨. get_game_state_reward()를 다시 부르면
            # reward machine transition·rm_reward_total이 스텝당 추가로 돌아가 wandb·total_reward와 불일치.
            "reward": dict(self.progress_reward),
            "reward_sum": float(sum(self.progress_reward.values())),
            # Remove padding
            "pokemon_exploration_map": self.explore_map,
            # "cut_exploration_map": self.cut_explore_map,
            "species": [pokemon.Species for pokemon in self.party],
            "levels": [pokemon.Level for pokemon in self.party],
            "moves": [list(int(m) for m in pokemon.Moves) for pokemon in self.party],
        }

    def start_episode_video(self) -> None:
        """에피소드 말미 tail 구간용 단일 화면 MP4 (이미 열려 있으면 무시)."""
        if self._episode_video_writer is not None:
            return
        rollout_dir = self.video_dir / f"rollout{int(self.env_id):03d}"
        rollout_dir.mkdir(parents=True, exist_ok=True)
        out_path = rollout_dir / f"saved_video_ep{int(self.reset_count)}.mp4"
        self._episode_video_writer = media.VideoWriter(
            str(out_path), (144, 160), fps=self.fps, input_format="gray"
        )
        self._episode_video_writer.__enter__()

    def add_video_frame(self) -> None:
        if self._episode_video_writer is None:
            return
        self._episode_video_writer.add_image(self.render()[:, :])

    def _close_episode_video(self) -> None:
        w = self._episode_video_writer
        if w is None:
            return
        try:
            w.close()
        finally:
            self._episode_video_writer = None

    def get_game_coords(self):
        return (self.read_m("wXCoord"), self.read_m("wYCoord"), self.read_m("wCurMap"))

    def get_near_tile_memory_8(self) -> npt.NDArray[np.uint8]:
        """wTileMap 기준 플레이어 주변 8방 타일 ID (상·하·좌·우·좌상·우상·좌하·우하)."""
        _, w_tile_map = self.pyboy.symbol_lookup("wTileMap")
        tile_map = np.array(self.pyboy.memory[w_tile_map : w_tile_map + 20 * 18], dtype=np.uint8)
        tile_map = tile_map.reshape(18, 20)
        py, px = NEAR_TILE_PLAYER_ROW, NEAR_TILE_PLAYER_COL
        coords = (
            (py - 2, px),
            (py + 2, px),
            (py, px - 2),
            (py, px + 2),
            (py - 2, px - 2),
            (py - 2, px + 2),
            (py + 2, px - 2),
            (py + 2, px + 2),
        )
        out = np.zeros(NEAR_TILE_MEMORY_DIM, dtype=np.uint8)
        for i, (ry, cx) in enumerate(coords):
            if 0 <= ry < tile_map.shape[0] and 0 <= cx < tile_map.shape[1]:
                out[i] = tile_map[ry, cx]
        return out

    def _menu_flags_obs(self) -> npt.NDArray[np.uint8]:
        return np.array(
            [
                int(self._start_menu_open),
                self.seen_pokemon_menu,
                self.seen_bag_menu,
                self.seen_stats_menu,
                int(self.is_start_menu_illegal_navigation_active()),
            ],
            dtype=np.uint8,
        )

    def _ui_lock_obs(self) -> npt.NDArray[np.uint8]:
        return np.array(
            [
                int(self.read_m("wJoyIgnore") != 0),
                int(self.read_m("wFontLoaded") != 0),
                int(self.read_m("wIsInBattle") != 0),
            ],
            dtype=np.uint8,
        )

    def get_tile_in_front_of_player(self) -> int:
        """앞칸 타일 ID (WRAM ``wTileInFrontOfPlayer``). obs·RM·cut_hook과 동일 소스."""
        return int(self.pyboy.memory[self._ram["wTileInFrontOfPlayer"]])

    def get_current_menu_item(self) -> int:
        """메뉴 커서 (``wCurrentMenuItem``). 스타트 메뉴 포켓몼 줄은 보통 1."""
        return int(self.pyboy.memory[self._ram["wCurrentMenuItem"]])

    def _party_slot_obs(self) -> dict[str, npt.NDArray[np.uint8]]:
        """파티 인원·종·4기술(HM 포함). 빈 슬롯은 0으로 마스크."""
        party_count = self._clamp_party_count()
        species = np.zeros(6, dtype=np.uint8)
        moves = np.zeros((6, 4), dtype=np.uint8)
        for i in range(party_count):
            species[i] = self.party[i].Species
            moves[i] = np.array(self.party[i].Moves, dtype=np.uint8)
        return {
            "party_count": np.array([party_count], dtype=np.uint8),
            "species": species,
            "moves": moves,
        }

    def get_party_hm_cap_obs(self) -> npt.NDArray[np.uint8]:
        """슬롯별 cut/surf/flash 기술 보유 (빈 슬롯은 0)."""
        cap = np.zeros(PARTY_HM_CAP_SHAPE, dtype=np.uint8)
        party_count = self._clamp_party_count()
        for i in range(party_count):
            moves = self.party[i].Moves
            cap[i, 0] = int(TmHmMoves.CUT.value in moves)
            cap[i, 1] = int(TmHmMoves.SURF.value in moves)
            cap[i, 2] = int(TmHmMoves.FLASH.value in moves)
        return cap

    def _supports_surf_tile_scan(self) -> bool:
        tileset = self.read_m("wCurMapTileset")  # infrequent; symbol_lookup ok
        in_overworld = tileset == Tilesets.OVERWORLD.value
        in_plateau = tileset == Tilesets.PLATEAU.value
        in_cavern = tileset == Tilesets.CAVERN.value
        return bool(in_overworld or in_plateau or (in_cavern and self.get_game_coords() in SEAFOAM_SURF_SPOTS))

    def _get_adjacent_directional_tiles(self) -> dict[str, npt.NDArray[np.uint8]] | None:
        if not self._supports_surf_tile_scan():
            return None

        _, wTileMap = self.pyboy.symbol_lookup("wTileMap")
        tile_map = np.array(self.pyboy.memory[wTileMap : wTileMap + 20 * 18], dtype=np.uint8)
        tile_map = np.reshape(tile_map, (18, 20))
        y, x = 8, 8
        return {
            "up": tile_map[y - 2 : y, x : x + 2],
            "down": tile_map[y + 2 : y + 4, x : x + 2],
            "left": tile_map[y : y + 2, x - 2 : x],
            "right": tile_map[y : y + 2, x + 2 : x + 4],
        }

    def get_adjacent_water_count(self) -> int:
        directional_tiles = self._get_adjacent_directional_tiles()
        if directional_tiles is None:
            return 0
        return sum(int(SURF_TILE_IN_FRONT in tiles) for tiles in directional_tiles.values())

    def player_faces_adjacent_water(self) -> bool:
        """바라보는 방향 인접 타일에 물(0x14)이 있으면 True. RM SURF_DETECTED·surf_if_attempt와 동일 기준."""
        directional_tiles = self._get_adjacent_directional_tiles()
        if directional_tiles is None:
            return False
        direction = self.read_m("wSpritePlayerStateData1FacingDirection")
        if direction == 0x4:
            return bool(SURF_TILE_IN_FRONT in directional_tiles["up"])
        if direction == 0x0:
            return bool(SURF_TILE_IN_FRONT in directional_tiles["down"])
        if direction == 0x8:
            return bool(SURF_TILE_IN_FRONT in directional_tiles["left"])
        if direction == 0xC:
            return bool(SURF_TILE_IN_FRONT in directional_tiles["right"])
        return False

    def _action_points_to_adjacent_water(self, action: WindowEvent) -> bool:
        directional_tiles = self._get_adjacent_directional_tiles()
        if directional_tiles is None:
            return False

        direction = self.read_m("wSpritePlayerStateData1FacingDirection")
        return bool(
            (
                direction == 0x4
                and action == WindowEvent.PRESS_ARROW_UP
                and SURF_TILE_IN_FRONT in directional_tiles["up"]
            )
            or (
                direction == 0x0
                and action == WindowEvent.PRESS_ARROW_DOWN
                and SURF_TILE_IN_FRONT in directional_tiles["down"]
            )
            or (
                direction == 0x8
                and action == WindowEvent.PRESS_ARROW_LEFT
                and SURF_TILE_IN_FRONT in directional_tiles["left"]
            )
            or (
                direction == 0xC
                and action == WindowEvent.PRESS_ARROW_RIGHT
                and SURF_TILE_IN_FRONT in directional_tiles["right"]
            )
        )

    def get_map_pal_offset(self) -> int:
        _, addr = self.pyboy.symbol_lookup("wMapPalOffset")
        return int(self.pyboy.memory[addr])

    def get_rm_flash_cycle_start(self) -> int:
        return 0

    def get_max_steps(self):
        return max(
            0,
            self.max_steps,
            self.max_steps
            * (len(self.required_events) + len(self.required_items))
            * self.max_steps_scaling,
        )

    def update_seen_coords(self):
        inc = 0.5 if (self.read_m("wMovementFlags") & 0b1000_0000) else self.exploration_inc

        x_pos, y_pos, map_n = self.get_game_coords()
        # self.seen_coords[(x_pos, y_pos, map_n)] = inc
        cur_map_tileset = self.read_m("wCurMapTileset")
        if cur_map_tileset not in self.seen_coords:
            self.seen_coords[cur_map_tileset] = {}
        self.seen_coords[cur_map_tileset][(x_pos, y_pos, map_n)] = min(
            self.seen_coords[cur_map_tileset].get((x_pos, y_pos, map_n), 0.0) + inc,
            self.exploration_max,
        )
        # TODO: Turn into a wrapper?
        self.explore_map[local_to_global(y_pos, x_pos, map_n)] = min(
            self.explore_map[local_to_global(y_pos, x_pos, map_n)] + inc,
            self.exploration_max,
        )
        self.reward_explore_map[local_to_global(y_pos, x_pos, map_n)] = min(
            self.explore_map[local_to_global(y_pos, x_pos, map_n)] + inc,
            self.exploration_max,
        ) * (self.map_id_scalefactor if self.scale_map_id(map_n) else 1.0)
        # self.seen_global_coords[local_to_global(y_pos, x_pos, map_n)] = 1
        self.seen_map_ids[map_n] = 1

    def update_a_press(self):
        if self.read_m("wIsInBattle") != 0 or self.read_m("wFontLoaded"):
            return

        direction = self.read_m("wSpritePlayerStateData1FacingDirection")
        x_pos, y_pos, map_n = self.get_game_coords()
        if direction == 0:
            y_pos += 1
        if direction == 4:
            y_pos -= 1
        if direction == 8:
            x_pos -= 1
        if direction == 0xC:
            x_pos += 1
        # if self.scale_map_id(map_n):
        self.a_press.add((x_pos, y_pos, map_n))

    def get_explore_map(self):
        explore_map = np.zeros(GLOBAL_MAP_SHAPE)
        for inner in self.seen_coords.values():
            for (x, y, map_n), v in inner.items():
                gy, gx = local_to_global(y, x, map_n)
                if gy < 0 or gy >= explore_map.shape[0] or gx < 0 or gx >= explore_map.shape[1]:
                    print(f"coord out of bounds! global: ({gx}, {gy}) game: ({x}, {y}, {map_n})")
                else:
                    explore_map[gy, gx] = v

        return explore_map

    def update_reward(self):
        # 보상 dict를 만들기 전에 서브클래스가 스텝당 1회만 해야 할 갱신(RM transition 등)을 수행.
        pre = getattr(self, "_before_progress_reward", None)
        if callable(pre):
            pre()
        self.progress_reward = self.get_game_state_reward()

        # 모든 보상 항목(step_penalty 포함)을 cumulative로 관리한다.
        # 스텝 보상 = Σ(누적값) 델타 → 어느 항목도 special-case 없이 동일하게 처리.
        new_total = sum(self.progress_reward.values())
        new_step = new_total - self.total_reward
        self.total_reward = new_total
        return new_step

    def read_m(self, addr: str | int) -> int:
        if isinstance(addr, str):
            val = self.pyboy.memory[self.pyboy.symbol_lookup(addr)[1]]
        else:
            val = self.pyboy.memory[addr]
        if isinstance(val, (bytes, bytearray)):
            return int(val[0]) if len(val) > 0 else 0
        return int(val)

    def read_short(self, addr: str | int) -> int:
        if isinstance(addr, str):
            _, addr = self.pyboy.symbol_lookup(addr)
        data = self.pyboy.memory[addr : addr + 2]
        return int(data[0] << 8) + int(data[1])

    def read_bit(self, addr: str | int, bit: int) -> bool:
        # add padding so zero will read '0b100000000' instead of '0b0'
        return bool(int(self.read_m(addr)) & (1 << bit))

    def read_event_bits(self):
        _, addr = self.pyboy.symbol_lookup("wEventFlags")
        return self.pyboy.memory[addr : addr + EVENTS_FLAGS_LENGTH]

    def get_badges(self):
        return self.read_m("wObtainedBadges").bit_count()

    def read_party(self) -> list[int]:
        _, addr = self.pyboy.symbol_lookup("wPartySpecies")
        party_length = self._clamp_party_count()
        if party_length == 0:
            return []
        return [int(x) for x in self.pyboy.memory[addr : addr + party_length]]

    @abstractmethod
    def get_game_state_reward(self):
        raise NotImplementedError()

    def get_reward_machine_state_id(self) -> int:
        reward_machine = getattr(self, "reward_machine", None)
        if reward_machine is None:
            return int(RewardMachineState.IDLE)
        return reward_machine.state_id

    def get_reward_machine_hm_target_id(self) -> int:
        reward_machine = getattr(self, "reward_machine", None)
        if reward_machine is None:
            return int(HMTarget.NONE)
        return int(reward_machine.hm_target)

    def get_hm_supervision_target_id(self) -> int:
        hm_supervision_target = getattr(self, "hm_supervision_target", None)
        if hm_supervision_target is not None:
            return int(hm_supervision_target)
        return self.get_reward_machine_hm_target_id()

    def update_max_op_level(self):
        # opp_base_level = 5
        # Defensive clamp: pokered enemy party slots are 1..6.
        # Some custom/debug states can transiently expose invalid counts (e.g. 7),
        # which would make symbol lookup request non-existent labels like
        # `wEnemyMon7Level`.
        enemy_party_count = min(max(int(self.read_m("wEnemyPartyCount")), 0), MAX_ENEMY_PARTY_SIZE)
        opponent_level = max(
            [0]
            + [self.read_m(f"wEnemyMon{i+1}Level") for i in range(enemy_party_count)]
        )
        # - opp_base_level

        self.max_opponent_level = max(0, self.max_opponent_level, opponent_level)
        return self.max_opponent_level

    def update_health(self):
        cur_health = self.read_hp_fraction()
        # if health increased and party size did not change
        if cur_health > self.last_health and self.read_m("wPartyCount") == self.party_size:
            if self.last_health > 0:
                self.total_heal_health += cur_health - self.last_health
            else:
                self.died_count += 1

    def update_pokedex(self):
        # TODO: Make a hook
        _, wPokedexOwned = self.pyboy.symbol_lookup("wPokedexOwned")
        _, wPokedexOwnedEnd = self.pyboy.symbol_lookup("wPokedexOwnedEnd")
        _, wPokedexSeen = self.pyboy.symbol_lookup("wPokedexSeen")
        _, wPokedexSeenEnd = self.pyboy.symbol_lookup("wPokedexSeenEnd")

        caught_mem = self.pyboy.memory[wPokedexOwned:wPokedexOwnedEnd]
        seen_mem = self.pyboy.memory[wPokedexSeen:wPokedexSeenEnd]
        self.caught_pokemon = np.unpackbits(np.array(caught_mem, dtype=np.uint8))
        self.seen_pokemon = np.unpackbits(np.array(seen_mem, dtype=np.uint8))

    def update_tm_hm_obtained_move_ids(self):
        # TODO: Make a hook
        # Scan party
        for i in range(self._clamp_party_count()):
            _, addr = self.pyboy.symbol_lookup(f"wPartyMon{i+1}Moves")
            for move_id in self.pyboy.memory[addr : addr + 4]:
                # if move_id in TM_HM_MOVES:
                self.obtained_move_ids[move_id] = 1
        """
        # Scan current box (since the box doesn't auto increment in pokemon red)
        num_moves = 4
        box_struct_length = 25 * num_moves * 2
        for i in range(self.pyboy.memory[0xDA80)):
            offset = i * box_struct_length + 0xDA96
            if self.pyboy.memory[offset) != 0:
                for j in range(4):
                    move_id = self.pyboy.memory[offset + j + 8)
                    if move_id != 0:
                        self.obtained_move_ids[move_id] = 1
        """

    def remove_all_nonuseful_items(self):
        _, wNumBagItems = self.pyboy.symbol_lookup("wNumBagItems")
        if self.pyboy.memory[wNumBagItems] == MAX_ITEM_CAPACITY:
            _, wBagItems = self.pyboy.symbol_lookup("wBagItems")
            bag_items = self.pyboy.memory[wBagItems : wBagItems + MAX_ITEM_CAPACITY * 2]
            # Fun fact: The way they test if an item is an hm in code is by testing the item id
            # is greater than or equal to 0xC4 (the item id for HM_01)

            # TODO either remove or check if guard has been given drink
            # guard given drink are 4 script pointers to check, NOT an event
            new_bag_items = [
                (item, quantity)
                for item, quantity in zip(bag_items[::2], bag_items[1::2])
                if Items(item) in KEY_ITEMS | REQUIRED_ITEMS | HM_ITEMS
            ]
            # Write the new count back to memory
            self.pyboy.memory[wNumBagItems] = len(new_bag_items)
            # 0 pad
            new_bag_items += [(255, 255)] * (20 - len(new_bag_items))
            # now flatten list
            new_bag_items = list(sum(new_bag_items, ()))
            # now write back to list
            self.pyboy.memory[wBagItems : wBagItems + len(new_bag_items)] = new_bag_items

            _, wBagSavedMenuItem = self.pyboy.symbol_lookup("wBagSavedMenuItem")
            _, wListScrollOffset = self.pyboy.symbol_lookup("wListScrollOffset")
            # TODO: Make this point to the location of the last removed item
            # Should be something like the current location - the number of items
            # that have been removed - 1
            self.pyboy.memory[wBagSavedMenuItem] = 0
            self.pyboy.memory[wListScrollOffset] = 0

    def reverse_damage(self):
        for i in range(self._clamp_party_count()):
            _, wPartyMonHP = self.pyboy.symbol_lookup(f"wPartyMon{i+1}HP")
            _, wPartymonMaxHP = self.pyboy.symbol_lookup(f"wPartyMon{i+1}MaxHP")
            self.pyboy.memory[wPartyMonHP] = 0
            self.pyboy.memory[wPartyMonHP + 1] = 128
            self.pyboy.memory[wPartymonMaxHP] = 0
            self.pyboy.memory[wPartymonMaxHP + 1] = 128

    def read_hp_fraction(self):
        party_size = self._clamp_party_count()
        hp_sum = sum(self.read_short(f"wPartyMon{i+1}HP") for i in range(party_size))
        max_hp_sum = sum(self.read_short(f"wPartyMon{i+1}MaxHP") for i in range(party_size))
        max_hp_sum = max(max_hp_sum, 1)
        return hp_sum / max_hp_sum

    def update_map_progress(self):
        map_idx = self.read_m(0xD35E)
        self.max_map_progress = max(0, self.max_map_progress, self.get_map_progress(map_idx))

    def get_map_progress(self, map_idx):
        if map_idx in self.essential_map_locations.keys():
            return self.essential_map_locations[map_idx]
        else:
            return -1

    def get_items_in_bag(self) -> Iterable[Items]:
        # Defensive handling for custom/corrupted save states.
        num_bag_items = min(
            max(int(self.pyboy.memory[self._ram["wNumBagItems"]]), 0), MAX_ITEM_CAPACITY
        )
        try:
            addr = self._ram["wBagItems"]
            raw_items = self.pyboy.memory[addr : addr + 2 * num_bag_items][::2]
        except Exception:
            return []

        bag_items: list[Items] = []
        for item in raw_items:
            try:
                bag_items.append(Items(item))
            except ValueError:
                continue
        return bag_items

    def get_hm_count(self) -> int:
        return len(HM_ITEMS.intersection(self.get_items_in_bag()))

    def get_levels_reward(self):
        # Level reward
        party_levels = self.read_party()
        self.max_level_sum = max(0, self.max_level_sum, sum(party_levels))
        if self.max_level_sum < 30:
            level_reward = 1 * self.max_level_sum
        else:
            level_reward = 30 + (self.max_level_sum - 30) / 4
        return level_reward

    def get_required_events(self) -> set[str]:
        return (
            set(
                event
                for event, v in zip(REQUIRED_EVENTS, self.events.get_events(REQUIRED_EVENTS))
                if v
            )
            | ({"rival3"} if (self.read_m("wSSAnne2FCurScript") == 4) else set())
            | ({"saffron_guard"} if self.flags.get_bit("BIT_GAVE_SAFFRON_GUARDS_DRINK") else set())
        )

    def get_required_items(self) -> set[str]:
        # Some custom/debug save states can have transiently invalid bag counters.
        # Clamp to legal bag capacity to avoid out-of-range PyBoy memory slicing.
        wNumBagItems = min(
            max(int(self.pyboy.memory[self._ram["wNumBagItems"]]), 0), MAX_ITEM_CAPACITY
        )
        try:
            wBagItems = self._ram["wBagItems"]
            bag_items = self.pyboy.memory[wBagItems : wBagItems + wNumBagItems * 2 : 2]
        except Exception:
            # Some handcrafted save states can momentarily expose invalid bag pointers.
            # In that case, skip required-item extraction for this step instead of crashing.
            return set()
        required_items = set()
        for item in bag_items:
            try:
                bag_item = Items(item)
            except ValueError:
                continue
            if bag_item in REQUIRED_ITEMS:
                required_items.add(bag_item.name)
        return required_items

    def get_events_sum(self):
        # adds up all event flags, exclude museum ticket
        return max(
            sum(
                [
                    self.read_m(i).bit_count()
                    for i in range(EVENT_FLAGS_START, EVENT_FLAGS_START + EVENTS_FLAGS_LENGTH)
                ]
            )
            - self.base_event_flags
            - int(self.read_bit(*MUSEUM_TICKET)),
            0,
        )

    def scale_map_id(self, map_n: int) -> bool:
        map_id = MapIds(map_n)
        if map_id not in MAP_ID_COMPLETION_EVENTS:
            return False
        after, until = MAP_ID_COMPLETION_EVENTS[map_id]

        if all(
            (item.startswith("EVENT_") and self.events.get_event(item))
            or (item.startswith("HS_") and self.missables.get_missable(item))
            or (item.startswith("BIT_") and self.flags.get_bit(item))
            for item in after
        ) and any(
            (item.startswith("EVENT_") and not self.events.get_event(item))
            or (item.startswith("HS_") and not self.missables.get_missable(item))
            or (item.startswith("BIT_") and not self.flags.get_bit(item))
            for item in until
        ):
            return True
        return False

    def check_num_bag_items(self):
        _, wBagItems = self.pyboy.symbol_lookup("wBagItems")
        _, wNumBagItems = self.pyboy.symbol_lookup("wNumBagItems")
        numBagItems = self.read_m(wNumBagItems)
        bag = np.array(self.pyboy.memory[wBagItems : wBagItems + 40], dtype=np.uint8)
        if numBagItems >= 20:
            print(
                f"WARNING: env id {int(self.env_id)} contains a full bag with items: {[Items(item) for item in bag[::2]]}"
            )

    def close(self):
        if getattr(self, "save_video", False):
            self._close_episode_video()