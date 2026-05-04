from collections import OrderedDict
import random
import gymnasium as gym
import numpy as np

from omegaconf import DictConfig
from pokemonred_puffer.environment import RedGymEnv
from pokemonred_puffer.global_map import local_to_global


def _decay_seen_coords(
    seen_coords: dict[int, dict[tuple[int, int, int], float | int]],
    factor: float,
    floor: float = 0.15,
) -> None:
    """seen_coords is nested: tileset_id -> (x, y, map_n) -> visit weight."""
    for inner in seen_coords.values():
        for key in list(inner.keys()):
            inner[key] = max(floor, float(inner[key]) * factor)


def _delete_seen_coord(
    seen_coords: dict[int, dict[tuple[int, int, int], float | int]],
    coord: tuple[int, int, int],
) -> None:
    for tileset, inner in list(seen_coords.items()):
        if coord in inner:
            del inner[coord]
            if not inner:
                del seen_coords[tileset]
            return


def _lower_seen_coords_positive(
    seen_coords: dict[int, dict[tuple[int, int, int], float | int]],
    value: float,
) -> None:
    for inner in seen_coords.values():
        for key in list(inner.keys()):
            if inner[key] > 0:
                inner[key] = value


class LRUCache:
    # initialising capacity
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def contains(self, key: tuple[int, int, int]) -> bool:
        if key not in self.cache:
            return False
        else:
            self.cache.move_to_end(key)
            return True

    def put(self, key: tuple[int, int, int]) -> tuple[int, int, int] | None:
        self.cache[key] = 1
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            return self.cache.popitem(last=False)[0]

    def clear(self):
        self.cache.clear()


# Yes. This wrapper mutates the env.
# Is that good? No.
# Am I doing it anyway? Yes.
# Why? To save memory
class DecayWrapper(gym.Wrapper):
    def __init__(self, env: RedGymEnv, reward_config: DictConfig):
        super().__init__(env)
        self.step_forgetting_factor = reward_config.step_forgetting_factor
        self.forgetting_frequency = reward_config.forgetting_frequency

    def step(self, action):
        if self.env.unwrapped.step_count % self.forgetting_frequency == 0:
            self.step_forget_explore()

        return self.env.step(action)

    def step_forget_explore(self):
        factor = self.step_forgetting_factor
        _decay_seen_coords(self.env.unwrapped.seen_coords, factor["coords"])
        self.env.unwrapped.seen_map_ids *= factor["map_ids"]
        self.env.unwrapped.seen_npcs.update(
            (k, max(0.15, v * factor["npc"]))
            for k, v in self.env.unwrapped.seen_npcs.items()
        )
        self.env.unwrapped.seen_warps.update(
            (k, max(0.15, v * factor["coords"]))
            for k, v in self.env.unwrapped.seen_warps.items()
        )
        self.env.unwrapped.explore_map *= factor["explore"]
        self.env.unwrapped.explore_map[self.env.unwrapped.explore_map > 0] = np.clip(
            self.env.unwrapped.explore_map[self.env.unwrapped.explore_map > 0], 0.15, 1
        )
        self.env.unwrapped.reward_explore_map *= factor["explore"]
        self.env.unwrapped.reward_explore_map[self.env.unwrapped.explore_map > 0] = np.clip(
            self.env.unwrapped.reward_explore_map[self.env.unwrapped.explore_map > 0], 0.15, 1
        )
        # Optional keys: only decay if present in config (not needed for HM agent)
        if "hidden_objs" in factor:
            self.env.unwrapped.seen_hidden_objs.update(
                (k, max(0.15, v * factor["hidden_objs"]))
                for k, v in self.env.unwrapped.seen_hidden_objs.items()
            )
        if "signs" in factor:
            self.env.unwrapped.seen_signs.update(
                (k, max(0.15, v * factor["signs"]))
                for k, v in self.env.unwrapped.seen_signs.items()
            )
        if "safari_zone_steps" in factor:
            self.env.unwrapped.safari_zone_steps.update(
                (k, max(0.15, v * factor["safari_zone_steps"]))
                for k, v in self.env.unwrapped.safari_zone_steps.items()
            )

        if self.env.unwrapped.read_m("wIsInBattle") == 0:
            self.env.unwrapped.seen_start_menu *= factor["start_menu"]
            self.env.unwrapped.seen_pokemon_menu *= factor["pokemon_menu"]
            self.env.unwrapped.seen_stats_menu *= factor["stats_menu"]
            self.env.unwrapped.seen_bag_menu *= factor["bag_menu"]
            self.env.unwrapped.seen_action_bag_menu *= factor["action_bag_menu"]


class MaxLengthWrapper(gym.Wrapper):
    def __init__(self, env: RedGymEnv, reward_config: DictConfig):
        super().__init__(env)
        self.capacity = reward_config.capacity
        self.cache = LRUCache(capacity=self.capacity)

    def step(self, action):
        if self.env.unwrapped.step_count >= self.env.unwrapped.get_max_steps():
            self.cache.clear()

        step = self.env.step(action)
        player_x, player_y, map_n = self.env.unwrapped.get_game_coords()
        # Walrus operator does not support tuple unpacking
        if coord := self.cache.put((player_x, player_y, map_n)):
            x, y, n = coord
            _delete_seen_coord(self.env.unwrapped.seen_coords, (x, y, n))
            self.env.unwrapped.explore_map[local_to_global(y, x, n)] = 0
            self.env.unwrapped.reward_explore_map[local_to_global(y, x, n)] = 0
        return step


class OnResetExplorationWrapper(gym.Wrapper):
    def __init__(self, env: RedGymEnv, reward_config: DictConfig):
        super().__init__(env)
        self.full_reset_frequency = reward_config.full_reset_frequency
        self.jitter = reward_config.jitter
        self.counter = 0

    def step(self, action):
        if self.env.unwrapped.step_count >= self.env.unwrapped.get_max_steps():
            if (self.counter + random.randint(0, self.jitter)) >= self.full_reset_frequency:
                self.counter = 0
                self.env.unwrapped.explore_map *= 0
                self.env.unwrapped.reward_explore_map *= 0
                self.env.unwrapped.cut_explore_map *= 0
                self.env.unwrapped.seen_coords.clear()
                self.env.unwrapped.seen_map_ids *= 0
                self.env.unwrapped.seen_npcs.clear()
                # HM 사용 좌표(valid_/invalid_*_coords, *_tiles)는 소프트 리셋에서 비우지 않는다.
                # 소프트 리셋마다 비우면 RM이 같은 나무/물 타일을 다시 SUCCESS로 인식해
                # rm_*_success_count가 cut_count(에피소드 말미 dict 크기)와 수십 배 어긋난다.
                # 진짜 에피소드 reset(reset_mem)에서만 비우도록 둔다.
                self.env.unwrapped.seen_warps.clear()
                self.env.unwrapped.seen_hidden_objs.clear()
                self.env.unwrapped.seen_signs.clear()
                self.env.unwrapped.safari_zone_steps.update(
                    (k, 0) for k in self.env.unwrapped.safari_zone_steps.keys()
                )
            self.counter += 1
        return self.env.step(action)


class OnResetLowerToFixedValueWrapper(gym.Wrapper):
    def __init__(self, env: RedGymEnv, reward_config: DictConfig):
        super().__init__(env)
        self.fixed_value = reward_config.fixed_value

    def step(self, action):
        if self.env.unwrapped.step_count >= self.env.unwrapped.get_max_steps():
            _lower_seen_coords_positive(
                self.env.unwrapped.seen_coords, self.fixed_value["coords"]
            )
            self.env.unwrapped.seen_map_ids[self.env.unwrapped.seen_map_ids > 0] = self.fixed_value[
                "map_ids"
            ]
            self.env.unwrapped.seen_npcs.update(
                (k, self.fixed_value["npc"])
                for k, v in self.env.unwrapped.seen_npcs.items()
                if v > 0
            )
            self.env.unwrapped.valid_cut_coords.update(
                (k, self.fixed_value["valid_cut"])
                for k, v in self.env.unwrapped.valid_cut_coords.items()
                if v > 0
            )
            self.env.unwrapped.invalid_cut_coords.update(
                (k, self.fixed_value["invalid_cut"])
                for k, v in self.env.unwrapped.invalid_cut_coords.items()
                if v > 0
            )
            self.env.unwrapped.valid_pokeflute_coords.update(
                (k, self.fixed_value["valid_pokeflute"])
                for k, v in self.env.unwrapped.valid_pokeflute_coords.items()
                if v > 0
            )
            self.env.unwrapped.invalid_pokeflute_coords.update(
                (k, self.fixed_value["invalid_pokeflute"])
                for k, v in self.env.unwrapped.invalid_pokeflute_coords.items()
                if v > 0
            )
            self.env.unwrapped.valid_surf_coords.update(
                (k, self.fixed_value["valid_surf"])
                for k, v in self.env.unwrapped.valid_surf_coords.items()
                if v > 0
            )
            self.env.unwrapped.invalid_surf_coords.update(
                (k, self.fixed_value["invalid_surf"])
                for k, v in self.env.unwrapped.invalid_surf_coords.items()
                if v > 0
            )
            self.env.unwrapped.explore_map[self.env.unwrapped.explore_map > 0] = self.fixed_value[
                "explore"
            ]
            self.env.unwrapped.reward_explore_map[self.env.unwrapped.reward_explore_map > 0] = (
                self.fixed_value["explore"]
            )
            self.env.unwrapped.cut_explore_map[self.env.unwrapped.cut_explore_map > 0] = (
                self.fixed_value["invalid_cut"]
            )
            self.env.unwrapped.seen_warps.update(
                (k, self.fixed_value["coords"])
                for k, v in self.env.unwrapped.seen_warps.items()
                if v > 0
            )
            self.env.unwrapped.seen_hidden_objs.update(
                (k, self.fixed_value["hidden_objs"])
                for k, v in self.env.unwrapped.seen_hidden_objs.items()
                if v > 0
            )
            self.env.unwrapped.seen_signs.update(
                (k, self.fixed_value["signs"])
                for k, v in self.env.unwrapped.seen_signs.items()
                if v > 0
            )
            self.env.unwrapped.safari_zone_steps.update(
                (k, self.fixed_value["safari_zone_steps"])
                for k, v in self.env.unwrapped.safari_zone_steps.items()
                if v > 0
            )
        return self.env.unwrapped.step(action)
