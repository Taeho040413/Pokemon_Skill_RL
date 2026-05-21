import pufferlib.emulation
import pufferlib.models
import pufferlib.pytorch
import torch
from torch import nn

from pokemonred_puffer.environment import PIXEL_VALUES, VALID_ACTIONS_STR
from pokemonred_puffer.rewards.reward_machine import (
    HMTarget,
    RewardMachineState,
    START_MENU_POKEMON_CURSOR,
    hm_supervision_label_from_rm_state,
)


HM_ACTIONS = ("cut", "surf", "flash", "none")
HM_FEATURE_COUNT = len(HMTarget)
HM_LATENT_DIM = 256
PARTY_HM_CAP_FLAT = 6 * 3
PARTY_COUNT_DIM = 1
PARTY_SPECIES_DIM = 6
PARTY_MOVES_FLAT = 6 * 4
PARTY_SLOT_FLAT = PARTY_COUNT_DIM + PARTY_SPECIES_DIM + PARTY_MOVES_FLAT
FACING_DIR_COUNT = 4  # obs direction: wFacingDirection // 4 → 0 down, 1 up, 2 left, 3 right

_LEGACY_RM_TO_HM_SUPERVISION = torch.tensor(
    [hm_supervision_label_from_rm_state(i) for i in range(len(RewardMachineState))],
    dtype=torch.long,
)


def one_hot(tensor, num_classes):
    index = torch.arange(0, num_classes, device=tensor.device)
    return (tensor.view([*tensor.shape, 1]) == index.view([1] * tensor.ndim + [num_classes])).to(
        torch.int64
    )


class MultiConvolutionalRNN(pufferlib.models.LSTMWrapper):
    def __init__(self, env, policy, input_size=512, hidden_size=512, num_layers=1):
        super().__init__(env, policy, input_size, hidden_size, num_layers)


class MultiConvolutionalPolicy(nn.Module):
    """HM 타워(상황) → z_hm. Policy 타워: z_hm + rm_state + α·hm_probs → z_policy → LSTM."""

    def __init__(
        self,
        env: pufferlib.emulation.GymnasiumPufferEnv,
        hidden_size: int = 512,
        hm_latent_size: int = HM_LATENT_DIM,
        rm_state_embedding_dim: int = 4,
        hm_hidden_size: int = 128,
        hm_feature_alpha_init: float = 0.1,
        hm_action_beta_init: float = 0.1,
        hm_menu_action_beta_init: float = 0.08,
        channels_last: bool = True,
        downsample: int = 1,
        use_screen: bool | None = None,
    ):
        super().__init__()
        self.dtype = pufferlib.pytorch.nativize_dtype(env.emulated)
        self.num_actions = env.single_action_space.n
        self.channels_last = channels_last
        self.downsample = downsample
        self.hm_latent_size = int(hm_latent_size)
        self.policy_hidden_size = int(hidden_size)

        if use_screen is None:
            use_screen = bool(getattr(env.unwrapped.env, "include_screen_obs", True))
        self.use_screen = use_screen

        if self.use_screen:
            self.screen_cnn = nn.Sequential(
                nn.LazyConv2d(32, 8, stride=2),
                nn.ReLU(),
                nn.LazyConv2d(64, 4, stride=2),
                nn.ReLU(),
                nn.LazyConv2d(64, 3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.LazyLinear(self.hm_latent_size),
                nn.ReLU(),
            )
        else:
            self.register_buffer(
                "_zero_screen_latent",
                torch.zeros(1, self.hm_latent_size),
                persistent=False,
            )

        self.encode_linear_hm = nn.Sequential(
            nn.LazyLinear(self.hm_latent_size),
            nn.ReLU(),
        )
        self.encode_linear_policy = nn.Sequential(
            nn.LazyLinear(self.policy_hidden_size),
            nn.ReLU(),
        )
        self.hm_head = nn.Sequential(
            nn.Linear(hm_hidden_size, hm_hidden_size),
            nn.ReLU(),
            nn.Linear(hm_hidden_size, HM_FEATURE_COUNT),
        )
        self.hm_pre_head = nn.LazyLinear(hm_hidden_size)

        self.hm_feature_alpha = nn.Parameter(torch.tensor(hm_feature_alpha_init))
        self.register_buffer(
            "hm_action_beta",
            torch.tensor(hm_action_beta_init, dtype=torch.float32),
            persistent=False,
        )
        self.last_hm_logits = None
        self.last_hm_probs = None
        self.last_hm_target = None
        self.last_rm_state = None
        self.last_menu_flags = None
        self.last_current_menu_item = None

        self.register_buffer(
            "hm_menu_action_beta",
            torch.tensor(hm_menu_action_beta_init, dtype=torch.float32),
            persistent=False,
        )
        menu_detected_bias = torch.zeros(self.num_actions, dtype=torch.float32)
        menu_scroll_bias = torch.zeros(self.num_actions, dtype=torch.float32)
        menu_open_bias = torch.zeros(self.num_actions, dtype=torch.float32)
        menu_detected_bias[VALID_ACTIONS_STR.index("start")] = 1.0
        menu_scroll_bias[VALID_ACTIONS_STR.index("down")] = 1.0
        menu_open_bias[VALID_ACTIONS_STR.index("a")] = 1.0
        self.register_buffer("hm_menu_detected_bias", menu_detected_bias, persistent=False)
        self.register_buffer("hm_menu_scroll_bias", menu_scroll_bias, persistent=False)
        self.register_buffer("hm_menu_open_bias", menu_open_bias, persistent=False)
        self.register_buffer(
            "_rm_menu_detected_states",
            torch.tensor(
                [
                    int(RewardMachineState.CUT_DETECTED),
                    int(RewardMachineState.SURF_DETECTED),
                    int(RewardMachineState.FLASH_DETECTED),
                ],
                dtype=torch.long,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_rm_menu_scroll_states",
            torch.tensor(
                [
                    int(RewardMachineState.CUT_START_MENU),
                    int(RewardMachineState.SURF_START_MENU),
                    int(RewardMachineState.FLASH_START_MENU),
                ],
                dtype=torch.long,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_rm_menu_open_states",
            torch.tensor(
                [
                    int(RewardMachineState.CUT_MENU_OPEN),
                    int(RewardMachineState.SURF_MENU_OPEN),
                    int(RewardMachineState.FLASH_MENU_OPEN),
                ],
                dtype=torch.long,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_rm_menu_party_scroll_states",
            torch.tensor(
                [
                    int(RewardMachineState.CUT_PARTY_MENU),
                    int(RewardMachineState.SURF_PARTY_MENU),
                    int(RewardMachineState.FLASH_PARTY_MENU),
                ],
                dtype=torch.long,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_rm_menu_confirm_states",
            torch.tensor(
                [
                    int(RewardMachineState.CUT_MON_SELECTED),
                    int(RewardMachineState.SURF_MON_SELECTED),
                    int(RewardMachineState.FLASH_MON_SELECTED),
                ],
                dtype=torch.long,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_rm_cut_mon_selected",
            torch.tensor([int(RewardMachineState.CUT_MON_SELECTED)], dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_rm_surf_mon_selected",
            torch.tensor([int(RewardMachineState.SURF_MON_SELECTED)], dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_rm_surf_detected_only",
            torch.tensor([int(RewardMachineState.SURF_DETECTED)], dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_rm_surf_party_only",
            torch.tensor([int(RewardMachineState.SURF_PARTY_MENU)], dtype=torch.long),
            persistent=False,
        )
        self.surf_menu_action_scale = 3.5
        self.register_buffer(
            "_rm_flash_hm_action_states",
            torch.tensor(
                [
                    int(RewardMachineState.FLASH_DETECTED),
                    int(RewardMachineState.FLASH_START_MENU),
                    int(RewardMachineState.FLASH_MENU_OPEN),
                    int(RewardMachineState.FLASH_PARTY_MENU),
                    int(RewardMachineState.FLASH_MON_SELECTED),
                ],
                dtype=torch.long,
            ),
            persistent=False,
        )

        self.actor = nn.LazyLinear(self.num_actions)
        self.value_fn = nn.LazyLinear(1)

        # Python int: decode_actions에서 .item() 쓰면 torch.compile graph break.
        self._hm_a_idx = VALID_ACTIONS_STR.index("a")
        self._hm_start_idx = VALID_ACTIONS_STR.index("start")
        self._hm_b_idx = VALID_ACTIONS_STR.index("b")
        self._hm_down_idx = VALID_ACTIONS_STR.index("down")
        self._hm_up_idx = VALID_ACTIONS_STR.index("up")
        self.register_buffer(
            "_menu_mask_scroll",
            torch.tensor(
                [self._hm_b_idx, self._hm_down_idx, self._hm_up_idx], dtype=torch.long
            ),
            persistent=False,
        )
        self.register_buffer(
            "_menu_mask_confirm",
            torch.tensor([self._hm_b_idx, self._hm_a_idx], dtype=torch.long),
            persistent=False,
        )

        self.two_bit = env.unwrapped.env.two_bit
        self.use_global_map = env.unwrapped.env.use_global_map

        if self.use_screen:
            self.register_buffer(
                "screen_buckets", torch.tensor(PIXEL_VALUES, dtype=torch.uint8), persistent=False
            )
            self.register_buffer(
                "unpack_mask",
                torch.tensor([0xC0, 0x30, 0x0C, 0x03], dtype=torch.uint8),
                persistent=False,
            )
            self.register_buffer(
                "unpack_shift", torch.tensor([6, 4, 2, 0], dtype=torch.uint8), persistent=False,
            )

        self.map_embeddings = nn.Embedding(0xFF, 4, dtype=torch.float32)
        self.rm_state_embeddings = nn.Embedding(
            len(RewardMachineState), rm_state_embedding_dim, dtype=torch.float32
        )

    @staticmethod
    def _hm_supervision_targets_from_obs(observations) -> torch.Tensor:
        if "hm_aux_label" in observations:
            return (
                observations["hm_aux_label"]
                .long()
                .reshape(-1)
                .clamp(0, HM_FEATURE_COUNT - 1)
            )
        if "hm_supervision_target" in observations:
            return (
                observations["hm_supervision_target"]
                .long()
                .reshape(-1)
                .clamp(0, HM_FEATURE_COUNT - 1)
            )
        rm_idx = observations["rm_state"].long().reshape(-1).clamp(
            0, len(RewardMachineState) - 1
        )
        return _LEGACY_RM_TO_HM_SUPERVISION.to(rm_idx.device)[rm_idx]

    def _decode_screen(self, screen: torch.Tensor) -> torch.Tensor:
        restored_shape = (
            screen.shape[0],
            screen.shape[1],
            screen.shape[2] * 4,
            screen.shape[3],
        )
        if self.two_bit:
            screen = torch.index_select(
                self.screen_buckets,
                0,
                ((screen.reshape((-1, 1)) & self.unpack_mask) >> self.unpack_shift)
                .flatten()
                .int(),
            ).reshape(restored_shape)
        image = screen
        if self.channels_last:
            image = image.permute(0, 3, 1, 2)
        if self.downsample > 1:
            image = image[:, :, :: self.downsample, :: self.downsample]
        return image

    def _party_slot_features(self, observations) -> torch.Tensor:
        """파티 인원 + 종(6) + 기술 ID(6×4, HM·비전머신 포함)."""
        party_count = observations["party_count"].float() / 6.0
        species = observations["species"].float() / 255.0
        moves = (
            observations["moves"]
            .float()
            .reshape(observations["moves"].shape[0], PARTY_MOVES_FLAT)
            / 255.0
        )
        return torch.cat((party_count, species, moves), dim=-1)

    def _facing_direction_features(self, observations) -> torch.Tensor:
        direction = observations["direction"].long().reshape(-1).clamp(0, FACING_DIR_COUNT - 1)
        return one_hot(direction, FACING_DIR_COUNT).float()

    def _hm_state_features(self, observations) -> tuple[torch.Tensor, ...]:
        map_id = self.map_embeddings(observations["map_id"].int()).squeeze(1)
        near_tile = observations["near_tile"].float() / 255.0
        tile_in_front = observations["tile_in_front"].float() / 255.0
        direction = self._facing_direction_features(observations)
        menu_flags = observations["menu_flags"].float()
        party_slots = self._party_slot_features(observations)
        party_hm_cap = observations["party_hm_cap"].float().reshape(
            observations["party_hm_cap"].shape[0], PARTY_HM_CAP_FLAT
        )
        return near_tile, tile_in_front, direction, map_id, menu_flags, party_slots, party_hm_cap

    def forward(self, observations):
        hidden, lookup = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden, lookup)
        return actions, value

    def encode_observations(self, observations):
        observations = observations.type(torch.uint8)
        observations = pufferlib.pytorch.nativize_tensor(observations, self.dtype)

        near_tile, tile_in_front, direction, map_id, menu_flags, party_slots, party_hm_cap = (
            self._hm_state_features(observations)
        )

        if self.use_screen:
            screen = self._decode_screen(observations["screen"])
            screen_latent = self.screen_cnn(screen.float() / 255.0)
        else:
            batch = near_tile.shape[0]
            screen_latent = self._zero_screen_latent.expand(batch, -1)

        hm_state_cat = torch.cat(
            (
                screen_latent,
                near_tile,
                tile_in_front,
                direction,
                map_id,
                menu_flags,
                party_slots,
                party_hm_cap,
            ),
            dim=-1,
        )
        z_hm = self.encode_linear_hm(hm_state_cat)
        hm_hidden = self.hm_pre_head(z_hm)
        hm_logits = self.hm_head(hm_hidden)
        hm_probs = torch.softmax(hm_logits, dim=-1)
        self.last_hm_logits = hm_logits
        self.last_hm_probs = hm_probs
        self.last_hm_target = self._hm_supervision_targets_from_obs(observations)
        self.last_menu_flags = observations["menu_flags"]
        self.last_current_menu_item = observations["current_menu_item"].long().reshape(-1)

        rm_state_ids = observations["rm_state"].int().reshape(-1)
        self.last_rm_state = rm_state_ids
        rm_state = self.rm_state_embeddings(rm_state_ids).squeeze(1)

        policy_cat = torch.cat(
            (z_hm, rm_state, self.hm_feature_alpha * hm_probs.detach()),
            dim=-1,
        )
        z_policy = self.encode_linear_policy(policy_cat)
        return z_policy, {"hm_logits": hm_logits, "hm_probs": hm_probs}

    def _apply_row_action_mask(
        self, action: torch.Tensor, row_mask: torch.Tensor, allowed_idx: torch.Tensor
    ) -> torch.Tensor:
        """row_mask 행만 allowed_idx 액션을 남기고 나머지는 -inf."""
        if not row_mask.any():
            return action
        neg_inf = torch.finfo(action.dtype).min
        keep = torch.zeros(action.shape[1], dtype=torch.bool, device=action.device)
        keep[allowed_idx] = True
        rows = row_mask.unsqueeze(-1)
        return torch.where(rows, torch.where(keep.unsqueeze(0), action, neg_inf), action)

    def decode_actions(self, flat_hidden, lookup, concat=None):
        action = self.actor(flat_hidden)
        rm_state = self.last_rm_state
        if lookup is not None and "hm_probs" in lookup and rm_state is not None:
            hm_probs = lookup["hm_probs"].detach().to(action.dtype)
            rm = rm_state.reshape(-1, 1)
            device = rm.device
            a_idx = self._hm_a_idx
            start_idx = self._hm_start_idx
            beta = self.hm_action_beta.to(action.dtype)

            cut_mon = (rm == self._rm_cut_mon_selected.to(device)).any(dim=1)
            surf_mon = (rm == self._rm_surf_mon_selected.to(device)).any(dim=1)
            surf_det = (rm == self._rm_surf_detected_only.to(device)).any(dim=1)
            flash_hm = (rm == self._rm_flash_hm_action_states.to(device)).any(dim=1)

            action[:, a_idx] = action[:, a_idx] + beta * cut_mon.to(action.dtype) * hm_probs[:, 0]
            action[:, a_idx] = action[:, a_idx] + beta * surf_mon.to(action.dtype) * hm_probs[:, 1]
            action[:, start_idx] = (
                action[:, start_idx]
                + beta * surf_det.to(action.dtype) * hm_probs[:, 1] * self.surf_menu_action_scale
            )
            action[:, start_idx] = (
                action[:, start_idx] + beta * flash_hm.to(action.dtype) * hm_probs[:, 2]
            )
        if rm_state is not None:
            dtype = action.dtype
            menu_beta = self.hm_menu_action_beta.to(dtype)
            rm = rm_state.reshape(-1, 1)
            detected_mask = (rm == self._rm_menu_detected_states.to(rm.device)).any(dim=1)
            scroll_mask = (rm == self._rm_menu_scroll_states.to(rm.device)).any(dim=1)
            party_scroll_mask = (rm == self._rm_menu_party_scroll_states.to(rm.device)).any(
                dim=1
            )
            open_mask = (rm == self._rm_menu_open_states.to(rm.device)).any(dim=1)
            confirm_mask = (rm == self._rm_menu_confirm_states.to(rm.device)).any(dim=1)
            action = action + detected_mask.unsqueeze(-1).to(dtype) * (
                menu_beta * self.hm_menu_detected_bias.to(dtype)
            )
            action = action + scroll_mask.unsqueeze(-1).to(dtype) * (
                menu_beta * self.hm_menu_scroll_bias.to(dtype)
            )
            action = action + party_scroll_mask.unsqueeze(-1).to(dtype) * (
                menu_beta * self.hm_menu_scroll_bias.to(dtype)
            )
            action = action + open_mask.unsqueeze(-1).to(dtype) * (
                menu_beta * self.hm_menu_open_bias.to(dtype)
            )
            action = action + confirm_mask.unsqueeze(-1).to(dtype) * (
                menu_beta * self.hm_menu_open_bias.to(dtype)
            )
            surf_scale = self.surf_menu_action_scale
            surf_det = (rm == self._rm_surf_detected_only.to(rm.device)).any(dim=1)
            surf_party = (rm == self._rm_surf_party_only.to(rm.device)).any(dim=1)
            action = action + surf_det.unsqueeze(-1).to(dtype) * (
                menu_beta * surf_scale * self.hm_menu_detected_bias.to(dtype)
            )
            action = action + surf_party.unsqueeze(-1).to(dtype) * (
                menu_beta * surf_scale * self.hm_menu_scroll_bias.to(dtype)
            )

        # 메뉴 네비게이션은 hard mask:
        # - START_MENU 단계: [B, down, up]만 허용 (포켓몬 줄까지 이동/취소)
        # - MENU_OPEN/PARTY/MON_SELECTED 단계: [B, A]만 허용 (진입/확정/취소)
        # 그 외 액션은 매우 작은 값으로 눌러 샘플링에서 제외한다.
        if rm_state is not None:
            rm = rm_state.reshape(-1, 1)
            device = rm.device
            dtype = action.dtype
            rm_start_menu = self._rm_menu_scroll_states.to(device)
            rm_menu_chain = torch.cat(
                (
                    self._rm_menu_open_states.to(device),
                    self._rm_menu_party_scroll_states.to(device),
                    self._rm_menu_confirm_states.to(device),
                )
            )
            start_menu_mask = (rm == rm_start_menu).any(dim=1)
            menu_chain_mask = (rm == rm_menu_chain).any(dim=1)
            action = self._apply_row_action_mask(
                action, start_menu_mask, self._menu_mask_scroll
            )
            action = self._apply_row_action_mask(
                action, menu_chain_mask, self._menu_mask_confirm
            )

        # 일반 START 메뉴 hard mask:
        # - 포켓몬 줄이 아니면 [B, down, up]만
        # - 포켓몬 줄이면 [B, A]만
        # (RM 상태와 무관하게 menu_flags/current_menu_item 기준으로 항상 적용)
        if self.last_menu_flags is not None and self.last_current_menu_item is not None:
            menu_flags = self.last_menu_flags
            start_menu_open = menu_flags[:, 0] > 0
            on_pokemon_row = self.last_current_menu_item == int(START_MENU_POKEMON_CURSOR)

            start_non_pokemon = start_menu_open & (~on_pokemon_row)
            action = self._apply_row_action_mask(
                action, start_non_pokemon, self._menu_mask_scroll
            )
            start_on_pokemon = start_menu_open & on_pokemon_row
            action = self._apply_row_action_mask(
                action, start_on_pokemon, self._menu_mask_confirm
            )
        value = self.value_fn(flat_hidden)
        return action, value
