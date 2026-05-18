import pufferlib.emulation
import pufferlib.models
import pufferlib.pytorch
import torch
from torch import nn

from pokemonred_puffer.data.items import Items
from pokemonred_puffer.environment import PIXEL_VALUES, VALID_ACTIONS_STR
from pokemonred_puffer.rewards.reward_machine import (
    HMTarget,
    RewardMachineState,
    hm_supervision_label_from_rm_state,
)


HM_ACTIONS = ("cut", "surf", "flash", "none")
HM_FEATURE_COUNT = len(HMTarget)
HM_LOCAL_HINT_DIM = 8  # near_tile 8방향
HM_LOCAL_HINT_HIDDEN_SIZE = 16
HM_LOCAL_HINT_MAX_BIAS = 0.1

# Rollouts without obs `hm_supervision_target` (old checkpoints / buffers).
_LEGACY_RM_TO_HM_SUPERVISION = torch.tensor(
    [hm_supervision_label_from_rm_state(i) for i in range(len(RewardMachineState))],
    dtype=torch.long,
)


# Because torch.nn.functional.one_hot cannot be traced by torch as of 2.2.0
def one_hot(tensor, num_classes):
    index = torch.arange(0, num_classes, device=tensor.device)
    return (tensor.view([*tensor.shape, 1]) == index.view([1] * tensor.ndim + [num_classes])).to(
        torch.int64
    )


class MultiConvolutionalRNN(pufferlib.models.LSTMWrapper):
    def __init__(self, env, policy, input_size=512, hidden_size=512, num_layers=1):
        super().__init__(env, policy, input_size, hidden_size, num_layers)


# We dont inherit from the pufferlib convolutional because we wont be able
# to easily call its __init__ due to our usage of lazy layers
# All that really means is a slightly different forward
class MultiConvolutionalPolicy(nn.Module):
    def __init__(
        self,
        env: pufferlib.emulation.GymnasiumPufferEnv,
        hidden_size: int = 512,
        rm_state_embedding_dim: int = 4,
        hm_hidden_size: int = 128,
        hm_feature_alpha_init: float = 0.1,
        channels_last: bool = True,
        downsample: int = 1,
    ):
        super().__init__()
        self.dtype = pufferlib.pytorch.nativize_dtype(env.emulated)
        self.num_actions = env.single_action_space.n
        self.channels_last = channels_last
        self.downsample = downsample
        self.screen_network = nn.Sequential(
            nn.LazyConv2d(32, 8, stride=2),
            nn.ReLU(),
            nn.LazyConv2d(64, 4, stride=2),
            nn.ReLU(),
            nn.LazyConv2d(64, 3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.hm_screen_network = nn.Sequential(
            nn.LazyConv2d(16, 5, stride=2),
            nn.ReLU(),
            nn.LazyConv2d(32, 3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(128),
            nn.ReLU(),
        )

        self.encode_linear_hm = nn.Sequential(
            nn.LazyLinear(hidden_size),
            nn.ReLU(),
        )
        self.encode_linear_policy = nn.Sequential(
            nn.LazyLinear(hidden_size),
            nn.ReLU(),
        )
        self.hm_head = nn.Sequential(
            nn.LazyLinear(hm_hidden_size),
            nn.ReLU(),
            nn.LazyLinear(HM_FEATURE_COUNT),
        )
        self.hm_hint_network = nn.Sequential(
            nn.Linear(HM_LOCAL_HINT_DIM, HM_LOCAL_HINT_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HM_LOCAL_HINT_HIDDEN_SIZE, HM_FEATURE_COUNT),
        )
        self.hm_feature_alpha = nn.Parameter(torch.tensor(hm_feature_alpha_init))
        self.register_buffer("hm_action_beta", torch.tensor(0.1, dtype=torch.float32), persistent=False)
        self.last_hm_logits = None
        self.last_hm_probs = None
        self.last_hm_target = None

        self.actor = nn.LazyLinear(self.num_actions)
        self.value_fn = nn.LazyLinear(1)

        action_map = torch.zeros((HM_FEATURE_COUNT, self.num_actions), dtype=torch.float32)
        a_idx = VALID_ACTIONS_STR.index("a")
        start_idx = VALID_ACTIONS_STR.index("start")
        action_map[HMTarget.CUT, a_idx] = 1.0
        action_map[HMTarget.SURF, a_idx] = 1.0
        action_map[HMTarget.FLASH, start_idx] = 1.0
        self.register_buffer("hm_action_map", action_map, persistent=False)

        self.two_bit = env.unwrapped.env.two_bit
        self.use_global_map = env.unwrapped.env.use_global_map

        if self.use_global_map:
            self.global_map_network = nn.Sequential(
                nn.LazyConv2d(32, 8, stride=4),
                nn.ReLU(),
                nn.LazyConv2d(64, 4, stride=2),
                nn.ReLU(),
                nn.LazyConv2d(64, 3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.LazyLinear(480),
                nn.ReLU(),
            )

        self.register_buffer(
            "screen_buckets", torch.tensor(PIXEL_VALUES, dtype=torch.uint8), persistent=False
        )
        self.register_buffer(
            "linear_buckets", torch.tensor([0, 64, 128, 255], dtype=torch.uint8), persistent=False
        )
        self.register_buffer(
            "unpack_mask",
            torch.tensor([0xC0, 0x30, 0x0C, 0x03], dtype=torch.uint8),
            persistent=False,
        )
        self.register_buffer(
            "unpack_shift", torch.tensor([6, 4, 2, 0], dtype=torch.uint8), persistent=False
        )

        self.map_embeddings = nn.Embedding(0xFF, 4, dtype=torch.float32)
        item_count = max(Items._value2member_map_.keys())
        self.item_embeddings = nn.Embedding(
            item_count, int(item_count**0.25 + 1), dtype=torch.float32
        )

        self.party_network = nn.Sequential(nn.LazyLinear(6), nn.ReLU(), nn.Flatten())
        self.species_embeddings = nn.Embedding(0xBE, int(0xBE**0.25) + 1, dtype=torch.float32)
        self.type_embeddings = nn.Embedding(0x1A, int(0x1A**0.25) + 1, dtype=torch.float32)
        self.moves_embeddings = nn.Embedding(0xA4, int(0xA4**0.25) + 1, dtype=torch.float32)

        self.rm_state_embeddings = nn.Embedding(
            len(RewardMachineState), rm_state_embedding_dim, dtype=torch.float32
        )

    @staticmethod
    def _hm_supervision_targets_from_obs(observations) -> torch.Tensor:
        """HM aux CE labels: env latch (`hm_supervision_target`), not raw `rm_state`."""
        if "hm_supervision_target" in observations:
            return (
                observations["hm_supervision_target"]
                .long()
                .reshape(-1)
                .clamp(0, HM_FEATURE_COUNT - 1)
            )
        # Legacy rollouts / checkpoints without the obs key.
        rm_idx = observations["rm_state"].long().reshape(-1).clamp(
            0, len(RewardMachineState) - 1
        )
        return _LEGACY_RM_TO_HM_SUPERVISION.to(rm_idx.device)[rm_idx]

    @staticmethod
    def _feature_width(features: tuple[torch.Tensor, ...]) -> int:
        return sum(int(feature.shape[-1]) for feature in features)

    @staticmethod
    def _expected_input_width(block: nn.Sequential) -> int | None:
        first_layer = block[0]
        has_uninitialized = getattr(first_layer, "has_uninitialized_params", None)
        if callable(has_uninitialized) and has_uninitialized():
            return None

        weight = getattr(first_layer, "weight", None)
        if weight is None:
            return None
        return int(weight.shape[1])

    def forward(self, observations):
        hidden, lookup = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden, lookup)
        return actions, value

    def encode_observations(self, observations):
        observations = observations.type(torch.uint8)
        observations = pufferlib.pytorch.nativize_tensor(observations, self.dtype)

        screen = observations["screen"]
        restored_shape = (screen.shape[0], screen.shape[1], screen.shape[2] * 4, screen.shape[3])
        global_map = observations.get("global_map") if self.use_global_map else None
        if self.use_global_map and global_map is not None:
            restored_global_map_shape = (
                global_map.shape[0],
                global_map.shape[1],
                global_map.shape[2] * 4,
                global_map.shape[3],
            )

        if self.two_bit:
            screen = torch.index_select(
                self.screen_buckets,
                0,
                ((screen.reshape((-1, 1)) & self.unpack_mask) >> self.unpack_shift).flatten().int(),
            ).reshape(restored_shape)
            if self.use_global_map and global_map is not None:
                global_map = torch.index_select(
                    self.linear_buckets,
                    0,
                    ((global_map.reshape((-1, 1)) & self.unpack_mask) >> self.unpack_shift)
                    .flatten()
                    .int(),
                ).reshape(restored_global_map_shape)

        map_id = self.map_embeddings(observations["map_id"].int()).squeeze(1)
        items = (
            self.item_embeddings(observations["bag_items"].int())
            * (observations["bag_quantity"].float().unsqueeze(-1) / 100.0)
        ).squeeze(1)

        image_observation = screen
        if self.channels_last:
            image_observation = image_observation.permute(0, 3, 1, 2)
            if self.use_global_map and global_map is not None:
                global_map = global_map.permute(0, 3, 1, 2)
        if self.downsample > 1:
            image_observation = image_observation[:, :, :: self.downsample, :: self.downsample]

        hm_screen = observations.get("hm_screen")
        if hm_screen is None:
            hm_screen = screen
        if self.channels_last:
            hm_screen = hm_screen.permute(0, 3, 1, 2)

        species = self.species_embeddings(observations["species"].int()).float().squeeze(1)
        status = one_hot(observations["status"].int(), 7).float().squeeze(1)
        type1 = self.type_embeddings(observations["type1"].int()).squeeze(1)
        type2 = self.type_embeddings(observations["type2"].int()).squeeze(1)
        moves = (
            self.moves_embeddings(observations["moves"].int())
            .squeeze(1)
            .float()
            .reshape((-1, 6, 4 * self.moves_embeddings.embedding_dim))
        )
        party_obs = torch.cat(
            (
                species,
                observations["hp"].float().unsqueeze(-1) / 714.0,
                status,
                type1,
                type2,
                observations["level"].float().unsqueeze(-1) / 100.0,
                observations["maxHP"].float().unsqueeze(-1) / 714.0,
                moves,
            ),
            dim=-1,
        )
        party_latent = self.party_network(party_obs)

        rm_state = self.rm_state_embeddings(observations["rm_state"].int()).squeeze(1)
        screen_latent = self.screen_network(image_observation.float() / 255.0).squeeze(1)
        hm_screen_latent = self.hm_screen_network(hm_screen.float() / 255.0)

        near_tile_feats = observations["near_tile"].float() / 255.0

        shared_suffix = (
            (self.global_map_network(global_map.float() / 255.0).squeeze(1),)
            if self.use_global_map and global_map is not None
            else ()
        )

        hm_feature_prefix = (
            screen_latent,
            hm_screen_latent,
            one_hot(observations["direction"].int(), 4).float().squeeze(1),
            map_id.squeeze(1),
            items.flatten(start_dim=1),
            party_latent,
            near_tile_feats,
        )
        policy_feature_prefix = (
            screen_latent,
            one_hot(observations["direction"].int(), 4).float().squeeze(1),
            map_id.squeeze(1),
            items.flatten(start_dim=1),
            party_latent,
            rm_state,
            near_tile_feats,
        )

        hm_base_features = hm_feature_prefix + shared_suffix
        policy_base_features = policy_feature_prefix + shared_suffix

        cat_obs_hm = torch.cat(hm_base_features, dim=-1)
        z_hm = self.encode_linear_hm(cat_obs_hm)
        hm_logits = self.hm_head(z_hm)
        hm_hint_network = getattr(self, "hm_hint_network", None)
        if hm_hint_network is not None:
            hm_hint_logits = HM_LOCAL_HINT_MAX_BIAS * torch.tanh(
                hm_hint_network(near_tile_feats.to(hm_logits.dtype))
            )
            hm_logits = hm_logits + hm_hint_logits
        hm_probs = torch.softmax(hm_logits, dim=-1)
        self.last_hm_logits = hm_logits
        self.last_hm_probs = hm_probs

        self.last_hm_target = self._hm_supervision_targets_from_obs(observations)

        cat_obs_policy = torch.cat(policy_base_features, dim=-1)
        z_policy = self.encode_linear_policy(cat_obs_policy)
        z_aug = torch.cat((z_policy, self.hm_feature_alpha * hm_probs.detach()), dim=-1)
        return z_aug, {"hm_logits": hm_logits, "hm_probs": hm_probs}

    def decode_actions(self, flat_hidden, lookup, concat=None):
        action = self.actor(flat_hidden)
        if lookup is not None and "hm_probs" in lookup:
            action_bias = self.hm_action_beta * torch.matmul(
                lookup["hm_probs"].detach().to(action.dtype), self.hm_action_map.to(action.dtype)
            )
            action = action + action_bias
        value = self.value_fn(flat_hidden)
        return action, value
