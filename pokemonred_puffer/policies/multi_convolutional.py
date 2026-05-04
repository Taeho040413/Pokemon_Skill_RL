import pufferlib.emulation
import pufferlib.models
import pufferlib.pytorch
import torch
from torch import nn

from pokemonred_puffer.data.events import EVENTS_IDXS
from pokemonred_puffer.data.items import Items
from pokemonred_puffer.environment import PIXEL_VALUES, VALID_ACTIONS_STR
from pokemonred_puffer.rewards.reward_machine import HMTarget, RewardMachineState


HM_ACTIONS = ("cut", "surf", "flash", "pokeflute", "none")
HM_FEATURE_COUNT = len(HMTarget)


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
        hm_action_beta_init: float = 0.1,
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
        # if channels_last:
        #     self.screen_network = self.screen_network.to(memory_format=torch.channels_last)

        # HM head는 RM을 "복사"하지 않도록 RM state 없이 학습하고,
        # actor/value는 RM state를 포함해서 상위 진행을 반영하도록
        # 인코더를 분리합니다.
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
        self.hm_feature_alpha = nn.Parameter(torch.tensor(hm_feature_alpha_init))
        # HM action bias는 학습 파라미터가 아니라 고정 상수로 유지합니다.
        # (nn.Parameter 제거)
        self.register_buffer("hm_action_beta", torch.tensor(0.1, dtype=torch.float32), persistent=False)
        self.last_hm_logits = None
        self.last_hm_probs = None
        self.last_hm_target = None

        self.actor = nn.LazyLinear(self.num_actions)
        self.value_fn = nn.LazyLinear(1)

        # Environment action order: down, left, right, up, A, B, Start.
        action_map = torch.zeros((HM_FEATURE_COUNT, self.num_actions), dtype=torch.float32)
        a_idx = VALID_ACTIONS_STR.index("a")
        start_idx = VALID_ACTIONS_STR.index("start")
        action_map[HMTarget.CUT, a_idx] = 1.0
        action_map[HMTarget.SURF, a_idx] = 1.0
        action_map[HMTarget.FLASH, start_idx] = 1.0
        action_map[HMTarget.POKEFLUTE, a_idx] = 1.0
        self.register_buffer("hm_action_map", action_map, persistent=False)

        self.two_bit = env.unwrapped.env.two_bit
        self.skip_safari_zone = env.unwrapped.env.skip_safari_zone
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
            # if channels_last:
            #     self.global_map_network = self.global_map_network.to(
            #         memory_format=torch.channels_last
            #    )

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
        self.register_buffer(
            "unpack_bytes_mask",
            torch.tensor([0x80, 0x40, 0x20, 0x10, 0x8, 0x4, 0x2, 0x1], dtype=torch.uint8),
            persistent=False,
        )
        self.register_buffer(
            "unpack_bytes_shift",
            torch.tensor([7, 6, 5, 4, 3, 2, 1, 0], dtype=torch.uint8),
            persistent=False,
        )
        # self.register_buffer("badge_buffer", torch.arange(8) + 1, persistent=False)

        # pokemon has 0xF7 map ids
        # Lets start with 4 dims for now. Could try 8
        self.map_embeddings = nn.Embedding(0xFF, 4, dtype=torch.float32)
        # N.B. This is an overestimate
        item_count = max(Items._value2member_map_.keys())
        self.item_embeddings = nn.Embedding(
            item_count, int(item_count**0.25 + 1), dtype=torch.float32
        )

        # Party layers
        self.party_network = nn.Sequential(nn.LazyLinear(6), nn.ReLU(), nn.Flatten())
        self.species_embeddings = nn.Embedding(0xBE, int(0xBE**0.25) + 1, dtype=torch.float32)
        self.type_embeddings = nn.Embedding(0x1A, int(0x1A**0.25) + 1, dtype=torch.float32)
        self.moves_embeddings = nn.Embedding(0xA4, int(0xA4**0.25) + 1, dtype=torch.float32)

        # event embeddings
        n_events = env.env.observation_space["events"].shape[0]
        self.event_embeddings = nn.Embedding(n_events, int(n_events**0.25) + 1, dtype=torch.float32)
        self.rm_state_embeddings = nn.Embedding(
            len(RewardMachineState), rm_state_embedding_dim, dtype=torch.float32
        )

    def forward(self, observations):
        hidden, lookup = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden, lookup)
        return actions, value

    def encode_observations(self, observations):
        observations = observations.type(torch.uint8)  # Undo bad cleanrl cast
        observations = pufferlib.pytorch.nativize_tensor(observations, self.dtype)

        screen = observations["screen"]
        visited_mask = observations["visited_mask"]
        restored_shape = (screen.shape[0], screen.shape[1], screen.shape[2] * 4, screen.shape[3])
        if self.use_global_map:
            global_map = observations["global_map"]
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
            visited_mask = torch.index_select(
                self.linear_buckets,
                0,
                ((visited_mask.reshape((-1, 1)) & self.unpack_mask) >> self.unpack_shift)
                .flatten()
                .int(),
            ).reshape(restored_shape)
            if self.use_global_map:
                global_map = torch.index_select(
                    self.linear_buckets,
                    0,
                    ((global_map.reshape((-1, 1)) & self.unpack_mask) >> self.unpack_shift)
                    .flatten()
                    .int(),
                ).reshape(restored_global_map_shape)
        # badges = self.badge_buffer <= observations["badges"]
        map_id = self.map_embeddings(observations["map_id"].int()).squeeze(1)
        # The bag quantity can be a value between 1 and 99
        # TODO: Should items be positionally encoded? I dont think it matters
        items = (
            self.item_embeddings(observations["bag_items"].int())
            * (observations["bag_quantity"].float().unsqueeze(-1) / 100.0)
        ).squeeze(1)

        # image_observation = torch.cat((screen, visited_mask, global_map), dim=-1)
        image_observation = torch.cat((screen, visited_mask), dim=-1)
        if self.channels_last:
            image_observation = image_observation.permute(0, 3, 1, 2)
            # image_observation = image_observation.to( memory_format=torch.channels_last)
            if self.use_global_map:
                global_map = global_map.permute(0, 3, 1, 2)
                # global_map = global_map.to(memory_format=torch.channels_last)
        if self.downsample > 1:
            image_observation = image_observation[:, :, :: self.downsample, :: self.downsample]

        # party network
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

        # event_obs = (
        #     observations["events"].float() @ self.event_embeddings.weight
        # ) / self.event_embeddings.weight.shape[0]
        events_obs = (
            (
                (
                    (observations["events"].reshape((-1, 1)) & self.unpack_bytes_mask)
                    >> self.unpack_bytes_shift
                )
                .flatten()
                .reshape((observations["events"].shape[0], -1))[:, EVENTS_IDXS]
            )
            .float()
            .squeeze(1)
        )

        rm_state = self.rm_state_embeddings(observations["rm_state"].int()).squeeze(1)
        screen_latent = self.screen_network(image_observation.float() / 255.0).squeeze(1)

        cat_obs_hm = torch.cat(
            (
                screen_latent,
                one_hot(observations["direction"].int(), 4).float().squeeze(1),
                map_id.squeeze(1),
                items.flatten(start_dim=1),
                party_latent,
                events_obs,
                observations["game_corner_rocket"].float(),
                observations["saffron_guard"].float(),
                observations["lapras"].float(),
                (observations["tile_in_front"].float() / 255.0).reshape(
                    observations["tile_in_front"].shape[0], -1
                ),
            )
            + (
                ()
                if self.skip_safari_zone
                else (
                    (observations["safari_steps"].float() / 502.0).reshape(
                        observations["safari_steps"].shape[0], -1
                    ),
                )
            )
            + (
                (self.global_map_network(global_map.float() / 255.0).squeeze(1),)
                if self.use_global_map
                else ()
            ),
            dim=-1,
        )
        z_hm = self.encode_linear_hm(cat_obs_hm)
        hm_logits = self.hm_head(z_hm)
        hm_probs = torch.softmax(hm_logits, dim=-1)
        self.last_hm_logits = hm_logits
        self.last_hm_probs = hm_probs
        # Stash hm_target from the observation dict so the PPO loss path (which
        # only sees the flat tensor) can supervise hm_logits via CE.
        self.last_hm_target = observations["hm_target"].long().reshape(-1)

        # actor/value는 rm_state를 포함해서 학습합니다.
        cat_obs_policy = torch.cat(
            (
                screen_latent,
                one_hot(observations["direction"].int(), 4).float().squeeze(1),
                map_id.squeeze(1),
                items.flatten(start_dim=1),
                party_latent,
                events_obs,
                rm_state,
                observations["game_corner_rocket"].float(),
                observations["saffron_guard"].float(),
                observations["lapras"].float(),
                (observations["tile_in_front"].float() / 255.0).reshape(
                    observations["tile_in_front"].shape[0], -1
                ),
            )
            + (
                ()
                if self.skip_safari_zone
                else (
                    (observations["safari_steps"].float() / 502.0).reshape(
                        observations["safari_steps"].shape[0], -1
                    ),
                )
            )
            + (
                (self.global_map_network(global_map.float() / 255.0).squeeze(1),)
                if self.use_global_map
                else ()
            ),
            dim=-1,
        )
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
