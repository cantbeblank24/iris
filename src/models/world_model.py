from dataclasses import dataclass
from typing import Any, Optional, Tuple
import random

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

from dataset import Batch
from .kv_caching import KeysValues
from .slicer import Embedder, Head
from .tokenizer import Tokenizer
from .transformer import Transformer, TransformerConfig
from utils import init_weights, LossWithIntermediateLosses


class EpisodeSplitter:
    def __init__(self, orig_vocab_size: int):
        self.orig_vocab_size = orig_vocab_size
        self.mappings = [bytes([i]) for i in range(self.orig_vocab_size)]
        self.pattern_length = [1 for _ in range(self.orig_vocab_size)]

    @property
    def vocab_size(self):
        return len(self.mappings)

    def extend_vocab(self, dataset):
        def actions_from_batch(batch):
            action_str = batch['actions'].squeeze().numpy().astype(np.ubyte).tobytes()
            action_str = self.transform_actions(action_str)
            return np.frombuffer(action_str, dtype=np.ubyte)

        episodes = [actions_from_batch(batch) for batch in dataset.traverse(1, 32)]

        original_lengths = [len(episode) for episode in episodes]

        print('\'Training\' splitter on', sum(original_lengths), 'actions')

        i = self.vocab_size

        p = np.zeros((i, i), dtype=int)

        for actions in episodes:
            np.add.at(p, (actions[:-1], actions[1:]), 1)

        p[0, :] = 0
        p[:, 0] = 0
        # p[self.orig_vocab_size:, self.orig_vocab_size:] = 0

        if p.max() < 200:
            print('Not enough actions to extend action set')
            return

        a, b = np.unravel_index(p.argmax(), p.shape)

        self.mappings.append(bytes([a, b]))
        self.pattern_length.append(self.pattern_length[a] + self.pattern_length[b])

        for j in range(len(episodes)):
            action_str = episodes[j].tobytes()
            action_str = action_str.replace(bytes([a, b]), bytes([i]))
            episodes[j] = np.frombuffer(action_str, dtype=np.ubyte)

        print('Mappings:', self.mappings)

        frac = np.mean([
            len(episode) / original_len
            for episode, original_len in zip(episodes, original_lengths)
        ])
        print('Approx size reduction: ', 1 / frac)

    def transform_actions(self, action_str, sample=False):
        mapping_pairs = list(enumerate(self.mappings))[self.orig_vocab_size:]

        if sample:
            n = len(mapping_pairs)
            num_mappings = min(n, random.randint(0, int(1.5 * n)))

            mapping_pairs = random.sample(mapping_pairs, num_mappings)
            mapping_pairs = sorted(mapping_pairs)

        for token, mapping in mapping_pairs:
            if token < self.orig_vocab_size: continue
            action_str = action_str.replace(mapping, bytes([token]))

        return action_str

    def __call__(
        self,
        batch: Batch,
        sequence_length: int,
        sample_from_start: bool = True,
        sample_mappings: bool = False,
    ):
        batch_size = len(batch['actions'])

        assert batch['actions'].max() < self.vocab_size

        lengths = torch.zeros((batch_size,), dtype=int)

        for i in range(batch_size):
            action_str = batch['actions'][i].numpy().astype(np.ubyte).tobytes()
            action_str = self.transform_actions(action_str, sample=sample_mappings)
            actions = torch.from_numpy(np.frombuffer(action_str, dtype=np.ubyte))

            lengths[i] = len(actions)

            if not lengths[i] > sequence_length:
                breakpoint()

            batch['actions'][i] = -1
            batch['actions'][i, :lengths[i]] = actions

        def zero_pad(x):
            return torch.cat([torch.zeros_like(x[:, :1]), x], dim=-1)

        lut = torch.LongTensor(self.pattern_length)
        offsets = zero_pad(lut[batch['actions']].cumsum(dim=-1))

        if sample_from_start:
            start = torch.zeros_like(lengths)
            stop = torch.full_like(lengths, sequence_length)
        else:
            start = lengths - sequence_length
            stop = lengths

        actions = torch.zeros((batch_size, sequence_length), dtype=int)
        final_offsets = torch.zeros((batch_size, sequence_length + 1), dtype=int)

        for i in range(batch_size):
            actions[i] = batch['actions'][i, start[i]:stop[i]]
            assert not (actions[i] == -1).any()
            final_offsets[i] = offsets[i, start[i]:stop[i] + 1]

        batch['actions'] = actions
        offsets = final_offsets

        for i in range(batch_size):
            batch['observations'][i, :sequence_length] = batch['observations'][i, offsets[i, :-1]]
            batch['mask_padding'][i, :sequence_length] = batch['mask_padding'][i, offsets[i, :-1]]

        agg_rewards = zero_pad(batch['rewards'].cumsum(dim=-1))

        for i in range(batch_size):
            batch['rewards'][i, :sequence_length] = (
                agg_rewards[i, offsets[i, 1:]] - agg_rewards[i, offsets[i, :-1]]
            )

        agg_ends = zero_pad(batch['ends'].cumsum(dim=-1))

        for i in range(batch_size):
            batch['ends'][i, :sequence_length] = (
                agg_ends[i, offsets[i, :-1]] != agg_ends[i, offsets[i, 1:]]
            )

        for k in batch.keys():
            batch[k] = batch[k][:, :sequence_length]

        return batch, offsets

    def decode_action(self, action):
        if action < self.orig_vocab_size:
            return [action]
        else:
            return sum(
                (self.decode_action(subaction) for subaction in self.mappings[action]),
                start=[],
            )

@dataclass
class WorldModelOutput:
    output_sequence: torch.FloatTensor
    logits_observations: torch.FloatTensor
    logits_rewards: torch.FloatTensor
    logits_ends: torch.FloatTensor


class WorldModel(nn.Module):
    def __init__(
        self, obs_vocab_size: int, act_vocab_size: int, config: TransformerConfig
    ) -> None:
        super().__init__()
        self.obs_vocab_size, self.act_vocab_size = obs_vocab_size, act_vocab_size
        self.config = config
        self.transformer = Transformer(config)

        all_but_last_obs_tokens_pattern = torch.ones(config.tokens_per_block)
        all_but_last_obs_tokens_pattern[-2] = 0
        act_tokens_pattern = torch.zeros(self.config.tokens_per_block)
        act_tokens_pattern[-1] = 1
        obs_tokens_pattern = 1 - act_tokens_pattern

        self.pos_emb = nn.Embedding(config.max_tokens, config.embed_dim)

        self.embedder = Embedder(
            max_blocks=config.max_blocks,
            block_masks=[act_tokens_pattern, obs_tokens_pattern],
            embedding_tables=nn.ModuleList(
                [
                    nn.Embedding(act_vocab_size, config.embed_dim),
                    nn.Embedding(obs_vocab_size, config.embed_dim),
                ]
            ),
        )

        self.head_observations = Head(
            max_blocks=config.max_blocks,
            block_mask=all_but_last_obs_tokens_pattern,
            head_module=nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ReLU(),
                nn.Linear(config.embed_dim, obs_vocab_size),
            ),
        )

        self.head_rewards = Head(
            max_blocks=config.max_blocks,
            block_mask=act_tokens_pattern,
            head_module=nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ReLU(),
                nn.Linear(config.embed_dim, 3),
            ),
        )

        self.head_ends = Head(
            max_blocks=config.max_blocks,
            block_mask=act_tokens_pattern,
            head_module=nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ReLU(),
                nn.Linear(config.embed_dim, 2),
            ),
        )

        self.apply(init_weights)

    def __repr__(self) -> str:
        return "world_model"

    def forward(
        self, tokens: torch.LongTensor, past_keys_values: Optional[KeysValues] = None
    ) -> WorldModelOutput:
        num_steps = tokens.size(1)  # (B, T)
        assert num_steps <= self.config.max_tokens
        prev_steps = 0 if past_keys_values is None else past_keys_values.size

        sequences = self.embedder(tokens, num_steps, prev_steps) + self.pos_emb(
            prev_steps + torch.arange(num_steps, device=tokens.device)
        )

        x = self.transformer(sequences, past_keys_values)

        logits_observations = self.head_observations(
            x, num_steps=num_steps, prev_steps=prev_steps
        )
        logits_rewards = self.head_rewards(
            x, num_steps=num_steps, prev_steps=prev_steps
        )
        logits_ends = self.head_ends(x, num_steps=num_steps, prev_steps=prev_steps)

        return WorldModelOutput(x, logits_observations, logits_rewards, logits_ends)

    def compute_loss(
        self, batch: Batch, tokenizer: Tokenizer, **kwargs: Any
    ) -> LossWithIntermediateLosses:
        with torch.no_grad():
            obs_tokens = tokenizer.encode(
                batch["observations"], should_preprocess=True
            ).tokens  # (BL, K)

        act_tokens = rearrange(batch["actions"], "b l -> b l 1")
        tokens = rearrange(
            torch.cat((obs_tokens, act_tokens), dim=2), "b l k1 -> b (l k1)"
        )  # (B, L(K+1))

        outputs = self(tokens)

        (
            labels_observations,
            labels_rewards,
            labels_ends,
        ) = self.compute_labels_world_model(
            obs_tokens, batch["rewards"], batch["ends"], batch["mask_padding"]
        )

        logits_observations = rearrange(
            outputs.logits_observations[:, :-1], "b t o -> (b t) o"
        )
        loss_obs = F.cross_entropy(logits_observations, labels_observations)
        loss_rewards = F.cross_entropy(
            rearrange(outputs.logits_rewards, "b t e -> (b t) e"), labels_rewards
        )
        loss_ends = F.cross_entropy(
            rearrange(outputs.logits_ends, "b t e -> (b t) e"), labels_ends
        )

        return LossWithIntermediateLosses(
            loss_obs=loss_obs, loss_rewards=loss_rewards, loss_ends=loss_ends
        )

    def compute_labels_world_model(
        self,
        obs_tokens: torch.Tensor,
        rewards: torch.Tensor,
        ends: torch.Tensor,
        mask_padding: torch.BoolTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert torch.all(ends.sum(dim=1) <= 1)  # at most 1 done
        mask_fill = torch.logical_not(mask_padding)
        labels_observations = rearrange(
            obs_tokens.masked_fill(mask_fill.unsqueeze(-1).expand_as(obs_tokens), -100),
            "b t k -> b (t k)",
        )[:, 1:]
        labels_rewards = (
            (rewards.sign() + 1).masked_fill(mask_fill, -100).long()
        )  # Rewards clipped to {-1, 0, 1}
        labels_ends = ends.masked_fill(mask_fill, -100)
        return labels_observations.reshape(-1), labels_rewards.reshape(
            -1
        ), labels_ends.reshape(-1)
