import math
from pathlib import Path

import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn

from models.actor_critic import ActorCritic
from models.tokenizer import Tokenizer
from models.world_model import WorldModel, EpisodeSplitter
from utils import extract_state_dict


class Agent(nn.Module):
    def __init__(
        self,
        tokenizer: Tokenizer,
        world_model: WorldModel,
        actor_critic: ActorCritic,
        splitter: EpisodeSplitter,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.world_model = world_model
        self.actor_critic = actor_critic
        self.splitter = splitter

        self.queue = []

        self.raw_actions = True

    @property
    def device(self):
        return self.actor_critic.conv1.weight.device

    def load(
        self,
        path_to_checkpoint: Path,
        device: torch.device,
        load_tokenizer: bool = True,
        load_world_model: bool = True,
        load_actor_critic: bool = True,
    ) -> None:
        agent_state_dict = torch.load(path_to_checkpoint, map_location=device)
        if load_tokenizer:
            self.tokenizer.load_state_dict(
                extract_state_dict(agent_state_dict, "tokenizer")
            )
        if load_world_model:
            self.world_model.embedder.embedding_tables[0] = nn.Embedding(self.splitter.vocab_size, 256).to(self.device)
            self.world_model.load_state_dict(
                extract_state_dict(agent_state_dict, "world_model")
            )
        if load_actor_critic:
            # self.actor_critic.actor_linear = nn.Linear(512, self.splitter.vocab_size).to(self.device)
            self.actor_critic.load_state_dict(
                extract_state_dict(agent_state_dict, "actor_critic")
            )

    def extend_vocab(self, dataset):
        prev_size = self.splitter.vocab_size

        self.splitter.extend_vocab(dataset)

        old_embedding = self.world_model.embedder.embedding_tables[0]
        new_embedding = nn.Embedding(self.splitter.vocab_size, 256).to(self.device)

        self.world_model.embedder.embedding_tables[0] = new_embedding

        with torch.no_grad():
            new_embedding.weight[:prev_size] = old_embedding.weight[:prev_size]

    def extend_actor_vocab(self):
        print("Extending actor vocab")

        old_layer = self.actor_critic.actor_linear
        prev_size = old_layer.weight.shape[0]

        new_layer = nn.Linear(512, prev_size + 1).to(self.device)
        self.actor_critic.actor_linear = new_layer

        with torch.no_grad():
            new_layer.weight[:prev_size] = old_layer.weight
            new_layer.bias[:prev_size] = old_layer.bias

            j = self.splitter.mappings[prev_size][0]
            new_layer.weight[prev_size] = old_layer.weight[j]
            new_layer.bias[prev_size] = old_layer.bias[j]

    def act(
        self,
        obs: torch.FloatTensor,
        should_sample: bool = True,
        temperature: float = 1.0,
    ) -> torch.LongTensor:
        if not self.queue:
            input_ac = (
                obs
                if self.actor_critic.use_original_obs
                else torch.clamp(
                    self.tokenizer.encode_decode(
                        obs, should_preprocess=True, should_postprocess=True
                    ),
                    0,
                    1,
                )
            )
            logits_actions = self.actor_critic(input_ac).logits_actions[:, -1] / temperature
            act_token = (
                Categorical(logits=logits_actions).sample()
                if should_sample
                else logits_actions.argmax(dim=-1)
            )

            if self.raw_actions:
                self.queue = self.splitter.decode_action(act_token)
            else:
                self.queue = [act_token]

            print(self.queue)

        raw_action = torch.LongTensor(self.queue[:1])
        self.queue = self.queue[1:]

        return raw_action

        # input_ac = (
        #     obs
        #     if self.actor_critic.use_original_obs
        #     else torch.clamp(
        #         self.tokenizer.encode_decode(
        #             obs, should_preprocess=True, should_postprocess=True
        #         ),
        #         0,
        #         1,
        #     )
        # )
        # logits_actions = self.actor_critic(input_ac).logits_actions[:, -1] / temperature
        # act_token = (
        #     Categorical(logits=logits_actions).sample()
        #     if should_sample
        #     else logits_actions.argmax(dim=-1)
        # )

        # return act_token
