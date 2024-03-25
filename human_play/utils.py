# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MeltingPotEnv as a MultiAgentEnv wrapper to interface with RLLib."""

from typing import List, Optional, Tuple

import dm_env
import dmlab2d
from gymnasium import spaces
from ml_collections import config_dict
import numpy as np
from ray.rllib import algorithms
from ray.rllib.env import multi_agent_env
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils.typing import MultiAgentDict

# from examples import utils
from meltingpot import substrate
from meltingpot.utils.policies import policy

PLAYER_STR_FORMAT = "player_{index}"

# general utils from examples folder from original meltingpot
from typing import Any, List, Mapping

import dm_env
from gymnasium import spaces
import numpy as np
import tree

PLAYER_STR_FORMAT = 'player_{index}'
_WORLD_PREFIX = 'WORLD.'


def timestep_to_observations(timestep: dm_env.TimeStep,
                             individual_obs: List[str]) -> Mapping[str, Any]:
  gym_observations = {}
  for index, observation in enumerate(timestep.observation):
    gym_observations[PLAYER_STR_FORMAT.format(index=index)] = {
        key: value
        for key, value in observation.items()
        if key in individual_obs
    }
  return gym_observations


def remove_world_observations_from_space(
    observation: spaces.Dict, individual_obs: List[str]) -> spaces.Dict:
  return spaces.Dict({
      key: observation[key] for key in observation if key in individual_obs
  })


def spec_to_space(spec: tree.Structure[dm_env.specs.Array]) -> spaces.Space:
  """Converts a dm_env nested structure of specs to a Gym Space.

  BoundedArray is converted to Box Gym spaces. DiscreteArray is converted to
  Discrete Gym spaces. Using Tuple and Dict spaces recursively as needed.

  Args:
    spec: The nested structure of specs

  Returns:
    The Gym space corresponding to the given spec.
  """
  if isinstance(spec, dm_env.specs.DiscreteArray):
    return spaces.Discrete(spec.num_values)
  elif isinstance(spec, dm_env.specs.BoundedArray):
    return spaces.Box(spec.minimum, spec.maximum, spec.shape, spec.dtype)
  elif isinstance(spec, dm_env.specs.Array):
    if np.issubdtype(spec.dtype, np.floating):
      return spaces.Box(-np.inf, np.inf, spec.shape, spec.dtype)
    elif np.issubdtype(spec.dtype, np.integer):
      info = np.iinfo(spec.dtype)
      return spaces.Box(info.min, info.max, spec.shape, spec.dtype)
    else:
      raise NotImplementedError(f'Unsupported dtype {spec.dtype}')
  elif isinstance(spec, (list, tuple)):
    return spaces.Tuple([spec_to_space(s) for s in spec])
  elif isinstance(spec, dict):
    return spaces.Dict({key: spec_to_space(s) for key, s in spec.items()})
  else:
    raise ValueError(f'Unexpected spec of type {type(spec)}: {spec}')

class MeltingPotEnv(multi_agent_env.MultiAgentEnv):
  """An adapter between the Melting Pot substrates and RLLib MultiAgentEnv."""

  def _convert_spaces_tuple_to_dict(
      self,
      input_tuple: spaces.Tuple,
      remove_world_observations: bool = False) -> spaces.Dict:
    """Returns spaces tuple converted to a dictionary.

    Args:
      input_tuple: tuple to convert.
      remove_world_observations: If True will remove non-player observations.
    """
    return spaces.Dict({
        agent_id: (remove_world_observations_from_space(
            input_tuple[i], self._individual_obs)
                   if remove_world_observations else input_tuple[i])
        for i, agent_id in enumerate(self._ordered_agent_ids)
    })

  def _standard_rewards(self, reward):
    return {
        agent_id: reward[index]
        for index, agent_id in enumerate(self._ordered_agent_ids)
    }

  def _linear_reward_transfer(self, reward):
    return {
        agent_id:
        np.sum(np.array(reward, dtype=float) * self._reward_transfer[index])
        for index, agent_id in enumerate(self._ordered_agent_ids)
    }

  def __init__(self,
               env: dmlab2d.Environment,
               individual_obs: List[str] = ["RGB"],
               reward_transfer: Optional[List[List]] = None):
    """Initialize the instance

    Args:
      env: dmlab2d environment to wrap. Will be closed when this wrapper closes.
      individual_obs: the substrate observations to pass to the agents.
      reward_transfer: a matrix specifying the amount of reward column player i
        transfers to row player j.

    """
    self._env = env
    self._individual_obs = individual_obs
    self._num_players = len(self._env.observation_spec())
    self._ordered_agent_ids = [
        PLAYER_STR_FORMAT.format(index=index)
        for index in range(self._num_players)
    ]
    if reward_transfer is not None:
      self._reward_transfer = np.array(reward_transfer, dtype=float)
      assert self._reward_transfer.shape == (
          self._num_players, self._num_players
      ), "The reward gifting matrix must be of size (n_players,n_players)"
      self._reward_transfer_fn = self._linear_reward_transfer
    else:
      self._reward_transfer_fn = self._standard_rewards

    # RLLib requires environments to have the following member variables:
    # observation_space, action_space, and _agent_ids
    self._agent_ids = set(self._ordered_agent_ids)
    # RLLib expects a dictionary of agent_id to observation or action,
    # Melting Pot uses a tuple, so we convert
    self.observation_space = self._convert_spaces_tuple_to_dict(
        spec_to_space(self._env.observation_spec()),
        remove_world_observations=True)
    self.action_space = self._convert_spaces_tuple_to_dict(
        spec_to_space(self._env.action_spec()))

    self._action_space_in_preferred_format = True
    self._obs_space_in_preferred_format = True
    super().__init__()

  def reset(
      self,
      *,
      seed: Optional[int] = None,
      options: Optional[dict] = None,
  ) -> MultiAgentDict:
    """See base class."""
    timestep = self._env.reset()
    obs = timestep_to_observations(timestep, self._individual_obs)
    return obs

  def step(
      self, action_dict
  ) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
    """See base class."""
    actions = [action_dict[agent_id] for agent_id in self._ordered_agent_ids]
    timestep = self._env.step(actions)
    rewards = self._reward_transfer_fn(timestep.reward)
    done = {"__all__": timestep.last()}
    infos = {}

    obs = timestep_to_observations(timestep, self._individual_obs)
    return obs, rewards, done, infos

  def close(self):
    """See base class."""
    self._env.close()

  def get_dmlab2d_env(self):
    """Return the underlying DM Lab2D environment."""
    return self._env

  # Metadata is required by the gym `Env` class that we are extending, to show
  # which modes the `render` method supports.
  metadata = {"render.modes": ["rgb_array"]}

  def render(self, mode: str) -> np.ndarray:
    """Render the environment.

    This allows you to set `record_env` in your training config, to record
    videos of gameplay.

    Args:
        mode (str): The mode to render with (see
        `MeltingPotEnv.metadata["render.modes"]` for supported modes).

    Returns:
        np.ndarray: This returns a numpy.ndarray with shape (x, y, 3),
        representing RGB values for an x-by-y pixel image, suitable for turning
        into a video.
    """
    observation = self._env.observation()
    world_rgb = observation["WORLD.RGB"]

    # RGB mode is used for recording videos
    if mode == "rgb_array":
      return world_rgb
    else:
      return super().render()


def env_creator(env_config):
  """Outputs an environment for registering."""
  env_config = config_dict.ConfigDict(env_config)
  env = substrate.build_from_config(env_config, roles=env_config["roles"])
  env = MeltingPotEnv(env, env_config["individual_observation_names"],
                      env_config["reward_transfer"])
  return env


class RayModelPolicy(policy.Policy):
  """Policy wrapping an RLLib model for inference.

  Note: Currently only supports a single input, batching is not enabled
  """

  def __init__(self,
               model: algorithms.Algorithm,
               individual_obs: List[str],
               policy_id: str = DEFAULT_POLICY_ID) -> None:
    """Initialize a policy instance.

    Args:
      model: An rllib.trainer.Trainer checkpoint.
      individual_obs: observation keys for the agent (not global observations)
      policy_id: Which policy to use (if trained in multi_agent mode)
    """
    self._model = model
    self._individual_obs = individual_obs
    self._policy_id = policy_id
    self._prev_action = 0

  def step(self, timestep: dm_env.TimeStep,
           prev_state: policy.State) -> Tuple[int, policy.State]:
    """See base class."""
    observations = {
        key: value
        for key, value in timestep.observation.items()
        if key in self._individual_obs
    }

    action, state, _ = self._model.compute_single_action(
        observations,
        prev_state,
        policy_id=self._policy_id,
        prev_action=self._prev_action,
        prev_reward=timestep.reward)

    self._prev_action = action
    return action, state

  def initial_state(self) -> policy.State:
    """See base class."""
    self._prev_action = 0
    return self._model.get_policy(self._policy_id).get_initial_state()

  def close(self) -> None:
    """See base class."""



