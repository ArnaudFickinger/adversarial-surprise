### Adversarial Surprise
### Arnaud Fickinger, 2021

import numpy as np
import gym
from gym.spaces import Box
import pdb
import class_util as classu

class BaseSurpriseWrapper(gym.Wrapper):
    
    @classu.hidden_member_initialize
    def __init__(self, 
                 env, 
                 buffer,
                 time_horizon,
                 true_state_buffer = None,
                 add_true_rew=False,
                 smirl_rew_scale=None, 
                 buffer_type=None,
                 latent_obs_size=None,
                 obs_label=None,
                 obs_out_label=None,
                 thresh=300,
                 flattened_obs=True,
                 augmented_obs = True,
                 time_in_obs = True,
                 life_long_buffer = False,
                 episode_long_buffer = False,
                 is_cat = False,
                 alice_first = False,
                 no_reward_time = 0,
                 length_round = 16,
                 number_round = 2):
        '''
        params
        ======
        env (gym.Env) : environment to wrap

        buffer (Buffer object) : Buffer that tracks history and fits models
        '''
        
        gym.Wrapper.__init__(self, env)
        theta = self._buffer.get_params()
        self.previous_theta = theta
        self._num_steps = 0
        self.is_cat = is_cat
        self.thresh = thresh
        self.life_long_buffer = life_long_buffer
        self.episode_long_buffer = episode_long_buffer

        self.index_current_round = 0

        self.flattened_obs = flattened_obs
        self.augmented_obs = augmented_obs
        self.time_in_obs = time_in_obs

        # Add true reward to surprise

        # Gym spaces
        self.action_space = env.action_space
        self.env_obs_space = env.observation_space

        self.agent_changed = False

        self.alice_first = alice_first


        if self.alice_first:
            self.current_agent = 'alice'
        else:
            self.current_agent = 'bob'

        self.length_round = length_round
        self.number_round = number_round

        self.no_reward_time = no_reward_time

        self.observation_space = Box(
                    self.env_obs_space.low.flatten() if flattened_obs else self.env_obs_space.low,
            self.env_obs_space.high.flatten() if flattened_obs else self.env_obs_space.high
                )

        try:
            self.augmented_observation_space = Box(
                    np.concatenate(
                        (self.env_obs_space.low.flatten() if flattened_obs else self.env_obs_space.low,)+
                         (np.zeros(theta.shape),)
                    ),
                    np.concatenate(
                        (self.env_obs_space.high.flatten() if flattened_obs else self.env_obs_space.high,)+
                         (np.zeros(theta.shape),)
                    )
                )
        except: #for cat

            to = theta.reshape(*self.env_obs_space.shape, -1)
            self.augmented_observation_space = Box(
                np.concatenate(
                    (self.env_obs_space.low.flatten() if flattened_obs else self.env_obs_space.low,) +
                    (np.zeros((to.shape[0]*theta.shape[-1], to.shape[1], to.shape[2])),)
                ),
                np.concatenate(
                    (self.env_obs_space.high.flatten() if flattened_obs else self.env_obs_space.high,) +
                    (np.zeros((to.shape[0]*theta.shape[-1], to.shape[1], to.shape[2])),)
                )
            )

        self.already_reset = False

    def step(self, action):
        # Take Action
        obs, env_rew, envdone, info = self.env.step(action)
        info['task_reward'] = env_rew
        # Get wrapper outputs
        assert self._num_steps<=self.length_round
        if self.current_agent == 'bob':
            if self._num_steps>=self.no_reward_time*self.length_round:
                self.last_obs = self.encode_obs(obs)
                rew = self._buffer.logprob(self.last_obs)

                if self.thresh > 0:
                    assert False
                    rew = np.clip(rew, a_min=-self.thresh, a_max=self.thresh)
                self._buffer.add(self.last_obs)
                info['state_entropy_smirl'] = rew
                try:
                    info["theta_entropy"] = self._buffer.entropy()
                except:
                    pass
                if (self._smirl_rew_scale is not None):
                    assert False
                    rew = (rew * self._smirl_rew_scale)
                if self._true_state_buffer:
                    assert False
                    self.last_true_state = self.encode_obs(self.env.grid.encode())
                    self._true_state_buffer.add(self.last_true_state)
                    try:
                        info["true_state_entropy"] = self._true_state_buffer.entropy()
                    except:
                        pass
            else:
                rew = 0
                if not (self.episode_long_buffer or self.life_long_buffer):
                    assert self._buffer.buffer_size == 0
                elif self.episode_long_buffer:
                    if self.alice_first:
                        nb_round_bob_done = (self.index_current_round) // 2
                    else:
                        nb_round_bob_done = (self.index_current_round + 1) // 2
                    assert self._buffer.buffer_size == nb_round_bob_done * (1 - self.no_reward_time) * self.length_round

            self._num_steps += 1
        else:
            rew = 0
            if not(self.episode_long_buffer or self.life_long_buffer):
                assert self._buffer.buffer_size == 0
            elif self.episode_long_buffer:
                if self.alice_first:
                    nb_round_bob_done = (self.index_current_round) // 2
                else:
                    nb_round_bob_done = (self.index_current_round + 1) // 2
                assert self._buffer.buffer_size == nb_round_bob_done * (1 - self.no_reward_time) * self.length_round

        if self.augmented_obs:
            aug_obs = self.get_obs(self.encode_obs(obs))
            return {"obs": obs, "augmented_obs": aug_obs, "time": self._buffer.buffer_size}, rew, envdone, info
        else:
            return obs, rew, envdone, info

    def change_agent(self):

        if not (self.episode_long_buffer or self.life_long_buffer):
            assert self._num_steps==0
        else:
            self._num_steps = 0

        if self.current_agent == "bob":
            self.current_agent = "alice"
        elif self.current_agent == "alice":
            self.current_agent = "bob"
        else:
            assert False
        self.agent_changed = True

    def change_round(self):

        try:
            assert self.index_current_round<self.number_round-1
        except:
            print(self.index_current_round)
            print(self.number_round)
            import pdb; pdb.set_trace()
        self.index_current_round+=1

    def get_obs(self, obs, reset=False):
        '''
        Augment observation, perhaps with generative model params
        '''
        assert self.augmented_obs
        if reset or (self.current_agent=='bob' and self._num_steps>=self.no_reward_time*self.length_round) or (self.agent_changed and not (self.episode_long_buffer or self.life_long_buffer)):
            theta = self._buffer.get_params()
            if self.is_cat:
                if not self.flattened_obs:
                    theta = theta.reshape(*self.env_obs_space.shape, theta.shape[-1])
                concat = []
                for i in range(12):
                    concat.append(theta[:, :, :, i])
                theta = np.concatenate(concat)
            self.previous_theta = theta
            obs = np.concatenate([np.array(obs).flatten() if self.flattened_obs else np.array(obs), np.array(theta).flatten() if self.flattened_obs else np.array(theta)])
            if self.agent_changed:
                self.agent_changed = False
        else:
            obs = np.concatenate([np.array(obs).flatten() if self.flattened_obs else np.array(obs),
                                  np.array(self.previous_theta).flatten() if self.flattened_obs else np.array(self.previous_theta)])
        return obs


    def reset(self):
        '''
        Reset the wrapped env and the buffer
        '''
        if not self.already_reset:
            assert self._buffer.buffer_size==0
            self.already_reset = True
        else:
            assert self.index_current_round == self.number_round - 1
            if not self.episode_long_buffer:
                assert (self.current_agent=='alice' and self._buffer.buffer_size==0) or (self.current_agent=='bob' and self._buffer.buffer_size==(1-self.no_reward_time)*self.length_round)
            elif self.episode_long_buffer:
                nb_round_bob = self.number_round // 2 + 1 if self.number_round%2==1 else self.number_round // 2
                assert self._buffer.buffer_size == nb_round_bob*(1-self.no_reward_time) * self.length_round
        obs = self.env.reset()
        if not self.life_long_buffer:
            self._buffer.reset()
            if self._true_state_buffer:
                assert False
                self._true_state_buffer.reset()
            self._num_steps = 0
        self.index_current_round = 0
        if self.augmented_obs:
            aug_obs = self.get_obs(self.encode_obs(obs), reset=True)
            return {"obs": obs, "augmented_obs": aug_obs, "time": self._buffer.buffer_size}
        return obs

    def reset_buffer(self):
        assert not (self.life_long_buffer or self.episode_long_buffer)
        assert (self.current_agent=='alice' and self._buffer.buffer_size==0) or (self.current_agent=='bob' and self._buffer.buffer_size==(1-self.no_reward_time)*self.length_round)
        self._buffer.reset()
        if self._true_state_buffer:
            assert False
            self._true_state_buffer.reset()
        self._num_steps = 0

    def render(self, **kwargs):
        return self.env.render(**kwargs)

    def encode_obs(self, obs):
        '''
        Used to encode the observation before putting on the buffer
        '''
        if self._obs_label is None:
            return np.array(obs).flatten().copy() if self.flattened_obs else np.array(obs).copy()
        else:
            assert False
            return np.array(obs[self._obs_label]).flatten().copy() if self.flattened_obs else np.array(obs[self._obs_label]).copy()
