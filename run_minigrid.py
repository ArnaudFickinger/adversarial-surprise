### Adversarial Surprise
### Arnaud Fickinger, 2021

import argparse
import os
from model import *
from buffers import *
from base_surprise import *

parser = argparse.ArgumentParser()

## General parameters
parser.add_argument("--env", type=str, default="MiniGrid-MultiRoomAB-v0",
                    help="name of the environment to train on")
parser.add_argument("--model", default=None,
                    help="name of the model (default: {ENV}_{ALGO}_{TIME})")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--max_steps", type=int, default=64,
                    help="horizon")
parser.add_argument("--log-interval", type=int, default=1,
                    help="number of updates between two logs (default: 1)")
parser.add_argument("--save-interval", type=int, default=10,
                    help="number of updates between two saves (default: 10, 0 means no saving)")
parser.add_argument("--procs", type=int, default=16,
                    help="number of processes (default: 16)")
parser.add_argument("--frames", type=int, default=10 ** 7,
                    help="number of frames of training (default: 1e7)")

parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                    help='the name of this experiment')
parser.add_argument('--wandb-project-name', type=str, default="adversarial_surprise2",
                    help="the wandb's project name")

## Parameters for main algorithm
parser.add_argument("--epochs", type=int, default=4,
                    help="number of epochs for PPO (default: 4)")
parser.add_argument("--batch-size", type=int, default=256,
                    help="batch size for PPO (default: 256)")
parser.add_argument("--frames-per-proc", type=int, default=128,
                    help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--lr", type=float, default=0.001,
                    help="learning rate (default: 0.001)")
parser.add_argument("--gae-lambda", type=float, default=0.95,
                    help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
parser.add_argument("--entropy-coef", type=float, default=0.01,
                    help="entropy term coefficient (default: 0.01)")
parser.add_argument("--value-loss-coef", type=float, default=0.5,
                    help="value loss term coefficient (default: 0.5)")
parser.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="maximum norm of gradient (default: 0.5)")
parser.add_argument("--optim-eps", type=float, default=1e-8,
                    help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
parser.add_argument("--optim-alpha", type=float, default=0.99,
                    help="RMSprop optimizer alpha (default: 0.99)")
parser.add_argument("--clip-eps", type=float, default=0.2,
                    help="clipping epsilon for PPO (default: 0.2)")

parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model to handle text input")

parser.add_argument('--external-reward-alice', type=str, default='',
                    help='dictionary key of the external reward of Alice')

parser.add_argument('--norm-rew', type=int, default=1,
                    help='immediate is divided by the std of the immediate reward')
parser.add_argument('--norm-rew-after-clip', type=int, default=0)
parser.add_argument('--clip-rew', type=int, default=1)

parser.add_argument('--slurm-time', type=int, default=240)
parser.add_argument('--gpus-per-node', type=int, default=4)
parser.add_argument('--cpus-per-task', type=int, default=40)
parser.add_argument("--submitit", action="store_true", default=False,
                    help="use submitit")

parser.add_argument("--recurrence_alice", type=int, default=1,
                    help="number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory.")
parser.add_argument("--recurrence_bob", type=int, default=1,
                    help="number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory.")

parser.add_argument('--number-round', type=int, default=4,
                        help='total number of round per meta-episode (8 rounds = Alice plays 4 times)')
parser.add_argument('--length-round', type=int, default=32,
                    help='horizon of 1 round')

parser.add_argument('--alice-still', type=int, default=0,
                    help='alice is still')
parser.add_argument('--freeze-alice', type=int, default=0,
                    help='alice is not learning')
parser.add_argument('--freeze-bob', type=int, default=0,
                    help='bob is not learning')

parser.add_argument('--nb-epochs', type=int, default=6005)

parser.add_argument('--render-frequency', type=int, default=-1)
parser.add_argument('--save-initial-bob', type=int, default=0,
                    help='save bob at certain index') #0: do not save
parser.add_argument('--save-initial-alice', type=int, default=0,
                    help='save alice at certain index')
parser.add_argument('--load-initial-bob', type=int, default=0,
                    help='save bob at certain index') #0: do not save
parser.add_argument('--load-initial-alice', type=int, default=0,
                    help='save alice at certain index')
parser.add_argument('--save-bob', type=int, default=-1,
                    help='save bob at certain index') #-1: do not save
parser.add_argument('--save-alice', type=int, default=-1,
                    help='save alice at certain index')
parser.add_argument('--load-bob', type=int, default=-1,
                    help='load bob at certain index')
parser.add_argument('--load-alice', type=int, default=-1,
                    help='load bob at certain index')
parser.add_argument('--load-bob-epoch', type=int, default=-1,
                    help='load bob at certain epoch')
parser.add_argument('--load-alice-epoch', type=int, default=-1,
                    help='load bob at certain epoch')

parser.add_argument('--save-runningmean', type=int, default=0)
parser.add_argument('--index-save-runningmean', type=int, default=0)
parser.add_argument('--separated-norm', type=int, default=0)

#Buffer
parser.add_argument('--reset-buffer', type=int, default=1)
parser.add_argument('--alice-reward', type=int, default=0)
parser.add_argument('--thresh', type=int, default=-1)
parser.add_argument('--norm-r', type=int, default=1)
parser.add_argument('--clip-r', type=int, default=0)
parser.add_argument('--buffer', type=str, default="cat")
parser.add_argument('--init-buffer-with-zero', type=int, default=1)

#Wrapper
parser.add_argument('--flattened-obs', type=int, default=0)
parser.add_argument("--augmented-observation", type=int, default=1,
                    help="augmented obs for bob")

parser.add_argument('--bob-still', type=int, default=0,
                    help='bob is still')

parser.add_argument("--time-in-observation", type=int, default=1,
                    help="time in obs for bob and alice")
parser.add_argument("--position-in-observation", type=int, default=0,
                    help="position in obs for bob and alice")
parser.add_argument("--direction-in-observation", type=int, default=0,
                    help="direction in obs for bob and alice")

#Observation
parser.add_argument('--obs-key', type=str, default='lowdim_obs')

#Environment
parser.add_argument('--init-noisy', type=int, default=0)
parser.add_argument('--init-dark', type=int, default=1)
parser.add_argument('--curriculum-door', type=int, default=0)
parser.add_argument('--curriculum-key', type=int, default=0)
parser.add_argument('--frequency-curriculum', type=int, default=500)
parser.add_argument('--first-level-curriculum', type=int, default=0)
parser.add_argument('--init-locked', type=int, default=0)
parser.add_argument('--fixed-key', type=int, default=0)
parser.add_argument('--fixed-door', type=int, default=0)
parser.add_argument('--easy-lock', type=int, default=0)
parser.add_argument('--medium-lock', type=int, default=0)
parser.add_argument('--hard-lock', type=int, default=0)
parser.add_argument('--lock-in-noisy', type=int, default=1)

#Training
parser.add_argument('--only-bob', type=int, default=0)
parser.add_argument('--wandb', type=int, default=1)
parser.add_argument('--train-bob', type=int, default=1,
                    help='bob is learning')
parser.add_argument('--train-alice', type=int, default=1,
                    help='alice is learning')
parser.add_argument('--external-reward-bob', type=str, default='',
                    help='dictionary key of the external reward of Bob')


parser.add_argument('--size', type=int, default=8)

parser.add_argument('--no-reward-time', type=float, default=0.5) #proportion between 0 and 1

parser.add_argument("--lr-alice", type=float, default=0.0003,
                    help="learning rate (default: 0.001)")
parser.add_argument("--lr-bob", type=float, default=0.0003,
                    help="learning rate (default: 0.001)")

parser.add_argument('--state-count', type=int, default=0)
parser.add_argument('--alice-first', type=int, default=1)
parser.add_argument('--round-time', type=int, default=0)

parser.add_argument('--episodes-per-batch', type=int, default=4)
parser.add_argument('--nb-rooms', type=int, default=4)

parser.add_argument('--inverse-room', type=int, default=0)

parser.add_argument('--save-index', type=int, default=-1)
parser.add_argument('--load-index', type=int, default=-1)
parser.add_argument('--save-frequency', type=int, default=-1)
parser.add_argument('--load-epoch', type=int, default=-1)

parser.add_argument('--has-doors', type=int, default=1)
parser.add_argument('--has-switch', type=int, default=1)

parser.add_argument('--true-state-buffer', type=int, default=0)

parser.add_argument("--proba-dark", type=float, default=0.5)
parser.add_argument("--proba-ball", type=float, default=0)
parser.add_argument("--proba-floor", type=float, default=0.5)

parser.add_argument('--minNumRooms', type=int, default=4)
parser.add_argument('--maxNumRooms', type=int, default=4)
parser.add_argument('--maxRoomSize', type=int, default=8)
parser.add_argument('--minRoomSize', type=int, default=4)
parser.add_argument('--proportion-obst-floor', type=float, default=1)
parser.add_argument('--proportion-obst-ball', type=float, default=0)

parser.add_argument('--history-size', type=int, default=1)
parser.add_argument('--view-size', type=int, default=7)

parser.add_argument('--episode-long-buffer', type=int, default=1)
parser.add_argument('--life-long-buffer', type=int, default=0)


args = parser.parse_args()

args.mem_alice = args.recurrence_alice > 1
args.mem_bob = args.recurrence_bob > 1

if (not args.init_dark) and (not args.init_noisy) and args.first_level_curriculum==0:
    args.first_level_curriculum=1

def LWmain(args):

    if args.buffer=='circgaus':
        return
    if args.proba_dark+args.proba_ball+args.proba_floor!=1:
        return

    if args.init_dark and args.init_noisy:
        return
    import gym
    import gym_minigrid
    from multiprocessing import Process, Pipe

    from abc import ABC, abstractmethod

    import numpy as np

    import re

    import cv2

    from collections import deque

    import math

    import time
    import datetime

    import csv
    import os
    import torch
    import logging
    import sys

    import random
    import collections

    import pickle

    import copy

    if args.wandb:
        import wandb

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def synthesize(array):
        d = collections.OrderedDict()
        d["mean"] = np.mean(array)
        d["std"] = np.std(array)
        d["min"] = np.amin(array)
        d["max"] = np.amax(array)
        return d

    def get_obss_preprocessor(obs_space):
        # Check if obs_space is an image space
        if isinstance(obs_space, gym.spaces.Box):
            obs_space = {"image": obs_space.shape}

            def preprocess_obss(obss, device=None):
                return torch_ac.DictList({
                    "image": preprocess_images(obss, device=device)
                })

        # Check if it is a MiniGrid observation space
        elif isinstance(obs_space, gym.spaces.Dict) and list(obs_space.spaces.keys()) == ["image"]:
            obs_space = {"image": obs_space.spaces["image"].shape, "text": 100}

            vocab = Vocabulary(obs_space["text"])

            def preprocess_obss(obss, device=None):
                return torch_ac.DictList({
                    "image": preprocess_images([obs["image"] for obs in obss], device=device),
                    "text": preprocess_texts([obs["mission"] for obs in obss], vocab, device=device)
                })

            preprocess_obss.vocab = vocab

        else:
            raise ValueError("Unknown observation space: " + str(obs_space))

        return obs_space, preprocess_obss

    def preprocess_images(images, device=None):
        # Bug of Pytorch: very slow if not first converted to numpy array
        images = np.array(images)
        return torch.tensor(images, device=device, dtype=torch.float)

    def preprocess_texts(texts, vocab, device=None):
        var_indexed_texts = []
        max_text_len = 0

        for text in texts:
            tokens = re.findall("([a-z]+)", text.lower())
            var_indexed_text = np.array([vocab[token] for token in tokens])
            var_indexed_texts.append(var_indexed_text)
            max_text_len = max(len(var_indexed_text), max_text_len)

        indexed_texts = np.zeros((len(texts), max_text_len))

        for i, indexed_text in enumerate(var_indexed_texts):
            indexed_texts[i, :len(indexed_text)] = indexed_text

        return torch.tensor(indexed_texts, device=device, dtype=torch.long)

    class Vocabulary:
        """A mapping from tokens to ids with a capacity of `max_size` words.
        It can be saved in a `vocab.json` file."""

        def __init__(self, max_size):
            self.max_size = max_size
            self.vocab = {}

        def load_vocab(self, vocab):
            self.vocab = vocab

        def __getitem__(self, token):
            if not token in self.vocab.keys():
                if len(self.vocab) >= self.max_size:
                    raise ValueError("Maximum vocabulary capacity reached")
                self.vocab[token] = len(self.vocab) + 1
            return self.vocab[token]

    class RunningMeanStd(object):
        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        def __init__(self, epsilon=1e-4, shape=()):
            self.mean = np.zeros(shape, 'float64')
            self.var = np.ones(shape, 'float64')
            self.count = epsilon
            self.epsilon = epsilon

        def update(self, x):
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            batch_count = x.shape[0]
            self.update_from_moments(batch_mean, batch_var, batch_count)

        def update_from_moments(self, batch_mean, batch_var, batch_count):
            delta = batch_mean - self.mean
            tot_count = self.count + batch_count

            new_mean = self.mean + delta * batch_count / tot_count
            m_a = self.var * (self.count)
            m_b = batch_var * (batch_count)
            M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
            new_var = M2 / (self.count + batch_count)

            new_count = batch_count + self.count

            self.mean = new_mean
            self.var = new_var
            self.count = new_count

        def get_mean(self):
            return torch.tensor(self.mean, device=device, dtype=torch.float).to(device)

        def get_var(self):
            return torch.tensor(self.var, device=device, dtype=torch.float).to(device)

    class RunningMeanStdTensor(object):
        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        def __init__(self, epsilon=1e-4, shape=()):
            self.mean = torch.zeros(shape, device=device, dtype=torch.float).to(device)
            self.var = torch.ones(shape, device=device, dtype=torch.float).to(device)
            self.count = epsilon
            self.epsilon = epsilon

        def update(self, x):
            batch_mean = torch.mean(x, axis=0)
            batch_var = torch.var(x, axis=0)
            batch_count = x.shape[0]
            self.update_from_moments(batch_mean, batch_var, batch_count)

        def update_from_moments(self, batch_mean, batch_var, batch_count):
            delta = batch_mean - self.mean
            tot_count = self.count + batch_count

            new_mean = self.mean + delta * batch_count / tot_count
            m_a = self.var * (self.count)
            m_b = batch_var * (batch_count)
            M2 = m_a + m_b + torch.square(delta) * self.count * batch_count / (self.count + batch_count)
            new_var = M2 / (self.count + batch_count)

            new_count = batch_count + self.count

            self.mean = new_mean
            self.var = new_var
            self.count = new_count

    class NormalizedReturn():
        def __init__(self, num_procs, cliprew=10., gamma=0.99, epsilon=1e-8):

            self.ret_rms = RunningMeanStd(shape=(1,))
            self.cliprew = cliprew
            self.ret = np.zeros((num_procs,))
            self.gamma = gamma
            self.epsilon = epsilon

        def get_normalized_rew(self, rews):
            self.ret = self.ret * self.gamma + rews
            self.ret_rms.update(self.ret.reshape((-1,1)))

            if self.cliprew>0:
                rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
            else:
                rews = rews / np.sqrt(self.ret_rms.var + self.epsilon)
            return rews

        def reset(self, index):
            self.ret[index] = 0

    def make_env(env_key, reward_key, obs_key, max_steps, seed, index=0):
        env = gym.make(env_key, max_steps=max_steps,proba_dark=args.proba_dark,proba_ball=args.proba_ball,proba_floor=args.proba_floor,
        minNumRooms = args.minNumRooms,
        maxNumRooms = args.maxNumRooms,
        maxRoomSize = args.maxRoomSize,
        minRoomSize = args.minRoomSize,
                       view_size = args.view_size,
                       seed=seed,
                       proportion_obst_floor=args.proportion_obst_floor,
                       proportion_obst_ball=args.proportion_obst_ball
                       )

        # import pdb; pdb.set_trace()
        obs_dim = np.prod(env.observation_space.shape) if args.flattened_obs else env.observation_space.shape
        # if (variant["buffer_type"] == "Bernoulli"):
        if args.buffer == "bernoulli":
            buffer = BernoulliBuffer(obs_dim)
            if args.true_state_buffer:
                true_state_buffer = BernoulliBuffer((env.width, env.height, 3))
        elif args.buffer == "circgaus":
            buffer = GaussianCircularBuffer(obs_dim, size=64)
            if args.true_state_buffer:
                true_state_buffer = GaussianCircularBuffer((env.width, env.height, 3), size=64)
        elif args.buffer == "gaus":
            buffer = GaussianBufferIncremental(obs_dim)
            if args.true_state_buffer:
                true_state_buffer = GaussianBufferIncremental((env.width, env.height, 3))
        elif args.buffer == "cat":
            buffer = CategoricalBuffer(obs_dim, 12) #Minigrid Compact: obs key must be set to lowdim_obs
            if args.true_state_buffer:
                true_state_buffer = CategoricalBuffer((env.width, env.height, 3), 12)
        else:
            assert False
        env = BaseSurpriseWrapper(env, buffer, true_state_buffer = true_state_buffer if args.true_state_buffer else None,
                                  time_horizon=64, thresh=args.thresh,
                                  flattened_obs = args.flattened_obs,
                                  augmented_obs = args.augmented_observation,
                                  time_in_obs = args.time_in_observation,
                                  life_long_buffer = args.life_long_buffer,
                                  episode_long_buffer = args.episode_long_buffer,
                                  is_cat = args.buffer == "cat",
                                  alice_first=args.alice_first,
                                  no_reward_time=args.no_reward_time,
                                  length_round=args.length_round,
                                  number_round=args.number_round
                                  )

        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    def create_folders_if_necessary(path):
        dirname = os.path.dirname(path)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

    def get_storage_dir():
        if "RL_STORAGE" in os.environ:
            return os.environ["RL_STORAGE"]
        return "storage"

    def get_model_dir(model_name):
        return os.path.join(get_storage_dir(), model_name)

    def get_status_path(model_dir):
        return os.path.join(model_dir, "status.pt")

    def get_status(model_dir):
        path = get_status_path(model_dir)
        return torch.load(path)

    def save_status(status, model_dir):
        path = get_status_path(model_dir)
        create_folders_if_necessary(path)
        torch.save(status, path)

    def get_vocab(model_dir):
        return get_status(model_dir)["vocab"]

    def get_model_state(model_dir):
        return get_status(model_dir)["model_state"]

    def get_txt_logger(model_dir):
        path = os.path.join(model_dir, "log.txt")
        create_folders_if_necessary(path)

        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[
                logging.FileHandler(filename=path),
                logging.StreamHandler(sys.stdout)
            ]
        )

        return logging.getLogger()

    def get_csv_logger(model_dir):
        csv_path = os.path.join(model_dir, "log.csv")
        create_folders_if_necessary(csv_path)
        csv_file = open(csv_path, "a")
        return csv_file, csv.writer(csv_file)

    def worker(conn, env):
        while True:
            cmd, data = conn.recv()
            if cmd == "step":
                obs, reward, done, info = env.step(data)
                if done:
                    obs = env.reset()
                conn.send((obs, reward, done, info))
            elif cmd == "reset":
                obs = env.reset()
                conn.send(obs)
            elif cmd == "increase":
                level = env.increase_level_curriculum()
                conn.send(level)
            elif cmd == "position":
                position = env.get_scalar_position()
                conn.send(position)
            elif cmd == "dir":
                dir = env.agent_dir
                conn.send(dir)
            elif cmd == "resetb":
                env.reset_buffer()
                conn.send(None)
            elif cmd == "change":
                env.change_agent()
                conn.send(None)
            elif cmd == "change_round":
                env.change_round()
                conn.send(None)
            else:
                raise NotImplementedError

    class ParallelEnv(gym.Env):
        """A concurrent execution of environments in multiple processes."""

        def __init__(self, envs):
            assert len(envs) >= 1, "No environment given."

            self.envs = envs
            self.observation_space = self.envs[0].observation_space
            self.action_space = self.envs[0].action_space

            self.locals = []
            # import pdb; pdb.set_trace()
            for env in self.envs[1:]:
                local, remote = Pipe()
                self.locals.append(local)
                p = Process(target=worker, args=(remote, env))
                p.daemon = True
                p.start()
                remote.close()

        def reset(self):
            for local in self.locals:
                local.send(("reset", None))
            results = [self.envs[0].reset()] + [local.recv() for local in self.locals]
            return results

        def reset_buffer(self, index):
            if index == 0:
                self.envs[0].reset_buffer()
            else:
                self.locals[index - 1].send(("resetb", None))
                self.locals[index - 1].recv()

        def step(self, actions):
            for local, action in zip(self.locals, actions[1:]):
                local.send(("step", action))
            obs, reward, done, info = self.envs[0].step(actions[0])

            if done:
                obs = self.envs[0].reset()
            results = zip(*[(obs, reward, done, info)] + [local.recv() for local in self.locals])
            return results

        def render(self, mode='rgb_array', caption=None):
            if mode == 'human':
                return self.envs[0].render(mode='human')
            elif mode == 'rgb_array':
                # import pdb; pdb.set_trace()
                img = self.envs[0].render(mode='rgb_array')
                if caption is None:
                    return img
                else:
                    # setup text
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    text = caption

                    # get boundary of this text
                    textsize = cv2.getTextSize(text, font, 1, 2)[0]

                    # get coords based on boundary
                    textX = int((img.shape[1] - textsize[0]) / 2)
                    textY = int(textsize[1])

                    # add text centered on image
                    cv2.putText(img, text, (textX, textY), font, 0.8, (255, 255, 255), 2)
                    return img

        def get_scalar_position(self):
            for local in self.locals:
                local.send(("position", None))
            positions = [self.envs[0].get_scalar_position()] + [local.recv() for local in self.locals]
            return positions

        def get_dir(self):
            for local in self.locals:
                local.send(("dir", None))
            positions = [self.envs[0].agent_dir] + [local.recv() for local in self.locals]
            return positions

        def increase_level_curriculum(self):
            for local in self.locals:
                local.send(("increase", None))
            levels = [self.envs[0].increase_level_curriculum()] + [local.recv() for local in self.locals]
            return levels

        def change_agent(self, index):
            if index == 0:
                self.envs[0].change_agent()
            else:
                self.locals[index - 1].send(("change", None))
                self.locals[index - 1].recv()

        def change_round(self, index):
            if index == 0:
                self.envs[0].change_round()
            else:
                self.locals[index - 1].send(("change_round", None))
                self.locals[index - 1].recv()

    class DictList(dict):
        """A dictionnary of lists of same size. Dictionnary items can be
        accessed using `.` notation and list items using `[]` notation.

        Example:
            >>> d = DictList({"a": [[1, 2], [3, 4]], "b": [[5], [6]]})
            >>> d.a
            [[1, 2], [3, 4]]
            >>> d[0]
            DictList({"a": [1, 2], "b": [5]})
        """

        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__

        def __len__(self):
            return len(next(iter(dict.values(self))))

        def __getitem__(self, index):
            return DictList({key: value[index] for key, value in dict.items(self)})

        def __setitem__(self, index, d):
            for key, value in d.items():
                dict.__getitem__(self, key)[index] = value

    def default_preprocess_obss(obss, device=None):
        return torch.tensor(obss, device=device)

    class BaseAlgo(ABC):
        """The base class for RL algorithms."""

        def __init__(self, envs, agents, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                     value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward, obs_spaces):
            """
            Initializes a `BaseAlgo` instance.

            Parameters:
            ----------
            envs : list
                a list of environments that will be run in parallel
            acmodel : torch.Module
                the model
            num_frames_per_proc : int
                the number of frames collected by every process for an update
            discount : float
                the discount for future rewards
            lr : float
                the learning rate for optimizers
            gae_lambda : float
                the lambda coefficient in the GAE formula
                ([Schulman et al., 2015](https://arxiv.org/abs/1506.02438))
            entropy_coef : float
                the weight of the entropy cost in the final objective
            value_loss_coef : float
                the weight of the value loss in the final objective
            max_grad_norm : float
                gradient will be clipped to be at most this value
            recurrence : int
                the number of steps the gradient is propagated back in time
            preprocess_obss : function
                a function that takes observations returned by the environment
                and converts them into the format that the model can handle
            reshape_reward : function
                a function that shapes the reward, takes an
                (observation, action, reward, done) tuple as an input
            """

            # Store parameters

            self.env = ParallelEnv(envs)
            self.agents = agents
            self.device = device
            self.num_frames_per_proc = num_frames_per_proc
            self.discount = discount
            self.lr = lr
            self.gae_lambda = gae_lambda
            self.entropy_coef = entropy_coef
            self.value_loss_coef = value_loss_coef
            self.max_grad_norm = max_grad_norm
            self.recurrence = recurrence
            self.preprocess_obss = preprocess_obss or default_preprocess_obss
            self.reshape_reward = reshape_reward
            self.obs_spaces = obs_spaces

            self.samples = 0

            # Control parameters

            for name in agents:
                if agents[name] is not None:
                    assert agents[name].recurrent or self.recurrence[name] == 1
                    assert self.num_frames_per_proc//2 % self.recurrence[name] == 0

            self.episode_length = args.number_round*args.length_round
            assert self.num_frames_per_proc%self.episode_length==0
            assert (self.num_frames_per_proc // self.episode_length)%2 == 0

            # Configure acmodel

            for name in self.agents:
                if agents[name] is not None:
                    self.agents[name].to(self.device)
                    self.agents[name].train()

            # Store helpers values

            self.num_procs = len(envs)
            self.num_frames = self.num_frames_per_proc * self.num_procs

            if args.alice_first:
                assert args.number_round%2==0

            if args.only_bob:
                self.num_frames_per_agent = {name: self.num_frames for name in self.agents}
                self.num_frames_per_proc_per_agent = {name: self.num_frames_per_proc for name in self.agents}

            elif args.number_round%2==0:
                self.num_frames_per_agent = {name: self.num_frames//2 for name in self.agents}

                # Initialize experience values

                self.num_frames_per_proc_per_agent = {name: self.num_frames_per_proc//2 for name in self.agents}

            else:
                self.num_frames_per_agent = {'bob': (args.number_round // 2 + 1)*args.length_round*(self.num_frames // self.episode_length),
                                             'alice': (args.number_round // 2)*args.length_round*(self.num_frames // self.episode_length)}

                # Initialize experience values

                self.num_frames_per_proc_per_agent = {'bob': (args.number_round // 2 + 1)*args.length_round*(self.num_frames_per_proc // self.episode_length),
                                             'alice': (args.number_round // 2)*args.length_round*(self.num_frames_per_proc // self.episode_length)}

            shape = {name: (self.num_frames_per_proc_per_agent[name], self.num_procs) for name in self.agents}

            if args.history_size > 1:
                self.history = deque(maxlen=args.history_size)
                for _ in range(args.history_size):
                    self.history.append(np.zeros((self.num_procs, self.obs_spaces['alice' if args.alice_first else 'bob'].shape[0], *self.obs_spaces['alice' if args.alice_first else 'bob'].shape[1:])))

            self.obs = self.env.reset()

            self.obss = {name: torch.zeros(*shape[name], args.history_size*self.obs_spaces[name].shape[0], *self.obs_spaces[name].shape[1:], device=self.device, dtype=torch.float)
                         for name in self.agents}

            if args.time_in_observation:
                self.times = {name: torch.zeros(*shape[name], 1, device=self.device, dtype=torch.float) for name in self.agents}
                self.agent_time = torch.zeros(self.num_procs, 1, device=self.device, dtype=torch.float)
            if args.position_in_observation:
                self.positions = {name: torch.zeros(*shape[name], 1, device=self.device, dtype=torch.float) for name in self.agents}
                self.agent_position = torch.zeros(self.num_procs, 1, device=self.device, dtype=torch.float)
                for i_pst, pst in enumerate(self.env.get_scalar_position()):
                    self.agent_position[i_pst] = pst

            if args.direction_in_observation:
                self.directions = {name: torch.zeros(*shape[name], 1, device=self.device, dtype=torch.float) for name in self.agents}
                self.agent_direction = torch.zeros(self.num_procs, 1, device=self.device, dtype=torch.float)
                for i_pst, pst in enumerate(self.env.get_dir()):
                    self.agent_direction[i_pst] = pst
            self.memory = {
                name: torch.zeros(self.num_procs, self.agents[name].memory_size, device=self.device) if self.agents[name] is not None and self.agents[
                    name].recurrent else None for name in self.agents}
            self.memories = {
                name: torch.zeros(*shape[name], self.agents[name].memory_size, device=self.device) if self.agents[name] is not None and self.agents[
                    name].recurrent else None for name in self.agents}

            self.mask = {name: torch.ones(self.num_procs, device=self.device) for name in
                          self.agents}
            self.masks = {name: torch.zeros(*shape[name], device=self.device) for name in
                          self.agents}
            self.actions = {name: torch.zeros(*shape[name], device=self.device, dtype=torch.long) for name in
                            self.agents}
            self.values = {name: torch.zeros(*shape[name], device=self.device) if self.agents[name] is not None else None for name in
                           self.agents}
            self.rewards = {name: torch.zeros(*shape[name], device=self.device) for name in
                            self.agents}
            self.advantages = {name: torch.zeros(*shape[name], device=self.device) if self.agents[name] is not None else None for name in
                               self.agents}
            self.log_probs = {name: torch.zeros(*shape[name], device=self.device) if self.agents[name] is not None else None for name in
                              self.agents}

            # Initialize log values

            if args.norm_r:
                self.reward_normalizer = NormalizedReturn(self.num_procs, args.clip_r)

            self.log_episode_return = {name: torch.zeros(self.num_procs, device=self.device) for name in self.agents}
            self.log_episode_num_frames = {name: torch.zeros(self.num_procs, device=self.device) for name in self.agents}
            self.log_episode_num_rounds = {name: torch.zeros(self.num_procs, device=self.device) for name in self.agents}

            self.log_round_return = {name: torch.zeros(self.num_procs, device=self.device) for name in self.agents}
            self.log_round_num_frames = {name: torch.zeros(self.num_procs, device=self.device) for name in
                                           self.agents}

            self.log_done_counter = 0
            self.log_return = {name: [0] * self.num_procs for name in self.agents}
            self.log_num_frames = {name: [0] * self.num_procs for name in self.agents}
            self.log_num_rounds = {name: [0] * self.num_procs for name in self.agents}

            self.log_round_counter = {name: 0 for name in self.agents}
            self.log_return_per_round = {name: [0] * self.num_procs for name in self.agents}

            self.log_num_frames_per_round = {name: [0] * self.num_procs for name in self.agents}

            self.global_step = 0
            if args.alice_first:
                self.current_env_per_agent = {'alice': list(np.arange(self.num_procs)), 'bob': []}
                self.current_agent_per_environment = ['alice']*self.num_procs
            else:
                self.current_env_per_agent = {'alice': [], 'bob': list(
                    np.arange(self.num_procs))}
                self.current_agent_per_environment = ['bob'] * self.num_procs
            self.episode_step = np.array([0]*self.num_procs)
            self.current_round = np.array([0] * self.num_procs)
            self.round_step = np.array([0] * self.num_procs)

        def reverse_agent(self, name):
            if args.only_bob:
                return 'bob'
            if name=='alice':
                return 'bob'
            elif name=='bob':
                return 'alice'
            else:
                assert False

        def change_condition(self, i):
            return self.episode_step[i]%args.length_round==0

        def collect_experiences(self, index_update, render=False):

            """Collects rollouts and computes advantages.

            Runs several environments concurrently. The next actions are computed
            in a batch mode for all environments at the same time. The rollouts
            and advantages from all environments are concatenated together.

            Returns
            -------
            exps : DictList
                Contains actions, rewards, advantages etc as attributes.
                Each attribute, e.g. `exps.reward` has a shape
                (self.num_frames_per_proc * num_envs, ...). k-th block
                of consecutive `self.num_frames_per_proc` frames contains
                data obtained from the k-th environment. Be careful not to mix
                data from different environments!
            logs : dict
                Useful stats about the training process, including the average
                reward, policy loss, value loss, etc.
            """

            epoch_logger = {}
            indices_buffer = {name:np.zeros(self.num_procs) for name in self.agents}
            if render:
                if self.episode_step[0]==0:
                    gif = []
                    rendering=True
                else:
                    rendering = False
                render_done = False

            for _ in range(self.num_frames_per_proc):

                if args.augmented_observation:
                    dic_obs = self.obs
                    agent_obs = [ob["obs"] for ob in dic_obs]
                    agent_obs = np.array(agent_obs)
                    agent_obs = torch.tensor(agent_obs, device=self.device, dtype=torch.float)
                    agent_augmented_obs = [ob["augmented_obs"] for ob in dic_obs]
                    agent_augmented_obs = np.array(agent_augmented_obs)
                    if args.history_size > 1:
                        self.history.append(agent_augmented_obs)
                    agent_augmented_obs = torch.tensor(agent_augmented_obs if args.history_size <=1 else np.hstack(self.history), device=self.device, dtype=torch.float)
                else:
                    agent_obs = self.obs
                    agent_obs = np.array(agent_obs)
                    if args.history_size > 1:
                        self.history.append(agent_obs)
                    agent_obs = torch.tensor(agent_obs if args.history_size <=1 else np.hstack(self.history), device=self.device, dtype=torch.float)

                if render and rendering:
                    gif.append(self.env.render(mode='rgb_array', caption=self.current_agent_per_environment[0]).transpose(2, 0, 1))

                action = torch.zeros(self.num_procs, device=self.device, dtype=torch.long)
                value = {}
                memory = {}
                dist = {}
                for name in self.agents:
                    if len(self.current_env_per_agent[name])==0:
                        continue
                    if self.agents[name] is not None:
                        with torch.no_grad():
                            if self.agents[name].recurrent:
                                dist[name], value[name], memory[name] = self.agents[name](agent_augmented_obs[self.current_env_per_agent[name]] if args.augmented_observation else agent_obs[self.current_env_per_agent[name]], self.memory[name][self.current_env_per_agent[name]] * self.mask[name][self.current_env_per_agent[name]].unsqueeze(1), 
                                                                                          time=self.agent_time[self.current_env_per_agent[name]] if args.time_in_observation else None,
                                                                                          position=self.agent_position[
                                                                                              self.current_env_per_agent[
                                                                                                  name]] if args.position_in_observation else None,
                                                                                          direction=
                                                                                          self.agent_direction[
                                                                                              self.current_env_per_agent[
                                                                                                  name]] if args.direction_in_observation else None
                                                                                          )
                            else:

                                try:
                                    dist[name], value[name] = self.agents[name](agent_augmented_obs[self.current_env_per_agent[name]] if args.augmented_observation else agent_obs[self.current_env_per_agent[name]],
                                                                            time=self.agent_time[self.current_env_per_agent[name]] if args.time_in_observation else None,
                                                                            position=self.agent_position[
                                                                                self.current_env_per_agent[
                                                                                    name]] if args.position_in_observation else None,
                                                                            direction=
                                                                            self.agent_direction[
                                                                                self.current_env_per_agent[
                                                                                    name]] if args.direction_in_observation else None
                                                                            )
                                except:
                                    import pdb; pdb.set_trace()
                        try:
                            action[self.current_env_per_agent[name]] = dist[name].sample()
                        except:
                            import pdb; pdb.set_trace()

                    elif name=='bob' and args.bob_still:
                        action[self.current_env_per_agent[name]] = 6 * (torch.ones(len(self.current_env_per_agent[name]), dtype=torch.long).to(device))
                    elif name=='alice' and args.alice_still:
                        action[self.current_env_per_agent[name]] = 6 * (torch.ones(len(self.current_env_per_agent[name]), dtype=torch.long).to(device))
                    else:
                        assert False

                obs, reward, done, infos = self.env.step(action.cpu().numpy())
                self.samples+=args.procs

                if args.direction_in_observation:
                    for i_info, info in enumerate(infos):
                        self.agent_direction[i_info] = info["direction"]

                if args.position_in_observation:
                    for i_info, info in enumerate(infos):
                        self.agent_position[i_info] = info["position"]

                if args.external_reward_bob != '':
                    reward = [info[args.external_reward_bob] for info in infos]

                if args.no_reward_time>0:
                    reward_ = []
                    for i_stp,stp in enumerate(self.round_step):
                        if stp<args.no_reward_time*args.length_round:
                            reward_.append(0)
                        else:
                            reward_.append(reward[i_stp])
                    reward = reward_

                if args.norm_r:
                    reward = self.reward_normalizer.get_normalized_rew(reward)
                elif args.clip_r > 0:
                    reward = np.clip(reward, -args.clip_r, args.clip_r)

                self.episode_step += 1
                self.round_step += 1


                for name in self.agents:
                    if len(self.current_env_per_agent[name])==0:
                        continue
                    infos_agent = {key: np.mean([info[key] for j,info in enumerate(infos) if j in self.current_env_per_agent[name]]) for key in infos[0]}
                    for key in infos_agent:
                        if f"{key} {name}" not in epoch_logger:
                            epoch_logger[f"{key} {name}"] = []
                        epoch_logger[f"{key} {name}"].append(infos_agent[key])
                self.global_step += 1

                # Update experiences values

                for name in self.agents:
                    if len(self.current_env_per_agent[name]) == 0:
                        self.mask[name][torch.tensor(done, device=self.device, dtype=torch.float) == 1] = 0
                        continue
                    try:
                        self.obss[name][indices_buffer[name][self.current_env_per_agent[name]], self.current_env_per_agent[name]] = agent_augmented_obs[self.current_env_per_agent[name]] if args.augmented_observation else agent_obs[self.current_env_per_agent[name]]
                    except:
                        import pdb; pdb.set_trace()
                    if args.time_in_observation:
                        self.times[name][indices_buffer[name][self.current_env_per_agent[name]], self.current_env_per_agent[name]] = self.agent_time[self.current_env_per_agent[name]]
                    if args.position_in_observation:
                        self.positions[name][
                            indices_buffer[name][self.current_env_per_agent[name]], self.current_env_per_agent[
                                name]] = self.agent_position[self.current_env_per_agent[name]]
                    if args.direction_in_observation:
                        self.directions[name][
                            indices_buffer[name][self.current_env_per_agent[name]], self.current_env_per_agent[
                                name]] = self.agent_direction[self.current_env_per_agent[name]]

                    if self.agents[name] is not None and self.agents[name].recurrent:
                        self.memories[name][indices_buffer[name][self.current_env_per_agent[name]], self.current_env_per_agent[name]] = self.memory[name][self.current_env_per_agent[name]]
                        self.memory[name][self.current_env_per_agent[name]] = memory[name]

                    self.masks[name][indices_buffer[name][self.current_env_per_agent[name]], self.current_env_per_agent[name]] = self.mask[name][self.current_env_per_agent[name]]
                    self.mask[name][self.current_env_per_agent[name]] = 1 - torch.tensor(done, device=self.device, dtype=torch.float)[self.current_env_per_agent[name]]
                    self.mask[name][torch.tensor(done, device=self.device, dtype=torch.float) == 1] = 0
                    self.actions[name][indices_buffer[name][self.current_env_per_agent[name]], self.current_env_per_agent[name]] = action[self.current_env_per_agent[name]]
                    if self.agents[name] is not None:
                        self.values[name][indices_buffer[name][self.current_env_per_agent[name]], self.current_env_per_agent[name]] = value[name]
                    if name == 'bob':
                        self.rewards[name][indices_buffer[name][self.current_env_per_agent[name]], self.current_env_per_agent[name]] = torch.tensor(reward, device=self.device, dtype=torch.float)[self.current_env_per_agent[name]].squeeze()
                    elif name == 'alice':
                        self.rewards[name][indices_buffer[name][self.current_env_per_agent[name]], self.current_env_per_agent[name]] = (0*torch.tensor(reward, device=self.device, dtype=torch.float)[self.current_env_per_agent[name]]).squeeze()
                    else:
                        assert False
                    if self.agents[name] is not None:
                        self.log_probs[name][indices_buffer[name][self.current_env_per_agent[name]], self.current_env_per_agent[name]] = dist[name].log_prob(action[self.current_env_per_agent[name]])

                    if name == 'bob':
                        self.log_episode_return[name][self.current_env_per_agent[name]] += \
                        torch.tensor(reward, device=self.device, dtype=torch.float)[
                            self.current_env_per_agent[name]].squeeze()
                        self.log_round_return[name][self.current_env_per_agent[name]] += torch.tensor(reward,
                                                                                                      device=self.device,
                                                                                                      dtype=torch.float)[
                            self.current_env_per_agent[name]].squeeze()
                    self.log_episode_num_frames[name][self.current_env_per_agent[name]] += torch.ones(self.num_procs, device=self.device)[self.current_env_per_agent[name]]


                    self.log_round_num_frames[name][self.current_env_per_agent[name]] += torch.ones(self.num_procs, device=self.device)[self.current_env_per_agent[name]]


                self.obs = obs
                old_current_env_per_agent = copy.deepcopy(self.current_env_per_agent)

                if args.time_in_observation:
                    self.agent_time+=1

                for i, done_ in enumerate(done):
                    if done_:
                        if args.norm_r:
                            self.reward_normalizer.reset(i)

                        if args.wandb and i==0 and render:
                            if rendering:
                                wandb.log(
                                    {f"Epoch {index_update}": wandb.Video(np.stack(gif, axis=0), fps=8, format="gif")})
                                rendering = False
                                render_done = True
                            elif not render_done:
                                gif = []
                                rendering = True
                        self.episode_step[i]=0
                        self.round_step[i]=0
                        if not self.current_round[i]==args.number_round-1:
                            import pdb; pdb.set_trace()
                        self.current_round[i]=0
                        agent_done = self.current_agent_per_environment[i]
                        if agent_done!='bob':
                            import pdb; pdb.set_trace()
                        self.log_done_counter += 1
                        self.log_episode_num_rounds[agent_done][i] += 1
                        self.log_round_counter[agent_done] += 1
                        self.log_return_per_round[agent_done].append(self.log_round_return[agent_done][i].item())
                        self.log_num_frames_per_round[agent_done].append(self.log_round_num_frames[agent_done][i].item())
                        self.current_env_per_agent[agent_done].remove(i)
                        if args.alice_first and not args.only_bob:
                            self.current_agent_per_environment[i]='alice'
                            self.current_env_per_agent['alice'].append(i)
                            self.env.change_agent(i)
                        else:
                            self.current_agent_per_environment[i]='bob'
                            self.current_env_per_agent['bob'].append(i)
                        self.log_round_return['alice'][i]=-self.log_round_return['bob'][i]
                        self.log_episode_return['alice'][i]-=self.log_round_return['bob'][i]
                        self.rewards['alice'][int(indices_buffer['alice'][i]) - 1, i] = -self.log_round_return[agent_done][i]
                        self.log_return_per_round['alice'].append(-self.log_round_return[agent_done][i].item())
                        for name in self.agents:
                            self.log_return[name].append(self.log_episode_return[name][i].item())
                            self.log_num_frames[name].append(self.log_episode_num_frames[name][i].item())
                            self.log_num_rounds[name].append(self.log_episode_num_rounds[name][i].item())
                            self.log_round_num_frames[name][i] = 0
                            self.log_round_return[name][i] = 0
                            self.log_episode_return[name][i] = 0
                            self.log_episode_num_frames[name][i] = 0
                            self.log_episode_num_rounds[name][i] = 0
                        if args.time_in_observation:
                            self.agent_time[i]=0
                        if args.history_size > 1:
                            for h in range(args.history_size-1):
                                self.history[h][i] = np.zeros((self.obs_spaces['alice' if args.alice_first else 'bob'].shape[0], *self.obs_spaces['alice' if args.alice_first else 'bob'].shape[1:]))

                    elif self.change_condition(i):
                        agent_done = self.current_agent_per_environment[i]
                        self.round_step[i] = 0
                        self.log_episode_num_rounds[agent_done][i]+=1
                        self.log_round_counter[agent_done] += 1
                        self.log_num_frames_per_round[agent_done].append(
                            self.log_round_num_frames[agent_done][i].item())
                        if agent_done == 'bob':
                            self.log_return_per_round[agent_done].append(self.log_round_return[agent_done][i].item())
                            if self.current_round[i] > 0:
                                try:
                                    self.rewards['alice'][int(indices_buffer['alice'][i]-1), i] = -self.log_round_return[agent_done][i]
                                except:
                                    import pdb; pdb.set_trace()

                                self.log_episode_return['alice'][i] -= self.log_round_return['bob'][i]
                                self.log_return_per_round['alice'].append(-self.log_round_return[agent_done][i].item())

                            self.log_round_return[agent_done][i] = 0
                        else:
                            if not (self.log_round_return[agent_done][i] == 0):
                                import pdb; pdb.set_trace()
                            assert self.log_round_return[agent_done][i] == 0
                        self.log_round_num_frames[agent_done][i] = 0
                        if not args.only_bob:
                            self.current_agent_per_environment[i] = self.reverse_agent(agent_done)
                            self.current_env_per_agent[self.reverse_agent(agent_done)].append(i)
                            self.current_env_per_agent[agent_done].remove(i)
                        if args.separated_norm:
                            assert False
                            self.env.change_agent(i)
                        if not (args.episode_long_buffer or args.life_long_buffer):
                            self.env.reset_buffer(i)
                            if args.time_in_observation and args.round_time:
                                self.agent_time[i] = 0
                        self.current_round[i] += 1
                        self.env.change_round(i)
                        if not args.only_bob:
                            self.env.change_agent(i)


                for name in self.agents:
                    if len(old_current_env_per_agent[name])==0:
                        continue
                    indices_buffer[name][old_current_env_per_agent[name]]+=1


            if args.augmented_observation:
                dic_obs = self.obs
                agent_obs = [ob["obs"] for ob in dic_obs]
                agent_obs = np.array(agent_obs)
                agent_obs = torch.tensor(agent_obs, device=self.device, dtype=torch.float)
                agent_augmented_obs = [ob["augmented_obs"] for ob in dic_obs]
                agent_augmented_obs = np.array(agent_augmented_obs)
                if args.history_size > 1:
                    self.history.append(agent_augmented_obs)
                agent_augmented_obs = torch.tensor(
                    agent_augmented_obs if args.history_size <= 1 else np.hstack(self.history), device=self.device,
                    dtype=torch.float)
            else:
                agent_obs = self.obs
                agent_obs = np.array(agent_obs)
                if args.history_size > 1:
                    self.history.append(agent_obs)
                agent_obs = torch.tensor(agent_obs if args.history_size <= 1 else np.hstack(self.history),
                                         device=self.device, dtype=torch.float)

            next_value = {name: torch.zeros(self.num_procs, device=self.device) for name in self.agents}
            next_mask ={}
            next_advantage = {}
            for name in self.agents:
                if self.agents[name] is not None:
                    if len(self.current_env_per_agent[name])>0:
                        with torch.no_grad():
                            if self.agents[name].recurrent:
                                _, next_value[name][self.current_env_per_agent[name]], _ = self.agents[name](agent_augmented_obs[self.current_env_per_agent[name]] if args.augmented_observation else agent_obs[self.current_env_per_agent[name]] , self.memory[name][self.current_env_per_agent[name]] * self.mask[name][self.current_env_per_agent[name]].unsqueeze(1),
                                                                                                             time=self.agent_time[self.current_env_per_agent[name]] if args.time_in_observation else None,
                                                                                                             position=
                                                                                                             self.agent_position[
                                                                                                                 self.current_env_per_agent[
                                                                                                                     name]] if args.position_in_observation else None,
                                                                                                             direction=
                                                                                                             self.agent_direction[
                                                                                                                 self.current_env_per_agent[
                                                                                                                     name]] if args.direction_in_observation else None
                                                                                                             )

                            else:
                                _, next_value[name][self.current_env_per_agent[name]] = self.agents[name](agent_augmented_obs[self.current_env_per_agent[name]] if args.augmented_observation else agent_obs[self.current_env_per_agent[name]],
                                                                                                          time=self.agent_time[self.current_env_per_agent[name]] if args.time_in_observation else None,
                                                                                                          position=
                                                                                                          self.agent_position[
                                                                                                              self.current_env_per_agent[
                                                                                                                  name]] if args.position_in_observation else None,
                                                                                                          direction=
                                                                                                          self.agent_direction[
                                                                                                              self.current_env_per_agent[
                                                                                                                  name]] if args.direction_in_observation else None
                                                                                                          )
                    for i in reversed(range(self.num_frames_per_proc_per_agent[name])): #TODO replace by min
                        next_mask[name]= self.masks[name][i + 1] if i < self.num_frames_per_proc_per_agent[name] - 1 else self.mask[name]
                        next_value[name] = self.values[name][i + 1] if i < self.num_frames_per_proc_per_agent[name] - 1 else next_value[name]
                        next_advantage[name] = self.advantages[name][i + 1] if i < self.num_frames_per_proc_per_agent[name] - 1 else 0

                        delta = self.rewards[name][i] + self.discount * next_value[name] * next_mask[name] - self.values[name][i]
                        self.advantages[name][i] = delta + self.discount * self.gae_lambda * next_advantage[name] * next_mask[name]

            exps = {name: DictList() if self.agents[name] is not None else None for name in self.agents }
            for name in self.agents:
                if self.agents[name] is not None:
                    if self.agents[name].recurrent:
                        exps[name].memory = self.memories[name].transpose(0, 1).reshape(-1, *self.memories[name].shape[2:])
                        exps[name].mask = self.masks[name].transpose(0, 1).reshape(-1).unsqueeze(1)
                    exps[name].obs = self.obss[name].transpose(0, 1).reshape(-1, args.history_size*self.obs_spaces[name].shape[0], *self.obs_spaces[name].shape[1:])
                    if args.time_in_observation:
                        exps[name].time = self.times[name].transpose(0, 1).reshape(-1, 1)
                    if args.position_in_observation:
                        exps[name].position = self.positions[name].transpose(0, 1).reshape(-1, 1)
                    if args.direction_in_observation:
                        exps[name].direction = self.directions[name].transpose(0, 1).reshape(-1, 1)

                    exps[name].action = self.actions[name].transpose(0, 1).reshape(-1)
                    exps[name].value = self.values[name].transpose(0, 1).reshape(-1)
                    exps[name].reward = self.rewards[name].transpose(0, 1).reshape(-1)
                    exps[name].advantage = self.advantages[name].transpose(0, 1).reshape(-1)
                    exps[name].returnn = exps[name].value + exps[name].advantage
                    exps[name].log_prob = self.log_probs[name].transpose(0, 1).reshape(-1)

            keep = max(self.log_done_counter, self.num_procs)

            logs = {name: {
                "return_per_episode": self.log_return[name][-keep:],
                "num_frames_per_episode": self.log_num_frames[name][-keep:],

            } for name in self.agents}

            logs.update({"num_frames": self.num_frames,
                "global_step": self.global_step})

            epoch_logger = {f"Epoch Logger/{key}": np.mean(epoch_logger[key]) for key in epoch_logger}
            print(f"Epoch {index_update}: Total number of room visited {epoch_logger['Epoch Logger/nb_rooms_visited_in_life bob']}, Average number of room visited per episode {epoch_logger['Epoch Logger/nb_rooms_visited_in_episode bob']}, Average control per episode {epoch_logger['Epoch Logger/entropy_reduced bob']}")
            if args.wandb:
                for name in self.agents:
                    epoch_logger.update(
                        {f"Epoch Logger/Cumulative reward per episode {name}": np.mean(self.log_return[name][-keep:]),
                         f"Epoch Logger/Cumulative reward per round {name}": np.mean(self.log_return_per_round[name][-keep:]),

                         f"Epoch Logger/Number of frames per episodes {name}": np.mean(
                             self.log_num_frames[name][-keep:]),
                         f"Epoch Logger/Number of frames per round {name}": np.mean(
                             self.log_num_frames_per_round[name][-keep:]),
                         f"Epoch Logger/Total number of rounds {name}":
                             self.log_round_counter[name],
                         f"Epoch Logger/Number of rounds per episode {name}": np.mean(
                             self.log_num_rounds[name][-keep:]),
                         f"Epoch Logger/Total number of frames {name}": self.num_frames_per_agent,
                         })
                epoch_logger.update(
                    {"Epoch": index_update,
                     "Samples": self.samples,
                     f"Total number of episodes":
                         self.log_done_counter,
                     "Global Step": self.global_step})
                wandb.log(epoch_logger)



            self.log_done_counter = 0

            for name in self.agents:
                self.log_return[name] = self.log_return[name][-self.num_procs:]
                self.log_num_frames[name] = self.log_num_frames[name][-self.num_procs:]

            return exps, logs

        @abstractmethod
        def update_parameters(self):
            pass

    class PPOAlgo(BaseAlgo):
        """The Proximal Policy Optimization algorithm
        ([Schulman et al., 2015](https://arxiv.org/abs/1707.06347))."""

        def __init__(self, envs, agents, obs_spaces, device=None, num_frames_per_proc=None, discount=0.99, lr=0.001,
                     gae_lambda=0.95,
                     entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=None,
                     adam_eps=1e-8, clip_eps=0.2, epochs=4, batch_size=256, preprocess_obss=None,
                     reshape_reward=None):

            if recurrence is None:
                recurrence = {"alice": 1, "bob": 1}

            num_frames_per_proc = num_frames_per_proc or 128

            super().__init__(envs, agents, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                             value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward, obs_spaces)

            self.clip_eps = clip_eps
            self.epochs = epochs
            self.batch_size = batch_size

            for name in agents:
                if self.agents[name] is not None:
                    assert self.batch_size % self.recurrence[name] == 0

            lrs = {'alice': args.lr_alice, 'bob': args.lr_bob}

            self.optimizers = {name: torch.optim.Adam(self.agents[name].parameters(), lrs[name], eps=adam_eps) if self.agents[name] is not None else None for name in
                               self.agents}

            self.batch_num = {name: 0 for name in self.agents}

            self.init_time = time.time()

        def update_parameters(self, exps, index_update, name):
            # Collect experiences

            for _ in range(self.epochs):
                # Initialize log values

                log_entropies = []
                log_values = []
                log_policy_losses = []
                log_value_losses = []
                log_grad_norms = []
                log_advantage = []
                log_return = []

                for inds in self._get_batches_starting_indexes(name):
                    # Initialize batch values

                    batch_entropy = 0
                    batch_value = 0
                    batch_policy_loss = 0
                    batch_value_loss = 0
                    batch_loss = 0
                    batch_advantage = 0
                    batch_return = 0

                    # Initialize memory

                    if self.agents[name].recurrent:
                        memory = exps.memory[inds]

                    for i in range(self.recurrence[name]):
                        # Create a sub-batch of experience

                        sb = exps[inds + i]
                        # Compute loss

                        if self.agents[name].recurrent:
                            dist, value, memory = self.agents[name](sb.obs, memory * sb.mask, time = sb.time if args.time_in_observation else None,
                                                                    position=sb.position if args.position_in_observation else None,
                                                                    direction=sb.direction if args.direction_in_observation else None,
                                                                    )
                        else:
                            dist, value = self.agents[name](sb.obs, time = sb.time if args.time_in_observation else None,
                                                            position=sb.position if args.position_in_observation else None,
                                                            direction=sb.direction if args.direction_in_observation else None,
                                                            )

                        entropy = dist.entropy().mean()

                        ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
                        surr1 = ratio * sb.advantage
                        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.advantage
                        policy_loss = -torch.min(surr1, surr2).mean()

                        value_clipped = sb.value + torch.clamp(value - sb.value, -self.clip_eps, self.clip_eps)
                        surr1 = (value - sb.returnn).pow(2)
                        surr2 = (value_clipped - sb.returnn).pow(2)
                        value_loss = torch.max(surr1, surr2).mean()

                        loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss

                        # Update batch values

                        batch_entropy += entropy.item()
                        batch_value += value.mean().item()
                        batch_policy_loss += policy_loss.item()
                        batch_value_loss += value_loss.item()
                        batch_loss += loss
                        batch_advantage += sb.advantage.mean().item()
                        batch_return += sb.returnn.mean().item()

                        # Update memories for next epoch

                        if self.agents[name].recurrent and i < self.recurrence[name] - 1:
                            exps.memory[inds + i + 1] = memory.detach()

                    # Update batch values

                    batch_entropy /= self.recurrence[name]
                    batch_value /= self.recurrence[name]
                    batch_policy_loss /= self.recurrence[name]
                    batch_value_loss /= self.recurrence[name]
                    batch_loss /= self.recurrence[name]
                    batch_advantage /= self.recurrence[name]
                    batch_return /= self.recurrence[name]

                    # Update actor-critic

                    self.optimizers[name].zero_grad()
                    batch_loss.backward()
                    grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.agents[name].parameters()) ** 0.5
                    torch.nn.utils.clip_grad_norm_(self.agents[name].parameters(), self.max_grad_norm)
                    self.optimizers[name].step()

                    # Update log values

                    log_entropies.append(batch_entropy)
                    log_values.append(batch_value)
                    log_policy_losses.append(batch_policy_loss)
                    log_value_losses.append(batch_value_loss)
                    log_grad_norms.append(grad_norm)
                    log_advantage.append(batch_advantage)
                    log_return.append(batch_return)

            # Log some values

            logs = {
                "entropy": np.mean(log_entropies),
                "value": np.mean(log_values),
                "policy_loss": np.mean(log_policy_losses),
                "value_loss": np.mean(log_value_losses),
                "grad_norm": np.mean(log_grad_norms)
            }

            if args.wandb :

                plot = {"loss_policy": np.mean(log_policy_losses), "loss_value": logs["value_loss"],
                        "value": np.mean(log_values), "entropy": np.mean(log_entropies),
                        "advantage": np.mean(log_advantage), "return": np.mean(log_return)}

                plot = {f"Network Logger/{key} {name}": plot[key] for key in plot}
                plot.update({"Epoch": index_update, "Global Step": self.global_step,
                             "Samples": self.samples,
                             "Network Logger/Time": time.time() - self.init_time})
                wandb.log(plot)

            return logs

        def _get_batches_starting_indexes(self, name):
            """Gives, for each batch, the indexes of the observations given to
            the model and the experiences used to compute the loss at first.

            First, the indexes are the integers from 0 to `self.num_frames` with a step of
            `self.recurrence`, shifted by `self.recurrence//2` one time in two for having
            more diverse batches. Then, the indexes are splited into the different batches.

            Returns
            -------
            batches_starting_indexes : list of list of int
                the indexes of the experiences to be used at first for each batch
            """

            indexes = np.arange(0, self.num_frames_per_agent[name], self.recurrence[name])
            indexes = np.random.permutation(indexes)

            # Shift starting indexes by self.recurrence//2 half the time
            if self.batch_num[name] % 2 == 1:
                indexes = indexes[(indexes + self.recurrence[name]) % self.num_frames_per_proc_per_agent[name] != 0]
                indexes += self.recurrence[name] // 2
            self.batch_num[name] += 1

            num_indexes = self.batch_size // self.recurrence[name]
            batches_starting_indexes = [indexes[i:i + num_indexes] for i in range(0, len(indexes), num_indexes)]

            return batches_starting_indexes
    if args.wandb :
        wandb.init(project=args.wandb_project_name, name= args.exp_name, config=vars(args))
    # Set run dir

    date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    default_model_name = f"{args.env}_PPO_seed{args.seed}_{date}"

    model_name = args.model or default_model_name
    model_dir = get_model_dir(model_name)

    seed(args.seed)

    envs = []
    for i in range(args.procs):
        envs.append(make_env(env_key=args.env, seed=args.seed + 10000 * i, reward_key='',
                             obs_key=args.obs_key, max_steps = args.number_round*args.length_round, index=i))

    try:
        status = get_status(model_dir)
    except OSError:
        status = {"num_frames": 0, "update": 0}
    obs_space = envs[0].observation_space

    obs_space_alice = envs[0].observation_space if not args.augmented_observation else envs[0].augmented_observation_space


    obs_space_bob = envs[0].observation_space if not args.augmented_observation else envs[0].augmented_observation_space

    alice = LWACModel(obs_space_alice, envs[0].action_space, time=args.time_in_observation, maxpool= not(args.view_size==5 and args.obs_key=='lowdim_obs'),
                      use_memory=args.mem_alice, use_text=args.text,position=args.position_in_observation,
                      direction=args.direction_in_observation, history_size=args.history_size) if args.train_alice else None
    if alice is not None:
        alice.to(device)

    bob = LWACModel(obs_space_bob, envs[0].action_space, use_memory=args.mem_bob, use_text=args.text,
                    maxpool= not(args.view_size==5 and args.obs_key=='lowdim_obs'),
                    time=args.time_in_observation,position=args.position_in_observation,
                      direction=args.direction_in_observation, history_size=args.history_size) if args.train_bob else None
    if bob is not None:
        bob.to(device)

    agents = {"alice": alice, "bob": bob}
    recurrence = {"alice": args.recurrence_alice, "bob": args.recurrence_bob}

    # Load algo
    obs_spaces = {"alice": obs_space_alice, "bob": obs_space_bob}

    algo = PPOAlgo(envs, agents, obs_spaces,  device, args.episodes_per_batch*args.number_round*args.length_round, args.discount, args.lr, args.gae_lambda,
                   args.entropy_coef, args.value_loss_coef, args.max_grad_norm, recurrence,
                   args.optim_eps, args.clip_eps, args.epochs, args.batch_size)
    if args.load_index >= 0:
        algo.agents['bob'].load_state_dict(torch.load(f'agent_bob_{args.load_index}_{args.load_epoch}.pth'))
        algo.optimizers['bob'].load_state_dict(torch.load(f'opt_bob_{args.load_index}_{args.load_epoch}.pth'))
        algo.agents['alice'].load_state_dict(torch.load(f'agent_alice_{args.load_index}_{args.load_epoch}.pth'))
        algo.optimizers['alice'].load_state_dict(torch.load(f'opt_alice_{args.load_index}_{args.load_epoch}.pth'))

    if args.load_initial_bob:
        algo.agents['bob'].load_state_dict(torch.load(f"bob{'a' if args.augmented_observation else ''}{'t' if args.time_in_observation else ''}_0.pth"))
        algo.optimizers['bob'].load_state_dict(torch.load(f"bob{'a' if args.augmented_observation else ''}{'t' if args.time_in_observation else ''}_opt_0.pth"))
    elif args.save_initial_bob:
        torch.save(algo.agents['bob'].state_dict(),
                   f"bob{'a' if args.augmented_observation else ''}{'t' if args.time_in_observation else ''}_0.pth")
        torch.save(algo.optimizers['bob'].state_dict(),
                   f"bob{'a' if args.augmented_observation else ''}{'t' if args.time_in_observation else ''}_opt_0.pth")
    elif args.load_bob > 0:
        algo.agents['bob'].load_state_dict(torch.load(f"bob{'a' if args.augmented_observation else ''}{'t' if args.time_in_observation else ''}_{args.load_bob}_epoch_{args.load_bob_epoch}.pth"))
        algo.optimizers['bob'].load_state_dict(torch.load(f"bob{'a' if args.augmented_observation else ''}{'t' if args.time_in_observation else ''}_opt_{args.load_bob}_epoch_{args.load_bob_epoch}.pth"))

    if args.load_initial_alice:
        algo.agents['alice'].load_state_dict(torch.load(f"alice{'a' if args.augmented_observation else ''}{'t' if args.time_in_observation else ''}_0.pth"))
        algo.optimizers['alice'].load_state_dict(torch.load(f"alice{'a' if args.augmented_observation else ''}{'t' if args.time_in_observation else ''}_opt_0.pth"))
    elif args.save_initial_alice:
        torch.save(algo.agents['alice'].state_dict(),
                   f"alice{'a' if args.augmented_observation else ''}{'t' if args.time_in_observation else ''}_0.pth")
        torch.save(algo.optimizers['alice'].state_dict(),
                   f"alice{'a' if args.augmented_observation else ''}{'t' if args.time_in_observation else ''}_opt_0.pth")
    elif args.load_alice > 0:
        algo.agents['alice'].load_state_dict(torch.load(f"alice{'a' if args.augmented_observation else ''}{'t' if args.time_in_observation else ''}_{args.load_alice}_epoch_{args.load_alice_epoch}.pth"))
        algo.optimizers['alice'].load_state_dict(torch.load(f"alice{'a' if args.augmented_observation else ''}{'t' if args.time_in_observation else ''}_opt_{args.load_alice}_epoch_{args.load_alice_epoch}.pth"))

    update = 0

    freezes = {'alice': args.freeze_alice, 'bob': args.freeze_bob}

    save_runnningmean = [0,10,100,1000]
    index_runningmean = args.index_save_runningmean

    while update < args.nb_epochs:
        # Update model parameters

        if args.save_runningmean and update in save_runnningmean:
            algo.env.envs[0].save_rm(index_runningmean, update)

        exps, logs1 = algo.collect_experiences(update, render=args.render_frequency>0 and update%args.render_frequency==0)
        for name in ['alice', 'bob']:
            if agents[name] is not None:
                if not freezes[name]:
                    logs2 = algo.update_parameters(exps[name], update, name)

        update += 1

        if args.save_index >= 0 and update%args.save_frequency==0:
            torch.save(algo.agents['bob'].state_dict(),f'agent_bob_{args.save_index}_{update}.pth')
            torch.save(algo.optimizers['bob'].state_dict(),f'opt_bob_{args.save_index}_{update}.pth')
            torch.save(algo.agents['alice'].state_dict(), f'agent_alice_{args.save_index}_{update}.pth')
            torch.save(algo.optimizers['alice'].state_dict(), f'opt_alice_{args.save_index}_{update}.pth')


        if args.save_frequency>0 and update%args.save_frequency==0:
            if args.save_bob>0:
                torch.save(algo.agents['bob'].state_dict(),
                           f"bob{'a' if args.augmented_observation else ''}{'t' if args.time_in_observation else ''}_{args.save_bob}_epoch_{update}.pth")
                torch.save(algo.optimizers['bob'].state_dict(),
                           f"bob{'a' if args.augmented_observation else ''}{'t' if args.time_in_observation else ''}_opt_{args.save_bob}_epoch_{update}.pth")
            if args.save_alice>0:
                torch.save(algo.agents['alice'].state_dict(),
                           f"alice_{args.save_alice}_epoch_{update}.pth")
                torch.save(algo.optimizers['alice'].state_dict(),
                           f"alice_opt_{args.save_alice}_epoch_{update}.pth")

        if args.curriculum_door and update%args.frequency_curriculum == 0:
            algo.env.increase_level_curriculum()

LWmain(args)
