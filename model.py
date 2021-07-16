### Adversarial Surprise
### Arnaud Fickinger, 2021

import torch
import torch.nn as nn
import torch.nn.functional as F


from torch.nn import init

from torch.distributions.categorical import Categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class LWACModel(nn.Module):
    def __init__(self, obs_space, action_space, maxpool=True, use_memory=False, use_text=False, time=False,
                 position=False, direction=False, history_size=1):
        super().__init__()

        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory
        self.time = time
        self.position = position
        self.direction = direction
        self.history_size = history_size

        self.recurrent = use_memory

        init_relu_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                                    constant_(x, 0), nn.init.calculate_gain('relu'))
        init_tanh_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                                    constant_(x, 0), nn.init.calculate_gain('tanh'))
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        # Define image embedding
        if maxpool:
            self.image_conv = nn.Sequential(
                init_relu_(nn.Conv2d(3, 16, (2, 2))),
                nn.ReLU(),
                nn.MaxPool2d((2, 2)),
                init_relu_(nn.Conv2d(16, 32, (2, 2))),
                nn.ReLU(),
                init_relu_(nn.Conv2d(32, 64, (2, 2))),
                nn.ReLU()
            ).to(device)
        else:
            self.image_conv = nn.Sequential(
                init_relu_(nn.Conv2d(3, 16, (2, 2))),
                nn.ReLU(),
                init_relu_(nn.Conv2d(16, 32, (2, 2))),
                nn.ReLU(),
                init_relu_(nn.Conv2d(32, 64, (2, 2))),
                nn.ReLU()
            ).to(device)
        self.obs_dim = obs_space.shape
        if self.history_size>1:
            self.obs_dim = (self.history_size*self.obs_dim[0], *self.obs_dim[1:])
        self.image_embedding_size = self.feature_size()

        # Define memory
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size).to(device)

        # Define text embedding
        if self.use_text:
            assert False, 'no text'
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)
            self.text_embedding_size = 128
            self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)

        # Resize image embedding
        self.embedding_size = self.semi_memory_size
        if self.use_text:
            self.embedding_size += self.text_embedding_size

        # Define actor's model
        self.actor = nn.Sequential(
            init_tanh_(nn.Linear(self.embedding_size + int(time) + int(position) + int(direction), 64)),
            nn.Tanh(),
            init_(nn.Linear(64, action_space.n))
        ).to(device)

        # Define critic's model
        self.critic = nn.Sequential(
            init_tanh_(nn.Linear(self.embedding_size + int(time) + int(position) + int(direction), 64)),
            nn.Tanh(),
            init_(nn.Linear(64, 1))
        ).to(device)


    @property
    def memory_size(self):
        return 2 * self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def feature_size(self):
        tmp = self.image_conv(torch.zeros(1, *self.obs_dim).to(device).transpose(1, 3).transpose(2, 3))
        tmp = tmp.reshape(tmp.shape[0], -1)
        return tmp.size(1)

    def forward(self, obs, memory=None, time=None, position=None, direction=None):

        if self.time:
            assert time is not None
        if self.position:
            assert position is not None
        if self.direction:
            assert direction is not None
        x = obs.transpose(1, 3).transpose(2, 3)
        try:
            x = self.image_conv(x)
        except:
            import pdb;
            pdb.set_trace()
        x = x.reshape(x.shape[0], -1)

        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        if self.use_text:
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)

        if time is not None:
            assert self.time
            embedding = torch.cat((embedding, time), dim=1)

        if position is not None:
            assert self.position
            embedding = torch.cat((embedding, position), dim=1)

        if direction is not None:
            assert self.direction
            embedding = torch.cat((embedding, direction), dim=1)

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        if memory is not None:
            return dist, value, memory
        else:
            return dist, value

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]


