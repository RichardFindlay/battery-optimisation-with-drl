import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transistions = namedtuple('transistion', ('state','action', 'next_state', 'reward'))

class ReplayMemory(object):


 