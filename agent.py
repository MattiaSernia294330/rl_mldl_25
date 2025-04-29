import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal


def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

class Critic(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        """
            Critic network
        """
        # TASK 3: critic network for actor-critic algorithm
        input_dim = state_space + action_space
        self.conv1 = torch.nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2 = torch.nn.Conv1d(32, 64, kernel_size=3)
        self.conv3 = torch.nn.Conv1d(64, 64, kernel_size=3)

        conv_out_dim = input_dim - 6  # 3 kernel_size=3 convoluzioni
        self.fc1 = torch.nn.Linear(64 * conv_out_dim, 128)
        self.fc2 = torch.nn.Linear(128, 1)

        self.init_weights()

    def init_weights(self):
        for layer in [self.conv1, self.conv2, self.conv3, self.fc1, self.fc2]:
            torch.nn.init.zeros_(layer.weight)
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)

    def forward(self, state_action):
        x = state_action.unsqueeze(1)  # (batch_size, 1, input_dim)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        value = self.fc2(x)
        return value
class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()

        """
            Actor network
        """
        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)
        
        # Learned standard deviation for exploration at training time 
        self.sigma_activation = F.softplus
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space)+init_sigma)

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)


    def forward(self, x):
        """
            Actor
        """
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)

        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(action_mean, sigma)
        return normal_dist



class Agent(object):
    def __init__(self, policy,critic, device='cpu'):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.critic = critic.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-2)
        self.gamma = 0.99
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []


    def update_policy(self, state, action, td_error):
    # Recalculate log-prob of taken action
    action_mean = self.policy_net(state)
    action_std = torch.ones_like(action_mean).to(device)
    dist = torch.distributions.Normal(action_mean, action_std)
    log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)

    # Actor loss = -delta * log_prob
    actor_loss = -(td_error * log_prob).mean()

    self.policy_optimizer.zero_grad()
    actor_loss.backward()
    self.policy_optimizer.step()
      

    def update_critic(self, state, action, reward, next_state, next_action, done):
    state_action = torch.cat([state, action], dim=1)
    next_state_action = torch.cat([next_state, next_action], dim=1)

    current_Q = self.get_critic(state_action)
    next_Q = self.get_critic(next_state_action).detach()

    # TD target
    target_Q = reward if done else reward + self.gamma * next_Q

    # TD error
    td_error = target_Q - current_Q

    # Critic loss = TD error squared
    critic_loss = td_error.pow(2).mean()

    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    self.critic_optimizer.step()

    return td_error.detach()



    def get_action(self, state, evaluation=False):
        """ state -> action (3-d), action_log_densities """
        x = torch.from_numpy(state).float().to(self.train_device)

        normal_dist = self.policy(x)

        if evaluation:  # Return mean
            return normal_dist.mean, None

        else:   # Sample from the distribution
            action = normal_dist.sample()

            # Compute Log probability of the action [ log(p(a[0] AND a[1] AND a[2])) = log(p(a[0])*p(a[1])*p(a[2])) = log(p(a[0])) + log(p(a[1])) + log(p(a[2])) ]
            action_log_prob = normal_dist.log_prob(action).sum()

            return action, action_log_prob
    def get_critic(self, state_action):
        x = state_action.float().to(self.train_device)

        value= self.critic(x)
        return value

    def store_outcome(self, state, next_state, action_log_prob, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)

