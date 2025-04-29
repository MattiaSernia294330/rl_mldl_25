"""Train an RL agent on the OpenAI Gym Hopper environment using
    REINFORCE and Actor-critic algorithms
"""
import argparse

import torch
import gym

from env.custom_hopper import *
from agent import Agent, Policy, Critic


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=100000, type=int, help='Number of training episodes')
    parser.add_argument('--print-every', default=5000, type=int, help='Print info every <> episodes')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')

    return parser.parse_args()

args = parse_args()


def main():

	env = gym.make('CustomHopper-source-v0')
	# env = gym.make('CustomHopper-target-v0')

	print('Action space:', env.action_space)
	print('State space:', env.observation_space)
	print('Dynamics parameters:', env.get_parameters())


	"""
		Training
	"""
	observation_space_dim = env.observation_space.shape[-1]
	action_space_dim = env.action_space.shape[-1]

	policy = Policy(observation_space_dim, action_space_dim)
	critic = Critic(observation_space_dim, action_space_dim)
	agent = Agent(policy, critic, device=args.device)

    #
    # TASK 2 and 3: interleave data collection to policy updates
    #

	for episode in range(args.n_episodes):
		state = env.reset()
		done = False
		train_reward = 0
		action, action_log_prob = agent.get_action(state)
		while not done:
			prev_state = state
			prev_action = action
			prev_log_prob = action_log_prob
			state, reward, done, info = env.step(action.detach().cpu().numpy())
			agent.states.append(torch.from_numpy(prev_state).float())
			agent.action_log_probs.append(prev_log_prob)
			agent.update_policy(prev_action)
			action, action_log_prob = agent.get_action(state)
			agent.update_critic(prev_action, action, prev_state, state, reward, done)
			agent.rewards.append(torch.tensor([reward]))
			train_reward += reward
		if (episode + 1) % args.print_every == 0:
			print(f"Training episode: {episode+1}")
			print(f"Episode return: {train_reward:.2f}")
	torch.save(agent.policy.state_dict(), "model6.mdl")

	

if __name__ == '__main__':
	main()