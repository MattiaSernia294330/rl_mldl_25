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
		done = False
		train_reward = 0
		state = env.reset()  # Reset the environment and observe the initial state
        action, action_log_prob = agent.get_action(state)

        while not done:
            next_state, reward, done, _ = env.step(action.detach().cpu().numpy())
            next_action, next_action_log_prob = agent.get_action(next_state)

            # Update critic and policy using TD error
            td_error = agent.update_critic(state, action, reward, next_state, next_action, done)
            agent.update_policy(state, action, td_error)

            state = next_state
            action = next_action
            action_log_prob = next_action_log_prob

            train_reward += reward
		if (episode+1)%args.print_every == 0:
			print('Training episode:', episode)
			print('Episode return:', train_reward)


	torch.save(agent.policy.state_dict(), "model3.mdl")

	

if __name__ == '__main__':
	main()