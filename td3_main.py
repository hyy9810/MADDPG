import numpy as np
import torch
import gym
import argparse
import os
from common.arguments import get_args
from common.utils import make_env
from tqdm import tqdm
import utils
import supersuit as ss
from maddpg.TD3 import TD3

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment

class M2SWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        

    def raw_to_state(self,raw_state):
        state = np.array(list(raw_state.values()))/255
        state.resize((60,84,84))
        return state

    def step(self, action):
        raw_action = {f'piston_{i}': action[i].reshape(1) for i in range(len(action))}
        raw_next_state, raw_reward, r_done, info = self.env.step(raw_action)
        next_state = self.raw_to_state(raw_next_state)
        reward = np.mean(list(raw_reward.values()))
        self.done = done = any(r_done.values())
        self.cur_step += 1
        return next_state, reward, done, info
    def reset(self, **kwargs):
        # print("env reset")
        raw_state = self.env.reset(**kwargs)
        self.cur_step = 0
        self.done = False
        # state = np.array(list(raw_state.values()))/255
        return self.raw_to_state(raw_state)

def eval_policy(policy:TD3, env_name, seed, eval_episodes=1):
    args = get_args()
    eval_env,args = make_env(args)
    eval_env = M2SWrapper(eval_env)
    avg_reward = 0.
    print("start eval")
    cycles = eval_env.unwrapped.max_cycles
    with tqdm(total=eval_episodes*cycles) as pbar:
        for time_step in (range(eval_episodes)):
            state, done = eval_env.reset(), False
            while not done:
                action = policy.select_action(np.array(state))
                state, reward, done, _ = eval_env.step(action)
                avg_reward += reward
                pbar.update(1)

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


if __name__ == "__main__":

    args = get_args()
    

    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    # env = gym.make(args.env)

    # Set seeds
    from pettingzoo.butterfly import pistonball_v6
    env = pistonball_v6.parallel_env()
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 3, 0)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = (60,84,84)#env.observation_spaces['piston_0'].shape
    action_dim = 20#env.action_spaces['piston_0'].shape[0]
    max_action = float(env.action_spaces['piston_0'].high[0])
    args.high_action = max_action

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        "args": args
    }

    # Initialize policy
    if args.policy == "TD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        policy = TD3(**kwargs)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim, args=args)

    # Evaluate untrained policy
    evaluations = [eval_policy(policy, args.env, args.seed)]
    env = M2SWrapper(env)
    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in range(int(args.max_timesteps)):

        episode_timesteps += 1

        # Select action randomly or according to policy
        # if t < args.start_timesteps:
        #     action = env.action_space.sample()
        # else:
        action = (
            policy.select_action(np.array(state))
            + np.random.normal(0, max_action * args.expl_noise, size=action_dim).astype('float32')
        ).clip(-max_action, max_action)

        # Perform action
        next_state, reward, done, _ = env.step(action)
        done_bool = float(done)

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(
                f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            evaluations.append(eval_policy(policy, args.env, args.seed))
            np.save(f"./results/{file_name}", evaluations)
            if args.save_model:
                policy.save(f"./models/{file_name}")
