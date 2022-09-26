import argparse

"""
Here are the param for the training

"""


def get_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario-name", type=str, default="pistonball", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=100, help="maximum episode length")
    parser.add_argument("--time-steps", type=int, default=int(4e5), help="number of time steps")
    parser.add_argument("--device", type=str, default='cuda:3', help="used device")
    # # 一个地图最多env.n个agents，用户可以定义min(env.n,num-adversaries)个敌人，剩下的是好的agent
    # parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    # # Core training parameters
    # parser.add_argument("--lr-actor", type=float, default=1e-5, help="learning rate of actor")
    # parser.add_argument("--lr-critic", type=float, default=1e-4, help="learning rate of critic")
    # parser.add_argument("--epsilon", type=float, default=0.1, help="epsilon greedy")
    # parser.add_argument("--noise_rate", type=float, default=0.1, help="noise rate for sampling from a standard normal distribution ")
    # parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    # parser.add_argument("--tau", type=float, default=0.01, help="parameter for updating the target network")
    # parser.add_argument("--buffer-size", type=int, default=int(5e3), help="number of transitions can be stored in buffer")
    # parser.add_argument("--batch-size", type=int, default=128, help="number of episodes to optimize at the same time")
    # # Checkpointing
    # parser.add_argument("--save-dir", type=str, default="./model", help="directory in which training state and model should be saved")
    # parser.add_argument("--save-rate", type=int, default=100, help="save model once every time this many episodes are completed")
    # parser.add_argument("--model-dir", type=str, default="", help="directory in which training state and model are loaded")

    # # Evaluate
    # parser.add_argument("--evaluate-episodes", type=int, default=10, help="number of episodes for evaluating")
    # parser.add_argument("--evaluate-episode-len", type=int, default=100, help="length of episodes for evaluating")
    # parser.add_argument("--evaluate", type=bool, default=False, help="whether to evaluate the model")
    # parser.add_argument("--evaluate-rate", type=int, default=1000, help="how often to evaluate model")
    
    parser.add_argument("--policy", default="TD3")
    # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--seed", default=0, type=int)
    # OpenAI gym environment name
    parser.add_argument("--env", default="pistonball")
    # Time steps initial random policy is used
    parser.add_argument("--start_timesteps", default=25e1, type=int)
    # How often (time steps) we evaluate
    parser.add_argument("--eval_freq", default=5e2, type=int)
    # Max time steps to run environment
    parser.add_argument("--max_timesteps", default=1e6, type=int)
    # Std of Gaussian exploration noise
    parser.add_argument("--expl_noise", default=0.1)
    # Batch size for both actor and critic
    parser.add_argument("--batch_size", default=64, type=int)
    # Discount factor
    parser.add_argument("--discount", default=0.99)
    # Target network update rate
    parser.add_argument("--tau", default=0.005)
    # Noise added to target policy during critic update
    parser.add_argument("--policy_noise", default=0.2)
    # Range to clip target policy noise
    parser.add_argument("--noise_clip", default=0.5)
    # Frequency of delayed policy updates
    parser.add_argument("--policy_freq", default=2, type=int)
    # Save model and optimizer parameters
    parser.add_argument("--save_model", action="store_true")
    # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--load_model", default="")

    args = parser.parse_args()

    
    return args
