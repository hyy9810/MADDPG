from tqdm import tqdm
from agent import Agent, SAgent
from common.replay_buffer import Buffer
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from common.utils import writer


class Runner:
    def __init__(self, args, env):
        self.args = args
        self.noise = args.noise_rate
        self.epsilon = args.epsilon
        self.episode_limit = args.max_episode_len
        self.env = env
        self.agents = self._init_agents()
        self.buffer = Buffer(args)
        self.save_path = self.args.save_dir + '/' + self.args.scenario_name
        self.best_return = None
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def _init_agents(self):
        agents = []
        # self.agent = agent = Agent(0, self.args)
        self.agent = SAgent(0, self.args)
        # for i in range(self.args.n_agents):
        #     agents.append(agent)
        return agents

    def run(self):
        returns = []
        done = {f'piston_{i}':True for i in range(self.args.n_agents)}
        for time_step in tqdm(range(self.args.time_steps)):
            # reset the environment
            if any(done.values()):
                s = self.env.reset()
                s = list(s.values())
                done=False
            u = []
            actions = []
            agent_list = self.env.agents
            with torch.no_grad():
                actions = self.agent.select_actions(s,self.noise,self.epsilon)
                u = actions.copy()
                # for agent_id, agent in enumerate(self.agents):
                #     action = agent.select_action(s[agent_id], self.noise, self.epsilon)
                #     u.append(action)
                #     actions.append(action)
            # for i in range(self.args.n_agents, self.args.n_players):
            #     actions.append(np.array([np.random.rand(), np.random.rand(), np.random.rand(), np.random.rand(), 0],dtype='float32'))
            actions_dict = {agent_list[i]: actions[i]for i in range(len(actions))}
            s_next, r, done, info = self.env.step(actions_dict)
            
            r = list(r.values())
            s_next = list(s_next.values())
            self.buffer.store_episode(s[:self.args.n_agents], u, r[:self.args.n_agents], s_next[:self.args.n_agents])
            s = s_next
            if self.buffer.current_size >= self.args.batch_size and any(done.values()):
                transitions = self.buffer.sample(self.args.batch_size)
                self.agent.learn(transitions)
                # for agent in self.agents:
                #     other_agents = self.agents.copy()
                #     other_agents.remove(agent)
                #     agent.learn(transitions, other_agents)
            if time_step > 0 and time_step % self.args.evaluate_rate == 0:
                returns.append(self.evaluate())
                # 保存表现最好的
                if self.best_return == None or returns[-1] > self.best_return:
                    self.best_return = returns[-1]
                    self.agent.policy.save_model_in(f'best_in_{len(returns)}')
                plt.figure()
                plt.plot(range(len(returns)), returns)
                writer.add_scalar("eval/rewards", returns[-1], len(returns))
                plt.xlabel('episode * ' + str(self.args.evaluate_rate / self.episode_limit))
                plt.ylabel('average returns')
                plt.savefig(self.save_path + '/plt.png', format='png')
            self.noise = max(0.05, self.noise - 0.0000005)
            self.epsilon = max(0.05, self.epsilon - 0.0000005)
            np.save(self.save_path + '/returns.pkl', returns)

    def evaluate(self):
        returns = []
        for episode in range(self.args.evaluate_episodes):
            # reset the environment
            s = self.env.reset()
            s = list(s.values())
            rewards = 0
            done = {f'piston_{i}':True for i in range(self.args.n_agents)}
            for time_step in range(self.args.evaluate_episode_len):
                # self.env.render()
                actions = []
                with torch.no_grad():
                    actions = self.agent.select_actions(s,self.noise,self.epsilon)
                actions =  {self.env.agents[i]: actions[i]for i in range(len(actions))}
                s_next, r, done, info = self.env.step(actions)
                s_next = list(s_next.values())
                r = list(r.values())
                rewards += r[0]
                s = s_next
                if any(done.values()):
                    break
            returns.append(rewards)
            print('Returns is', rewards)
        s = self.env.reset()
        return sum(returns) / self.args.evaluate_episodes
