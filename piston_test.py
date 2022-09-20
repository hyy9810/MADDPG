import math
import numpy as np
import supersuit as ss
# from stable_baselines3 import PPO
# from stable_baselines3.ppo import CnnPolicy

from pettingzoo.butterfly import pistonball_v6

# from PIL import Image 

wall_h = 40
wall_w = 40

piston_anchor_color= np.array([68,76,77])
piston_highest = 385
piston_lowest = 449
interval = int(800/20)

def agent_state(state_img):
    global piston_anchor_color, piston_highest, piston_lowest
    cur_p_pos = np.min(np.where(state_img==piston_anchor_color)[0])
    cur_p_ratio = (piston_lowest - cur_p_pos) / (piston_lowest-piston_highest)
    return cur_p_ratio

def get_ball_pos(env):
    raw_pos = env.unwrapped.ball.position
    return [raw_pos[1]-40,raw_pos[0]-40]

def get_ball_near_3_p(env):
    ball_pos = get_ball_pos(env)
    bp_v = ball_pos[1]
    relat_pos = bp_v/interval
    left = math.floor(relat_pos)
    right = math.ceil(relat_pos)
    if relat_pos-left < right-relat_pos or right>=19:
        return [left-1,left,right]
    else:
        return [left,right,right+1]

def get_action(cur_pos, tar_pos):
    res = 0
    if cur_pos>tar_pos:
        res= -1
    elif cur_pos < tar_pos:
        res= 1
    return np.array([res],dtype='float32')

# def triangle(agents,cur_pos):
#     tar_pos = [0.2,0.5,0.7]
#     return {agent:get_action(cur_pos[i],tar_pos[i]) for i, agent in enumerate(agents)}

def get_all_cur_pos(state_img):
    res=[]
    for i in range(20):
        sta = int(i*interval)
        res.append(agent_state(state_img[:,sta:sta+interval,:]))
    return res
def get_all_action(env):
    state_img = env.state()[wall_h:-wall_h,wall_w:-wall_w,:]
    cur_pos = get_all_cur_pos(state_img)
    tar_pos = [0.2 for i in range(20)]
    near_3 = get_ball_near_3_p(env)
    near_3_pos = [0.2,0.5,0.7]
    for j,i in enumerate(near_3):
        tar_pos[i] = near_3_pos[j]
    for j in range(i+1,20):
        tar_pos[i] = 1
    
    return {f'piston_{i}':get_action(cur_pos[i],tar_pos[i]) for i in range(20)}


def main():
    # env = pistonball_v6.parallel_env(
    #     n_pistons=20,
    #     time_penalty=-0.1,
    #     continuous=True,
    #     random_drop=True,
    #     random_rotate=True,
    #     ball_mass=0.75,
    #     ball_friction=0.3,
    #     ball_elasticity=1.5,
    #     max_cycles=125,
    # )
    # env = ss.color_reduction_v0(env, mode="B")
    # env = ss.resize_v1(env, x_size=84, y_size=84)
    # env = ss.frame_stack_v1(env, 3)
    # env = ss.pettingzoo_env_to_vec_env_v1(env)
    # env = ss.concat_vec_envs_v1(
    #     env, 8, num_cpus=4, base_class="stable_baselines3")
    # model = PPO(
    #     CnnPolicy,
    #     env,
    #     verbose=3,
    #     gamma=0.95,
    #     n_steps=256,
    #     ent_coef=0.0905168,
    #     learning_rate=0.00062211,
    #     vf_coef=0.042202,
    #     max_grad_norm=0.9,
    #     gae_lambda=0.99,
    #     n_epochs=5,
    #     clip_range=0.3,
    #     batch_size=256,
    # )
    # model.learn(total_timesteps=200000)
    # model.save("policy")

    # Rendering

    # env = pistonball_v6.env()
    # env = ss.color_reduction_v0(env, mode="B")
    # env = ss.resize_v1(env, x_size=84, y_size=84)
    # env = ss.frame_stack_v1(env, 3)

    # model = PPO.load("policy")
    parallel_env = pistonball_v6.parallel_env(
        n_pistons=20,
        time_penalty=-0.1,
        continuous=True,
        random_drop=True,
        random_rotate=True,
        ball_mass=0.75,
        ball_friction=0.3,
        ball_elasticity=1.5,
        max_cycles=125,
    )
    parallel_env = ss.color_reduction_v0(parallel_env, mode="B")
    parallel_env.reset(), parallel_env.agents
    parallel_env = ss.resize_v1(parallel_env, x_size=84, y_size=84)
    parallel_env.reset(), parallel_env.agents
    parallel_env = ss.frame_stack_v1(parallel_env, 3)
    parallel_env.reset(), parallel_env.agents
    # parallel_env = ss.pettingzoo_env_to_vec_env_v1(parallel_env)
    # parallel_env.reset(), parallel_env.agents
    # parallel_env = ss.concat_vec_envs_v1(parallel_env, 8, num_cpus=4, base_class="stable_baselines3")
    # parallel_env.reset(), parallel_env.agents
    observations = parallel_env.reset()
    
    whole_img = parallel_env.state()[wall_h:-wall_h,wall_w:-wall_w,:]
    # Image.fromarray(whole_img).show()

    
    # for i in range(20):
    #     sta = int(i*interval)
    #     Image.fromarray(state0:=whole_img[:,sta:sta+interval,:]).show()
    #     break
    # print(agent_state(state0))
    max_cycles = 500
    # for step in range(max_cycles):
    #     actions = {agent: model.predict(observations[agent], deterministic=True)[0] for agent in parallel_env.agents}
    #     observations, rewards, dones, infos = parallel_env.step(actions)
    #     parallel_env.render()
    all_reward = 0
    for step in range(max_cycles):
        # actions = {agent: (np.array([-1 if agent=='piston_0' else 1],dtype='float32')) for agent in parallel_env.agents}
        actions = get_all_action(parallel_env)
        observations, rewards, dones, infos = parallel_env.step(actions)
        all_reward+=np.average(list(rewards.values()))
        parallel_env.render()
        if all(dones.values()):
            break
    if all(dones.values()):
        print("all done")
    print(f"cur step {step}")
    print(f"all reward: {all_reward}")
    # whole_img = parallel_env.state()[wall_h:-wall_h,wall_w:-wall_w,:]
    # Image.fromarray(whole_img).show()
    # print(agent_state(whole_img[:,:interval,:]))
    # print(agent_state(whole_img[:,-interval:,:]))
    # get_ball_near_3_p(parallel_env)
    # env.reset()
    # for agent in env.agent_iter():
    #     obs, reward, done, info = env.last()
    #     act = model.predict(obs, deterministic=True)[0] if not done else None
    #     env.step(act)
    #     env.render()
    pass


if __name__ == "__main__":
    main()
