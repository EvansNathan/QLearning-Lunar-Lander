import gymnasium as gym
import numpy as np
import time
import random
import cv2


env = gym.make("LunarLander-v2", render_mode="rgb_array")

LEARNING_RATE = 0.2
DISCOUNT = 0.99
EPISODES = 7000

DISCRETE_OS_SIZE = [10, 10, 10, 10, 10, 10, 1, 1]
DISCRETE_OS_STEP_SIZE = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

capture_interval = 1000
average_reward = 0
reward_update_interval = 100
export_fps = 30

q_table = np.random.uniform(low=0, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))


def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / DISCRETE_OS_STEP_SIZE
    discrete_state = np.clip(discrete_state, 0, np.array(DISCRETE_OS_SIZE) - 1)  # clamp values to range of q_table
    return tuple(discrete_state.astype(np.int32))


def get_action_from_discrete_state(discrete_state):
    return np.argmax(q_table[discrete_state])


for episode in range(EPISODES):
    seed = random.randint(0, 2**16 -1)
    state, info = env.reset(seed=seed)
    discrete_state = get_discrete_state(state)
    max_observations = 0
    terminated = False
    truncated = False
    total_reward_this_run = 0

    out_video = None
    if episode % capture_interval == 0:
        height, width, _ = env.render().shape
        out_video = cv2.VideoWriter(
            f"Moon Lander Iteration {episode}.mp4",
            cv2.VideoWriter_fourcc(*'mp4v'),
            60,
            (width, height))

    while not terminated and not truncated:
        action = get_action_from_discrete_state(discrete_state)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward_this_run += reward
        new_discrete_state = get_discrete_state(observation)

        # print(new_discrete_state, action, reward)

        if not terminated:
            if new_discrete_state in q_table:
                max_future_q = np.max(q_table[new_discrete_state])
            else:
                max_future_q = 0
            current_q = q_table[discrete_state + (action, )]
            new_q = current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q - current_q)
            q_table[discrete_state + (action, )] = new_q

            discrete_state = new_discrete_state

        if episode % capture_interval ==0:
            out_video.write(env.render())

    average_reward += total_reward_this_run

    if total_reward_this_run >= 200:
        print(f"Mission Success: Episode {episode}!!")

    if episode % capture_interval == 0:
        print(f"Writing Video for iteration {episode}...")
        out_video.release()

    if episode % reward_update_interval == 0:
        print(f"Average Reward in the last {reward_update_interval} runs: {average_reward/reward_update_interval}")
        average_reward = 0


env.close()
