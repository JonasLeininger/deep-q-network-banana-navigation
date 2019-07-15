import time
import numpy as np
from collections import deque

from unityagents import UnityEnvironment

from pytorch.double_dqn_agent import Agent

def main():
    env = UnityEnvironment(file_name="Banana_Linux/Banana.x86_64")
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    print('Number of agents:', len(env_info.agents))
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)
    state = env_info.vector_observations[0]
    print('States look like:', state)
    state_size = len(state)
    print('States have length:', state_size)

    # env_info = env.reset(train_mode=False)[brain_name]
    # state = env_info.vector_observations[0]
    # action_size = brain.vector_action_space_size
    # state_size = len(state)
    # score = 0 
    # while True:
    #     action = np.random.randint(action_size)
    #     env_info = env.step(action)[brain_name]
    #     next_state = env_info.vector_observations[0]
    #     reward = env_info.rewards[0]
    #     done = env_info.local_done[0]
    #     score += reward
    #     state = next_state
    #     time.sleep(0.0001)
    #     if done:
    #         break
        
    # print("Score: {}".format(score))

    env_info = env.reset(train_mode=True)[brain_name]
    state = env_info.vector_observations[0]
    action_size = brain.vector_action_space_size
    state_size = len(state)
    agent = Agent(state_size=state_size, action_size=action_size)
    episodes = 5000
    scores_window = deque(maxlen=100)
    scores = []
    for e in range(episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        state = np.reshape(state, [1, state_size])
        score = 0
        
        for t in range(500):
            action = agent.act(state)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            next_state = np.reshape(next_state, [1, state_size])
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                print("episode: {}/{}, score: {}".format(e, episodes, score))
                break
                
            if ((t+1)% 4) == 0:
                if agent.memory.__len__()>=64:
                    agent.replay()
            
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
        
        scores.append(score)
        scores_window.append(score)
        mean_score = np.mean(scores_window)
        if (e >= 100) and (mean_score >= 10.0):
            agent.save_checkpoint(epoch=e)
            np.save(file="double_dqn_final_scores.npy", arr=np.asarray(scores))
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(e, mean_score))
            break

        if (e)%100 == 0:
            mean_score = np.mean(scores_window)
            agent.save_checkpoint(epoch=e)
            np.save(file="double_dqn_saved_scores.npy", arr=np.asarray(scores))
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(e, mean_score))


if __name__=='__main__':
    main()