import time
import numpy as np
from collections import deque

from unityagents import UnityEnvironment

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

if __name__=='__main__':
    main()