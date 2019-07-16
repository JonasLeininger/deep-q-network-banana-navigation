# Report for solving the Banana Navigation task

## DQN

The first algorithm I worked on for this project is the [ Deep Q-Networks](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf).
You can see the neural network in `pytorch/dqn.py` and the DQN agent in `pytorch/dqn_agent.py`. The architecture is:
- Linear layer with environment states as input mapped to 64 nodes
- ReLu activation
- Linear layer with again 64 nodes/features
- ReLu activation
- Linear output layer that maps to the action dimension

The `run_dqn.py` script trains the dqn_agent and saves the weights in the `checkpoints` folder. I copied the last run to `weights/dqn`.
In the jupyter notebook the graph from the training scores is loaded and an agent with the trained weights.

## Double DQN

The [Double DQN](https://arxiv.org/abs/1509.06461) algorithm works a little bit better.
The Network is the same model as in the DQN model. The only change is how the update of the target is calculated.

The `run_doubledqn.py` script trains the dqn_agent and saves the weights in the `checkpoints` folder. I copied the last run to `weights/double_dqn`.
In the jupyter notebook the graph from the training scores is loaded and an agent with the trained weights.

## Further Research
To improve the results the next step would be to try out the dueling dqn and a [Preoritized Experience Replay](https://arxiv.org/abs/1511.05952)
