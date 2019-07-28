# Project context

Using a simplified version of the Unity Banana environment, an agent is trained to navigate (and collect bananas!) in a large, square world.

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

- 0 - move forward.
- 1 - move backward.
- 2 - turn left.
- 3 - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

# Learning algorithm
The learning algorithm is based on Deep Q-learning (a deep Q-network) which generalizes the approximation of the Q-value function. __Double Q-learning (DDQN)__ tackle the issue of overestimation of Q-values that basic DQNs have. To prevent this, double Q-Learning decouple the selection from the evaluation of an action.
One set of weights is used to determine the greedy policy and the other one to determine its values.

# Implementation details

## Loss function

## Optimization

## epsilon-greedy policy

# Architecture

# Ideas of future works
