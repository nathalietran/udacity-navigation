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
The learning algorithm is an extension of the Deep Q-Networks algorithm which generalizes the approximation of the Q-value function. __Double Q-learning (Double DQN)__ tackle the issue of overestimation of Q-values that basic DQNs have.

To prevent this, double Q-Learning decouple the selection from the evaluation of an action. One set of weights is used to determine the greedy policy and the other one to determine its values. The target network is the obvious candidate for this. This network is the same as the online network except that its parameters ![](https://latex.codecogs.com/svg.latex?\theta_{target}) are updated softly every __four__ steps from the
online network ![](https://latex.codecogs.com/svg.latex?\theta_{local}) as :

![](https://latex.codecogs.com/svg.latex?\theta_{target}&space;=&space;\tau&space;*&space;\theta_{local}&space;&plus;&space;(1&space;-&space;\tau)&space;*&space;\theta_{target})


In order to stabilize the training process, we apply replay buffer which memorizes experiences of the Agent. During learning, the Q-learning updates is applied on samples (or minibatches of size `BATCH_SIZE`) of experience drawn uniformly at random from the pool of stored samples of size `BUFFER_SIZE`.

| Parameter | |
|-|-|
|BATCH_SIZE| 64 |
| BUFFER_SIZE | 100 000 |
| ![](https://latex.codecogs.com/svg.latex?\tau)| 0.001 |
| discount factor ![](https://latex.codecogs.com/svg.latex?\gamma) | 0.99 |

# Architecture

The model architecture consists of a neural network of two layers network with 64 hidden units in each layer and input state size of 37 dimensions and output actions size of 4. All these layers are separated by Rectifier Linear Units (ReLu).

## Loss function
We use the Huber loss to further stabilize the DQN algorithm. It uses MSE for low values and MAE for large values.

Indeed, in the [DQN Nature paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) the authors write:
> “We also found it helpful to clip the error term from the update [...] to be between -1 and 1. This form of error clipping further improved the stability of the algorithm.”

The correct interpretation is given by OpenAI in this [post](https://openai.com/blog/openai-baselines-dqn/):
> There are two ways to interpret this statement — clip the objective, or clip the multiplicative term when computing gradient. The former seems more natural, but it causes the gradient to be zero on transitions with high error, which leads to suboptimal performance, as found in one DQN implementation. The latter is correct and has a simple mathematical interpretation — __Huber Loss__.

## Optimization
The optimization employed to train the
network is Adam with a learning rate set to `LR = 0.0005` and other default parameters from PyTorch library.

## epsilon-greedy policy
At the beginning, the Agent chooses a random action from the action space. Then, the exploration policy used is an epsilon-greedy policy with the ![](https://latex.codecogs.com/svg.latex?\epsilon) decreasing by a decay factor 0.995 from 1 to 0.01. That is, with the probability ![](https://latex.codecogs.com/svg.latex?\epsilon), the Agent selects a random action A and with probability ![](https://latex.codecogs.com/svg.latex?1-\epsilon), it selects an action that has a maximum Q value.

# Plot of Rewards

# Ideas for future works
There is still some room for improvements.

We can try other optimizers such as the RMSProp mentioned in the DQN Nature paper.

We can implement other extensions of the Deep Q-Networks (DQN) algorithm such as the __Dueling DQN__ which has the advantages that we do not need to calculate the value of each action at a state. DDQN can learn which states are (or are not) valuable without having to learn the effect of each action at each state. This new architecture speeds up the training and can more quickly identify the correct action for each action by decoupling the estimation of the state value and the advantage for each action between two streams.

Moreover using __Prioritized Experience Replay__ can make experience replay more efficient and effective than if all experiences are sampled uniformly by introducing a stochastic sampling method.

Finally, Google DeepMind's Rainbow DQN algorithm which incorporates all six existing extensions (each addressing a different issue with the original DQN algorithm) was shown to perform the best among all the other implementation. This algorithm can be used to further improve the performance of the Agent.
