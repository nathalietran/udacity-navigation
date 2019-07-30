from base import Agent, TAU
import torch.nn.functional as F

class DoubleDQNAgent(Agent):
    """Double Deep Q-Network Agent"""

    def __init__(self, state_size, action_size,
                 seed, hidden_sizes=[64, 64]):
        super(DDQNAgent, self).__init__(state_size,
                                        action_size,
                                        seed,
                                        hidden_sizes)

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple
            of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # selecting best actions with max(1)[1]
        # the greedy policy is estimated according the online network
        Q_actions_next = (self.qnetwork_local(next_states)
                          .max(1)[1]
                          .unsqueeze(1))
        # selecting best rewards according to the target QNetwork
        # fairly evaluate the value of this policy with the target network
        Q_targets_next = (self.qnetwork_target(next_states)
                          .detach().
                          gather(1, Q_actions_next))
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        # Compute huber loss
        loss = F.smooth_l1_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def __str__(self):
        return "double_dqn"
