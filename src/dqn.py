from .base import Agent, TAU
import torch.nn.functional as F


class DQNAgent(Agent):
    """Deep Q-Network Agent"""

    def __init__(self, state_size, action_size,
                 seed, hidden_sizes=[64, 64]):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        super(DQNAgent, self).__init__(state_size,
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

        # selecting best reward with max(1)[0]
        Q_targets_next = (self.qnetwork_target(next_states)
                          .detach()
                          .max(1)[0]
                          .unsqueeze(1))
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
        return "dqn"
