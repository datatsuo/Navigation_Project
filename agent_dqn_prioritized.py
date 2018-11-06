import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque, namedtuple

from model_dqn import DQN

LR = 0.001 # learning rate
GAMMA = 0.999 # discount factor for reward
UPDATE_EVERY = 5 # the update of weights is done every UPDATE_EVERY time steps
BATCH_SIZE = 128 # batch size for learning process
BUFFER_SIZE = 100000 # size of the replay buffer
TAU = 0.001 # parameter for soft-update of the target network

EPS_PRIORITIZED = 0.00000001 # epsilon parameter for prioritized replay
ALPHA_PRIORITIZED = 0.9 # alpha parameter for prioritized replay
BETA0_PRIORITIZED = 0.4 # initial beta parameter for prioritized replay

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

class Agent():
    """
    This module is used for interacting with and learn from the environement.

    """

    def __init__(self, state_size, action_size, is_double_Q, is_prioritized, seed = 456):
        """
        This is for initialization.

        (input)
        - state_size (int): size of a state
        - action_size (int): dim of the action space
        - seed (int): random seed
        - is_double_Q (bool): double Q-learning (True) or normal Q-learning
        - is_prioritized (bool): prioritized replay buffer (True) or normal replay buffer
        - seed (int): random seed

        """

        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size

        # Q-networks, local and target
        self.qnetwork_local = DQN(self.state_size, self.action_size, seed).to(device)
        self.qnetwork_target = DQN(self.state_size, self.action_size, seed).to(device)
        self.is_double_Q = is_double_Q

        # optimizer for learning process
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr = LR)

        # replay buffer for learning process
        self.is_prioritized = is_prioritized
        self.buffer = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, self.is_prioritized, seed)
        self.beta = BETA0_PRIORITIZED # power of the weights for the prioritized replay buffer
        self.learning_count = 0 # count how many times the learning process is done (used for the update of beta)

        # the number of time step (modulo UPDATED_EVERY)
        self.t_step = 0
        # loss
        self.loss = 0.0

    def reset(self):
        """
        This method is used for resetting time_step
        
        """
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        """
        This method saves an experience to the replay buffer.
        Then, after a fixed number of iterations, the Q-network learns from
        the experiences stored in the replay buffer if it contains more
        experiences than the batch size.

        (input)
        - state (float, dim = state_size): state vector
        - action (int, dim = action_size): action vector
        - reward (float, dim 1): reward
        - next_state(float, dim = state_size): state vector for the next state
        - done (bool): if the episode is done or not

        """

        # create tensors storing states (same for actions etc.)
        states = torch.from_numpy(np.vstack([state])).float().to(device)
        actions = torch.from_numpy(np.vstack([action])).long().to(device)
        rewards = torch.from_numpy(np.vstack([reward])).float().to(device)
        next_states = torch.from_numpy(np.vstack([next_state])).float().to(device)
        dones = torch.from_numpy(np.vstack([done]).astype(np.uint8)).float().to(device)

        # compute the TD error
        self.qnetwork_local.eval()
        if(self.is_double_Q != True): # for normal Q-learning
            # get the maxium Q-values of target network for next_state
            qsa_target_next_max = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        else: # for double Q-learning
            # get actions which maximize the Q-values of the local network for next_states.
            self.qnetwork_local.eval()
            with torch.no_grad():
                max_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
            self.qnetwork_local.train()
            # get the Q-values of target network for next_state, max_actions
            qsa_target_next_max = self.qnetwork_target(next_states).gather(1, max_actions)

        delta = rewards + GAMMA * qsa_target_next_max * (1-dones) - self.qnetwork_local(states).gather(1, actions)
        delta = delta.data.cpu().numpy()[0][0]
        self.qnetwork_local.train()

        # save experience in the replay buffer
        self.buffer.add(state, action, reward, next_state, done, delta)

        # update the weights of Q-networks every UPDATE_EVERY time steps
        # (and if the replay buffer contains more experiences thant the batch size)
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if(self.t_step == 0):
            if(len(self.buffer) > BATCH_SIZE):
                self.learning_count += 1
                # in case of prioritized buffer, update beta
                if self.is_prioritized:
                    self.beta = 1.0 + (BETA0_PRIORITIZED - 1.0)/self.learning_count

                # learn and update the weights of Q-networks
                self.learn(self.beta)

    def act(self, state, eps):
        """
        This method takes a state as an input, uses the policy defined by
        deep Q-Network and then select the next action based on
        epsilon-greedy algorithm.

        (input)
        - state (float, dim = state_size): state vector
        - eps (float): epsilon for the epsilon-greedy algorithm

        (output)
        - index for the next action (int)

        """

        # convert state a tensor
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        ## retrieve the action value
        self.qnetwork_local.eval() # evaluation mode
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train() # return to the training model

        # choose an action based on epsilon-greedy algorithm
        if(random.random() > eps):
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return np.argmax(random.choice(np.arange(self.action_size)))

    def learn(self, beta = 1.0):
        """
        This method updates the weights of the Q-network
        by learning from the experiences stored in the replay buffer.

        (input)
        - beta (float): beta index for the priority replay

        """

        # sampling from the replay buffer
        states, actions, rewards, next_states, dones, deltas = self.buffer.sample(beta)

        if(self.is_double_Q != True): # for normal Q-learning
            # get the maxium Q-values of target network for next_state
            qsa_target_next_max = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        else: # for double Q-learning
            # get actions which maximize the Q-value of local network for next_state
            self.qnetwork_local.eval()
            with torch.no_grad():
                max_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
            self.qnetwork_local.train()
            # get the Q-values of target network for next_state, max_actions
            qsa_target_next_max = self.qnetwork_target(next_states).gather(1, max_actions)

        # compute target/expected Q-values
        qsa_target = rewards + GAMMA * qsa_target_next_max * (1- dones)
        qsa_expect = self.qnetwork_local(states).gather(1, actions)

        # in case of the prioritized replay buffer, multiply the gradient
        # (and thus temporarily the target and expected Q-value) with weights
        if(self.is_prioritized):

            # multiply the weights with the target/expected Q-values
            # Note that since self.buffer.weights is the square-root of the
            # weights considered in the prioritized replay buffer paper,
            # the mean square error evaluated for qsa_target and qsa_expect below
            # gives an appropriate gradient for the update of the network weights
            # (i.e. multiplied by self.buffer.weights compared to the standard replay buffer case)
            qsa_target = qsa_target * self.buffer.weights
            qsa_expect = qsa_expect * self.buffer.weights

            # update the deltas and priorities in the sampled experiences in the replay buffer
            deltas = qsa_target - qsa_expect
            deltas = deltas.data.cpu().numpy().squeeze(1)
            for i, j in enumerate(self.buffer.id_experiences):
                self.buffer.memory[j]._replace(delta = deltas[i])
                self.buffer.priority[j] = np.power(np.abs(deltas[i]) + EPS_PRIORITIZED, ALPHA_PRIORITIZED)

        # compute the mean square error as a loss and do back-propagation
        self.loss = F.mse_loss(qsa_expect, qsa_target)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

        # soft-update of the parameters in the target network
        self.soft_update(TAU)

    def soft_update(self, tau):
        """
        This method carries out the soft-update of the parameters
        in the target network.

        (input)
        tau (float): parameter for the soft-update

        """
        for l_param, t_param in zip(self.qnetwork_local.parameters(),self.qnetwork_target.parameters()):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)


class ReplayBuffer():
    """
    This module defines the replay buffer used for learning process.
    The size of the replay buffer is fixed and the experience tuples are stored.

    """

    def __init__(self, buffer_size, batch_size, is_prioritized, seed = 246):
        """
        initialization.

        (input)
        - buffer_size (int): the size of the replay buffer
        - batch_size (int): batch size used for the learning process
        - is_prioritized (bool): priority replay buffer (True) or normal one
        - seed (int): random seed

        """

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.is_prioritized = is_prioritized
        self.seed = random.seed(seed)

        self.memory = deque(maxlen = self.buffer_size) # deque storing experienecs

        field_names = ['state', 'action', 'reward', 'next_state', 'done', 'delta']
        self.experience = namedtuple("Experience", field_names = field_names) # an experience as a named tuple

        # in case of the use of the replay buffer, introduce another deque storing the priorities
        # more exactly (|delta| + EPS_PRIORITIZED)**alpha
        if self.is_prioritized:
            self.priority = deque(maxlen = self.buffer_size)

    def add(self, state, action, reward, next_state, done, delta):
        """
        This method is for adding an experience tuple to self.memory
        and the corresponding priority to self.priority

        (input)
        - state (float, dim = state_size): state vector
        - action (int, dim = action_size): action vector
        - reward (float, dim 1): reward
        - next_state(float, dim = state_size): state vector for the next state
        - done (bool): if the episode is done or not
        - delta (float): TD error

        """

        # create a named tuple from the input
        e = self.experience(state, action, reward, next_state, done, delta)
        # append the tuple to the memory
        self.memory.append(e)

        # in case of the prioritized replay buffer, save the priority, too
        if self.is_prioritized:
            p = np.power(np.abs(delta) + EPS_PRIORITIZED, ALPHA_PRIORITIZED)
            self.priority.append(p)

    def sample(self, beta):
        """
        This method samples the batch_size numbers of experiences from the replay buffer
        and convert them to tensors.

        (input)
        - beta (float): beta index for the priority replay

        (output)
        tuple of state tensors, action tensors, reward tensors, next_state tensors
        done tensors and delta tensors

        """
        # sample experiences
        if self.is_prioritized: # for prioritized replay buffer

            # compute the probabilities for each experience
            sum_priority = np.sum(self.priority)
            probs_full = self.priority / sum_priority

            # sample experiences based on the computed probabilities
            self.id_experiences = np.random.choice(len(self.memory), size = self.batch_size, p = probs_full)
            experiences = []
            probs= []
            for i in self.id_experiences:
                experiences.append(self.memory[i])
                probs.append(probs_full[i])

            # compute the square-root of the weights used for the learning step
            weights = np.power(probs, -beta/2) * np.power(len(self.memory), -beta/2)
            max_weights = np.max(weights)
            weights = weights/max_weights
            self.weights = torch.from_numpy(np.vstack([w for w in weights if w is not None])).float().to(device)

        else: # for normal replay buffer
            experiences = random.sample(self.memory, k=self.batch_size)

        # create tensors storing batch_size states (same for actions etc.)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        deltas = torch.from_numpy(np.vstack([e.delta for e in experiences if e is not None])).float().to(device)
        return (states, actions, rewards, next_states, dones, deltas)

    def __len__(self):
        """
        This method returns the current size of the internal memoryself

        """
        return len(self.memory)
