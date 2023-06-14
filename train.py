import numpy as np
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory = []
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.discount_factor * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def get_reward(previous_competencies, current_competencies):
    competency_difference = current_competencies - previous_competencies
    if np.average(competency_difference) > 0:
        reward = 5  # Positive reward for improvement in competencies
    elif np.average(competency_difference) < 0:
        reward = -2  # Negative reward for a decrease in competencies
    else:
        reward = 0  # Neutral reward for no change in competencies

    return reward

state_size = 6 # Specify the size of the state list
action_size = 8 # Specify the number of actions (recommendations)
batch_size= 10
# Create an instance of the DQNAgent class
agent = DQNAgent(state_size, action_size)

# Specify the number of episodes and the maximum number of timesteps per episode
num_episodes = 100
max_steps = 100

for episode in range(num_episodes):
    state = [0,0,0,0,0,0]# Initialize the state list for the current episode
    state = np.reshape(state, [1, state_size])
    total_reward = 0

    for step in range(max_steps):
        # Select an action
        action = agent.act(state)
        # Perform the action and observe the next state and reward
        next_state = np.random.rand(1,state_size)
        reward = get_reward(state,next_state) # Calculate the reward based on the improvement in competencies

        # Update the agent's memory with the experience
        agent.remember(state, action, reward, next_state, False)

        # Transition to the next state
        state = next_state

        # Update the total reward
        total_reward += reward

    # Print the results of the current episode
    print("Episode:", episode, "Total Reward:", total_reward)
    print(state)

    # Train the agent with a batch of experiences
    agent.replay(batch_size)


agent.model.save('/home/vineethm/Documents/temp/model/trained')