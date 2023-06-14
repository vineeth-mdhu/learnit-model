import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from supabase import get_client

state_size=6

model = load_model('/home/vineethm/Documents/temp/model/trained')
supabase=get_client()


def get_data():
    response = supabase.table('state').select("*").execute()
    return response['data'][0]
    


class DQNAgent:
    def __init__(self,learning_rate=0.001, discount_factor=0.99):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.updated_experiences=get_data()
        self.model = model
        self.model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        self.update()

    def get_reward(self,previous_competencies, current_competencies):
        competency_difference = current_competencies - previous_competencies
        if np.average(competency_difference) > 0:
            reward = 5  
        elif np.average(competency_difference) < 0:
            reward = -2 
        else:
            reward = 0 

        return reward
    
    def act(self):
        state,action,next_state,done=self.updated_experiences
        next_state=np.reshape(next_state,[1,state_size])
        return np.argmax(self.model.predict(next_state))
    

    def update(self):
        state,action,next_state, done = self.updated_experiences
        state=np.reshape(state,[1,state_size])
        next_state=np.reshape(next_state,[1,state_size])
        reward=self.get_reward(state,next_state)
        # Calculate the target Q-value
        target = reward
        if not done:
            target = (reward + self.discount_factor * np.amax(model.predict(next_state)[0]))

        # Update the Q-value estimate for the selected action
        target_f = model.predict(state)
        target_f[0][action] = target

        # Fit the model with the updated target
        model.fit(state, target_f, epochs=1, verbose=0)
        self.model.save('/home/vineethm/Documents/temp/model/trained')

