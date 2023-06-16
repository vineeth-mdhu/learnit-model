import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from supabase_init import get_client

state_size=6
dir_path = os.path.dirname(os.path.realpath(__file__))
model = load_model(f'{dir_path}/trained')
supabase=get_client()


def get_data(student_id):
    response = supabase.table('state').select("*").eq('student_id',student_id).execute()
    print(response)
    return response
    # return (response['data'][0]['current_state'],response['data']['action'],response['data']['next_state'])
    


class DQNAgent:
    def __init__(self,student_id,course_id,learning_rate=0.001, discount_factor=0.99):
        self.student_id=student_id
        self.course_id=course_id
        print("student",student_id)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.updated_experiences=get_data(self.student_id)
        print(self.updated_experiences)
        print(type(self.updated_experiences))
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
        data,count = self.updated_experiences
        next_state=data[1][0]['next_state']
        next_state=np.reshape(next_state,[1,state_size])
        action=np.argmax(self.model.predict(next_state))
        print(type(action))
        data = supabase.table("state").update({"action": action.item()}).eq("student_id", self.student_id).execute()
        success = supabase.table("user_enrollment").update({"recommendation": action.item()}).eq("user_id", self.student_id).eq('course_id',self.course_id).execute()
        return action
    

    def update(self):
        data,count = self.updated_experiences
        state=data[1][0]['current_state']
        action=data[1][0]['action']
        next_state=data[1][0]['next_state']
        state=np.reshape(state,[1,state_size])
        next_state=np.reshape(next_state,[1,state_size])
        reward=self.get_reward(state,next_state)
        # Calculate the target Q-value
        target = reward
        
        target = (reward + self.discount_factor * np.amax(model.predict(next_state)[0]))

        # Update the Q-value estimate for the selected action
        target_f = model.predict(state)
        target_f[0][action] = target

        # Fit the model with the updated target
        model.fit(state, target_f, epochs=1, verbose=0)
        self.model.save(f'{dir_path}/trained')

