import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import random

np.random.seed(42)

print("=" * 50)
print("YOUR ALL-IN-ONE ML STARTER PROJECT")
print("=" * 50)

data = {
    'study_hours': np.random.randint(1, 10, 100),
    'sleep_hours': np.random.randint(4, 10, 100),
    'previous_score': np.random.randint(30, 100, 100)
}

data['exam_score'] = (5 * data['study_hours'] + 
                       3 * data['sleep_hours'] + 
                       0.5 * data['previous_score'] + 
                       np.random.normal(0, 5, 100))

df = pd.DataFrame(data)

print("\n📊 Your Dataset (first 5 rows):")
print(df.head())

X = df[['study_hours', 'sleep_hours', 'previous_score']]
y = df['exam_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = model.score(X_test, y_test)
error = mean_absolute_error(y_test, predictions)

print(f"\n🎓 SUPERVISED LEARNING - Predict Exam Scores")
print(f"Model Accuracy: {accuracy:.2%}")
print(f"Average Error: {error:.1f} points")
print(f"Study Hours Weight: {model.coef_[0]:.2f}")
print(f"Sleep Hours Weight: {model.coef_[1]:.2f}")
print(f"Previous Score Weight: {model.coef_[2]:.2f}")

print("\n" + "=" * 50)

cluster_data = df[['study_hours', 'sleep_hours', 'exam_score']]
kmeans = KMeans(n_clusters=3)
df['student_type'] = kmeans.fit_predict(cluster_data)

print(f"\n👥 UNSUPERVISED LEARNING - Student Groups")
for i in range(3):
    group = df[df['student_type'] == i]
    print(f"\nGroup {i+1}: {len(group)} students")
    print(f"  Avg Study Hours: {group['study_hours'].mean():.1f}")
    print(f"  Avg Sleep Hours: {group['sleep_hours'].mean():.1f}")
    print(f"  Avg Exam Score: {group['exam_score'].mean():.1f}")

print("\n" + "=" * 50)

class StudyBuddy:
    def __init__(self):
        self.q_values = {}
        self.learning_rate = 0.1
        self.epsilon = 0.3
    
    def get_reward(self, state, action):
        study, sleep = state
        
        if action == 0:
            study += 2
            sleep -= 0.5
        elif action == 1:
            sleep += 2
            study -= 0.5
        else:
            study += 1
            sleep += 1
        
        score = 5 * study + 3 * sleep
        score = max(0, min(100, score))
        
        if score >= 80:
            reward = 10
        elif score >= 50:
            reward = 5
        else:
            reward = -2
        
        return (study, sleep), reward, score >= 80
    
    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 2)
        
        if state in self.q_values:
            return max(self.q_values[state], key=self.q_values[state].get)
        return random.randint(0, 2)
    
    def learn(self, state, action, reward, next_state):
        if state not in self.q_values:
            self.q_values[state] = {0: 0, 1: 0, 2: 0}
        if next_state not in self.q_values:
            self.q_values[next_state] = {0: 0, 1: 0, 2: 0}
        
        current_q = self.q_values[state][action]
        max_next_q = max(self.q_values[next_state].values())
        self.q_values[state][action] = current_q + self.learning_rate * (reward + 0.9 * max_next_q - current_q)

print("\n🤖 REINFORCEMENT LEARNING - Smart Study Planner")
print("Actions: 0=Study More, 1=Sleep More, 2=Balance Both")
print("\nTraining...")

agent = StudyBuddy()
for episode in range(100):
    state = (5.0, 5.0)
    done = False
    
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = agent.get_reward(state, action)
        agent.learn(state, action, reward, next_state)
        state = next_state

state = (5.0, 5.0)
print(f"\nStarting: Study=5hrs, Sleep=5hrs")
print("AI recommends:")

for step in range(5):
    if state in agent.q_values:
        best_action = max(agent.q_values[state], key=agent.q_values[state].get)
    else:
        best_action = random.randint(0, 2)
    
    actions = {0: "📚 Study more", 1: "😴 Sleep more", 2: "⚖️ Balance both"}
    print(f"  Day {step+1}: {actions[best_action]}")
    
    next_state, _, _ = agent.get_reward(state, best_action)
    state = next_state

print("\n" + "=" * 50)
print("=" * 50)