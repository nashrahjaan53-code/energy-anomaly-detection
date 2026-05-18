import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt 
np.random.seed(42)
study_hours=np.array([1,2,3,4,5,6,7,8,9,10, 1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5, 2,3,4,5,6,7,8,9,10,11])
exam_scores=np.array([15, 25, 32, 38, 42, 52, 58, 62, 68, 75,
                        18, 28, 30, 40, 48, 55, 60, 65, 70, 78,
                        22, 30, 36, 45, 50, 55, 63, 68, 72, 80])
print("=" * 50)
print("the training data")
print("="* 50)
for i in range(len(study_hours)):
    print(f"Student{i+1}: Studied{study_hours[i]}hours → Got{exam_scores[i]}%")
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.scatter(study_hours,exam_scores,color='blue',alpha=0.6,s=50)
plt.xlabel('Study Hours')
plt.ylabel('Exam Scores(%)')
plt.title('Our Data Points')
plt.grid(True,alpha=0.3)

x=study_hours.reshape(-1,1)
y=exam_scores
model = LinearRegression()
model.fit(x,y)
m=model.coef_[0]
b =model.intercept_
print("/n" + "="*50)
print("model learned this")
print("="*50)
print(f"Formula: Score = {m:.2f} × StudyHours + {b:.2f}")
print(f"\nMeaning:")
print(f"  For every 1 hour of study, score increases by {m:.2f}%")
print(f"  If you study 0 hours, you'd still get {b:.2f}% (base score)")

plt.subplot(1, 2, 2)
plt.scatter(study_hours, exam_scores, color='blue', alpha=0.6, s=50, label='Actual Data')

line_x = np.array([0, 12])
line_y = m * line_x + b
plt.plot(line_x, line_y, color='red', linewidth=2, label='Best Fit Line')

plt.xlabel('Study Hours')
plt.ylabel('Exam Score (%)')
plt.title('The Line It Learned')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 12)
plt.ylim(0, 100)
plt.tight_layout()
plt.show()

print("\n" + "=" * 50)
print("LET'S PREDICT NEW SCORES")
print("=" * 50)

new_students = np.array([[3.5], [7.0], [9.5], [11.0]])
for hours in new_students:
    predicted_score = model.predict(hours.reshape(1, -1))[0]
    print(f"Study {hours[0]} hours → Predicted Score: {predicted_score:.1f}%")

print("\n" + "=" * 50)
print("HOW IT MADE THESE PREDICTIONS")
print("=" * 50)
test_hour = 7.0
manual_prediction = m * test_hour + b
print(f"For 7 hours: {m:.2f} × {test_hour} + {b:.2f} = {manual_prediction:.1f}%")
print(f"Just plug the number into the formula. That's all.")

errors = []
for i in range(len(study_hours)):
    actual = exam_scores[i]
    predicted = m * study_hours[i] + b
    error = abs(actual - predicted)
    errors.append(error)

print(f"\nAverage prediction error: {np.mean(errors):.1f}%")
print(f"The line isn't perfect but it's the best possible straight line.")