# Load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import codecademylib3

# Read in the data
codecademy = pd.read_csv('codecademy.csv')

# Print the first five rows
print(codecademy.head())
# Create a scatter plot of score vs completed
plt.scatter(codecademy.completed, codecademy.score)
# Show then clear plot
plt.show()
plt.clf()
# Fit a linear regression to predict score based on prior lessons completed
model = sm.OLS.from_formula('score ~ completed', data=codecademy)
results = model.fit()
print(results.params)
# Intercept interpretation:
# Slope interpretation:
"""
The student is likely to score 13 marks if they have not completed 
any prior course item on codecademy. A student has completed 1 course
on codecademy is likely to add 1.3 marks to their score on the quiz.
"""

# Plot the scatter plot with the line on top
# Show then clear plot
plt.scatter(codecademy.completed, codecademy.score)
plt.plot(codecademy.completed, ((results.params[1] * codecademy.completed) + results.params[0]))
plt.show()
plt.clf()

# Predict score for learner who has completed 20 prior lessons
learner_20_content_score = results.predict({'completed':20})
learner_20_content_score2 = results.params[1] * 20 + results.params[0]

print(learner_20_content_score)
print(learner_20_content_score2)
# Calculate fitted values
fitted_values = results.predict(codecademy)
# Calculate residuals
residuals = codecademy.score - fitted_values
# Check normality assumption
plt.hist(residuals)
# Show then clear the plot
plt.show()
plt.clf()
# Check homoscedasticity assumption
plt.scatter(fitted_values, residuals)
# Show then clear the plot
plt.show()
plt.clf()
# Create a boxplot of score vs lesson
sns.boxplot(x=codecademy.lesson, y=codecademy.score, data=codecademy)
# Show then clear plot
plt.show()
plt.clf()
# Fit a linear regression to predict score based on which lesson they took
model_les = sm.OLS.from_formula('score ~ lesson', data=codecademy)
results = model_les.fit()
print(results.params)
# Calculate and print the group means and mean difference (for comparison)
print(codecademy.groupby('lesson').mean().score)
print(59.220 - 47.578)
# Use `sns.lmplot()` to plot `score` vs. `completed` colored by `lesson`
sns.lmplot(x = 'completed', y = 'score', hue = 'lesson', data = codecademy)
plt.show()
