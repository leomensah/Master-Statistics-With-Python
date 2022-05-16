#import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import codecademylib3

#load data
forests = pd.read_csv('forests.csv')

#check multicollinearity with a heatmap
corr_grid = forests.corr()
sns.heatmap(corr_grid, vmin=-1, vmax=1, xticklabels=forests.columns, yticklabels=forests.columns, annot=True, center=0, cmap='Purples')
plt.show()
plt.clf()
#plot humidity vs temperature
sns.lmplot(x='temp', y='humid', hue='region', data=forests, ci=None, fit_reg=False, markers=['o', 'x'], palette='Set2')
plt.show()
plt.clf()
#model predicting humidity
modelH = sm.OLS.from_formula('humid ~ temp + region', data=forests).fit()
print(modelH.params)

#equations
#'humid = 142.58 - 2.39*temp - 7.25*region'
#humid_Banaija = 135.33 - 2.39*temp
#humid_Sidi = 142.58 - 2.39*temp

#interpretations
## Coefficient on temp:
# The coefficient of temperature (-2.39) indicates that holding all 
# variables constant, humidity decreases by -2.39 by every 1 degree 
# change in temperature.
# the intercept 142.58 indicates that, there is 142.58 humidity at 0 degrees at Bejaia region.
## For Bejaia equation:
## For Bejaia region, relative humidity is 142.58 % when temperature 
## is at zero degrees (this doesn't make sense since humidity is not above 100%)
## For Sidi Bel-abbes equation:
## At Sidi Bel-abbes region, the humidity is at 135% when temperature 
## is at zero degrees celsius. This also doesn't make sense since the 
## humidity is beyond 100%.

#plot regression lines
sns.lmplot(x='temp', y='humid', hue='region', data=forests, ci=None, fit_reg=False, markers=['o', 'x'], palette='Set2')
plt.plot(forests.temp, modelH.params[0]+modelH.params[1]*0+modelH.params[2]*forests.temp, color='orange', linewidth=5, label='Bejaia_forest')
plt.plot(forests.temp, modelH.params[0]+modelH.params[1]*1+modelH.params[2]*forests.temp, color='red', linewidth=5, label='Sidi Bel')
plt.show()
plt.clf()

#plot FFMC vs temperature
sns.lmplot(x='temp', y='FFMC', hue='fire', markers=['o', 'x'], data=forests, fit_reg=False, ci=None)
plt.show()
plt.clf()
#model predicting FFMC with interaction
resultsF = sm.OLS.from_formula('FFMC ~ temp + fire + temp:fire', data=forests).fit()
print(resultsF.params)
#equations
## Full equation:
## FFMC = -8.10889 + 2.45*temp + 76.79*fire - 1.89*temp*fire
## For locations without fire:
## FFMC = -8.10889 + 2.45*temp
## For locations with fire:
## 71.68 + 0.56*temp

#interpretations
## For locations without fire:
## for locations without fire, there is a slope of 2.45 which 
## indicates that, 1 degree increase in temperature can cause 
## a 2.45 increase in FFMC.
## For locations with fire:
## The slope reduces with the place with fire to 0.56, this shows 
## that fire has an impact on the measure of FFMC.

#plot regression lines
sns.lmplot(x='temp', y='FFMC', hue='fire', markers=['o', 'x'], data=forests, fit_reg=False, ci=None)
plt.plot(forests.temp, resultsF.params[0]+(resultsF.params[1]*0)+(resultsF.params[2]*forests.temp)+(resultsF.params[3]*forests.temp*0), color='blue', label='No Fire')
plt.plot(forests.temp, resultsF.params[0]+(resultsF.params[1]*1)+(resultsF.params[2]*forests.temp)+(resultsF.params[3]*forests.temp*1), label='Fire')
plt.show()
plt.clf()

#plot FFMC vs humid
sns.lmplot(x='humid', y='FFMC', data=forests, fit_reg=False, ci=None)
plt.show()
plt.clf()
#polynomial model predicting FFMC
modelP = sm.OLS.from_formula('FFMC ~ humid + np.power(humid, 2)', data=forests).fit()
print(modelP.params)
#regression equation
# FFMC = 77.634 + 0.7521*humid - 0.01*14humid^2

#sample predicted values
print(modelP.params[0] + modelP.params[1]*25 + modelP.params[2]*np.power(25,2))
print(modelP.params[0] + modelP.params[1]*35 + modelP.params[2]*np.power(35,2))
print(modelP.params[0] + modelP.params[1]*60 + modelP.params[2]*np.power(60,2))
print(modelP.params[0] + modelP.params[1]*70 + modelP.params[2]*np.power(70,2))

#interpretation of relationship
# For lower humidity levels, increases in relative humidity are associated with very small increases in FFMC score, until about 35% relative humidity. After this point increases in humidity are associated with increasingly bigger decreases in FFMC score.

#multiple variables to predict FFMC
modelFFMC = sm.OLS.from_formula('FFMC ~ temp + rain + wind + humid',data=forests).fit()
print(modelFFMC.params)
#predict FWI from ISI and BUI
modelFWI = sm.OLS.from_formula('FWI ~ ISI + BUI',data=forests).fit()
print(modelFWI.params)