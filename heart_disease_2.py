# import libraries
import codecademylib3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy.stats import ttest_ind
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import chi2_contingency

# load data
heart = pd.read_csv('heart_disease.csv')
# print(heart.head())

# sns.boxplot(heart.heart_disease, heart.thalach)
# plt.show()

thalach_hd = heart.thalach[(heart.heart_disease == 'presence')]
thalach_no_hd = heart.thalach[(heart.heart_disease == 'absence')]

thalach_typical = heart.thalach[(heart.cp == 'typical angina')]
thalach_asymptom = heart.thalach[(heart.cp == 'asymptomatic')]
thalach_nonangin = heart.thalach[(heart.cp == 'non-anginal pain')]
thalach_atypical = heart.thalach[(heart.cp == 'atypical angina')]

'****** ONE WAY ANOVA TEST **********'
pstats, pval= f_oneway(thalach_typical, thalach_asymptom, thalach_nonangin, thalach_atypical)

# print(pval)

'******** TUKEY RESULTS TEST *******************'
tukey_results = pairwise_tukeyhsd(heart.thalach, heart.cp, 0.05)
# print(tukey_results)

'******** CONTINGENCY OR CROSSTABULATION TABLE **********************'
Xtab = pd.crosstab(heart.cp, heart.heart_disease)
print(Xtab)

'****** CHI-SQUARE TEST *************'
chi2, pval, dof, expected = chi2_contingency(Xtab)
print(pval)


mean_diff = np.mean(thalach_hd) - np.mean(thalach_no_hd)
median_diff = np.median(thalach_hd) - np.median(thalach_no_hd)
# print(mean_diff)
# print(median_diff)


'******** Independent t-test *************'

tstat, pval = ttest_ind(thalach_hd, thalach_no_hd)
print(pval)

# first box plot:
sns.boxplot(x=heart.heart_disease, y=heart.thalach)
plt.show()
 
# second box plot:
plt.clf()
sns.boxplot(x=heart.heart_disease, y=heart.age)
plt.show()
plt.clf()
sns.boxplot(x=heart.heart_disease, y=heart.age)
plt.show()
plt.clf()


sns.boxplot(x=heart.cp, y=heart.thalach)
plt.show()




