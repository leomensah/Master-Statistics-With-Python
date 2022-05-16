# Import libraries
import codecademylib3
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, binom_test

# Read in the `clicks.csv` file as `abdata`
abdata = pd.read_csv('clicks.csv')
# print(abdata.head())
# print(abdata.shape)
Xtab = pd.crosstab(abdata.group, abdata.is_purchase)
# print(Xtab)

chi2, pval, dof, expected = chi2_contingency(Xtab)
# print(pval)

"the p value is significant and hence we reject the null hypothesis and conclude that, there is a significant difference in the purchase rate for groups"

# print(abdata.head())
num_visits = len(abdata)
# print(num_visits)

num_sales_needed_099 = 1000 / 0.99
num_sales_needed_199 = 1000 / 1.99
num_sales_needed_499 = 1000 / 4.99

p_sales_needed_099 = num_sales_needed_099/num_visits
p_sales_needed_199 = num_sales_needed_199/num_visits
p_sales_needed_499 = num_sales_needed_499/num_visits

samp_size_099 = np.sum(abdata.group == 'A')
sales_099 = np.sum((abdata.group == 'A') & (abdata.is_purchase == 'Yes'))

samp_size_199 = np.sum(abdata.group == 'B')
sales_199 = np.sum((abdata.group == 'B') & (abdata.is_purchase == 'Yes'))

samp_size_499 = np.sum(abdata.group == 'C')
sales_499 = np.sum((abdata.group == 'C') & (abdata.is_purchase == 'Yes'))

pvalueA = binom_test(x=sales_099, n=samp_size_099, p=p_sales_needed_099, alternative = 'greater')
print(pvalueA)

pvalueB = binom_test(x=sales_199, n=samp_size_199, p=p_sales_needed_199, alternative = 'greater')
print(pvalueB)

pvalueC = binom_test(x=sales_499, n=samp_size_499, p=p_sales_needed_499, alternative = 'greater')
print(pvalueC)

# print(num_sales_needed_099)
# print(num_sales_needed_199)
# print(num_sales_needed_499)
# print(p_sales_needed_099)
# print(p_sales_needed_199)
# print(p_sales_needed_499)