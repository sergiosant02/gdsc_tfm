from scipy.stats import kstest, anderson
stat, p = kstest(y["LN_IC50"], 'norm', args=(y["LN_IC50"].mean(), y["LN_IC50"].std()))
print(stat, p) 

# >> Output: 0.07784598710281665 0.0

result = anderson(y["LN_IC50"], dist='norm')
print(result)

# >> Output
# >> AndersonResult(statistic=3269.8765859772393, critical_values=array([0.576, 0.656, 0.787, 0.918, 1.092]), significance_level=array([15. , 10. ,  5. ,  2.5,  1. ]), fit_result=  params: FitParams(loc=2.7822776442090498, scale=2.8346892731645923)
# >>  success: True
# >>  message: '`anderson` successfully fit the distribution to the data.')
# >> 