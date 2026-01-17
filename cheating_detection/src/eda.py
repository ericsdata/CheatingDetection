
import pandas as pd
import os
from ydata_profiling import ProfileReport

 # For Jupyter
# profile.to_file("sample_profile.html")  # To save as HTML


dat = pd.read_csv(r'..\data\train.csv')
target = 'is_cheating'

dat_rows = dat.shape[0]

dat['strat_sample'] = dat['high_conf_clean'].fillna('UNK').astype(str) +'_'+ dat[target].fillna('UNK').astype(str)
data_sample = dat.groupby('strat_sample', group_keys=False).apply( lambda x: x.sample(frac=0.4, random_state=22) )
data_sample.drop(columns = ['strat_sample'],inplace=True)
 
profile = ProfileReport(
data_sample,
    title="Sample Pandas DataFrame Profile",
    explorative=True,
   vars={ "target":target },
)

profile.to_file("..\output\data_sample.html")

import XGBpipeline 


feat_cols = dat.columns[1:19]
moddat = dat[~dat[target].isna()]

xg = XGBpipeline.XGBoostPipeline(task = "classification", test_size=.3)

train_x,test_x, train_y,  test_y = xg.split(moddat[feat_cols], y = moddat[target])

#xg.tune(train_x, train_y)

some_good_params = {'eta': 0.06378770346414234,
 'max_depth': 5,
 'subsample': 0.7121449346735086,
 'colsample_bytree': 0.8013918460685786,
 'lambda': 6.233162208541431,
 'alpha': 0.5197118721606118}

xg.train(train_x, train_y, params = some_good_params)
xg.evaluate(test_x,test_y)
xg.shap_analysis(train_x, plot_path =r'..\output\ConfirmedCase_baseModel.png' )