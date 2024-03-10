import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp


data = pd.read_csv('cs-training.csv')
data.info()

###################fitting logistic regression model#######################
missing_vars = ['MonthlyIncome', 'NumberOfDependents']
data['flag_MonthlyIncome'] = data.MonthlyIncome.isnull().astype(int)
data['flag_NumberOfDependents'] = data.NumberOfDependents.isnull().astype(int)
data[missing_vars] = data[missing_vars].fillna(data[missing_vars].mean())
missing_vars_clean = missing_vars + ['flag_' + name for name in missing_vars]
data['cus_id'] = range(1, len(data) + 1)  # add an id column with sequential integers
data[missing_vars_clean].head()

data.isnull().sum()
cols = list(data.columns)

rem = ['SeriousDlqin2yrs', 'cus_id', 'DebtRatio', 
       'RevolvingUtilizationOfUnsecuredLines', 'flag_MonthlyIncome', 
       'flag_NumberOfDependents',
       'NumberOfOpenCreditLinesAndLoans']

for it in rem:
    if it in cols:
        cols.remove(it)    

X_train, X_test, y_train, y_test = \
    train_test_split(data[cols], data['SeriousDlqin2yrs'], \
    test_size=0.3, random_state=11)     

logist_model = LogisticRegression(random_state=0).fit(X_train[cols], y_train)
proba_pred = logist_model.predict_proba(X_test[cols])[:,1]
X_test['predict'] = list(proba_pred)

pred = logist_model.predict(X_test[cols])
X_test['predict_vote'] = list(pred)
auc = roc_auc_score(y_test, proba_pred)  

#######################

# Correlation checker for cutting down noise in our dummies
def getcorr_cut(Y, df, varnamelist, thresh):
    # Keeping things simple: we're after correlation and average values
    corr, meanv = [], []
    for vname in varnamelist: 
            X, C = df[vname], np.corrcoef(X, Y)      
            corr.append(np.round(C[1, 0],4))
            meanv.append(round(X.mean(), 6))
    
    # Crafting our correlation DataFrame, weeding out the weak links
    corrdf = pd.DataFrame({'varname': varnamelist, 'correlation': corr, 'mean': meanv}).assign(abscorr=lambda x: np.abs(x['correlation'])).sort_values(['abscorr'], ascending=False)
    corrdf = corrdf.assign(order=range(1, len(corrdf) + 1), meanabs=np.abs(corrdf['mean'])).query("abscorr >= @thresh")

    return corrdf

# Dummy generation for top frequent levels of categorical variables
def get_top_n_levels(data, column, threshold=0.7, max_levels=10):
    # Identifying top N levels based on frequency
    freq, cum_freq = data[column].value_counts(normalize=True), freq.cumsum()
    top_n_levels = freq.head(max_levels).index.tolist() if cum_freq.iloc[0] >= threshold else cum_freq[cum_freq < threshold].index.tolist()
    
    # Adding one more level if it reaches the threshold
    top_n_levels += [cum_freq.index[len(top_n_levels)]] if len(cum_freq) > len(top_n_levels) else []
    
    return top_n_levels

# Process and generate dummies, keeping our model lean and mean
dummy_frames, dummy_cols = [], []
for column in cats:
    top_n_levels = get_top_n_levels(data, column)
    dummies = pd.get_dummies(data[column], prefix=column)[[column + '_' + str(level) for level in top_n_levels if column + '_' + str(level) in dummies.columns]]
    dummy_frames.append(dummies)
    dummy_cols.extend(dummies.columns)

# Merging dummies with key data, trimming the fat with correlation study
DF_new = pd.concat([data[['cus_id', 'SeriousDlqin2yrs']] + dummy_frames], axis=1)
cor = getcorr_cut(DF_new.SeriousDlqin2yrs, DF_new, dummy_cols, 0.05)
DF_new = DF_new[['cus_id', 'SeriousDlqin2yrs'] + cor.varname.tolist()]

# Process each column in categorical and generate dummies for top N levels
cats = ['NumberOfTime30-59DaysPastDueNotWorse',
'NumberOfOpenCreditLinesAndLoans',  'NumberOfTimes90DaysLate',
 'NumberRealEstateLoansOrLines',  'NumberOfTime60-89DaysPastDueNotWorse']

# Final merge, keeping our model focused and ready to predict
df = pd.merge(DF_new, data.drop(cats + ['SeriousDlqin2yrs'], axis=1), on='cus_id', how='inner')

# remove those insignificant variables we have confirmed
rem = ['SeriousDlqin2yrs', 'cus_id', 'DebtRatio',  'flag_MonthlyIncome', 'RevolvingUtilizationOfUnsecuredLines', , 'flag_NumberOfDependents']

for it in rem:
    if it in cols:
        cols.remove(it)   

df = df[['SeriousDlqin2yrs', 'cus_id'] + cols]

#######################################

df[cols] = (df[cols] - df[cols].min()) / (df[cols].max() - df[cols].min())

#####################################

logist_model = LogisticRegression(penalty = 'l1', C = 0.5, solver = 'saga').fit(X_train[cols], y_train)
intercept = logist_model.intercept_[0]
coefficients = logist_model.coef_[0]  # Extract coefficients
features =  logist_model.feature_names_in_ 
features_df = pd.DataFrame(zip(features, coefficients))
features_df.columns = ['feature', 'coef']
features_df['abs_coef'] = features_df['coef'].abs()
features_df = features_df.sort_values('abs_coef')

#######################################

def feature_selection_backward(data, target, features_df,  reduce_th = 0.003,
                               folds = 5):

    def ks_stat(y, yhat):
         return ks_2samp(yhat[y==1], yhat[y!=1]).statistic   
     
    final_features = features_df['feature'].tolist()
    best_auc = 0
    best_ks = 0
    best_features = []
    for feature in features_df['feature']:
        current_features = [f for f in final_features if f != feature]
        X = data[current_features]
        y = data[target]
        cv = StratifiedKFold(n_splits=folds)
        
        model = LogisticRegression(solver='liblinear')  # Using liblinear for binary classification
        auc_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
        avg_auc = np.mean(auc_scores)
        
        # Fitting model to calculate KS statistic (Not part of cross-validation)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict_proba(X_test)[:, 1]
        ks_statistic = ks_stat(y_test, predictions)
        
        if avg_auc > best_auc - reduce_th:  # Using 0.3% AUC reduction as threshold
            best_auc = avg_auc
            best_ks = ks_statistic
            best_features = current_features[:]
        else:
            break  # Stop if AUC is reduced more than threshold
        
        final_features.remove(feature)  # Remove feature for next iteration

    # Refitting the best model with selected features
    best_model = LogisticRegression(solver='liblinear').fit(data[best_features], data[target])
    X_test['pred'] = list(predictions)
    X_test['actual'] = list(y_test)
    
    return best_model, best_auc, best_ks, best_features, X_test

# Example usage (Uncomment the real `data`, `target`, and `feature_df` before running)
best_model, final_auc, final_ks, selected_features, X_test = feature_selection_backward(df, 'SeriousDlqin2yrs', features_df, reduce_th = 0.001, folds = 5)
# print(f"Final AUC: {final_auc}, KS: {final_ks}, Selected Features: {selected_features}")

len(cols) # 26
len(selected_features) # 13

#######################################

def lift_chart(X_test, pred, actual, decn):
    def makerkgrp(D, gp):
        if D['grp'] >= gp:
            x = gp - 1
        else:
            x = D['grp']
        
        return x    
    
    DATA = X_test.sort_values([pred])
    DATA['rk'] = list(range(1, 1+ len(DATA)))
    z = len(DATA) / decn
    DATA['grp'] = np.floor((DATA['rk'] / z))
    DATA['grp'].value_counts()
    DATA['grp'] = DATA.apply(makerkgrp, axis = 1, args=([decn]))
    DATA['cnt'] = 1
    DATA['non_actual'] = 1 - DATA[actual]

    grouped_data = DATA.groupby('grp')
    score_dec = pd.concat([
        grouped_data['cnt'].sum().rename('grp_cnt'),
        grouped_data[pred].mean().rename('pred_target_rate'),
        grouped_data[actual].mean().rename('target_rate'),
        grouped_data[actual].sum().sort_index(ascending=False).cumsum().rename('target_cumrate') / DATA[actual].sum(),
        grouped_data['non_actual'].sum().sort_index(ascending=False).cumsum().rename('nontarget_cumrate') / (len(DATA) - DATA[actual].sum())
    ], axis=1).reset_index()

    score_dec['grp'] = decn - score_dec['grp']
    score_dec = score_dec.sort_values('grp')
    score_dec = score_dec[score_dec['grp']>0]

    avg_target_rate = DATA[actual].mean()
    score_dec['lift'] = (100 * score_dec['target_rate'] / avg_target_rate).round(3)
    score_dec['ks'] = (np.abs(score_dec['target_cumrate'] - score_dec['nontarget_cumrate'])).round(3)
    
    return score_dec

X_test.actual.mean()
lift_chart = lift_chart(X_test, 'pred', 'actual', 10)

# 1) Plotting Lift Chart
plt.figure(figsize=(10, 6))
plt.bar(lift_chart['grp'], lift_chart['lift'], color='blue')
plt.axhline(y=100, color='r', linestyle='--')
plt.xlabel('Group')
plt.ylabel('Lift')
plt.title('Lift Chart')
plt.show()

# 2) Plotting Predicted Target Rate vs Actual Target Rate
plt.figure(figsize=(10, 6))
bar_width = 0.35
index = np.arange(len(lift_chart['grp']))

plt.bar(index, lift_chart['pred_target_rate'], bar_width, label='Predicted Target Rate')
plt.bar(index + bar_width, lift_chart['target_rate'], bar_width, label='Actual Target Rate')

plt.axhline(y=0.0662, color='r', linestyle='--')
plt.xlabel('Group')
plt.ylabel('Rate')
plt.title('Predicted vs Actual Target Rate')
plt.xticks(index + bar_width / 2, lift_chart['grp'])
plt.legend()
plt.show()

#####################################################

def feature_chart(X_test, pred, actual, feature, decn):
    def makerkgrp(D, gp):
        if D['grp'] >= gp:
            x = gp - 1
        else:
            x = D['grp']
        
        return x    
    
    DATA = X_test.sort_values([pred])
    DATA['rk'] = list(range(1, 1+ len(DATA)))
    z = len(DATA) / decn
    DATA['grp'] = np.floor((DATA['rk'] / z))
    DATA['grp'].value_counts()
    DATA['grp'] = DATA.apply(makerkgrp, axis = 1, args=([decn]))
    DATA['cnt'] = 1
    DATA['non_actual'] = 1 - DATA[actual]

    grouped_data = DATA.groupby('grp')
    score_dec = pd.concat([
        grouped_data['cnt'].sum().rename('grp_cnt'),
        grouped_data[feature].mean().rename(feature),
        grouped_data[actual].mean().rename('target_rate'),
        grouped_data[actual].sum().sort_index(ascending=False).cumsum().rename('target_cumrate') / DATA[actual].sum(),
        grouped_data['non_actual'].sum().sort_index(ascending=False).cumsum().rename('nontarget_cumrate') / (len(DATA) - DATA[actual].sum())
    ], axis=1).reset_index()

    score_dec['grp'] = decn - score_dec['grp']
    score_dec = score_dec.sort_values('grp')
    score_dec = score_dec[score_dec['grp']>0]
          
    return score_dec
 
list(X_test.columns)           
f_chart = feature_chart(X_test, 'pred', 'actual', 'NumberOfTimes90DaysLate_0',  10)            

plt.figure(figsize=(10, 6))
bar_width = 0.35
index = np.arange(len(f_chart['grp']))
plt.bar(index, f_chart['NumberOfTimes90DaysLate_0'], bar_width, label='Number Of Times90Days Late is 0')
plt.bar(index + bar_width, f_chart['target_rate'], bar_width, label='Actual Target Rate')

plt.axhline(y=0.0662, color='r', linestyle='--')
plt.xlabel('Group')
plt.ylabel('rate')
plt.title('NumberOfTimes90DaysLate_0 vs Actual Target Rate')
plt.xticks(index + bar_width / 2, f_chart['grp'])
plt.legend()
plt.show()

#########################################

import seaborn as sns
import matplotlib.pyplot as plt
            
df = X_test[['actual', 'pred']]
df['good'] = 1 - df['pred']
df['percentile_rank'] = df['good'].rank(pct=True)

# Assign scores based on percentile ranks with smooth transitions
def smooth_score_assignment(row, bands, transitions):
    for i, band in enumerate(bands):
        if row <= band:
            return np.interp(row, [bands[i-1] if i > 0 else 0, band], transitions[i])

    return transitions[-1][-1]

# Define bands and transitions
bands = [0.15, 0.35, 0.6, 0.85, 1.0]
transitions = [[300, 500], [501, 650],  
              [651, 720], [721, 800],[801, 900]]

df['score'] = df['percentile_rank'].apply(lambda x: smooth_score_assignment(x, bands, transitions))

def assign_type(score):
    if score <= 500:
        return 'LL'
    elif score <= 650:
        return 'LM'
    elif score <= 720:
        return 'MM'
    elif score <= 800:
        return 'HM'
    else:
        return 'HH'

df['type'] = df['score'].apply(assign_type)

# Output the transformed DataFrame
df.head(), df['type'].value_counts(), df['score'].hist()
plt.figure(figsize=(10, 6))
sns.kdeplot(df['score'], color='blue', lw=3) 
plt.title('Score Distribution with Smooth Curve')
plt.xlabel('Score')
plt.ylabel('Density')
plt.grid(True)
plt.show()

score_ranges = ['300_500', '501_650', '651_720', '721_800', '801_900']
data = []
for score_range in score_ranges:
    lower, upper = map(int, score_range.split('_'))
    filtered_df = df[(df['score'] >= lower) & (df['score'] <= upper)]
    avg_score = round(filtered_df['score'].mean())
    cnt_per = (len(filtered_df) / len(df)) * 100
    data.append({'score_range': score_range, 'avg_score': avg_score, 'cnt_per': cnt_per})

summary_df = pd.DataFrame(data)
summary_df['cnt_per'] = summary_df['cnt_per'].round(0)
print(summary_df)


#############################################

# Your variables list and the model training
varslist = ['NumberOfTime60-89DaysPastDueNotWorse_0', 'NumberOfDependents',
            'NumberOfTime30-59DaysPastDueNotWorse_0', 'NumberOfTimes90DaysLate_0', 'age']
X = df[varslist]
y = df['SeriousDlqin2yrs']

model = LogisticRegression(solver='liblinear')
model.fit(X, y)

# Initialize a DataFrame to store the elasticities
elasticities = pd.DataFrame(columns=['Variable', 'Elasticity'])

# Calculate the original predicted probabilities
original_pred_prob = model.predict_proba(X)[:, 1]

for var in varslist:
    X_perturbed = X.copy()
    perturbation = X_perturbed[var].mean() * 0.01  # 1% of the mean
    X_perturbed[var] += perturbation
    
    # Calculate the new predicted probabilities
    new_pred_prob = model.predict_proba(X_perturbed)[:, 1]
    
    # Calculate the percentage change in predictions
    change_in_pred = ((new_pred_prob - original_pred_prob) / original_pred_prob).mean()
    
    # Calculate the elasticity
    elasticity = change_in_pred / 0.01  # Because we used a 1% perturbation
    elasticities = elasticities.append({'Variable': var, 'Elasticity': elasticity}, ignore_index=True)

elasticities

DF1 = pd.DataFrame(index=X.index)
for var in varslist:
    avg_x = X[var].mean()
    y = (X[var] / avg_x) - 1
    elasticity_value = elasticities[elasticities['Variable'] == var]['Elasticity'].values[0]
    DF1[var] = elasticity_value * y

DF2 = DF1.sample(frac = 0.1) 

# Using a more efficient method to find top two columns and their values
reason_df = DF2.apply(lambda x: pd.Series(x.nlargest(2).index, index=['Top1_Var', 'Top2_Var']).append(x.nlargest(2).reset_index(drop=True)), axis=1)

# Renaming columns for clarity
reason_df.columns = ['Top1_Var', 'Top1_Value', 'Top2_Var', 'Top2_Value']

# This creates a new DataFrame with the names of the top two variables and their corresponding values for each row
reason_df.head(10)
