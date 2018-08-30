# pandas and numpy for data manipulation
import pandas as pd
import numpy as np

# featuretools for automated feature engineering
# import featuretools as ft

# matplotlit and seaborn for visualizations
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 22
import seaborn as sns

# Suppress warnings from pandas
import warnings

warnings.filterwarnings('ignore')

# modeling
import lightgbm as lgb

# utilities
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder


def plot_feature_importances(df, threshold=0.9):
    """
    Plots 15 most important features and the cumulative importance of features.
    Prints the number of features needed to reach threshold cumulative importance.

    Parameters
    --------
    df : dataframe
        Dataframe of feature importances. Columns must be feature and importance
    threshold : float, default = 0.9
        Threshold for prining information about cumulative importances

    Return
    --------
    df : dataframe
        Dataframe ordered by feature importances with a normalized column (sums to 1)
        and a cumulative importance column

    """

    plt.rcParams['font.size'] = 18

    # Sort features according to importance
    df = df.sort_values('importance', ascending=False).reset_index()

    # Normalize the feature importances to add up to one
    df['importance_normalized'] = df['importance'] / df['importance'].sum()
    df['cumulative_importance'] = np.cumsum(df['importance_normalized'])

    # Make a horizontal bar chart of feature importances
    plt.figure(figsize=(10, 6))
    ax = plt.subplot()

    # Need to reverse the index to plot most important on top
    ax.barh(list(reversed(list(df.index[:15]))),
            df['importance_normalized'].head(15),
            align='center', edgecolor='k')

    # Set the yticks and labels
    ax.set_yticks(list(reversed(list(df.index[:15]))))
    ax.set_yticklabels(df['feature'].head(15))

    # Plot labeling
    plt.xlabel('Normalized Importance');
    plt.title('Feature Importances')
    plt.show()

    # Cumulative importance plot
    plt.figure(figsize=(8, 6))
    plt.plot(list(range(len(df))), df['cumulative_importance'], 'r-')
    plt.xlabel('Number of Features');
    plt.ylabel('Cumulative Importance');
    plt.title('Cumulative Feature Importance');
    plt.show();

    importance_index = np.min(np.where(df['cumulative_importance'] > threshold))
    print('%d features required for %0.2f of cumulative importance' % (importance_index + 1, threshold))

    return df


# # memory management
# import gc
#
# train_bureau = pd.read_csv('../data/input/kernel/train_bureau_raw.csv', nrows=1000)
# test_bureau = pd.read_csv('../data/input/kernel/test_bureau_raw.csv', nrows=1000)
#
# train_previous = pd.read_csv('../data/input/kernel/train_previous_raw.csv', nrows=1000)
# test_previous = pd.read_csv('../data/input/kernel/test_previous_raw.csv', nrows=1000)
#
# # All columns in dataframes
# bureau_columns = list(train_bureau.columns)
# previous_columns = list(train_previous.columns)
#
# # Bureau only features
# bureau_features = list(set(bureau_columns) - set(previous_columns))
#
# # Previous only features
# previous_features = list(set(previous_columns) - set(bureau_columns))
#
# # Original features will be in both datasets
# original_features = list(set(previous_columns) & set(bureau_columns))
#
# print('There are %d original features.' % len(original_features))
# print('There are %d bureau and bureau balance features.' % len(bureau_features))
# print('There are %d previous Home Credit loan features.' % len(previous_features))
#
# train_labels = train_bureau['TARGET']
# previous_features.append('SK_ID_CURR')
#
# train_ids = train_bureau['SK_ID_CURR']
# test_ids = test_bureau['SK_ID_CURR']
#
# # Merge the dataframes avoiding duplicating columns by subsetting train_previous
# train = train_bureau.merge(train_previous[previous_features], on='SK_ID_CURR')
# test = test_bureau.merge(test_previous[previous_features], on='SK_ID_CURR')
#
# print('Training shape: ', train.shape)
# print('Testing shape: ', test.shape)
#
# # One hot encoding
# train = pd.get_dummies(train)
# test = pd.get_dummies(test)
#
# # Match the columns in the dataframes
# train, test = train.align(test, join='inner', axis=1)
# print('Training shape: ', train.shape)
# print('Testing shape: ', test.shape)
#
# cols_with_id = [x for x in train.columns if 'SK_ID_CURR' in x]
# cols_with_bureau_id = [x for x in train.columns if 'SK_ID_BUREAU' in x]
# cols_with_previous_id = [x for x in train.columns if 'SK_ID_PREV' in x]
# print('There are %d columns that contain SK_ID_CURR' % len(cols_with_id))
# print('There are %d columns that contain SK_ID_BUREAU' % len(cols_with_bureau_id))
# print('There are %d columns that contain SK_ID_PREV' % len(cols_with_previous_id))
#
# train = train.drop(columns=cols_with_id)
# test = test.drop(columns=cols_with_id)
# print('Training shape: ', train.shape)
# print('Testing shape: ', test.shape)
#
# ###
# # Remove Collinear Variables
# ###
# threshold = 0.9
#
# # Absolute value correlation matrix
# corr_matrix = train.corr().abs()
# corr_matrix.head()
#
# # Upper triangle of correlations
# upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
# print(upper.head())
#
# # Select columns with correlations above threshold
# to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
# print('There are %d columns to remove.' % (len(to_drop)))
#
# train = train.drop(columns=to_drop)
# test = test.drop(columns=to_drop)
#
# print('Training shape: ', train.shape)
# print('Testing shape: ', test.shape)

#  Read in Full Dataset
train = pd.read_csv('../data/input/kernel/m_train_combined.csv')
test = pd.read_csv('../data/input/kernel/m_test_combined.csv')

###
# Remove Missing Values
###
# Train missing values (in percent)
train_missing = (train.isnull().sum() / len(train)).sort_values(ascending=False)
print(train_missing.head())

# test missing values (in percent)
test_missing = (test.isnull().sum() / len(test)).sort_values(ascending=False)
print(test_missing.head())

# Identify missing values above threshold
train_missing = train_missing.index[train_missing > 0.75]
test_missing = test_missing.index[test_missing > 0.75]

all_missing = list(set(set(train_missing) | set(test_missing)))

print('There are %d columns with more than 75%% missing values' % len(all_missing))
# Need to save the labels because aligning will remove this column
train_labels = train["TARGET"]
train_ids = train['SK_ID_CURR']
tjkest_ids = test['SK_ID_CURR']

train = pd.get_dummies(train.drop(columns=all_missing))
test = pd.get_dummies(test.drop(columns=all_missing))

train, test = train.align(test, join='inner', axis=1)

print('Training set full shape: ', train.shape)
print('Testing set full shape: ', test.shape)

###
# Feature Selection through Feature Importance
###
# Initialize an empty array to hold feature importances
feature_importances = np.zeros(train.shape[1])

# Create the model with several hyperparameters
model = lgb.LGBMClassifier(objective='binary',
                           boosting_type='goss',
                           n_estimators=10000,
                           class_weight='balanced')

# Fit the model twice to avoid overfitting
for i in range(2):
    # Split into training and validation set
    train_features, valid_features, train_y, valid_y = train_test_split(train,
                                                                        train_labels,
                                                                        test_size=0.25,
                                                                        random_state=i)

    # Train using early stopping
    model.fit(train_features,
              train_y,
              early_stopping_rounds=100,
              eval_set=[(valid_features, valid_y)],
              eval_metric='auc', verbose=200)

    # Record the feature importances
    feature_importances += model.feature_importances_

# Make sure to average feature importances!
feature_importances = feature_importances / 2
feature_importances = pd.DataFrame(
    {'feature': list(train.columns),
     'importance': feature_importances}
).sort_values('importance',
              ascending=False)
print(feature_importances.head())

zero_features = list(feature_importances[feature_importances['importance'] == 0.0]['feature'])
print('There are %d features with 0.0 importance' % len(zero_features))
feature_importances.tail()

norm_feature_importances = plot_feature_importances(feature_importances)

train = train.drop(columns=zero_features)
test = test.drop(columns=zero_features)

print('Training shape: ', train.shape)
print('Testing shape: ', test.shape)


def identify_zero_importance_features(train, train_labels, iterations=2):
    """
    Identify zero importance features in a training dataset based on the
    feature importances from a gradient boosting model.

    Parameters
    --------
    train : dataframe
        Training features

    train_labels : np.array
        Labels for training data

    iterations : integer, default = 2
        Number of cross validation splits to use for determining feature importances
    """

    # Initialize an empty array to hold feature importances
    feature_importances = np.zeros(train.shape[1])

    # Create the model with several hyperparameters
    model = lgb.LGBMClassifier(objective='binary', boosting_type='goss', n_estimators=10000, class_weight='balanced')

    # Fit the model multiple times to avoid overfitting
    for i in range(iterations):
        # Split into training and validation set
        train_features, valid_features, train_y, valid_y = train_test_split(train, train_labels, test_size=0.25,
                                                                            random_state=i)

        # Train using early stopping
        model.fit(train_features, train_y, early_stopping_rounds=100, eval_set=[(valid_features, valid_y)],
                  eval_metric='auc', verbose=200)

        # Record the feature importances
        feature_importances += model.feature_importances_ / iterations

    feature_importances = pd.DataFrame(
        {'feature': list(train.columns), 'importance': feature_importances}
    ).sort_values('importance', ascending=False)

    # Find the features with zero importance
    zero_features = list(feature_importances[feature_importances['importance'] == 0.0]['feature'])
    print('\nThere are %d features with 0.0 importance' % len(zero_features))

    return zero_features, feature_importances


second_round_zero_features, feature_importances = identify_zero_importance_features(train,
                                                                                    train_labels)

norm_feature_importances = plot_feature_importances(feature_importances, threshold=0.95)
