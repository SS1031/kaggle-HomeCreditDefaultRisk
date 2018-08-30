import gc

import pandas as pd
import const

import matplotlib.pyplot as plt

import warnings

plt.rcParams['font.size'] = 22
warnings.filterwarnings('ignore')


def agg_numeric(df, group_var, df_name):
    """Aggregates the numeric values in a dataframe. This can
    be used to create features for each instance of the grouping variable.

    Parameters
    --------
        df (dataframe):
            the dataframe to calculate the statistics on
        group_var (string):
            the variable by which to group df
        df_name (string):
            the variable used to rename the columns

    Return
    --------
        agg (dataframe):
            a dataframe with the statistics aggregated for
            all numeric columns. Each instance of the grouping variable will have
            the statistics (mean, min, max, sum; currently supported) calculated.
            The columns are also renamed to keep track of features created.

    """

    # First calculate counts
    counts = pd.DataFrame(df.groupby(group_var, as_index=False)[df.columns[1]].count()).rename(
        columns={df.columns[1]: '%s_counts' % df_name})

    # Group by the specified variable and calculate the statistics
    agg = df.groupby(group_var).agg(['mean', 'max', 'min', 'sum']).reset_index()

    # Need to create new column names
    columns = [group_var]

    # Iterate through the variables names
    for var in agg.columns.levels[0]:
        # Skip the grouping variable
        if var != group_var:
            # Iterate through the stat names
            for stat in agg.columns.levels[1][:-1]:
                # Make a new column name for the variable and stat
                columns.append('%s_%s_%s' % (df_name, var, stat))

    #  Rename the columns
    agg.columns = columns

    # Merge with the counts
    agg = agg.merge(counts, on=group_var, how='left')

    return agg


def count_categorical(df, group_var, df_name):
    """Computes counts and normalized counts for each observation
    of `group_var` of each unique category in every categorical variable

    Parameters
    --------
    df : dataframe
        The dataframe to calculate the value counts for.

    group_var : string
        The variable by which to group the dataframe. For each unique
        value of this variable, the final dataframe will have one row

    df_name : string
        Variable added to the front of column names to keep track of columns


    Return
    --------
    categorical : dataframe
        A dataframe with counts and normalized counts of each unique category in every categorical variable
        with one row for every unique value of the `group_var`.

    """

    # Select the categorical columns
    categorical = pd.get_dummies(df.select_dtypes('object'))

    # Make sure to put the identifying id on the column
    categorical[group_var] = df[group_var]

    # Groupby the group var and calculate the sum and mean
    categorical = categorical.groupby(group_var).agg(['sum', 'mean'])

    column_names = []

    # Iterate through the columns in level 0
    for var in categorical.columns.levels[0]:
        # Iterate through the stats in level 1
        for stat in ['count', 'count_norm']:
            # Make a new column name
            column_names.append('%s_%s_%s' % (df_name, var, stat))

    categorical.columns = column_names

    return categorical.reset_index()


# Function to calculate missing values by column# Funct
def missing_values_table(df, print_info=False):
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

    if print_info:
        # Print some summary information
        print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
                                                                  "There are " + str(
            mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns


def remove_missing_columns(train, test, threshold=90):
    # Calculate missing stats for train and test (remember to calculate a percent!)
    train_miss = pd.DataFrame(train.isnull().sum())
    train_miss['percent'] = 100 * train_miss[0] / len(train)

    test_miss = pd.DataFrame(test.isnull().sum())
    test_miss['percent'] = 100 * test_miss[0] / len(test)

    # list of missing columns for train and test
    missing_train_columns = list(train_miss.index[train_miss['percent'] > threshold])
    missing_test_columns = list(test_miss.index[test_miss['percent'] > threshold])

    # Combine the two lists together
    missing_columns = list(set(missing_train_columns + missing_test_columns))

    # Print information
    print('There are %d columns with greater than %d%% missing values.' % (len(missing_columns), threshold))

    # Drop the missing columns and return
    train = train.drop(columns=missing_columns)
    test = test.drop(columns=missing_columns)

    return train, test


def aggregate_client(df, group_vars, df_names):
    """Aggregate a dataframe with data at the loan level
    at the client level

    Args:
        df (dataframe): data at the loan level
        group_vars (list of two strings): grouping variables for the loan
        and then the client (example ['SK_ID_PREV', 'SK_ID_CURR'])
        names (list of two strings): names to call the resulting columns
        (example ['cash', 'client'])

    Returns:
        df_client (dataframe): aggregated numeric stats at the client level.
        Each client will have a single row with all the numeric data aggregated
    """

    # Aggregate the numeric columns
    df_agg = agg_numeric(df, group_var=group_vars[0], df_name=df_names[0])

    # If there are categorical variables
    if any(df.dtypes == 'object'):

        # Count the categorical columns
        df_counts = count_categorical(df, group_var=group_vars[0], df_name=df_names[0])

        # Merge the numeric and categorical
        df_by_loan = df_counts.merge(df_agg, on=group_vars[0], how='outer')

        gc.enable()
        del df_agg, df_counts
        gc.collect()

        # Merge to get the client id in dataframe
        df_by_loan = df_by_loan.merge(df[[group_vars[0], group_vars[1]]], on=group_vars[0], how='left')

        # Remove the loan id
        df_by_loan = df_by_loan.drop(columns=[group_vars[0]])

        # Aggregate numeric stats by column
        df_by_client = agg_numeric(df_by_loan, group_var=group_vars[1], df_name=df_names[1])


    # No categorical variables
    else:
        # Merge to get the client id in dataframe
        df_by_loan = df_agg.merge(df[[group_vars[0], group_vars[1]]], on=group_vars[0], how='left')

        gc.enable()
        del df_agg
        gc.collect()

        # Remove the loan id
        df_by_loan = df_by_loan.drop(columns=[group_vars[0]])

        # Aggregate numeric stats by column
        df_by_client = agg_numeric(df_by_loan, group_var=group_vars[1], df_name=df_names[1])

    # Memory management
    gc.enable()
    del df, df_by_loan
    gc.collect()

    return df_by_client


##
# Introduction to Manual Feature Engineering
##
train = pd.read_feather(const.in_app_trn)
test = pd.read_feather(const.in_app_tst)

bureau = pd.read_feather(const.in_bur)
bureau_agg = agg_numeric(bureau.drop(columns=['SK_ID_BUREAU']),
                         group_var='SK_ID_CURR', df_name='bureau')
bureau_counts = count_categorical(bureau, group_var='SK_ID_CURR', df_name='bureau')

# Read in bureau balance
bureau_balance = pd.read_feather(const.in_bbl)
# Counts of each type of status for each previous loan
bureau_balance_counts = count_categorical(bureau_balance,
                                          group_var='SK_ID_BUREAU', df_name='bureau_balance')

# Calculate value count statistics for each `SK_ID_CURR`
bureau_balance_agg = agg_numeric(bureau_balance,
                                 group_var='SK_ID_BUREAU', df_name='bureau_balance')

# Dataframe grouped by the loan
bureau_by_loan = bureau_balance_agg.merge(bureau_balance_counts,
                                          right_index=True, left_on='SK_ID_BUREAU', how='outer')
# Merge to include the SK_ID_CURR
bureau_by_loan = bureau_by_loan.merge(bureau[['SK_ID_BUREAU', 'SK_ID_CURR']],
                                      on='SK_ID_BUREAU', how='left')

bureau_balance_by_client = agg_numeric(bureau_by_loan.drop(columns=['SK_ID_BUREAU']),
                                       group_var='SK_ID_CURR', df_name='client')

# Merge with the value counts of bureau
train = train.merge(bureau_counts, on='SK_ID_CURR', how='left')
test = test.merge(bureau_counts, on='SK_ID_CURR', how='left')

# Merge with the stats of bureau
train = train.merge(bureau_agg, on='SK_ID_CURR', how='left')
test = test.merge(bureau_agg, on='SK_ID_CURR', how='left')

# Merge with the monthly information grouped by client
train = train.merge(bureau_balance_by_client, on='SK_ID_CURR', how='left')
test = test.merge(bureau_balance_by_client, on='SK_ID_CURR', how='left')

##
# Introduction to Manual Feature Engineering, Part 2
##
cash = pd.read_feather(const.in_pcb)
cash_by_client = aggregate_client(cash, group_vars=['SK_ID_PREV', 'SK_ID_CURR'],
                                  df_names=['cash', 'client'])

print('Cash by Client Shape: ', cash_by_client.shape)
train = train.merge(cash_by_client, on='SK_ID_CURR', how='left')
test = test.merge(cash_by_client, on='SK_ID_CURR', how='left')

gc.enable()
del cash, cash_by_client
gc.collect()

installments = pd.read_feather(const.in_inp)
installments_by_client = aggregate_client(installments,
                                          group_vars=['SK_ID_PREV', 'SK_ID_CURR'],
                                          df_names=['installments', 'client'])
print('Installments by client shape: ', installments_by_client.shape)

train = train.merge(installments_by_client, on='SK_ID_CURR', how='left')
test = test.merge(installments_by_client, on='SK_ID_CURR', how='left')

train, test = remove_missing_columns(train, test)

gc.enable()
del installments, installments_by_client
gc.collect()
print('Final Training Shape: ', train.shape)
print('Final Testing Shape: ', test.shape)
