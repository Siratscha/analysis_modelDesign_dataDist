# this script calculates the IS

import pandas as pd
import numpy as np
def aggregate(df,labels, byGender):
     #Initialize an empty dictionary to store the aggregation functions
    aggregation_functions = {}

     #Iterate over the columns and add them to the aggregation functions dictionary
    for column in labels:
        aggregation_functions[column] = 'sum'
    if byGender:
         #Perform the dynamic aggregation
        result = df.groupby(['gender']).agg(aggregation_functions)
    else:
        result = pd.DataFrame(df.agg(aggregation_functions)).T

    return result

def format_with_commas(num):
    return '{:,.2f}'.format(num).replace('.', ',')



# %%
def compute_IR(df):

    # take an aggregated dataframe which contains the count of the labels per gender and disease and calculate the IR
    columns = list(df.columns)
    is_c = []

    # IS_C: calculate IS per class
    for column_outer in columns[:]:
        max_min_values = []
        for column_inner in columns[:]:
            if column_outer == column_inner:
                continue
            else:
            # Calculate the max/min values for each combination of columns
                max_min_values.append(df.apply(lambda x: max(x[column_inner], x[column_outer])/min(x[column_inner], x[column_outer]),
                          axis=1).values)
        is_c.append([sum(elements) for elements in zip(*max_min_values)])

    is_c_disease = pd.DataFrame(data=is_c,index=columns, columns=df.index.values)

    rows = df.index
    is_group = []

    #IS_G: calculate IS per group for each class
    for column in columns[:]:
        sum_row_result = []
        for row_outer in rows:
            max_min_values = []
            for row_inner in rows:
                if row_outer >= row_inner:
                    continue
                else:
                    max_min_values.append( max(df.loc[row_outer, column], df.loc[row_inner, column]) / min(df.loc[row_outer, column], df.loc[row_inner, column]) )
            if len(max_min_values) > 0:
                sum_row_result.append(sum(max_min_values) )
        is_group.append(sum(sum_row_result))
        

    is_g_df = pd.DataFrame(is_group, index=columns).T
   
    return is_c_disease, is_g_df

# %%
def mmult(df): 
    # take an aggregated df and calculate the matrix multiplication between class (disease) and group (gender) imbalance
    # returns 
    df_IR_disease, df_IR_gender = compute_IR(df)
    df_IR_disease.to_numpy()
    df_IR_gender.to_numpy()
    matrix_mul = np.matmul(df_IR_gender,df_IR_disease)
    # substract 1 per class to obtain 0 if all classes and groups are balanced since IR per default returns 1 in the balanced case
    correction = (df_IR_disease.shape[0] - 1) * (df_IR_gender.shape[1]) 
    result = matrix_mul - correction
    result_df = pd.DataFrame(result)
    normalized_df = result_df.copy()

    # normalize by number of classes and groups
    normalized_df = normalized_df.sum(axis=1) / (len(df.columns) * len(df))
   
    return normalized_df#, result_df.applymap(format_with_commas) #<- uncomment to obtain IS per group

# %%
data = {
    'Edema': [15000, 15000],
    'Cardiomegaly': [45000, 15000],
    'Atelectasis': [15000, 15000]
}

index = ['0', '1']

df = pd.DataFrame(data, index=index)

# %%
result = mmult(df)
print("IS: ", result[0])