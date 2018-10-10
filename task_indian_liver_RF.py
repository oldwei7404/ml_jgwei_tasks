import numpy as np
import pandas as pd 
from pandas import DataFrame 
import matplotlib.pyplot as plot 

from sklearn.cross_validation import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_squared_error

## source data
file_name = "indian_liver_patient.csv"

## some parameters
row_drop_na = True 
col_drop_na = not(row_drop_na)

## now starts
data_df = pd.read_csv(file_name)

## null check
has_null = data_df.isnull().any().any()

if (has_null):
    print ("\n***** Warning: NaN or Null entry found *****\n")
    data_df_null_check_arr = data_df.isnull()
    row_null_stats = np.zeros(data_df.shape[0], dtype = int)
    col_null_stats = np.zeros(data_df.shape[1], dtype = int)

    for i_ in range (data_df_null_check_arr.shape[0]):
        for j_ in range (data_df_null_check_arr.shape[1]):
            if data_df_null_check_arr.iloc[i_, j_] == True:
                row_null_stats[i_] = row_null_stats[i_] +1
                col_null_stats[j_] = col_null_stats[j_] +1

    row_to_drop = []
    col_to_drop = []
    print (" Row no.\t no. of Null entry\n")
    for index in range(len(row_null_stats)):
        if row_null_stats[index] > 0:
            print (index, "\t", row_null_stats[index])
            row_to_drop.append(index)
        

    print (" \n\nCol no./header \t no. of Null entry\n")
    for index in range(len(col_null_stats)):
        if col_null_stats[index] > 0:
            print (str(index)+"/"+data_df.columns[index], "\t", col_null_stats[index])            
            col_to_drop.append(index)
    ## clean up, remove incomplete examples
    if row_drop_na:
        data_df.drop(row_to_drop, inplace = True)
        print (" Rows with Null entry removed, row number now: %d\n", data_df.shape[0])
    elif col_drop_na:
        data_df.drop(data_df.columns[col_to_drop], axis = 1, inplace = True )
        print (" Cols with Null entry removed, col number now: %d\n", data_df.shape[1])

## done cleaning data, next start training
label_df = data_df.iloc[:, -1]
attri_df = data_df.drop(labels = 'Dataset', axis = 1)

xList  = attri_df.values
labels = label_df.values

row_xList = xList.shape[0]
col_xList = xList.shape[1]

cols_coded = np.zeros((row_xList, 2), dtype = int)
xList = np.hstack((xList, cols_coded))

for x_record in xList:
    if x_record[1] == 'Male': xList[-1] = 1.0
    elif x_record[1] == 'Female': xList[-2] = 1.0

xList = np.delete(xList, 1, 1)  ## remove gender column after coding 

xTrain, xTest, yTrain, yTest = train_test_split(xList, labels, test_size=0.30, random_state=531)

headerNames = data_df.columns.values.tolist()
headerNames.remove('Gender')
headerNames.remove('Dataset')
headerNames.append('Sex_FeMale')
headerNames.append('Sex_Male')
headerNames = np.array(headerNames)

#train random forest at a range of ensemble sizes in order to see how the mse changes
mseOos = []
nTreeList = range(50, 500, 10)
for iTrees in nTreeList:
    depth = None
    maxFeat  = 4 #try tweaking
    indianLiverRFModel = ensemble.RandomForestRegressor(n_estimators=iTrees, max_depth=depth, max_features=maxFeat,
                                                 oob_score=False, random_state=531)
    indianLiverRFModel.fit(xTrain,yTrain)

    #Accumulate mse on test set
    prediction = indianLiverRFModel.predict(xTest)
    mseOos.append(mean_squared_error(yTest, prediction))


print("MSE" )
print(mseOos[-1])


#plot training and test errors vs number of trees in ensemble
plot.plot(nTreeList, mseOos)
plot.xlabel('Number of Trees in Ensemble')
plot.ylabel('Mean Squared Error')
#plot.ylim([0.0, 1.1*max(mseOob)])
plot.show()


# Plot feature importance
featureImportance = indianLiverRFModel.feature_importances_

# normalize by max importance
featureImportance = featureImportance / featureImportance.max()
sortedIdx = np.argsort(featureImportance)
barPos = np.arange(sortedIdx.shape[0]) + .5
plot.barh(barPos, featureImportance[sortedIdx], align='center')
plot.yticks(barPos, headerNames[sortedIdx])
plot.xlabel('Variable Importance')
plot.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)
plot.show()
