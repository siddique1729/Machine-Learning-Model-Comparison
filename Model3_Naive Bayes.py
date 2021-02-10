"""
Author: The code is written by Jawwad Shadman Siddique | R11684947
Date of Submission: 11 / 16 / 2020
The model uses Naive Bayes Algorithm
The features are selected by the author from Assignment 02 based on correlation plot &
Mutual Information plot 
Input Parameters: 4
    1. channel condition 2. Structural evaluation 3. Age 4. Age Square
Output Parameter: 1
    Culvert condtion - 0 for Satisfactory & 1 for Unsatisfactory
"""

# Loading Libraries
import os
import pandas as pd
import numpy as np
import mixed_naive_bayes as mnb
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns
from matplotlib import pyplot as plt
from imblearn.over_sampling import SMOTE

# Checking Directory
"""
os.getcwd()
os.chdir('D:\Python Practice Programs')
os.getcwd()
"""

# Reading the Dataset

a = pd.read_csv('strat_clean.csv') # stratified & cleaned dataset file
features = ['struc_eval','channel_cond','SVCYR','SVCYR_sq'] # 4 input parameters
X = a[features] # Dataframe of the input features
Y = a['culv_cond'] # Dataframe of the output feature

# Transforming the dataset using SMOTE analysis
# The Unsatisfactory culvert condition data are the minority data 

oversample = SMOTE()
X, Y = oversample.fit_resample(X, Y) # generating the new dataset X,Y 

# Splitting the dataset into 60% training and 40% testing data

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.40, random_state = 10)

# Training the model
clf = mnb.MixedNB(None) # No binary variable present in the input feature
clf.fit(X_train,Y_train) # fitting the training data
clf.predict(X_train)

# Predicting testing data
y_pred = clf.predict(X_test) # predicting testing data
yprob = clf.predict_proba(X_test) #output probabilities of the test data

# Model Evaluation using the following parameters:
# Accuracy - Total accuracy of the model - (TP+TN) / (TP+FP+TN+FN)
# Precision - Total Positive rate (Satisfactory culvert condition) - TP/(TP+FP)
# Recall - Total negative rate (Unsatisfactory culvert condition) - TN/(TN+FN)
# F1 Score - Depends of Precision & Recall - 2 * (Precision*Recall) / (Precision+Recall)

print("Accuracy of the model: ", metrics.accuracy_score( Y_test, y_pred)) #overall accuracy
print("Precision of the model: ", metrics.precision_score( Y_test, y_pred)) #0 (Satisfactory)
print("Recall of the model: ", metrics.recall_score( Y_test, y_pred)) #1 (Unsatisfactory)
print("F1 Score of the model: ", metrics.f1_score( Y_test, y_pred)) #depends on precision+recall

"""
Defining the function for confusion matrix
function source: https://github.com/DTrimarchi10/confusion_matrix/blob/master/cf_matrix.py

"""

def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.
    '''


    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""


    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)
    
    if title:
        plt.title(title)


# Creating the ROC Curve

y_pred_proba = clf.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(Y_test, y_pred_proba)
auc = metrics.roc_auc_score(Y_test, y_pred_proba) # creaing the area under the curve
plt.plot(fpr,tpr,label="data 1(channel cond, struc eval, age, age square), data2, auc="+str(round(auc,4)))
plt.legend(loc=4)
plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity')
plt.title("Model 3 - Naive Bayes with identified variables")
plt.grid()
plt.show()

# Creating the confusion matrix and plotting it

cnf_matrix = metrics.confusion_matrix(Y_test, y_pred)
cnf_matrix # Y_test is going to be rows(abs), y_pred is going to be cols
labels = ['True Satisfac Culv','False Satisfac Culv',
          'False Unsatisfac Culv','True Unsatisfac Culv']
categories = ['0', '1']
make_confusion_matrix(cnf_matrix, group_names=labels,
                      categories=categories, cmap='Paired',
                      title="Model 3 - Naive Bayes with identified variables")