
# coding: utf-8

# In[92]:


import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split 
#from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier 
#from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report 
import matplotlib.pyplot as plt


# In[97]:


#Training data
data=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data', sep= ',', header = None)
# Printing the dataset shape 
data.columns = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "label"]
print ("Dataset Length: ", len(data)) 
print ("Dataset Shape: ", data.shape) 

# Printing the dataset obseravtions 
print ("Dataset: ",data.head())

#Preprocessing the data
data['buying'] = pd.Categorical(data['buying'])
data['buying_code'] = data['buying'].cat.codes

data['maint'] = pd.Categorical(data['maint'])
data['maint_code'] = data['maint'].cat.codes

data['lug_boot'] = pd.Categorical(data['lug_boot'])
data['lug_boot_code'] = data['lug_boot'].cat.codes

data['safety'] = pd.Categorical(data['safety'])
data['safety_code'] = data['safety'].cat.codes

data['doors'] = pd.Categorical(data['doors'])
data['d_code'] = data['doors'].cat.codes

data['persons'] = pd.Categorical(data['persons'])
data['p_code'] = data['persons'].cat.codes

data['label'] = pd.Categorical(data['label'])
data['label_code'] = data['label'].cat.codes

data = data[['buying_code', 'maint_code', 'lug_boot_code', 'safety_code', 'd_code', 'p_code', 'label', 'label_code']]
data.head()


# In[98]:


#Testing data
data_test = pd.read_csv('test.data', sep="\t", header=None)
data_test.columns = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "label"]
data_test

#Preprocessing testing data
data_test['buying'] = pd.Categorical(data_test['buying'])
data_test['buying_code'] = data_test['buying'].cat.codes

data_test['maint'] = pd.Categorical(data_test['maint'])
data_test['maint_code'] = data_test['maint'].cat.codes

data_test['lug_boot'] = pd.Categorical(data_test['lug_boot'])
data_test['lug_boot_code'] = data_test['lug_boot'].cat.codes

data_test['safety'] = pd.Categorical(data_test['safety'])
data_test['safety_code'] = data_test['safety'].cat.codes

data_test['doors'] = pd.Categorical(data_test['doors'])
data_test['d_code'] = data_test['doors'].cat.codes

data_test['persons'] = pd.Categorical(data_test['persons'])
data_test['p_code'] = data_test['persons'].cat.codes

data_test['label'] = pd.Categorical(data_test['label'])
data_test['label_code'] = data_test['label'].cat.codes

test_data = data_test[['buying_code', 'maint_code', 'lug_boot_code', 'safety_code', 'd_code', 'p_code', 'label', 'label_code']]
test_data.head()


# In[106]:


print('BAYESIAN CLASSIFICATION')

#X_train, X_test = train_test_split(data, test_size=0.01)
gnb = GaussianNB()
used_features =[
    "buying_code",
    "maint_code",
    "lug_boot_code",
    "safety_code",
    "d_code",
    "p_code"
]
# Train classifier
gnb.fit(data[used_features], data.label)
y_pred = gnb.predict(test_data[used_features])
y_pred


# In[107]:


accuracy = gnb.score(test_data[used_features], test_data.label)
accuracy


# In[108]:


print(confusion_matrix(test_data.label, y_pred))
conf_mat_nb = confusion_matrix(test_data.label, y_pred)
print('\nClasification report', classification_report(test_data.label, y_pred))


# In[109]:


print('DECISION TREE USING GINI INDEX')
clf_gini = DecisionTreeClassifier(criterion = "gini", 
            random_state = 100,max_depth=3, min_samples_leaf=7) 
  
clf_gini.fit(data[used_features].values, data.label)
y_pred_dt1 = clf_gini.predict(test_data[used_features])
y_pred_dt1


# In[110]:


accuracy_gini = clf_gini.score(test_data[used_features], test_data.label)
print("Accuracy for Decision tree using GINI INDEX: ", accuracy_gini)
conf_mat_gini = confusion_matrix(test_data.label, y_pred_dt1)
print(confusion_matrix(test_data.label, y_pred_dt1))
print('Clasification report\n', classification_report(test_data.label, y_pred_dt1))


# In[111]:


print('DECISION TREE USING ENTROPY')
clf_entropy = DecisionTreeClassifier(criterion = "entropy")
clf_entropy.fit(data[used_features].values, data.label)
y_pred_dt2 = clf_entropy.predict(test_data[used_features])
y_pred_dt2


# In[112]:


accuracy_infogain = clf_entropy.score(test_data[used_features], test_data.label)
print("Accuracy for Decision tree using INFO GAIN:  ", accuracy_infogain)
print(confusion_matrix(test_data.label, y_pred_dt2))
conf_mat_infogain = confusion_matrix(test_data.label, y_pred_dt2)
print('\nClasification report\n', classification_report(test_data.label, y_pred_dt2))


# In[118]:


print('PERFORMANCE')

FP = conf_mat_nb.sum(axis=0) - np.diag(conf_mat_nb)  
FN = conf_mat_nb.sum(axis=1) - np.diag(conf_mat_nb)
TP = np.diag(conf_mat_nb)
TN = conf_mat_nb.sum() - (FP + FN + TP)

print("\n1. NAIVE BAYES PERFORMANCE: ")

# Sensitivity
TPR = TP/(TP+FN)
print("a. Sensitivity: ", TPR)
# Specificity
TNR = TN/(TN+FP)
print("b. Specificity", TNR)
# Precision
PPV = TP/(TP+FP)
print("c. Precision", PPV)
#Accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)
print("d. Accuracy", ACC)


# In[119]:


FP = conf_mat_gini.sum(axis=0) - np.diag(conf_mat_gini)  
FN = conf_mat_gini.sum(axis=1) - np.diag(conf_mat_gini)
TP = np.diag(conf_mat_gini)
TN = conf_mat_gini.sum() - (FP + FN + TP)

print("2. GINI INDEX PERFORMANCE: ")

# Sensitivity
TPR = TP/(TP+FN)
print("a. Sensitivity: ", TPR)
# Specificity
TNR = TN/(TN+FP)
print("b. Specificity", TNR)
# Precision
PPV = TP/(TP+FP)
print("c. Precision", PPV)
#Accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)
print("d. Accuracy", ACC)


# In[120]:


FP = conf_mat_infogain.sum(axis=0) - np.diag(conf_mat_infogain)  
FN = conf_mat_infogain.sum(axis=1) - np.diag(conf_mat_infogain)
TP = np.diag(conf_mat_infogain)
TN = conf_mat_infogain.sum() - (FP + FN + TP)

print("3. ENTROPY PERFORMANCE: ")

# Sensitivity
TPR = TP/(TP+FN)
print("a. Sensitivity: ", TPR)
# Specificity
TNR = TN/(TN+FP)
print("b. Specificity", TNR)
# Precision
PPV = TP/(TP+FP)
print("c. Precision", PPV)
#Accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)
print("d. Accuracy", ACC)

