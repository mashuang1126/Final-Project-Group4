import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import  RandomForestRegressor
from sklearn.ensemble import  GradientBoostingRegressor
from sklearn.linear_model import  Ridge
from sklearn.svm import SVR

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from pandas.core.frame import DataFrame

from sklearn.decomposition import PCA
from sklearn import metrics
import numpy as np

import matplotlib.pyplot as plt
import numpy.matlib
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

import pickle
from sklearn.linear_model import Ridge,Lasso,LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer
from sklearn.model_selection import validation_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor





#load dataset
url="https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
names=['Sex','Length','Diameter','Height','Whole weight','Shucked weight','Viscera weight','Shell weight','Rings']
#abalonedata = pd.read_csv('new_abalone.csv', names=names)
#abalonedata = pd.read_csv('abalone.csv', names=names,header=0)
#abalonedata = pd.read_csv('new_abalone_scale.csv', names=names)
abalonedata = pd.read_csv(url, names=names,header=0)
# print(abalonedata.head())
abalonedata.info()
print(abalonedata.describe())

#browse data type of all column generally
# abalonedata['Length']=abalonedata['Length'].astype('float64')
number_f = abalonedata.select_dtypes(include=[np.number]).columns
object_f = abalonedata.select_dtypes(include=[np.object]).columns
print(number_f)
print(object_f)

# make hist of dataset
abalonedata.hist(figsize=(20,20),grid=True, layout=(2,4),bins=30)
plt.show()



#checking missing value
mv_abalone=abalonedata.isnull().sum().sort_values(ascending=False)
pmv_abalone=(mv_abalone/len(abalonedata))*100
missing_abalone=pd.concat([mv_abalone,pmv_abalone],axis=1,keys=['Missing value','% Missing'])
print(missing_abalone)

#impute missing value
# print(abalonedata.isnull().sum().sort_values(ascending=False))
# print('\n')
# null_data=abalonedata.loc[:,abalonedata.isna().any()]
# print(null_data.isnull().sum().sort_values(ascending=False))
# print(null_data.isnull().sum().sort_values(ascending=False)/len(abalonedata)*100)
# numeric_cols=null_data.select_dtypes(include=[np.number]).columns
# cat_cols=null_data.columns.difference(numeric_cols)
# imp=SimpleImputer(strategy='most_frequent')
# null_data_imp=null_data.copy(deep=True)
# null_data_imp[numeric_cols]=imp.fit_transform(null_data_imp[numeric_cols])
# null_data_imp.replace(np.NaN,0,inplace=True)
# null_data_imp[cat_cols]=null_data_imp[cat_cols].astype('str')
# le=LabelEncoder()
# null_data_imp[cat_cols]=null_data_imp[cat_cols].apply(le.fit_transform)
#
# catimputer=SimpleImputer(missing_values='NaN',strategy='most_frequent')

#analyze sex column
sns.countplot(x='Sex',data=abalonedata)
plt.show()
print("\nSex count in percentage")
print(abalonedata.Sex.value_counts(normalize=True))
print("\nSex count in numbers")
print(abalonedata.Sex.value_counts())

#grouping dataset by sex
print(abalonedata.groupby('Sex')[['Length','Diameter',
                            'Height','Whole weight',
                            'Shucked weight','Viscera weight',
                            'Shell weight','Rings']].mean().sort_values(by='Rings', ascending=False))

#target column analysis
print("Value count of rings column")
print(abalonedata.Rings.value_counts())
print("\nPercentage of rings column")
print(abalonedata.Rings.value_counts(normalize=True))
print("the number of value of rings: ",len(abalonedata.Rings.unique()))

# visualization
# abalonedata['Age']=abalonedata['Rings']+1.5
# print(abalonedata['Age'].head(5))
# plt.figure(figsize=(20,7))
sns.swarmplot(x='Sex', y='Rings', data=abalonedata, hue='Sex')
sns.violinplot(x='Sex', y='Rings', data=abalonedata)
plt.show()

# pairlplot
sns.pairplot(abalonedata[number_f])
plt.show()

# # heatmap
# plt.figure(figsize=(20,7))
# sns.heatmap(abalonedata[number_f].corr(),annot=True)
# plt.show()

#check outlier
#abalonedata_boxplot=pd.get_dummies(abalonedata)
abalonedata.boxplot(rot=90,figsize=(20,5))
plt.show()

#plot each column with rings
plt.scatter(abalonedata['Viscera weight'],abalonedata['Rings'])
plt.xlabel('Viscera weight')
plt.ylabel('Rings')
plt.grid()
plt.show()

#drop possible outliers
abalonedata.drop(abalonedata[(abalonedata['Viscera weight']>0.5) & (abalonedata['Rings']<15)].index, inplace=True)
abalonedata.drop(abalonedata[(abalonedata['Viscera weight']<0.5) & (abalonedata['Rings']>25)].index, inplace=True)

plt.scatter(abalonedata['Shell weight'],abalonedata['Rings'])
plt.xlabel('Shell weight')
plt.ylabel('Rings')
plt.grid()
plt.show()

abalonedata.drop(abalonedata[(abalonedata['Shell weight']>0.8) & (abalonedata['Rings']<20)].index, inplace=True)
abalonedata.drop(abalonedata[(abalonedata['Shell weight']<0.8) & (abalonedata['Rings']>20)].index, inplace=True)

plt.scatter(abalonedata['Shucked weight'],abalonedata['Rings'])
plt.xlabel('Shucked weight')
plt.ylabel('Rings')
plt.grid()
plt.show()

abalonedata.drop(abalonedata[(abalonedata['Shucked weight']>1.0) & (abalonedata['Rings']<15)].index, inplace=True)
abalonedata.drop(abalonedata[(abalonedata['Shucked weight']<1.0) & (abalonedata['Rings']>17.5)].index, inplace=True)

plt.scatter(abalonedata['Whole weight'],abalonedata['Rings'])
plt.xlabel('Whole weight')
plt.ylabel('Rings')
plt.grid()
plt.show()

abalonedata.drop(abalonedata[(abalonedata['Whole weight']>2.0) & (abalonedata['Rings']<17.5)].index, inplace=True)
abalonedata.drop(abalonedata[(abalonedata['Whole weight']<2.0) & (abalonedata['Rings']>17.5)].index, inplace=True)

plt.scatter(abalonedata['Diameter'],abalonedata['Rings'])
plt.xlabel('Diameter')
plt.ylabel('Rings')
plt.grid()
plt.show()

abalonedata.drop(abalonedata[(abalonedata['Diameter']<=0.1) & (abalonedata['Rings']<4)].index, inplace=True)
abalonedata.drop(abalonedata[(abalonedata['Diameter']>=0.6) & (abalonedata['Rings']<=16)].index, inplace=True)
abalonedata.drop(abalonedata[(abalonedata['Diameter']<0.6) & (abalonedata['Rings']>=16)].index, inplace=True)

plt.scatter(abalonedata['Height'],abalonedata['Rings'])
plt.xlabel('Height')
plt.ylabel('Rings')
plt.grid()
plt.show()

abalonedata.drop(abalonedata[(abalonedata['Height']>0.4) & (abalonedata['Rings']<15)].index, inplace=True)
abalonedata.drop(abalonedata[(abalonedata['Height']<=0) & (abalonedata['Rings']<10)].index, inplace=True)

plt.scatter(abalonedata['Length'],abalonedata['Rings'])
plt.xlabel('Length')
plt.ylabel('Rings')
plt.grid()
plt.show()

abalonedata.drop(abalonedata[(abalonedata['Length']<0.2) & (abalonedata['Rings']<4)].index, inplace=True)
abalonedata.drop(abalonedata[(abalonedata['Length']>0.6) & (abalonedata['Rings']<=8)].index, inplace=True)



#label encoding catagorical data
#abalonedata['Sex']=LabelEncoder().fit_transform(abalonedata['Sex'].tolist())
sex_mapping={'M':1,'F':2,'I':3}
abalonedata['Sex']=abalonedata['Sex'].map(sex_mapping)
print(abalonedata['Sex'].head())



def mulg(a): ### group the target
    if  a<5 :
        return 0
    elif  a>=5 and a<10:
        return 1
    elif  a>=10 and a<15:
        return 2
    elif  a>=15 and a<20:
        return 3
    else:
        return 4
#creat new target column
abalonedata['mapRings']=abalonedata.apply(lambda x: mulg(x.Rings),axis=1)
print(abalonedata['mapRings'])


#extract column
#original target
x=abalonedata.iloc[:,0:7]
# y=abalonedata.iloc[:,8]

#5 group target
# y=abalonedata['mapRings']

#binary target
abalonedata['newRings']=np.where(abalonedata['Rings']>10,1,0)
y=abalonedata['newRings']


pca=PCA(whiten=True)
pca.fit(x)
variance=pd.DataFrame(pca.explained_variance_ratio_)
print(variance)

pca=PCA(n_components=7,whiten=True)
pca=pca.fit(x)
dataPCA=pca.transform(x)

dataPCA=pd.DataFrame(dataPCA)
dataPCA.boxplot(rot=90)
plt.show()




#split training set and test set
x_train, x_test, y_train, y_test = train_test_split(dataPCA, y, test_size = 0.20)

#apply standscaler
# scaler = StandardScaler()
# scaler.fit(x_train)
#
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

#apply normalization
# scaler=Normalizer()
# scaler.fit(x_train)
#
# x_train=scaler.transform(x_train)
# x_test=scaler.transform(x_test)

#apply robust
# scaler=RobustScaler()
# scaler.fit(x_train)
#
# x_train=scaler.transform(x_train)
# x_test=scaler.transform(x_test)

#apply maxabs
# scaler=MaxAbsScaler()
# scaler.fit(x_train)
#
# x_train=scaler.transform(x_train)
# x_test=scaler.transform(x_test)

# apply minmax
scaler=MinMaxScaler()
scaler.fit(x_train)

x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

#----------------------------------------------------------------------------------------
#apply mlp


mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=5000,solver='adam',learning_rate='adaptive')
# gs=GridSearchCV(mlp,)
mlp.fit(x_train, y_train)

#return prediction of mlp
print("\nprint result of mlp:")
predictions = mlp.predict(x_test)


#show the matrix and plot of mlp
print(y_test.unique(),"target")
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print("accuracy score: ",accuracy_score(y_test,predictions))
# print(mlp.score(x_test,y_test))
print("mse: ",mean_squared_error(y_test,predictions))
print("rms: ",np.sqrt(mean_squared_error(y_test,predictions)))
print("r^2: ",r2_score(y_test,predictions))
score=cross_val_score(mlp,x_test,y_test,cv=10,scoring='accuracy').mean()
print("cross validation: ",score)
ID=np.arange(0,len(y_test),1)
plt.scatter(ID,y_test,color="red",label="training set")
plt.title("actual set")
# plt.legend()
# plt.show()
plt.scatter(ID,predictions,color="blue",label="prediction")
plt.title("prediction of mlp")
plt.legend()
plt.show()


y_test=np.array(y_test)
predictions=np.array(predictions)
cm=confusion_matrix(y_test,predictions)
plt.matshow(cm)
plt.title('Confusion matrix of mlp')
plt.colorbar()
plt.show()




#----------------------------------------------------------------------------------------
#apply logistic regression
#create a new target column for logistic regression
abalonedata['newRings']=np.where(abalonedata['Rings']>10,1,0)

#extract column
x_lr=abalonedata.iloc[:,0:7]

#original target
y_lr=abalonedata.iloc[:,8]

#binary target
# y_lr=abalonedata['newRings']

#5 group target
# y_lr=abalonedata['mapRings']
x_train1,x_test1,y_train1,y_test1=train_test_split(x_lr,y_lr,test_size=0.2)

logreg=LogisticRegression()
logreg.fit(x_train1,y_train1)
lrpredictions=logreg.predict(x_test1)

#show result and plot of logistic regression
print("\nshow the result of logistic regression:")
# print(y_test.shape)
# print(predictions.shape)
print(confusion_matrix(y_test1,lrpredictions))
print(classification_report(y_test,predictions))
print("mse: ",mean_squared_error(y_test1,lrpredictions))
print("rms: ",np.sqrt(mean_squared_error(y_test1,lrpredictions)))
print("r^2: ",r2_score(y_test1,lrpredictions))
print("accuracy_score: ",accuracy_score(y_test1,lrpredictions))
score=cross_val_score(logreg,x_test,y_test,cv=10,scoring='accuracy').mean()
print("cross validation: ",score)

ID=np.arange(0,len(y_test1),1)
plt.scatter(ID,y_test1,color="red",label="actual set")
plt.title("actual set of logistic regression")
plt.legend()
plt.show()


plt.scatter(ID,lrpredictions,color="yellow",label="prediction")
plt.title("prediction of logistic regression")
plt.legend()
plt.show()

y_test1=np.array(y_test1)
lrpredictions=np.array(lrpredictions)
cm1=confusion_matrix(y_test1,lrpredictions)
plt.matshow(cm1)
plt.title('Confusion matrix of logistic regression')
plt.colorbar()
plt.show()


