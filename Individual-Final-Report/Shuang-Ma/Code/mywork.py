import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import pickle
from sklearn.linear_model import Ridge,Lasso,LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer,accuracy_score,confusion_matrix,classification_report
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import validation_curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn import metrics
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
names = ['Sex','Length','Diameter','Height','Whole Weight','Shucked Weight',
         'Viscera Weight','Shell Weight','Rings']
dataset = pd.read_csv(url,names=names)

# dataset = pd.read_csv('/Users/ShuangMa/Desktop/GWU_Documents/DATS_6202/'
#                       'Final_Project/new_abalone.csv')
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)

# print(dataset.shape)
# print(dataset.isnull().sum())
# print(dataset.dtypes)
print(dataset.describe())

### Pie chart for Sex distribution###
# n = len(dataset['Sex'].unique())
# labels = [dataset['Sex'].unique()[i] for i in range(n)]
# fraces = [dataset['Sex'].value_counts()[i] for i in range(n)]
# explode = [0.1,0,0]
# plt.rcParams['font.sans-serif']=['SimHei']
# plt.title('Abalone percentage by Sex')
# wedges, texts, autotexts = plt.pie(x=fraces,labels=labels,autopct='%0.2f%%',
#                                    explode=explode,shadow=True)
# plt.legend(wedges,labels,fontsize=10,title='Sex',
#            loc="center left",bbox_to_anchor=(0.9,0,0.3,1))
# plt.show()


### Mean for each features by Sex ###
# gp_age=dataset.drop('Rings',axis=1).groupby('Sex').mean()
# gp_age.plot(kind='bar',grid=False)
# plt.title('Mean for each features by Sex')
# plt.legend(loc='best')
# plt.show()


dataset['Age'] = dataset['Rings']+1.5
dataset.drop('Rings',axis=1,inplace=True)

# print(dataset.shape)
# print(dataset.isnull().sum())
# print(dataset.dtypes)
# print(dataset.describe())

# dataset.hist(figsize=(20,20),grid=False,layout=(2,4),bins = 35)
# plt.show()

### Scatter plot Lenghth vs Age ###
dataset = dataset[(dataset['Age']< 28) & (dataset['Length']>0.1)]
# plt.scatter(dataset['Length'],dataset['Age'],c=dataset['Age'])
# plt.xlabel('Abalone Length')
# plt.ylabel('Age')
# plt.title('Age vs Length')
# plt.show()



### Scatter plot Diameter vs Age ###

# plt.scatter(dataset['Diameter'],dataset['Age'],c=dataset['Age'])
# plt.xlabel('Abalone Diameter')
# plt.ylabel('Age')
# plt.title('Age vs Diameter')
# plt.show()



### Scatter plot Height vs Age ###

dataset = dataset[dataset['Height']<0.4]

# plt.scatter(dataset['Height'],dataset['Age'],c=dataset['Age'])
# plt.xlabel('Abalone Height')
# plt.ylabel('Age')
# plt.title('Age vs Height')
# plt.show()


### Scatter plot Whole Weight vs Age ###
# plt.scatter(dataset['Whole Weight'],dataset['Age'],c=dataset['Age'])
# plt.xlabel('AWhole Weight')
# plt.ylabel('Age')
# plt.title('Age vs Whole Weight')
# plt.show()


### Scatter plot Shucked Weight vs Age ###
dataset = dataset[dataset['Shucked Weight']<1.4]
# plt.scatter(dataset['Shucked Weight'],dataset['Age'],c=dataset['Age'])
# plt.xlabel('Shucked Weight')
# plt.ylabel('Age')
# plt.title('Age vs Shucked Weight')
# plt.show()


### Scatter plot Viscera Weight vs Age ###
dataset = dataset[dataset['Viscera Weight']<0.58]
# plt.scatter(dataset['Viscera Weight'],dataset['Age'],c=dataset['Age'])
# plt.xlabel('Viscera Weight')
# plt.ylabel('Age')
# plt.title('Age vs Viscera Weight')
# plt.show()


### Scatter plot Shell Weight vs Age ###
dataset = dataset[dataset['Shell Weight']<1]
# plt.scatter(dataset['Shell Weight'],dataset['Age'],c=dataset['Age'])
# plt.xlabel('Shell Weight')
# plt.ylabel('Age')
# plt.title('Age vs Shell Weight')
# plt.show()



# # plt.hist(dataset['Length'],bins=20,facecolor='blue',edgecolor='black',alpha=0.7)
# # plt.title('Length distribution')
# # plt.show()
#
#
# # plt.hist(dataset['Diameter'],bins=20,facecolor='blue',edgecolor='black',alpha=0.7)
# # plt.title('Diameter distribution')
# # plt.show()
# #
# # plt.hist(dataset['Height'],bins=20,facecolor='blue',edgecolor='black',alpha=0.7)
# # plt.title('Height distribution')
# # plt.show()
# #
# # plt.hist(dataset['Whole Weight'],bins=20,facecolor='blue',edgecolor='black',alpha=0.7)
# # plt.title('Whole weight distribution')
# # plt.show()
# #
# # plt.hist(dataset['Shucked Weight'],bins=20,facecolor='blue',edgecolor='black',alpha=0.7)
# # plt.title('Shucked weight distribution')
# # plt.show()
# #
# # plt.hist(dataset['Viscera Weight'],bins=20,facecolor='blue',edgecolor='black',alpha=0.7)
# # plt.title('Viscera weight distribution')
# # plt.show()
# #
# # plt.hist(dataset['Shell Weight'],bins=20,facecolor='blue',edgecolor='black',alpha=0.7)
# # plt.title('Shell weight distribution')
# # plt.show()

# plt.hist(dataset['Age'],bins=20,facecolor='blue',edgecolor='black',alpha=0.7)
# plt.title('Age distribution')
# plt.show()


# plt.figure(figsize=(10,6))
# info = dataset.iloc[:,1:8].values
# plt.boxplot(info)
# plt.title('Boxplot for All features from Lengh to Shell Weight')
# plt.show()

#
#
# dataset = dataset[dataset['Length']>0.1]
#

### Box plot for Length ###
# length_box = pd.DataFrame(dataset['Length'])
# length_box.plot.box(title='Length')
# plt.grid(linestyle='--',alpha = 0.7)
# plt.title('Boxplot for Length')
# plt.show()
#


### Box plot for Diameter ###
# diameter_box = pd.DataFrame(dataset['Diameter'])
# diameter_box.plot.box(title='Diameter')
# plt.grid(linestyle='--',alpha = 0.7)
# plt.title('Boxplot for Diameter')
# plt.show()




### Box plot for Height ###
dataset = dataset[dataset['Height']<0.25]
dataset = dataset[dataset['Height']>0]
# height_box = pd.DataFrame(dataset['Height'])
# height_box.plot.box(title='Height')
# plt.grid(linestyle='--',alpha = 0.7)
# plt.title('Boxplot for Height')
# plt.show()





### Box plot for Whole Weight ###
dataset = dataset[dataset['Whole Weight']< 2.75]
# ww_box = pd.DataFrame(dataset['Whole Weight'])
# ww_box.plot.box(title='Whole Weight')
# plt.grid(linestyle='--',alpha = 0.7)
# plt.title('Boxplot for Whole Weight')
# plt.show()



### Box plot for Shucked Weight ###
dataset = dataset[dataset['Shucked Weight']< 1.3]
# sw_box = pd.DataFrame(dataset['Shucked Weight'])
# sw_box.plot.box(title='Shucked Weight')
# plt.grid(linestyle='--',alpha = 0.7)
# plt.title('Boxplot for Shucked Weight')
# plt.show()


### Box plot for Viscera Weight ###
dataset = dataset[dataset['Viscera Weight']< 0.55]
# vw_box = pd.DataFrame(dataset['Viscera Weight'])
# vw_box.plot.box(title='Viscera Weight')
# plt.grid(linestyle='--',alpha = 0.7)
# plt.title('Boxplot for Viscera Weight')
# plt.show()




### Box plot for Shell Weight ###
dataset = dataset[dataset['Shell Weight']< 0.75]
# sw_box = pd.DataFrame(dataset['Shell Weight'])
# sw_box.plot.box(title='Shell Weight')
# plt.grid(linestyle='--',alpha = 0.7)
# plt.title('Boxplot for Shell Weight')
# plt.show()



# dataset.to_csv('/Users/ShuangMa/Desktop/GWU_Documents/DATS_6202/Final_Project/new_abalone.csv')
# print(dataset.shape)




# plt.scatter(dataset['Height'],dataset['Age'],c=dataset['Age'])
# plt.xlabel('Abalone Height')
# plt.ylabel('Age')
# plt.title('Age vs Height')
# plt.show()


### Pairplot for all features ###
# sns.set()
# cols = ['Length','Diameter','Height','Whole Weight', 'Shucked Weight','Viscera Weight', 'Shell Weight','Age']
# sns.pairplot(dataset[cols], height = 2.5)
# plt.show();




# dataset['Sex']= dataset['Sex'].fillna('')
# labelencoder = LabelEncoder()
# dataset['SexEncoded'] = labelencoder.fit_transform(dataset['Sex'])
# dataset_scaled = minmax_scale(dataset.drop(['Sex'],axis=1))

# dataset_scaled = pd.DataFrame(dataset_scaled,columns= names )
# print(dataset_scaled.describe())
# dataset_scaled.to_csv('/Users/ShuangMa/Desktop/GWU_Documents/DATS_6202/'
#                       'Final_Project/new_abalone_scale.csv')




dataset['Sex']= LabelEncoder().fit_transform(dataset['Sex'].tolist())

print(dataset['Sex'])

x, y = dataset.iloc[:,0:8].values,dataset.iloc[:,8].values
# print(x)
# print(type(y))
# print(y)
# print(y.shape)


### Standard Scaler ###
# s_scaler = StandardScaler()
# x_s_scaler = s_scaler.fit_transform(x)
# y_s_scaler = s_scaler.fit_transform(y.reshape(-1,1))
# y_s_scaler = y_s_scaler.reshape(1,-1).flatten()

### Normalizer Scaler ###
# normal = preprocessing.Normalizer()
# x_norm_scaler = normal.fit_transform(x)


### Minmax Scaler: Default range (0,1) ###
minmax_scaler = MinMaxScaler()
x_minmax_scaler = minmax_scaler.fit_transform(x)
# y_minmax_scaler = minmax_scaler.fit_transform(y.reshape(-1,1))
# print(y_minmax_scaler)
# y_minmax_scaler = y_minmax_scaler.reshape(1,-1).flatten()

# print(x_minmax_scaler)
# print(y_minmax_scaler.flatten())
# print(type(y_minmax_scaler))
# print(y_minmax_scaler.shape)


# dataset_scaled = s_scaler.fit_transform(dataset)
# names = ['Sex','Length','Diameter','Height','Whole Weight','Shucked Weight',
#          'Viscera Weight','Shell Weight','Age']
# dataset_scaled = pd.DataFrame(dataset_scaled,columns= names )
# dataset[['Sex','Length','Diameter','Height','Whole Weight','Shucked Weight',
#          'Viscera Weight','Shell Weight','Age']] = dataset_scaled[['Sex','Length','Diameter','Height','Whole Weight','Shucked Weight',
#          'Viscera Weight','Shell Weight','Age']]
# print(dataset.describe())



### PCA ###

# pca = PCA(n_components=4)
# x_train_pca = pca.fit_transform(x_minmax_scaler)
# print((abs(pca.components_)).astype(int))
# print(np.around(abs(pca.components_),decimals=3))
# print("Explained Variance:\n")
# print(pca.explained_variance_ratio_)
# print(pca.coef_)



### Visualization for individual explained variance
### and Cumulative explained variance
# plt.bar(range(1,9),pca.explained_variance_ratio_,alpha=0.5,align='center',
#         label ='individual explained variance')
# plt.step(range(1,9),np.cumsum(pca.explained_variance_ratio_),where='mid',
#          label = 'Cumulative explained variance')
# plt.xlabel('Principal components')
# plt.ylabel('Explained variance ratio')
#
# for a,b in zip(range(1,9),pca.explained_variance_ratio_):
#     plt.text(a,b+0.05,'%.3f'%b,ha='center',va='bottom',fontsize=7)
# plt.legend(loc='center')
# plt.tight_layout()
# plt.show()


cv = ShuffleSplit(n_splits=5,test_size=0.3,random_state=4)
x_train, x_test, y_train, y_test = train_test_split(x_minmax_scaler,y,test_size=0.3,random_state=4)

#
# print(x_train)
# print(y_train)


### Random Forest ###

rf = RandomForestRegressor(n_estimators=1380)
rf.fit(x_train,y_train)
y_pred_rf = rf.predict(x_test)
rms_test = (y_test-y_pred_rf).std()
print('Root Mean Square for test',rms_test)
print(cross_val_score(rf,x_train,y_train,cv=cv).mean())

g_range = range(len(y_pred_rf))
plt.plot(g_range,y_pred_rf,color='b',label='Predicted Value')
plt.plot(g_range,y_test,color='r',label='True Vlue')
plt.xlabel('')
plt.ylabel('')
plt.legend(loc='best')
plt.show()


# print(y_test)
# print(y_pred_rf)


### Paramitor Optimization by plot ###
# k_range = range(1350,1400,10)
# k_score=[]
# for k in k_range:
#     rf = RandomForestRegressor(n_estimators=k)
#     # y_pred = knn.predict(x_test)
#     cv = ShuffleSplit(n_splits=5,test_size=0.3,random_state=0)
#     # score = cross_val_score(rf,x_minmax_scaler,y,cv=cv,scoring='accuracy') # for classification
#     # print(score)
#     loss = -cross_val_score(rf, x_minmax_scaler, y, cv=cv, scoring='neg_root_mean_squared_error') # for regression
#     k_score.append(loss.mean())
#
# plt.plot(k_range,k_score)
# plt.xlabel('Value of K for Random Forest')
# plt.ylabel('Cross Validated Accuracy')
# plt.show()
# print(k_score)




###  Check coefficient and interception for the model ###
# print(model.coef_)
# print(model.intercept_)
# print(model.get_params())


### Look at importance for each feature###
# importances = rf.feature_importances_
# print(importances)
# indices = np.argsort(importances)[::-1]
# feat_labels = dataset.columns[0:]
# for f in range(x_train.shape[1]):
#     print("%2d) %-*s %f" % (f+1, 30, feat_labels[indices[f]], importances[indices[f]]))