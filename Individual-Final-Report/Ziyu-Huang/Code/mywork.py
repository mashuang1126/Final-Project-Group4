#analyze sex column
sns.countplot(x='Sex',data=traindata)
plt.show()
print("\nSex count in percentage")
print(traindata.Sex.value_counts(normalize=True))
print("\nSex count in numbers")
print(traindata.Sex.value_counts())

# pairlplot
sns.pairplot(traindata[number_f])
plt.show()

#split training set and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)

#apply standard scaler
scaler = StandardScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print('StandardScaler_train',x_train)
print('StandardScaler_test',x_test)

#apply normalization
# scaler=Normalizer()
# scaler.fit(x_train)
#
# x_train=scaler.transform(x_train)
# x_test=scaler.transform(x_test)
# print('Normalization_train',x_train)
# print('Normalization_test',x_test)

#apply linear regression
lr=LinearRegression()
lr.fit(x_train,y_train)

#return prediction of linear regression
print("\nprint result of lr:")
lr_predictions = lr.predict(x_test)
# print(accuracy_score(y_test,lr_predictions))
print("score: ",lr.score(x_test,y_test))
print("mse: ",mean_squared_error(y_test,lr_predictions))
print("r^2: ",r2_score(y_test,lr_predictions))
# score=cross_val_score(lr,x_test,y_test,cv=10,scoring='accuracy')
# print("cross validation: ",score)

# show the plot of linear regression
plt.scatter(ID,y_test,color="red",label="training set")
plt.title("actual set")
plt.legend()
plt.show()
plt.scatter(ID,predictions,color="green",label="prediction")
plt.title("prediction of linear regression")
plt.legend()
plt.show()












