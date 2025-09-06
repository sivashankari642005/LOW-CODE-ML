import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv('playtennis1.csv')
df=df.apply(LabelEncoder().fit_transform)

x=df.drop('PlayTennis',axis=1)
y=df['PlayTennis']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
model=CategoricalNB().fit(x_train,y_train)
y_pred=model.predict(x_test)

print("prediction",y_pred)
print("actual",y_test.values)
print("accuracy",accuracy_score(y_test,y_pred))
