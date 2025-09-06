from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data=load_iris()
x,y=data.data,data.target
names=data.target_names

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
model=KNeighborsClassifier(n_neighbors=2).fit(x_train,y_train)
pred=model.predict(x_test)

print("correct prediction")
for i,(p,a) in enumerate(zip(y_test,pred)):
    if p==a:
        print(f"sample{i}:prediction={names[p]},actual={names[a]}")

print("wrong prediction")
for i,(p,a) in enumerate(zip(y_test,pred)):
    if p!=a:
        print(f"sample{i}:prediction={names[p]},actual={names[a]}")
