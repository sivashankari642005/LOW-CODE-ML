import pandas as pd
from sklearn.tree import DecisionTreeClassifier,plot_tree
import matplotlib.pyplot as plt

df=pd.read_csv('playtennis.csv').dropna()
x = pd.get_dummies(df.drop('PlayTennis', axis=1))
y = df['PlayTennis'].map({'Yes': 1, 'No': 0})

model=DecisionTreeClassifier(criterion='entropy').fit(x,y)

plt.figure(figsize=(10,6))
plot_tree(model,feature_names=x.columns,class_names=['No','Yes'],filled=True)
plt.show()
