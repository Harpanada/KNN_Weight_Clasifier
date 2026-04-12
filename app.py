import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score
from sklearn.preprocessing import StandardScaler,LabelEncoder


dataset=pd.read_csv("D:\Project\Python_ML\Classification\Data\Obesity Classification\Obesity Classification.csv")
dataset=dataset.drop(columns=['ID'])

print(dataset.info())

numeric= dataset.select_dtypes(include='number').columns
category= dataset.select_dtypes(include=['object','str']).columns


scaler=StandardScaler()
dataset[numeric]=scaler.fit_transform(dataset[numeric])

joblib.dump(scaler,'KNN/model_body/scaler.joblib')
joblib.dump(numeric,'KNN/model_body/numeric_val.joblib')


lb_encoder=LabelEncoder()
df=pd.DataFrame(dataset)


for col in category:
    df[col]=lb_encoder.fit_transform(df[col])

joblib.dump(lb_encoder,'KNN/model_body/encoder.joblib')
joblib.dump(category,'KNN/model_body/category_val.joblib')

X=df.drop(columns=['Label'])
y=df['Label']

x_train,x_test,y_train,y_test=train_test_split(X,y, test_size= 0.2 , random_state=1)

knn= KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2, weights="uniform")
model= knn.fit(x_train,y_train)

joblib.dump(list(X.columns), 'KNN/model_body/feature_names.pkl')
joblib.dump(model,'KNN/model_body/model.joblib')
y_pred= model.predict(x_test)


accuration=accuracy_score(y_test,y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
print("Accuration:", accuration)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)