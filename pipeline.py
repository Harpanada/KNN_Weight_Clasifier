import joblib
import pandas as pd
model= joblib.load("KNN/model_body/model.joblib")
encoder=joblib.load("KNN/model_body/encoder.joblib")
scaler=joblib.load("KNN/model_body/scaler.joblib")
category_val=joblib.load("KNN/model_body/category_val.joblib")
numeric_val=joblib.load("KNN/model_body/numeric_val.joblib")
feature_name=joblib.load("KNN/model_body/feature_names.pkl")


def predict(user_input):
   # hitung BMI
   height_m = user_input['Height'] / 100
   bmi = user_input['Weight'] / (height_m ** 2)

    # tambahkan ke input
   user_input['BMI'] = round(bmi, 2)
  
   input_df= pd.DataFrame([user_input])

   for col in category_val:
      if col in user_input:
         try:
            input_df[col]=encoder.transform(input_df[col].astype[str])
         except:
            input_df[col]=0

   input_df[numeric_val]=scaler.transform(input_df[numeric_val])
   encoded_prediction= model.predict(input_df[feature_name])
   
   prediction=encoder.inverse_transform(encoded_prediction)
   return prediction[0]


user_input={
   'Age': 17,
   'Gender': 'Male',
   'Height':184,
   'Weight': 76,
}
prediction=predict(user_input)
print(f'This Person has: {prediction}')