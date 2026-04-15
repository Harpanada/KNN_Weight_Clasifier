import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify


app=Flask(__name__)

model= joblib.load("model_body/model.joblib")
encoder=joblib.load("model_body/encoder.joblib")
scaler=joblib.load("model_body/scaler.joblib")
category_val=joblib.load("model_body/category_val.joblib")
numeric_val=joblib.load("model_body/numeric_val.joblib")
feature_name=joblib.load("model_body/feature_names.pkl")


@app.route('/')
def home():
   return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
   user_input=request.get_json()
   print(f"Received from HTML: {user_input}")
  

   # hitung BMI
   height_m = float(user_input['Height']) / 100
   bmi = float(user_input['Weight']) / (height_m ** 2)

    # tambahkan ke input
   user_input['BMI'] = round(bmi, 2)
   user_input['Age']=float(user_input['Age'])
   
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
   return jsonify({
      'bodyweight':prediction[0],
      'status': 'success'
   })

if __name__ == '__main__':
    app.run(debug=True)

# user_input={
#    'Age': 17,
#    'Gender': 'Male',
#    'Height':184,
#    'Weight': 76,
# }
# prediction=predict(user_input)
# print(f'This Person has: {prediction}')