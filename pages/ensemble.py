import streamlit as st
from sklearn.metrics import confusion_matrix
import joblib
import pandas as pd
import numpy as np

#x = st.slider("Select a value")
#st.write(x, "squared is", x * x)

knn_path = "pages/ensemble/knn_model.sav"
svm_path = "pages/ensemble/svm_model.sav"
d3_path = "pages/ensemble/D3_model.sav"

knn = joblib.load(knn_path)
svm = joblib.load(svm_path)
d3 = joblib.load(d3_path)

weight = np.array([0.73,0.86,0.9])
converted_Ans = np.array(["Insufficient_Weight" , "Normal_Weight", "Obesity_Type_I", "Obesity_Type_II" , "Obesity_Type_III" , "Overweight_Level_I" , "Overweight_Level_II"  ])

column_min_max = {
    'Gender': {'min': 0, 'max': 1},
    'Age': {'min': 0, 'max': 100},
    'Height': {'min': 0, 'max': 2.5},
    'Weight': {'min': 0, 'max': 200},
    'family_history': {'min': 0, 'max': 1},
    'FAVC': {'min': 0, 'max': 1},
    'FCVC': {'min': 0, 'max': 3},
    'NCP': {'min': 0, 'max': 4},
    'CAEC': {'min': 0, 'max': 3},
    'SMOKE': {'min': 0, 'max': 1},
    'CH2O': {'min': 1, 'max': 3},
    'SCC': {'min': 0, 'max': 1},
    'FAF': {'min': 0, 'max': 3},
    'TUE': {'min': 0, 'max': 2},
    'CALC': {'min': 0, 'max': 3},
    'MTRANS': {'min': 0, 'max': 4},
}

gender_cat = ["Male","Female"]
family_cat = ["no","yes"]
favc_cat = ["no","yes"]
caec_cat = ["no","Sometimes","Frequently","Always"]
smoke_cat = ["no","yes"]
scc_cat = ["no","yes"]
calc_cat = ["no","Sometimes","Frequently","Always"]
mtrans_cat = ["Walking","Bike","Public_Transportation","Automobile","Motorbike"]

data = pd.read_csv("pages\ensemble\OB.csv")
random_row = data.sample(n=1).iloc[0]

st.title("Obesity Prediction Model")

st.write("Input Data")

Gender = st.selectbox('Select Gender', gender_cat,index=gender_cat.index(random_row['Gender']))
Age = st.number_input("Age", value=random_row['Age'],min_value=0.0, max_value=100.0)
Height = st.number_input("Height (in meters)", value=random_row['Height'],min_value=0.00, max_value=2.50)
Weight = st.number_input("Weight (in kg)", value=random_row['Weight'],min_value=0.00, max_value=200.00)
family_history = st.selectbox('Any Family History of Overweight', family_cat,index=family_cat.index(random_row['family_history']))
FAVC = st.selectbox('FAVC', favc_cat,index=favc_cat.index(random_row['FAVC']))
FCVC = st.number_input("FCVC", value=random_row['FCVC'],min_value=0.0, max_value=3.0)
NCP = st.number_input("NCP", value=random_row['NCP'],min_value=0.0, max_value=4.0)
CAEC = st.selectbox('CAEC', caec_cat,index=caec_cat.index(random_row['CAEC']))
SMOKE = st.selectbox('SMOKE', smoke_cat,index=smoke_cat.index(random_row['SMOKE']))
CH2O = st.number_input("CH2O", value=random_row['CH2O'],min_value=1.0, max_value=3.0)
SCC = st.selectbox('SCC', scc_cat,index=scc_cat.index(random_row['SCC']))
FAF = st.number_input("FAF", value=random_row['FAF'],min_value=0.0, max_value=3.0)
TUE = st.number_input("TUE", value=random_row['TUE'],min_value=0.0, max_value=2.0)
CALC = st.selectbox('CALC', calc_cat,index=calc_cat.index(random_row['CALC']))
MTRANS = st.selectbox('MTRANS', mtrans_cat,index=mtrans_cat.index(random_row['MTRANS']))

if "show_expander" not in st.session_state:
    st.session_state.show_expander = False


if st.button("Predict"):
    #Convert Data into Numerical
    df = np.array([[Gender, Age, Height, Weight, family_history, FAVC, FCVC, NCP, CAEC,
                SMOKE, CH2O, SCC, FAF, TUE, CALC, MTRANS]], dtype=object)

    # Convert categorical features to numerical encoding safely
    df[0][0] = gender_cat.index(df[0][0]) if df[0][0] in gender_cat else 0  # Encode Gender
    df[0][4] = family_cat.index(df[0][4]) if df[0][4] in family_cat else 0  # Encode Family History
    df[0][5] = favc_cat.index(df[0][5]) if df[0][5] in favc_cat else 0  # Encode FAVC
    df[0][8] = caec_cat.index(df[0][8]) if df[0][8] in caec_cat else 0  # Encode CAEC
    df[0][9] = smoke_cat.index(df[0][9]) if df[0][9] in smoke_cat else 0  # Encode SMOKE
    df[0][11] = scc_cat.index(df[0][11]) if df[0][11] in scc_cat else 0  # Encode SCC
    df[0][14] = calc_cat.index(df[0][14]) if df[0][14] in calc_cat else 0  # Encode CALC
    df[0][15] = mtrans_cat.index(df[0][15]) if df[0][15] in mtrans_cat else 0  # Encode MTRANS

    # Normalize numerical features
    def normalize(value, column):
        return (float(value) - column_min_max[column]['min']) / (column_min_max[column]['max'] - column_min_max[column]['min'])

    df[0][1] = normalize(df[0][1], 'Age')  # Normalize Age
    df[0][2] = normalize(df[0][2], 'Height')  # Normalize Height
    df[0][3] = normalize(df[0][3], 'Weight')  # Normalize Weight
    df[0][6] = normalize(df[0][6], 'FCVC')  # Normalize FCVC
    df[0][7] = normalize(df[0][7], 'NCP')  # Normalize NCP
    df[0][10] = normalize(df[0][10], 'CH2O')  # Normalize CH2O
    df[0][12] = normalize(df[0][12], 'FAF')  # Normalize FAF
    df[0][13] = normalize(df[0][13], 'TUE')  # Normalize TUE

    # Convert to float array before using it in the model
    df = df.astype(float)

    knnAns = knn.predict_proba(df)
    svmAns = svm.predict_proba(df)
    d3Ans = d3.predict_proba(df)

    all_probas = np.array([knnAns, svmAns, d3Ans])
    weighted_proba = np.average(all_probas, axis=0, weights=weight)

    final_prediction = np.argmax(weighted_proba, axis=1)
    final_prediction_labels = converted_Ans[final_prediction]
    print(final_prediction_labels[0])

    st.markdown("# Prediction : {}".format(final_prediction_labels[0]))


