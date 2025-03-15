import streamlit as st

st.title("Obesity Prediction Model")

st.write("This model categorizes the prediction into 7 categories")
st.write("1.Insufficient Weight")
st.write("2.Normal Weight")
st.write("3.Obesity Type I")
st.write("4.Obesity Type II")
st.write("5.Obesity Type III")
st.write("6.Overweight Level I")
st.write("7.Overweight Level II")

st.write("By using 16 features as an input which consists of :")
st.write("Gender,Age,Height,Weight,Family History(of obesity or overweight),FAVC,FCVC,NCP,CAEC,SMOKE,CH2O,SCC,FAF,TUE,CALC,MTRANS")

st.write("Which I definitely doesn't have a clue of what it's about")
st.write("Except for the ones that are obvious like age or height")


st.title("Technical Details")
st.write("This is an ensemble model consists of 3 different models which are KNN,SVM and D3(Decision Tree).")
st.write("KNN is a model that uses the number(K) of the input neighbors to classify the categories")
st.write("Which I set the Hyperparameter(K) to be 1 as it gives me the most accuracy of 73.79% among the single digit numbers")
st.write("")
st.write("SVM is a model that try to use a mathematical function to seperate a the data.")
st.write("The model will try to minimize the difference of distance between the line and each categories.")
st.write("This model have 2 hyperparameters 'Degree' and 'C'. Degree is use to determine which polynomial function can be used and C is use to allow some error or misclassification on the dataset to allow more flexibility and prevent overfitting.")
st.write("And so, I use 5 and 10 as the value of the hyperparameter respectively which give me the result of 86.82% accuracy.")
st.write("")
st.write("D3 is a simple model that abuses the 'ifs and else' concept.")
st.write("As the inputs are entering, it goes down the staircase of 'ifs and else' until it reaches the bottom which is where the answer,or in this case, the categories that the model predicts.")
st.write("I didn't use any hyperparameter for this model and it gives me the result of 90.51% accuracy. Which is the highest of all three.")
st.write("")
st.write("For the final prediction, I use weight voting to determine the final answer.")
st.write("Each model have a different weight based on their accuracy which is 0.73,0.86 and 0.9")
st.write("And thus, the model will reach its final prediction which is the maximum score of all the possible answer.")
st.write("")
st.write("As for the dataset, I download it from kaggle. Unfortunately, the requirements implies that the dataset cannot already be cleanse(as I understand), so I manually inserts missing and/or null values into the dataset")
st.write("Which I, of course, cleanse the data before using it to train the model")
st.write("Dataset Source : https://www.kaggle.com/datasets/ruchikakumbhar/obesity-prediction")