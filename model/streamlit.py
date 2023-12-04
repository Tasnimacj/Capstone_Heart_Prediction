# Imports 

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib

#######################################################################################################################################
### LAUNCHING THE APP ON THE LOCAL MACHINE
### 1. Save your *.py file (the file and the dataset should be in the same folder)
### 2. Open git bash (Windows) or Terminal (MAC) and navigate (cd) to the folder containing the *.py and *.csv files
### 3. Execute... streamlit run <name_of_file.py>
### 4. The app will launch in your browser. A 'Rerun' button will appear every time you SAVE an update in the *.py file



#######################################################################################################################################
### Create a title

#Let's write a title to be displayed in the app
st.title("Heart Disease Risk Flagger  🫀")

## You can also use markdown syntax using st.write("#...") 
st.write("#### Using ML for Coronary heart disease predictions")

### To position text and color, you can use html syntax
st.markdown("<h3 style='text-align: center; color: blue;'>Not actual medical advice</h3>", unsafe_allow_html=True)


#######################################################################################################################################
### DATA LOADING

#defining function to load data
def load_data(path, num_rows):
    df = pd.read_csv(path, nrows=num_rows,index_col=0)
    
    return df

# Load data
df = load_data("data_sample.csv",4)
df.rename(index={139919: 'Patient 1',11071 : 'Patient 2',78993: 'Patient 3',84662: 'Patient 4'},inplace = True)
df.drop(['preds'],axis=1,inplace=True)

# Display the dataframe in the app
st.dataframe(df)


####################################################################################################################################
#DATA
column=[
       'Female', 'GeneralHealth', 'PhysicalHealthDays', 'MentalHealthDays',
       'LastCheckupTime', 'PhysicalActivities', 'SleepHours', 'RemovedTeeth',
       'HadHeartAttack', 'HadStroke', 'HadAsthma', 'HadSkinCancer', 'HadCOPD',
       'HadDepressiveDisorder', 'HadKidneyDisease', 'HadArthritis',
       'HadDiabetes', 'DeafOrHardOfHearing', 'BlindOrVisionDifficulty',
       'DifficultyConcentrating', 'DifficultyWalking',
       'DifficultyDressingBathing', 'DifficultyErrands', 'SmokerStatus',
       'ECigaretteUsage', 'ChestScan', 'AgeCategory', 'HeightInMeters',
       'WeightInKilograms', 'BMI', 'AlcoholDrinkers', 'HIVTesting',
       'FluVaxLast12', 'PneumoVaxEver', 'TetanusLast10Tdap',
       'HighRiskLastYear', 'CovidPos', 'RaceEthnicity_Black only',
       'RaceEthnicity_Hispanic', 'RaceEthnicity_Multiracial',
       'RaceEthnicity_Other race only'
]
# max = pd.DataFrame(data=(0,5,30,0,1,1,10,0,0,0,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,11,1.75,68.04,22.15,0,0,1,1,1,0,1,0,0,0,0), index=None, columns=column)






#######################################################################################################################################
### MODEL 
# Subheader:
st.subheader("Lets see how the model performs")

df.drop(['actual'],axis=1,inplace=True)


max = df.iloc[0, :].to_numpy().reshape(1, -1)
claire = df.iloc[1, :].to_numpy().reshape(1, -1)
peter = df.iloc[2, :].to_numpy().reshape(1, -1)
jane = df.iloc[3, :].to_numpy().reshape(1, -1)

prediction = 2

# Load the model using joblib

model = joblib.load("fitted_RF_sm.pkl")

col1, col2, col3,col4 = st.columns([1,1,1,1])

with col1:
    if st.button('Patient 1'):
     prediction = model.predict(max)
with col2:
    if st.button('Patient 2'):
     prediction = model.predict(claire)
with col3:
    if st.button('Patient 3'):
     prediction = model.predict(peter)
with col4:
   if  st.button('Patient 4'):
    prediction = model.predict(jane)

url = "https://www.nhs.uk/conditions/angina/"
# based on prediction display something to user
if prediction == 1:
    st.write("## You are at risk of Heart Disease")
    st.write("### Here you can learn more:  ")
    st.write("[NHS](%s)" % url)
elif prediction == 0:
    st.write("## You are not at risk of Heart Disease")
    
else:
    pass

