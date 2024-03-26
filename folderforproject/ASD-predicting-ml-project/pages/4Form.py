import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load dataset
st.title(":bookmark_tabs: :blue[Autism data assessment]")
st.write("---")
st.write("Fill the form below to check if your child is suffering from ASD")
autism_dataset = pd.read_csv('asd_data_csv.csv')

# Preprocessing
X = autism_dataset.drop(columns='Outcome', axis=1)
Y = autism_dataset['Outcome']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, stratify=Y, random_state=2)

# Model training
classifier = SVC(kernel='linear', C=1.0)  # Adjust regularization parameter C
classifier.fit(X_train, Y_train)

# User input
input_data = {}
form_layout = {
    "Social Responsiveness": list(range(11)),
    "Age": list(range(19)),
    "Speech Delay": ["No", "Yes"],
    "Learning disorder": ["No", "Yes"],
    "Genetic disorders": ["No", "Yes"],
    "Depression": ["No", "Yes"],
    "Intellectual disability": ["No", "Yes"],
    "Social/Behavioural issues": ["No", "Yes"],
    "Anxiety disorder": ["No", "Yes"],
    "Suffers from Jaundice": ["No", "Yes"],
    "Family member history with ASD": ["No", "Yes"],
    "Gender": ["Female", "Male"]
}

for feature, options in form_layout.items():
    if feature == "Gender":
        input_data[feature] = st.selectbox(feature, options)
    else:
        input_data[feature] = st.selectbox(feature, options)

# Convert categorical inputs to numerical values
for feature in ["Speech Delay", "Learning disorder", "Genetic disorders", "Depression",
                "Intellectual disability", "Social/Behavioural issues", "Anxiety disorder",
                "Suffers from Jaundice", "Family member history with ASD"]:
    input_data[feature] = 1 if input_data[feature] == "Yes" else 0

# Convert gender to numerical value
input_data["Gender"] = 1 if input_data["Gender"] == "Male" else 0

# Make prediction
input_features = np.array([list(input_data.values())])
prediction = classifier.predict(input_features)

# Display results
with st.expander("Analyze provided data"):
    st.subheader("Results:")
    if prediction[0] == 0:
        st.warning('The person is with Autism spectrum disorder')  # Changed color to warning (yellow)
    else:
        st.info('The person is not with Autism spectrum disorder')  # Changed color to info (blue)

# Debugging Inputs
st.write("---")
st.subheader("Debugging Information:")
st.write("User Input Data:")
st.write(input_data)

# Debugging Input Features
st.write("Input Features for Prediction:")
st.write(input_features)

# Debugging Model
st.write("Trained Model:")
st.write(classifier)

# Debugging Predictions
st.write("Prediction Result:")
st.write(prediction)