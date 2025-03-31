import re
import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import csv

# Load the training and testing data
training = pd.read_csv('Training.csv')
testing = pd.read_csv('Testing.csv')

# Preparing the data
cols = training.columns
cols = cols[:-1]
x = training[cols]
y = training['prognosis']
y1 = y

# Reduced data for symptoms per disease
reduced_data = training.groupby(training['prognosis']).max()

# Encoding string labels to numeric values
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

# Split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# Gradient Boosting Classifier Model
gbc = GradientBoostingClassifier()
gbc.fit(x_train, y_train)

# Train score
train_score = gbc.score(x_train, y_train)
print(f"Gradient Boosting Classifier score: {train_score}")

# Function for secondary prediction
def sec_predict(symptoms_exp):
    df = pd.read_csv('Training.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)
    symptoms_dict = {symptom: index for index, symptom in enumerate(X.columns)}
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
        input_vector[symptoms_dict[item]] = 1
    return rf_clf.predict([input_vector])

# Dictionaries to store severity, description, and precaution data
severityDictionary = dict()
description_list = dict()
precautionDictionary = dict()
symptoms_dict = {}

# Populating the symptoms dictionary with symptom indices
for index, symptom in enumerate(x):
    symptoms_dict[symptom] = index

# Calculate the severity of conditions based on symptoms
def calc_condition(exp, days):
    sum = 0
    for item in exp:
        sum += severityDictionary[item]
    if (sum * days) / (len(exp) + 1) > 13:
        print("You should consult a doctor.")
    else:
        print("It may not be severe, but you should take precautions.")

# Load symptom description data from CSV
def getDescription():
    global description_list
    with open('symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            _description = {row[0]: row[1]}
            description_list.update(_description)

# Load symptom severity data from CSV
def getSeverityDict():
    global severityDictionary
    with open('Symptom_severity.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        try:
            for row in csv_reader:
                _diction = {row[0]: int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass

# Load symptom precautions data from CSV
def getprecautionDict():
    global precautionDictionary
    with open('symptom_precaution.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            _prec = {row[0]: [row[1], row[2], row[3], row[4]]}
            precautionDictionary.update(_prec)

# Function to collect user info and greet
def getInfo():
    print("-----------------------------------AI Medical ChatBot-----------------------------------")
    print("\nYour Name? \t\t\t\t", end="-> ")
    name = input("")
    print("Hello, ", name)

# Function to check for a pattern match in symptoms
def check_pattern(dis_list, inp):
    pred_list = []
    inp = inp.replace(' ', '_')
    patt = f"{inp}"
    regexp = re.compile(patt)
    pred_list = [item for item in dis_list if regexp.search(item)]
    if len(pred_list) > 0:
        return 1, pred_list
    else:
        return 0, []

# Function for disease prediction
def disease_prediction(symptoms_exp):
    input_vector = np.zeros(len(x.columns))
    for symptom in symptoms_exp:
        input_vector[x.columns.get_loc(symptom)] = 1
    prediction = gbc.predict([input_vector])
    return le.inverse_transform(prediction)[0]

# Main function
def main():
    getInfo()
    symptoms_exp = set()
    while True:
        symptom = input("Enter the symptom you are experiencing (or type 'done' if finished) \t\t-> ").strip().lower()
        if symptom == 'done':
            break
        elif symptom in symptoms_dict:
            symptoms_exp.add(symptom)
            days = input(f"For how many days have you experienced {symptom}? -> ")
        else:
            print("Enter a valid symptom.")
    if not symptoms_exp:
        print("No symptoms entered. Exiting.")
        return
    
    disease = disease_prediction(symptoms_exp)
    print(f"\033[1mYou may have {disease}\033[0m")  # Disease in bold
    print(f"{description_list.get(disease, 'No description available.')}")
    print(f"Severity: {severityDictionary.get(disease, 'Unknown')}")
    print("Take the following precautions:")
    for idx, precaution in enumerate(precautionDictionary.get(disease, []), 1):
        print(f"{idx}) {precaution}")

    second_prediction = sec_predict(symptoms_exp)
    if disease != second_prediction[0]:
        print(f"\033[1mAlternatively, you may have {second_prediction[0]}\033[0m")  # Alternative disease in bold
if __name__ == "__main__":
    getDescription()
    getSeverityDict()
    getprecautionDict()
    main()
