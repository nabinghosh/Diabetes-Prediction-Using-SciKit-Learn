from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# To run do this
# .\env\Scripts\activate
# pip install flask scikit-learn pandas numpy seaborn matplotlib pillow
# flask run

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get user input and convert to DataFrame
        user_data = pd.DataFrame({
            'Age': [request.form['age']],
            'BMI': [request.form['bmi']],
            'DiabetesPedigreeFunction': [request.form['dpf']]
        })

        # Load dataset and split into training and testing sets
        df = pd.read_csv('diabetes.csv')
        x = df.drop('Outcome', axis=1)
        y = df['Outcome']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

        # Train RandomForestClassifier and make prediction
        rf = RandomForestClassifier()
        rf.fit(x_train, y_train)
        user_result = rf.predict(user_data)

        # Create scatterplots and save as base64 strings
        fig_bmi = plt.figure()
        sns.scatterplot(x='Age', y='BMI', data=df, hue='Outcome', palette='rainbow')
        sns.scatterplot(x=user_data['Age'], y=user_data['BMI'], s=150, color='red' if user_result[0] == 1 else 'green')
        plt.xticks(np.arange(10,100,5))
        plt.yticks(np.arange(0,70,5))
        plt.title('0 - Healthy & 1 - Unhealthy')
        img_bmi = BytesIO()
        fig_bmi.savefig(img_bmi, format='png')
        img_bmi = base64.b64encode(img_bmi.getvalue()).decode()

        fig_dpf = plt.figure()
        sns.scatterplot(x='Age', y='DiabetesPedigreeFunction', data=df, hue='Outcome', palette='YlOrBr')
        sns.scatterplot(x=user_data['Age'], y=user_data['DiabetesPedigreeFunction'], s=150, color='red' if user_result[0] == 1 else 'green')
        plt.xticks(np.arange(10,100,5))
        plt.yticks(np.arange(0,3,0.2))
        plt.title('0 - Healthy & 1 - Unhealthy')
        img_dpf = BytesIO()
        fig_dpf.savefig(img_dpf, format='png')
        img_dpf = base64.b64encode(img_dpf.getvalue()).decode()

        # Calculate accuracy
        accuracy = accuracy_score(y_test, rf.predict(x_test)) * 100

        return render_template('result.html', bmi=img_bmi, dpf=img_dpf, result=user_result[0], accuracy=accuracy)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
    