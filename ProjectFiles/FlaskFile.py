from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load the trained TF-IDF vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

# Load the trained Random Forest classifier
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the column names of the feature matrix
with open('column_names.pkl', 'rb') as f:
    column_names = pickle.load(f)

@app.route('/')
def home():
    return render_template('input.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        Review = request.form['Review']
        Overall_Rating = float(request.form['Overall_Rating'])
        Cabin_Staff_Service = float(request.form['Cabin_Staff_Service'])
        Food_Beverages = float(request.form['Food_Beverages'])

        # Create a DataFrame with the input data
        input_df = pd.DataFrame({
            'Review': [Review],
            'Overall_Rating': [Overall_Rating],
            'Cabin_Staff_Service': [Cabin_Staff_Service],
            'Food_Beverages': [Food_Beverages]
        })

        # Preprocess the input data
        rev_tfidf = tfidf_vectorizer.transform(input_df["Review"])
        x_input = pd.concat([pd.DataFrame(rev_tfidf.toarray()), input_df.drop(columns=["Review"])], axis=1)

        # After preprocessing your input data, before making predictions, convert feature names to strings
        x_input.columns = x_input.columns.astype(str)

        # Predict "Recommended" for the input data
        predictions = model.predict(x_input).tolist()

        # Predict "Recommended" for the input data
        predictions = model.predict(x_input).tolist()

        return render_template('output.html', prediction_result=predictions[0])
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
