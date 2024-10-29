from flask import Flask, request, jsonify, render_template
import pickle

# Step 1: Load the trained model and vectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

# Step 2: Initialize the Flask app
app = Flask(__name__)

# Step 3: List of recent Indian movies with posters
movies = [
    {
        "title": "Leo",
        "poster": "/static/images/leo.jpg"
    },
    {
        "title": "Jawan",
        "poster": "/static/images/jawan.jpeg"
    },
    {
        "title": "Gadar 2",
        "poster": "/static/images/gaddar2.jpeg"
    },
    {
        "title": "Salaar",
        "poster": "/static/images/salaar.jpeg"
    }
]

# Step 4: Define the home route to render posters with a review form
@app.route('/')
def home():
    return render_template('index.html', movies=movies)

# Step 5: Handle prediction requests via AJAX
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    review = data['review']
    
    # Vectorize and predict sentiment
    review_vector = vectorizer.transform([review])
    prediction = model.predict(review_vector)[0]
    sentiment = 'Positive' if prediction == 1 else 'Negative'
    
    return jsonify({'sentiment': sentiment})

# Step 6: Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
