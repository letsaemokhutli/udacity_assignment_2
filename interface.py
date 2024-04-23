import pickle
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

# Load the model
with open('model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Define a function for preprocessing input data
def preprocess_input(data):
    # Perform any necessary preprocessing here
    # Example: Tokenization, vectorization, scaling, etc.
    return data

@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get input data from the form
        input_data = [request.form['input_data']]
        print(input_data)
        # Preprocess the input data
        #preprocessed_data = preprocess_input(input_data)
        # Make prediction
        prediction = list(loaded_model.predict(input_data))  # Assuming your model expects a list of inputs
        prediction = prediction[0].tolist()
        print(prediction)
        return render_template("index.html", data=prediction)
    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)


