#importing libaries
import pickle
from flask import Flask, render_template, request
import pandas as pd
import plotly.graph_objs as go 

app = Flask(__name__)

# Load the model pickle file
with open('final_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

@app.route("/", methods=['GET', 'POST'])

def home():
    genre_counts = pd.read_csv("genre_count.csv")
    # Create Plotly bar chart
    genre_bar_chart = go.Bar(x=genre_counts['genre'], y=genre_counts['count'])
    genre_bar_chart_layout = go.Layout(title='Genre Counts')
    genre_bar_chart_fig = go.Figure(data=[genre_bar_chart], layout=genre_bar_chart_layout)       
    # Convert Plotly figure to JSON for passing to HTML
    genre_bar_chart_json = genre_bar_chart_fig.to_json()

    if request.method == 'POST':
        # Get input data from the form  
        input_data = [request.form['input_data']]
        prediction = list(loaded_model.predict(input_data)) 
        prediction = prediction[0].tolist()
        return render_template("go.html", data=prediction,genre_count = genre_counts,genre_bar_chart_json=genre_bar_chart_json)
    else:
        return render_template("go.html",genre_count = genre_counts,genre_bar_chart_json=genre_bar_chart_json)
if __name__ == "__main__":
    app.run(debug=True)


