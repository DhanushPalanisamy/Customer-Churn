from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("model/churn_model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""

    if request.method == "POST":
        try:
            gender = int(request.form["gender"])
            age = int(request.form["age"])
            tenure = int(request.form["tenure"])
            monthly = float(request.form["monthly"])
            total = float(request.form["total"])

            input_data = np.array([[gender, age, tenure, monthly, total]])
            result = model.predict(input_data)[0]

            if result == 1:
                prediction = "❌ Customer Will Churn"
            else:
                prediction = "✅ Customer Will Stay"

        except Exception as e:
            prediction = "⚠️ Error in input values"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
