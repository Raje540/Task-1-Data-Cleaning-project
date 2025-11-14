from flask import Flask, request, render_template_string
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load("model.pkl")

html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Student Score Predictor</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: linear-gradient(135deg, #4f46e5, #6d28d9);
            height: 100vh;
            margin: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            color: white;
        }

        .container {
            background: white;
            color: #333;
            width: 380px;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.2);
            text-align: center;
        }

        h2 {
            margin-bottom: 10px;
            color: #4f46e5;
        }

        label {
            font-weight: bold;
            float: left;
            margin-bottom: 5px;
        }

        input {
            width: 100%;
            padding: 10px;
            margin-bottom: 18px;
            border-radius: 8px;
            border: 1px solid #aaa;
            font-size: 14px;
        }

        button {
            width: 100%;
            background: #4f46e5;
            color: white;
            padding: 12px;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            cursor: pointer;
            transition: 0.3s;
        }

        button:hover {
            background: #4338ca;
        }

        .result-box {
            margin-top: 20px;
            padding: 15px;
            background: #eef2ff;
            border-left: 5px solid #4f46e5;
            border-radius: 8px;
            color: #333;
        }

        .footer {
            font-size: 12px;
            margin-top: 10px;
            color: #ddd;
        }
    </style>
</head>
<body>

    <div class="container">
        <h2>🎓 Student Score Predictor</h2>
        <p>Enter Student details to predict Future Score</p>

        <form action="/" method="post">
            <label>Hours Studied:</label>
            <input type="number" step="0.1" name="hours" required>

            <label>Attendance (%):</label>
            <input type="number" name="attendance" required>

            <label>Previous Score:</label>
            <input type="number" name="previous" required>

            <button type="submit">Predict Score</button>
        </form>

        {% if result %}
            <div class="result-box">
                <h3>Predicted Final Score: <strong>{{ result }}</strong></h3>
            </div>
        {% endif %}

        <div class="footer">Project: Student Performance ML Model</div>
    </div>

</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
        hours = float(request.form['hours'])
        attendance = float(request.form['attendance'])
        previous = float(request.form['previous'])

        study_eff = hours * (previous / 100.0)

        arr = np.array([[hours, attendance, previous, study_eff]])
        pred = model.predict(arr)[0]

        result = round(pred, 2)

    return render_template_string(html, result=result)

if __name__ == "__main__":
    app.run(debug=True)
