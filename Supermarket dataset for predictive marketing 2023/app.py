from flask import Flask , render_template, request
import numpy as np 
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def Home():
    return render_template('index.html')

@app.route("/predict",methods=["POST"])
def predict():
    if request.method == "POST":
        user_id = float(request.form["user_id"])
        order_number = float(request.form["order_number"])
        order_dow = float(request.form["order_dow"])
        order_hour_of_day = float(request.form["order_hour_of_day"])
        days_since_prior_order = float(request.form["days_since_prior_order"])
        add_to_cart_order= float(request.form["add_to_cart_order"])
        reordered = float(request.form["reordered"])
        values = np.array([[ user_id,order_number, order_dow,order_hour_of_day,days_since_prior_order,add_to_cart_order,reordered]]).reshape(1,-1)
        prediction =model.predict(values)
        return render_template('index.html',prediction_text=f"the most preferabble item will be{prediction}")




if __name__ == "__main__":
    app.run(debug=True)