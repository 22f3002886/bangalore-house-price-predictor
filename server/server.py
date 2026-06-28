"""Flask server for the Bangalore House Price Predictor."""
from flask import Flask, request, jsonify, render_template
import util

app = Flask(__name__)
util.load_artifacts()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/locations")
def locations():
    return jsonify({"locations": util.get_location_names()})


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    try:
        sqft = float(data["total_sqft"])
        bhk = int(data["bhk"])
        bath = int(data["bath"])
        location = data.get("location", "other")
    except (KeyError, ValueError, TypeError):
        return jsonify({"error": "Please provide valid square feet, BHK, and bathrooms."}), 400

    if sqft <= 0 or bhk <= 0 or bath <= 0:
        return jsonify({"error": "Values must be greater than zero."}), 400
    if sqft / bhk < 300:
        return jsonify({"error": f"{sqft:.0f} sqft for {bhk} BHK is unrealistically small (need ~300+ sqft per room)."}), 400

    price = util.predict_price(location, sqft, bath, bhk)
    return jsonify({
        "estimated_price_lakh": price,
        "estimated_price_inr": round(price * 100000),
        "per_sqft": round(price * 100000 / sqft),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
