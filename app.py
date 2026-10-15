import os
import numpy as np
from PIL import Image
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename


# ============================================================
# CREATE FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# NEW TFLITE MODEL
MODEL_PATH = os.path.join(BASE_DIR, "model.tflite")

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ============================================================
# LOAD TFLITE MODEL
# ============================================================

print("Loading TFLite model...")

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)

# Allocate memory for model tensors
interpreter.allocate_tensors()

# Get model input and output information
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("TFLite model loaded successfully!")
print("Input details:", input_details)
print("Output details:", output_details)


# ============================================================
# CLASS NAMES
# IMPORTANT: Must match training class order
# ============================================================

CLASS_NAMES = [
    "bacterial_blight",
    "curl_virus",
    "fussarium_wilt",
    "healthy"
]


# ============================================================
# DISEASE DATABASE
# ============================================================

REMEDIES_DATABASE = {

    "bacterial_blight": {

        "names": {
            "english": "Bacterial Blight",
            "hindi": "जीवाणु झुलसा",
            "bengali": "ব্যাকটেরিয়াল ব্লাইট"
        },

        "remedies": [
            "Apply Copper Oxychloride (0.3%).",
            "Use Streptocycline (100 ppm) at recommended intervals.",
            "Remove and destroy infected crop debris.",
            "Avoid excessive irrigation and keep the field clean."
        ]
    },


    "curl_virus": {

        "names": {
            "english": "Leaf Curl Virus",
            "hindi": "पत्ता मरोड़ वायरस",
            "bengali": "পাতা কোঁকড়ানো ভাইরাস"
        },

        "remedies": [
            "Control whiteflies because they can spread Leaf Curl Virus.",
            "Use Neem Oil (1% to 2%) as recommended.",
            "Use yellow sticky traps to monitor whiteflies.",
            "Remove severely infected plants and nearby weed hosts."
        ]
    },


    "fussarium_wilt": {

        "names": {
            "english": "Fusarium Wilt",
            "hindi": "फ्यूसेरियम विल्ट",
            "bengali": "ফিউজেরিয়াম উইল্ট"
        },

        "remedies": [
            "Apply Trichoderma viride to the soil.",
            "Pseudomonas fluorescens may be used as a bio-agent.",
            "Practice crop rotation.",
            "Maintain proper drainage and avoid waterlogging."
        ]
    },


    "healthy": {

        "names": {
            "english": "Healthy Cotton Leaf",
            "hindi": "स्वस्थ कपास का पत्ता",
            "bengali": "সুস্থ তুলো পাতা"
        },

        "remedies": [
            "No disease detected.",
            "Continue regular crop monitoring.",
            "Maintain balanced irrigation.",
            "Use balanced NPK fertilization."
        ]
    }
}


# ============================================================
# ALLOWED IMAGE TYPES
# ============================================================

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg",
    "webp"
}


def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image_path):

    # Open image
    img = Image.open(image_path).convert("RGB")

    # Same size used during training
    img = img.resize((224, 224))

    # Convert to numpy array
    img_array = np.array(img, dtype=np.float32)

    # Normalize exactly like training
    img_array = img_array / 255.0

    # Add batch dimension
    # (224,224,3) -> (1,224,224,3)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# ============================================================
# TFLITE PREDICTION FUNCTION
# ============================================================

def predict_with_tflite(img_array):

    # Get input tensor information
    input_index = input_details[0]["index"]

    # Get output tensor information
    output_index = output_details[0]["index"]

    # Make sure datatype matches model input
    input_dtype = input_details[0]["dtype"]

    img_array = img_array.astype(input_dtype)

    # Give image to TFLite model
    interpreter.set_tensor(
        input_index,
        img_array
    )

    # Run prediction
    interpreter.invoke()

    # Get prediction result
    predictions = interpreter.get_tensor(
        output_index
    )

    return predictions


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template("index.html")


# ============================================================
# PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        print("\n==============================")
        print("Prediction request received")
        print("==============================")


        # ----------------------------------------------------
        # CHECK FILE
        # ----------------------------------------------------

        if "file" not in request.files:

            return jsonify({
                "success": False,
                "error": "No image received."
            }), 400


        file = request.files["file"]


        if file.filename == "":

            return jsonify({
                "success": False,
                "error": "Please select an image."
            }), 400


        # ----------------------------------------------------
        # CHECK FILE EXTENSION
        # ----------------------------------------------------

        if not allowed_file(file.filename):

            return jsonify({
                "success": False,
                "error":
                    "Only PNG, JPG, JPEG and WEBP images are allowed."
            }), 400


        # ----------------------------------------------------
        # SAVE IMAGE
        # ----------------------------------------------------

        filename = secure_filename(file.filename)

        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )

        file.save(filepath)

        print("Uploaded image:", filename)


        # ----------------------------------------------------
        # PREPROCESS IMAGE
        # ----------------------------------------------------

        img_array = preprocess_image(filepath)

        print("Image shape:", img_array.shape)


        # ----------------------------------------------------
        # TFLITE MODEL PREDICTION
        # ----------------------------------------------------

        predictions = predict_with_tflite(
            img_array
        )

        print("Raw prediction:")
        print(predictions)


        # ----------------------------------------------------
        # FIND PREDICTED CLASS
        # ----------------------------------------------------

        predicted_index = int(
            np.argmax(predictions[0])
        )


        if predicted_index >= len(CLASS_NAMES):

            return jsonify({
                "success": False,
                "error":
                    "Model output does not match CLASS_NAMES."
            }), 500


        predicted_class = CLASS_NAMES[
            predicted_index
        ]


        confidence = float(
            predictions[0][predicted_index]
        )


        print("Predicted index:", predicted_index)
        print("Predicted class:", predicted_class)
        print("Confidence:", confidence)


        # ----------------------------------------------------
        # GET DISEASE DETAILS
        # ----------------------------------------------------

        disease_data = REMEDIES_DATABASE[
            predicted_class
        ]


        # ----------------------------------------------------
        # RETURN RESULT TO index.html
        # ----------------------------------------------------

        return jsonify({

            "success": True,

            "predicted_class":
                predicted_class,

            "confidence":
                f"{confidence * 100:.2f}%",

            "image_url":
                "/static/uploads/" + filename,

            "names": {

                "english":
                    disease_data["names"]["english"],

                "hindi":
                    disease_data["names"]["hindi"],

                "bengali":
                    disease_data["names"]["bengali"]
            },

            "remedies":
                disease_data["remedies"]
        })


    except Exception as e:

        print("\n==============================")
        print("PREDICTION ERROR")
        print("==============================")

        print(str(e))


        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


# ============================================================
# RUN FLASK
# ============================================================

if __name__ == "__main__":

    print("\nStarting CottiShield AI...")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )