from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.vgg16 import preprocess_input
import os
from tensorflow.keras.preprocessing import image
import random

app = Flask(__name__)
model = load_model("model.h5")
target_img = os.path.join(os.getcwd(), "static/images")


@app.route("/")
def index_view():
    return render_template("index.html")


ALLOWED_EXT = set(["jpg", "jpeg", "png"])


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1] in ALLOWED_EXT


def read_image(filename):
    img = load_img(filename, target_size=(250, 167))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        file = request.files["file"]
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join("static/images", filename)
            file.save(file_path)
            img = read_image(file_path)
            class_prediction = model.predict(img)
            classes_x = np.argmax(class_prediction, axis=1)
            if classes_x == 0:
                damage = "Crack"
                price = random.randint(20000, 80000)
            elif classes_x == 1:
                damage = "Dent"
                price = random.randint(10000, 200000)
            elif classes_x == 2:
                damage = "Glass shatter"
                price = random.randint(20000, 150000)
            elif classes_x == 3:
                damage = "Lamp broken"
                price = random.randint(15000, 120000)
            elif classes_x == 4:
                damage = "Scratch"
                price = random.randint(5000, 40000)
            elif classes_x == 5:
                damage = "Tire flat"
                price = random.randint(4000, 70000)
            else:
                damage = "No damage"
                price = 0
            return render_template(
                "predict.html",
                damage=damage,
                repairs=price,
                prob=class_prediction,
                user_image=file_path,
            )
        else:
            return "Unable to read the file. Please check file extension"


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, port=8000)
