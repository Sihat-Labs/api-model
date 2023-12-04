from flask import Flask, request, jsonify
from tensorflow import keras
import os
import numpy as np
from werkzeug.datastructures import FileStorage
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import MinMaxScaler

import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

matplotlib.use('agg')

scaler = MinMaxScaler()

sihat = keras.models.load_model("latest.hdf5", compile=False)


def preprocess_files(files: list[FileStorage]):
    list = sorted(files)

    processed_images = []

    for img in range(len(list)):

        temp_image = nib.load(list[img]).get_fdata()
        temp_image = scaler.fit_transform(
            temp_image.reshape(-1, temp_image.shape[-1])).reshape(temp_image.shape)

        processed_images.append(temp_image)

    stacked_images = np.stack(processed_images, axis=3)
    stacked_images = stacked_images[56:184, 56:184, 13:141]

    # np.save('image/image.npy', stacked_images)
    return stacked_images


def predict(x, n_slice: int):
    img = preprocess_files(x)
    img_input = np.expand_dims(img, axis=0)
    prediction = sihat.predict(img_input)
    prediction_argmax = np.argmax(prediction, axis=4)[0, :, :, :]

    fig = Figure([12, 8])
    image = fig.add_subplot(2, 3, 1)
    image.set_title("Image")
    image.imshow(img[:, :, n_slice, 1], cmap='gray')

    mask = fig.add_subplot(2, 3, 2)
    mask.set_title("Result")
    mask.imshow(prediction_argmax[:, :, n_slice])

    pngImage = io.BytesIO()
    FigureCanvas(fig).print_png(pngImage)

    pngImageB64String = "data:image/png;base64,"
    pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')

    return pngImageB64String


app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/", methods=["POST"])
def index():
    if request.method == "POST":
        files = request.files.getlist('file')

        # Default to 25 if not provided
        n_slice = int(request.form.get('n_slice', 25))

        if not files:
            return "No files provided for upload"

        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        filesList = []
        try:
            for file in files:
                if file and file.filename.endswith('.nii'):
                    file_path = os.path.join(
                        app.config['UPLOAD_FOLDER'], file.filename)
                    file.save(file_path)
                    filesList.append(file_path)

            image_base64 = predict(filesList, n_slice)

            return jsonify({"image_base64": image_base64})
        except Exception as e:
            return jsonify({"error": str(e)})
    return "OK"


if __name__ == "__main__":
    app.run(debug=True)
