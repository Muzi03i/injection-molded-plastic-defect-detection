import os
import sys
import io
import base64

import torch

from flask import (
    Flask,
    render_template,
    request,
    jsonify
)

from PIL import Image


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

APP_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.dirname(
    APP_DIR
)

SRC_PATH = os.path.join(
    PROJECT_ROOT,
    "src"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "efficientnet_defect_v2.pth"
)

if SRC_PATH not in sys.path:
    sys.path.append(
        SRC_PATH
    )


from model import get_model
from dataset import get_transforms
from gradcam import (
    GradCAM,
    annotate_defect_region
)


# --------------------------------------------------
# FLASK SETUP
# --------------------------------------------------

app = Flask(
    __name__
)

app.config[
    "MAX_CONTENT_LENGTH"
] = 10 * 1024 * 1024


# --------------------------------------------------
# MODEL CONFIGURATION
# --------------------------------------------------

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

CLASS_NAMES = [
    "Burn Mark",
    "Flash",
    "Short Shot",
    "Sink Mark"
]

# Final threshold selected using validation Macro F1.
THRESHOLD = 0.40


# --------------------------------------------------
# LOAD EFFICIENTNET-B0 V2
# --------------------------------------------------

model = get_model(
    num_classes=4,
    pretrained=False
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.to(
    DEVICE
)

model.eval()

transform = get_transforms(
    is_train=False
)

gradcam = GradCAM(
    model,
    model.features[-1]
)


# --------------------------------------------------
# IMAGE HELPERS
# --------------------------------------------------

def image_to_base64(
    image
):

    buffer = io.BytesIO()

    image.convert(
        "RGB"
    ).save(
        buffer,
        format="JPEG",
        quality=92
    )

    return base64.b64encode(
        buffer.getvalue()
    ).decode(
        "utf-8"
    )


def load_uploaded_image(
    uploaded_file
):

    image = Image.open(
        uploaded_file.stream
    )

    image.load()

    return image.convert(
        "RGB"
    )


# --------------------------------------------------
# MODEL PREDICTION
# --------------------------------------------------

def predict_image(
    image,
    generate_localization=True
):

    image = image.convert(
        "RGB"
    )

    input_tensor = (
        transform(
            image
        )
        .unsqueeze(0)
        .to(
            DEVICE
        )
    )

    if generate_localization:

        model.zero_grad(
            set_to_none=True
        )

        logits = model(
            input_tensor
        )

    else:

        with torch.no_grad():

            logits = model(
                input_tensor
            )

    probabilities = torch.sigmoid(
        logits
    )[0]

    probability_values = (
        probabilities
        .detach()
        .cpu()
        .numpy()
    )

    results = []

    for class_index, (
        class_name,
        probability
    ) in enumerate(
        zip(
            CLASS_NAMES,
            probability_values
        )
    ):

        probability = float(
            probability
        )

        detected = (
            probability
            >= THRESHOLD
        )

        cam_image = None

        if (
            detected
            and generate_localization
        ):

            heatmap = gradcam.generate(
                logits,
                class_index,
                image.size
            )

            annotated_image = (
                annotate_defect_region(
                    image,
                    heatmap,
                    class_name
                )
            )

            cam_image = image_to_base64(
                annotated_image
            )

        results.append({
            "class_name":
                class_name,

            "probability":
                probability,

            "percentage":
                probability * 100,

            "detected":
                bool(
                    detected
                ),

            "cam_image":
                cam_image
        })

    return results


# --------------------------------------------------
# OVERALL RESULT
# --------------------------------------------------

def get_overall_result(
    results
):

    any_detected = any(
        result[
            "detected"
        ]
        for result
        in results
    )

    if any_detected:

        return (
            "DEFECT DETECTED"
        )

    return (
        "NO SELECTED DEFECT DETECTED"
    )


# --------------------------------------------------
# MAIN WEB PAGE
# --------------------------------------------------

@app.route(
    "/",
    methods=[
        "GET",
        "POST"
    ]
)
def index():

    results = None
    error = None
    image_preview = None
    overall_result = None

    if request.method == "POST":

        camera_file = (
            request.files.get(
                "camera_image"
            )
        )

        gallery_file = (
            request.files.get(
                "gallery_image"
            )
        )

        uploaded_file = None


        if (
            camera_file is not None
            and camera_file.filename != ""
        ):

            uploaded_file = (
                camera_file
            )


        elif (
            gallery_file is not None
            and gallery_file.filename != ""
        ):

            uploaded_file = (
                gallery_file
            )


        if uploaded_file is None:

            error = (
                "Please take a photo or "
                "upload an image."
            )


        else:

            try:

                image = (
                    load_uploaded_image(
                        uploaded_file
                    )
                )

                results = predict_image(
                    image,
                    generate_localization=True
                )

                image_preview = (
                    image_to_base64(
                        image
                    )
                )

                overall_result = (
                    get_overall_result(
                        results
                    )
                )

            except Exception as exc:

                print(
                    "Image processing error:",
                    exc
                )

                error = (
                    "The selected image could "
                    "not be processed. Please "
                    "use a valid JPG, JPEG or "
                    "PNG image."
                )


    return render_template(
        "index.html",
        results=results,
        error=error,
        threshold=THRESHOLD,
        image_preview=image_preview,
        overall_result=overall_result
    )


# --------------------------------------------------
# FLASK REST API
# --------------------------------------------------

@app.route(
    "/api/predict",
    methods=[
        "POST"
    ]
)
def api_predict():

    uploaded_file = (
        request.files.get(
            "image"
        )
    )


    if uploaded_file is None:

        return jsonify({
            "error":
                "No image was provided.",

            "expected_field":
                "image"
        }), 400


    if uploaded_file.filename == "":

        return jsonify({
            "error":
                "No image was selected."
        }), 400


    try:

        image = (
            load_uploaded_image(
                uploaded_file
            )
        )

        results = predict_image(
            image,
            generate_localization=False
        )

        predictions = {}

        detected_classes = []


        for result in results:

            class_name = (
                result[
                    "class_name"
                ]
            )

            probability = float(
                result[
                    "probability"
                ]
            )

            detected = bool(
                result[
                    "detected"
                ]
            )


            predictions[
                class_name
            ] = {

                "probability":
                    round(
                        probability,
                        4
                    ),

                "percentage":
                    round(
                        probability * 100,
                        2
                    ),

                "detected":
                    detected
            }


            if detected:

                detected_classes.append(
                    class_name
                )


        return jsonify({

            "model":
                "EfficientNet-B0 V2",

            "task":
                "multi-label defect classification",

            "classes":
                CLASS_NAMES,

            "threshold":
                THRESHOLD,

            "defect_detected":
                len(
                    detected_classes
                ) > 0,

            "detected_classes":
                detected_classes,

            "predictions":
                predictions
        })


    except Exception as exc:

        print(
            "API prediction error:",
            exc
        )

        return jsonify({
            "error":
                "The uploaded image could "
                "not be processed."
        }), 400


# --------------------------------------------------
# API INFORMATION
# --------------------------------------------------

@app.route(
    "/api",
    methods=[
        "GET"
    ]
)
def api_info():

    return jsonify({

        "name":
            "PlasticVision AI API",

        "model":
            "EfficientNet-B0 V2",

        "endpoint":
            "/api/predict",

        "method":
            "POST",

        "image_field":
            "image",

        "classes":
            CLASS_NAMES,

        "threshold":
            THRESHOLD
    })


# --------------------------------------------------
# LARGE FILE ERROR
# --------------------------------------------------

@app.errorhandler(
    413
)
def file_too_large(
    error
):

    return render_template(
        "index.html",
        results=None,
        error=(
            "The selected image is too large. "
            "Maximum upload size is 10 MB."
        ),
        threshold=THRESHOLD,
        image_preview=None,
        overall_result=None
    ), 413


# --------------------------------------------------
# RUN APPLICATION
# --------------------------------------------------

if __name__ == "__main__":

    print(
        "=" * 60
    )

    print(
        "PLASTICVISION AI"
    )

    print(
        "=" * 60
    )

    print(
        f"Running on device: "
        f"{DEVICE}"
    )

    print(
        f"Model: "
        f"{MODEL_PATH}"
    )

    print(
        f"Decision threshold: "
        f"{THRESHOLD:.2f}"
    )

    print(
        "Web interface:"
    )

    print(
        "  http://127.0.0.1:5000"
    )

    print(
        "API endpoint:"
    )

    print(
        "  POST /api/predict"
    )


    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )