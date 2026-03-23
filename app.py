import os
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from PIL import Image
import requests
from gtts import gTTS

app = Flask(__name__)

# Upload folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static")

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# ---------------- MODELS ----------------

leaf_detector = tf.lite.Interpreter(model_path="models/leaf_or_not.tflite")
leaf_detector.allocate_tensors()

disease_model = tf.lite.Interpreter(model_path="models/leaf_disease.tflite")
disease_model.allocate_tensors()

leaf_input_details = leaf_detector.get_input_details()
leaf_output_details = leaf_detector.get_output_details()

disease_input_details = disease_model.get_input_details()
disease_output_details = disease_model.get_output_details()


# ---------------- DATASET CLASSES ----------------

CLASS_NAMES = [
"Apple - Black Rot","Apple - Apple Scab","Apple - Cedar Apple Rust",
"Apple - Healthy","Blueberry - Healthy",
"Cherry (including sour) - Powdery Mildew",
"Cherry (including sour) - Healthy","Corn - Common Rust",
"Corn (Maize) - Cercospora Leaf Spot / Gray Leaf Spot",
"Corn (Maize) - Northern Leaf Blight","Corn (Maize) - Healthy",
"Grape - Black Rot","Grape - Esca",
"Grape - Leaf Blight (Isariopsis Leaf Spot)","Grape - Healthy",
"Orange - Huanglongbing (Citrus Greening)",
"Peach - Bacterial Spot","Peach - Healthy",
"Pepper (Bell) - Healthy","Pepper (Bell) - Bacterial Spot",
"Potato - Late Blight","Potato - Early Blight","Potato - Healthy",
"Raspberry - Healthy","Soybean - Healthy",
"Squash - Powdery Mildew","Strawberry - Leaf Scorch",
"Strawberry - Healthy","Tomato - Mosaic Virus",
"Tomato - Bacterial Spot","Tomato - Early Blight",
"Tomato - Late Blight","Tomato - Leaf Mold",
"Tomato - Septoria Leaf Spot",
"Tomato - Spider Mites (Two-spotted Spider Mite)",
"Tomato - Target Spot",
"Tomato - Tomato Yellow Leaf Curl Virus",
"Tomato - Healthy"
]


LANGUAGE_CODES = {
"English":"en",
"Hindi":"hi",
"Gujarati":"gu"
}


# ---------------- IMAGE PREPROCESS ----------------

def preprocess_image(image, input_shape):
    image = image.resize((input_shape[2], input_shape[1]))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0).astype(np.float32)
    return image


# ---------------- SARVAM ADVISORY ----------------

def generate_advisory(prediction, language):

    api_key = os.getenv("SARVAM_API_KEY")

    if not api_key:
        return "⚠ SARVAM_API_KEY not set."

    try:

        url = "https://api.sarvam.ai/v1/chat/completions"

        prompt = f"""
Provide agricultural advisory for {prediction}.

IMPORTANT:
- Write bullet points in {language}
- BUT keep section titles EXACTLY in English

STRICT FORMAT:

Cause:
- point
- point
- point
- point
- point

Cure:
- point
- point
- point
- point
- point

Prevention:
- point
- point
- point
- point
- point
"""

        headers={
            "Authorization":f"Bearer {api_key}",
            "Content-Type":"application/json"
        }

        payload={
            "model":"sarvam-m",
            "messages":[{"role":"user","content":prompt}],
            "temperature":0.3
        }

        response=requests.post(url,headers=headers,json=payload,timeout=20)

        if response.status_code==200:

            result=response.json()
            advisory=result["choices"][0]["message"]["content"]

            for tag in ["<think>", "</think>", "<analysis>", "</analysis>"]:
                advisory=advisory.replace(tag,"")

            return advisory

        else:
            return f"API Error {response.status_code}"

    except Exception as e:
        return str(e)
    

def parse_advisory(text):

    cause=[]
    cure=[]
    prevention=[]

    section=None

    for line in text.split("\n"):

        line=line.strip()

        if "Cause" in line:
            section="cause"
            continue

        elif "Cure" in line:
            section="cure"
            continue

        elif "Prevention" in line:
            section="prevention"
            continue

        if line.startswith("-"):

            point=line.replace("-","").strip()

            if section=="cause":
                cause.append(point)

            elif section=="cure":
                cure.append(point)

            elif section=="prevention":
                prevention.append(point)

    return cause,cure,prevention


import uuid

def generate_voice(text, language):

    lang_code = LANGUAGE_CODES.get(language, "en")

    filename = f"advisory_{uuid.uuid4().hex}.mp3"

    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    tts = gTTS(text=text, lang=lang_code)

    tts.save(audio_path)

    return filename


# ---------------- ROUTE ----------------

@app.route('/', methods=['GET','POST'])
def index():

    prediction=None
    confidence=None
    advisory=None
    image_filename=None

    if request.method=='POST':

        language=request.form.get("language")
        file=request.files.get("image")
        manual_query=request.form.get("manual_query")


        # -------- MANUAL SEARCH --------

        if manual_query:

            prediction=manual_query
            confidence=100

            if "Healthy" in prediction:
                advisory = None
            else:
                advisory = generate_advisory(prediction, language)


        # -------- IMAGE DATASET SEARCH --------

        elif file:

            image_filename=file.filename
            image_path=os.path.join(app.config['UPLOAD_FOLDER'],image_filename)
            file.save(image_path)

            image=Image.open(image_path).convert("RGB")

            leaf_input=preprocess_image(image,leaf_input_details[0]['shape'])
            leaf_detector.set_tensor(leaf_input_details[0]['index'],leaf_input)
            leaf_detector.invoke()

            leaf_prediction=leaf_detector.get_tensor(leaf_output_details[0]['index'])

            if leaf_prediction[0][0] > 0.5:

                disease_input=preprocess_image(image,disease_input_details[0]['shape'])
                disease_model.set_tensor(disease_input_details[0]['index'],disease_input)
                disease_model.invoke()

                prediction_array=disease_model.get_tensor(disease_output_details[0]['index'])

                probs=prediction_array[0]
                index=np.argmax(probs)

                prediction=CLASS_NAMES[index]
                confidence=float(probs[index])*100

                advisory=generate_advisory(prediction, language)

            else:

                prediction="Not a leaf"
                confidence=0
                advisory="Please upload a clear leaf image."


    cause=[]
    cure=[]
    prevention=[]

    if advisory:
        cause,cure,prevention=parse_advisory(advisory)


    voice_file = None

    if advisory and ("Healthy" not in prediction):

        voice_text = ""

        for item in cause:
            voice_text += item + ". "

        for item in cure:
            voice_text += item + ". "

        for item in prevention:
            voice_text += item + ". "

        voice_file = generate_voice(voice_text, language) if voice_text.strip() else None    

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        cause=cause,
        cure=cure,
        prevention=prevention,
        image_filename=image_filename,
        voice_file=voice_file
    )


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)