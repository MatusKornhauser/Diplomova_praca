import json
import shutil

import PIL
from docutils.nodes import caption
from exceptiongroup import catch
from flask import Flask, render_template, request, jsonify, url_for, redirect, session
import os
import cv2
import torch
from PIL import Image
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib import patches
import matplotlib

# from interaktivnaos import timeline

matplotlib.use('Agg') # Pre použitie Matplotlib bez GUI
from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig, CLIPModel, CLIPProcessor
from werkzeug.utils import secure_filename

# Flask setup
app = Flask(__name__)
app.secret_key = "secret_key_for_session"  # Kľúč na šifrovanie session


UPLOAD_FOLDER = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'mov', 'avi'}
# VQA SETUP
from transformers import AutoConfig, AutoModelForCausalLM
import torch
from transformers import AutoProcessor

# Determine if a GPU is available and set the device accordingly
# device_VQA = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# config_VQA = AutoConfig.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
#
# model_VQA  = AutoModelForCausalLM.from_pretrained(
#     "microsoft/Florence-2-large",
#     config=config_VQA,
#     trust_remote_code=True
# ).to(device_VQA)
# # VivekChauhan06/Florence-2-FT-OK-VQA  "fauzail/Florence-2-VQA",
# processor_VQA  = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)

def predict(image, question):
    # Predpokladám, že 'processor' a 'model' sú už nainštalované a pripravené
    inputs = processor(text=[question], images=[image], return_tensors="pt", padding=True).to('cuda', torch.float16)
    outputs = model.generate(**inputs, max_length=200)  # Pôvodne býva okolo 20-30, zvýš na 100


    # Dekódovanie výsledku
    answer = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# Paligemma Model Setup
from transformers import AutoProcessor as PaliProcessor, PaliGemmaForConditionalGeneration

paligemma_model = PaliGemmaForConditionalGeneration.from_pretrained(
    "paligemma", torch_dtype=torch.bfloat16, device_map="cuda:0", revision="bfloat16"
).eval()
paligemma_processor = PaliProcessor.from_pretrained("paligemma")
paligemma_size = (224, 224)

#paligemma ft setup
paligemmaFT_model = PaliGemmaForConditionalGeneration.from_pretrained(
    "paligemma", torch_dtype=torch.bfloat16, device_map="cuda:0", revision="bfloat16"
).eval()
paligemmaFT_processor = PaliProcessor.from_pretrained("paligemma")
MODEL_PATH = "paligemma-weights.pth"  # Tvoj súbor
paligemmaFT_model.load_state_dict(torch.load("paligemma-weights.pth"), strict=False)

#florence2 ft setup
florence2modelFT= AutoModelForCausalLM.from_pretrained("allmodel", trust_remote_code=True, torch_dtype='auto').eval().cuda()
florence2processorFT = AutoProcessor.from_pretrained("allmodel", trust_remote_code=True)

model= AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base-ft", trust_remote_code=True, torch_dtype='auto').eval().cuda()
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base-ft", trust_remote_code=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# def process_image(image_path):
#     image = Image.open(image_path).convert("RGB")
#     width, height = image.size
#     crop_percentage = 0.8
#     new_width = int(width * crop_percentage)
#     new_height = int(height * crop_percentage)
#     start_x = (width - new_width) // 2
#     start_y = (height - new_height) // 2
#     cropped_image = image.crop((start_x, start_y, start_x + new_width, start_y + new_height))
#     cropped_image = cropped_image.resize(paligemma_size)
#     return np.array(cropped_image)


def process_image(image):
    # image = Image.open(image_path).convert("RGB")
    width, height = image.size
    crop_percentage = 0.5
    new_width = int(width * crop_percentage)
    new_height = int(height * crop_percentage)
    start_x = (width - new_width) // 2
    start_y = (height - new_height) // 2
    cropped_image = image.crop((start_x, start_y, start_x + new_width, start_y + new_height))
    resized_image = cropped_image.resize(paligemma_size)
    return np.array(resized_image)

def read_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image.resize(paligemma_size)
    return image

def process_image_video(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    width, height = image.size
    crop_percentage = 0.5
    new_width = int(width * crop_percentage)
    new_height = int(height * crop_percentage)
    start_x = (width - new_width) // 2
    start_y = (height - new_height) // 2
    cropped_image = image.crop((start_x, start_y, start_x + new_width, start_y + new_height))
    resized_image = cropped_image.resize((224, 224))
    image = np.array(resized_image)
    if image.shape[2] == 4:
        image = image[:, :, :3]
    return image

# def process_image_video(frame):
#     image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     width, height = image.size
#     crop_percentage = 0.7
#     new_width = int(width * crop_percentage)
#     new_height = int(height * crop_percentage)
#     start_x = (width - new_width) // 2
#     start_y = (height - new_height) // 2
#     cropped_image = image.crop((start_x, start_y, start_x + new_width, start_y + new_height))
#     resized_image = cropped_image.resize((224,224))
#     image = np.array(resized_image)
#     if image.shape[2] == 4:
#         image = image[:, :, :3]
#     return image


# Funkcia na vytvorenie obrázka časovej osi pre daný prompt
def visualize_answers(answers, prompt_index):
    timeline_length = len(answers)
    colors = ["green" if answer == "yes" else "red" for answer in answers]

    fig, ax = plt.subplots(figsize=(12, 2))
    ax.set_xlim(0, timeline_length)
    ax.set_ylim(0, 1)

    for i, color in enumerate(colors):
        ax.plot([i, i + 1], [0.5, 0.5], color=color, solid_capstyle="butt", lw=6)

    ax.axis("off")

    # Vytvoríme unikátny názov obrázka pre každý prompt
    img_filename = f"timeline_{prompt_index}.png"
    img_path = os.path.join(app.config['RESULT_FOLDER'], img_filename)
    plt.savefig(img_path, bbox_inches='tight')
    plt.close(fig)

    return img_filename  # Vraciame iba názov súboru


def parse_bbox_and_labels(detokenized_output: str):
    matches = re.finditer(
        '<loc(?P<y0>\d\d\d\d)><loc(?P<x0>\d\d\d\d)><loc(?P<y1>\d\d\d\d)><loc(?P<x1>\d\d\d\d)>'
        ' (?P<label>.+?)( ;|$)',
        detokenized_output,
    )
    labels, boxes = [], []
    fmt = lambda x: float(x) / 1024.0
    for m in matches:
        d = m.groupdict()
        boxes.append([fmt(d['y0']), fmt(d['x0']), fmt(d['y1']), fmt(d['x1'])])
        labels.append(d['label'])
    return np.array(boxes), np.array(labels)

#povodne
# def plot_bbox_florence(image, data):
#     fig, ax = plt.subplots()
#
#     # Display the image
#     ax.imshow(image)
#
#     # Plot each bounding box
#     for bbox, label in zip(data['bboxes'], data['labels']):
#         # Unpack the bounding box coordinates
#         x1, y1, x2, y2 = bbox
#         # Create a Rectangle patch
#         rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
#         # Add the rectangle to the Axes
#         ax.add_patch(rect)
#         # Annotate the label
#         plt.text(x1, y1, label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))
#
#
#         # Uložte obrázok do adresára "static/uploads"
#     output_path = os.path.join(app.config['UPLOAD_FOLDER'], "output_detect.png")
#     plt.axis('off')  # Skryte osy
#     plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
#     plt.close(fig)
#
#     # Vytvorte správnu relatívnu cestu
#     relative_path = os.path.relpath(output_path, start='static/')
#     relative_path = relative_path.replace("\\", "/")  # Oprava spätného lomítka na predné
#
#     print(f"Output path (corrected): {relative_path}")
#     return relative_path
app.static_folder = "static"
import uuid
def plot_bbox_florence(image, data):
    fig, ax = plt.subplots()
    ax.imshow(image)

    for bbox, label in zip(data['bboxes'], data['labels']):
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(x1, y1, label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))

    # Vytvorenie priečinka, ak neexistuje
    detect_folder = os.path.join(app.static_folder, 'detect')
    os.makedirs(detect_folder, exist_ok=True)

    # Generovanie unikátneho názvu súboru
    unique_filename = f"detect_{uuid.uuid4().hex}.png"
    output_path = os.path.join(detect_folder, unique_filename)

    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # Relatívna cesta pre frontend
    relative_path = f"/static/detect/{unique_filename}"

    print(relative_path)
    return relative_path

#
# def display_boxes(image, boxes, labels):
#     width, height = image.size
#     print(f"Original image size: {width}x{height}")
#
#     # Nastav veľkosť figúry podľa obrázka
#     fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
#     ax = fig.add_axes([0, 0, 1, 1])  # Žiadne okraje
#     ax.imshow(image)
#     ax.axis('off')
#
#     for i in range(boxes.shape[0]):
#         y0, x0, y1, x1 = boxes[i] * np.array([height, width, height, width])
#         rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=2, edgecolor='r', facecolor='none')
#         ax.add_patch(rect)
#         ax.text(x0, y0, labels[i], color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
#
#     # Uloženie do súboru
#     unique_filename = f"detect_{uuid.uuid4().hex}.png"
#     output_path = os.path.join('static', 'detect', unique_filename)
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#
#     plt.savefig(output_path, dpi=100, bbox_inches=None, pad_inches=0)
#     plt.close(fig)
#
#     relative_path = f"/static/detect/{unique_filename}"
#     print(f"✅ Uložené ako: {relative_path}")
#     return relative_path


def display_boxes(image, boxes, labels):
    width, height = paligemma_size
    print(width, height)
    fig, ax = plt.subplots()
    ax.imshow(image)

    # Nakreslíme bounding boxy a pridáme labely
    for i in range(boxes.shape[0]):
        y0, x0, y1, x1 = boxes[i] * np.array([height, width, height, width])  # Rozmerovanie boxov
        rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x0, y0, labels[i], color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    # Vytvoríme názov súboru s UUID
    unique_filename = f"detect_{uuid.uuid4().hex}.png"
    output_path = os.path.join('static', 'detect', unique_filename)

    # Uložíme obrázok do static/detect
    plt.axis("off")  # Skryjeme osi
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    relative_path = f"/static/detect/{unique_filename}"
    # Vytvárame relatívnu cestu
    # relative_path = os.path.relpath(output_path, start='static/')
    # relative_path = relative_path.replace("\\", "/")  # Oprava spätného lomítka na predné

    print(f"Output path (corrected): {relative_path}")
    return relative_path



# def display_boxes(image, boxes, labels):
#     width, height = paligemma_size
#     fig, ax = plt.subplots()
#     ax.imshow(image)
#     for i in range(boxes.shape[0]):
#         y0, x0, y1, x1 = boxes[i] * np.array([height, width, height, width])
#         rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=2, edgecolor='r', facecolor='none')
#         ax.add_patch(rect)
#         ax.text(x0, y0, labels[i], color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
#
#     # Uložte obrázok do adresára "static/uploads"
#     output_path = os.path.join(app.config['UPLOAD_FOLDER'], "output_detect.png")
#     plt.axis("off")
#     plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
#     plt.close(fig)
#
#     # Vytvorte správnu relatívnu cestu (použijeme '/' namiesto '\\')
#     relative_path = os.path.relpath(output_path, start='static/')
#     relative_path = relative_path.replace("\\", "/")  # Oprava spätného lomítka na predné
#
#     print(f"Output path (corrected): {relative_path}")
#     return relative_path

def run_example(image, task_prompt, text_input=None):
    prompt = task_prompt if text_input is None else task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt").to('cuda', torch.float16)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"].cuda(),
        pixel_values=inputs["pixel_values"].cuda(),
        max_new_tokens=2056,
        early_stopping=False,
        do_sample=False,
        repetition_penalty=1.2,  # Menej opakovaní slov
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )
    return parsed_answer

def run_exampleFT(image, task_prompt, text_input=None):
    prompt = task_prompt if text_input is None else task_prompt + text_input
    inputs = florence2processorFT(text=prompt, images=image, return_tensors="pt").to('cuda', torch.float16)
    generated_ids = florence2modelFT.generate(
        input_ids=inputs["input_ids"].cuda(),
        pixel_values=inputs["pixel_values"].cuda(),
        max_new_tokens=2056,
        early_stopping=False,
        do_sample=False,
        repetition_penalty=1.2,  # Menej opakovaní slov
        num_beams=3,
    )
    generated_text = florence2processorFT.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = florence2processorFT.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )
    return parsed_answer

# def process_video(video_path):
#     cap = cv2.VideoCapture(video_path)
#     frames = []
#     frame_index = 0  # Počítadlo framov
#
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break  # Koniec videa
#
#         if frame_index % 30 == 0:  # Spracovať iba každý 30. frame
#             processed_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#             # results = run_example(processed_frame, '<CAPTION_TO_PHRASE_GROUNDING>', text_input="Gun.")
#             # print("som v proccess video")
#             # print(results)
#             # print("------------------------------------------")
#             frames.append(processed_frame)  # Uloženie spracovaného framu
#
#         frame_index += 1  # Zvýšenie počítadla framov
#
#     cap.release()
#     return frames  # Vrátime iba vybrané framy

import cv2
from PIL import Image

import cv2
from PIL import Image


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    timestamps = []
    frame_indexes = []

    fps = cap.get(cv2.CAP_PROP_FPS)  # FPS videa
    print(f"FPS: {fps}")

    frame_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Koniec videa

        if frame_index % 30 == 0:  # Spracovať iba každý 30. frame
            # Presný časový výpočet z milisekúnd
            time_in_milliseconds = cap.get(cv2.CAP_PROP_POS_MSEC)  # Získa čas v milisekundách
            time_in_seconds = time_in_milliseconds / 1000.0  # Prevod na sekundy

            processed_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(processed_frame)
            timestamps.append(round(time_in_seconds, 3))  # Zaokrúhlenie na 3 desatinné miesta
            frame_indexes.append(frame_index)

        frame_index += 1

    cap.release()
    return frames, timestamps, frame_indexes


# def process_video(video_path):
#     cap = cv2.VideoCapture(video_path)
#     ret, frame = cap.read()
#     cap.release()
#     if ret:
#         return process_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     return None

def run_paligemma(image, prompt):
    model_inputs = paligemma_processor(text=prompt, images=image, return_tensors="pt").to("cuda")
    input_len = model_inputs["input_ids"].shape[-1]
    generation = paligemma_model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
    decoded = paligemma_processor.decode(generation[0][input_len:], skip_special_tokens=True)
    # print(f"Decoded: {decoded}")
    if "detect" in prompt.lower():
        boxes, labels = parse_bbox_and_labels(decoded)
        output_path = display_boxes(image, boxes, labels)
        return {"type": "image", "filename": output_path}
    else:
        return {"type": "text", "content": decoded}

def run_paligemmaFT_timeline(image, prompt):
    model_inputs = paligemmaFT_processor(text=prompt, images=image, return_tensors="pt").to("cuda")
    input_len = model_inputs["input_ids"].shape[-1]
    generation = paligemmaFT_model.generate(**model_inputs, max_new_tokens=100, do_sample=False,num_beams=3)
    decoded = paligemmaFT_processor.decode(generation[0][input_len:], skip_special_tokens=False)
    return  decoded

def run_paligemma_timeline(image, prompt):
    image = process_image(image)
    model_inputs = paligemma_processor(text=prompt, images=image, return_tensors="pt").to("cuda")
    input_len = model_inputs["input_ids"].shape[-1]
    generation = paligemma_model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
    decoded = paligemma_processor.decode(generation[0][input_len:], skip_special_tokens=True)

    return  decoded
def classify_text(text):
    from transformers import pipeline

    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", framework="pt")

    sequence = (text)
    candidate_labels = ["FALL", "GRAB", "GUN", "HIT", "KICK", "LYING DOWN", "RUN" ,"SIT", "STAND", "SNEAK", "STRUGGLE", "THROW", "WALK", "ARREST"]


    result = classifier(sequence, candidate_labels=candidate_labels)

    return result

def classify_frames(frames):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    labels = ["FALL", "GRAB", "GUN", "HIT", "KICK", "LYING DOWN", "RUN" ,"SIT", "STAND", "SNEAK", "STRUGGLE", "THROW", "WALK", "ARREST"]

    scores = {label: [] for label in labels}  # Slovník na ukladanie skóre

    for image in frames:
        # Spracovanie obrázka a textových labelov
        inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)

        # Predikcia
        with torch.no_grad():
            outputs = model(**inputs)

        # Výpočet pravdepodobností pre každý label
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)  # Normalizácia pravdepodobností

        # Uloženie výsledkov do slovníka
        for label, prob in zip(labels, probs[0]):
            scores[label].append(prob.item())

    return scores


def classify_frame(frame):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    labels = ["FALL", "GRAB", "GUN", "HIT", "KICK", "LYING DOWN", "RUN" ,"SIT", "STAND", "SNEAK", "STRUGGLE", "THROW", "WALK", "ARREST"]

    scores = {label: [] for label in labels}  # Slovník na ukladanie skóre


    # Spracovanie obrázka a textových labelov
    inputs = processor(text=labels, images=frame, return_tensors="pt", padding=True)

    # Predikcia
    with torch.no_grad():
        outputs = model(**inputs)

    # Výpočet pravdepodobností pre každý label
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)  # Normalizácia pravdepodobností

    # Uloženie výsledkov do slovníka
    for label, prob in zip(labels, probs[0]):
        scores[label].append(prob.item())

    return scores



def get_final_label(scores):
    # Spočítať priemer pravdepodobností pre každý label
    avg_scores = {label: np.mean(values) for label, values in scores.items() if values}

    # Vybrať najpravdepodobnejší label
    best_label = max(avg_scores, key=avg_scores.get)

    return best_label, avg_scores

# def run_florence(image, prompt):
#     inputs = florence_processor(text=prompt, images=image, return_tensors="pt").to("cuda", torch.float16)
#     generated_ids = florence_model.generate(
#         input_ids=inputs["input_ids"].cuda(),
#         pixel_values=inputs["pixel_values"].cuda(),
#         max_new_tokens=1024,
#         num_beams=3,
#     )
#     generated_text = florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
#     return florence_processor.post_process_generation(generated_text, task="<VQA>", image_size=image.size)

def run_florence(image, prompt):
    try:
        # Príprava vstupov pomocou procesora
        inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda", torch.float16)

        # Generovanie odpovede modelom
        generated_ids = model.generate(
            input_ids=inputs["input_ids"].cuda(),
            pixel_values=inputs["pixel_values"].cuda(),
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )

        # Dekódovanie výstupu
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        parsed_answer = processor.post_process_generation(
            generated_text,
            task=prompt,
            image_size=paligemma_size,
        )
        # print(f"Generated answer: {parsed_answer}")
        # return parsed_answer
        print("toto je prompt:" + prompt.lower())
        if "<OD>" in prompt:
            output_path = plot_bbox_florence(image, parsed_answer['<OD>'])
            return {"type": "image", "filename": output_path}
        else:
            return {"type": "text", "filename": parsed_answer}
    except Exception as e:
        print(f"Error during Florence run: {e}")
        return None

def run_florence_timeline(image, task_prompt, text_input=None):
        prompt = task_prompt if text_input is None else task_prompt + text_input
        # Príprava vstupov pomocou procesora
        inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda", torch.float16)

        # Generovanie odpovede modelom
        generated_ids = model.generate(
            input_ids=inputs["input_ids"].cuda(),
            pixel_values=inputs["pixel_values"].cuda(),
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )

        # Dekódovanie výstupu
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        parsed_answer = processor.post_process_generation(
            generated_text,
            task=prompt,
            image_size=(image.width, image.height),
        )
        # print(f"Generated answer: {parsed_answer}")
        # return parsed_answer
        return  parsed_answer

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     result = None
#     filename = None
#     model = None
#     prompt_text = None
#
#     if request.method == 'POST':
#         if 'file' in request.files:
#             file = request.files['file']
#             if file and allowed_file(file.filename):
#                 filename = file.filename
#                 file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#                 file.save(file_path)
#             else:
#                 return "Invalid file type. Please upload an image or video."
#         else:
#             filename = request.form['filename']
#             file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             model = request.form['model']  # Získanie hodnoty modelu z formulára
#             prompt_type = request.form['prompt_type']
#             prompt_text = request.form['prompt_text']
#
#             if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
#                 image = process_image(file_path)
#             elif filename.lower().endswith(('.mp4', '.mov', '.avi')):
#                 image = process_video(file_path)
#             else:
#                 return "Unsupported file format."
#
#             if model == "paligemma":
#                 prompt = f"<image> <bos>{prompt_type}: {prompt_text}"
#                 result = run_paligemma(image, prompt)
#             elif model == "florence2":
#                 if prompt_type == "VQA":
#                     prompt = f"<{prompt_type}> + {prompt_text}"
#                 else:
#                     prompt = f"<{prompt_type}>"
#                 result = run_florence(image, prompt)
#             else:
#                 result = "Invalid model selection."
#
#     # Odoslanie všetkých potrebných hodnôt (filename, result, model) do šablóny
#     return render_template('index.html', filename=filename, result=result, model=model, prompt_text=prompt_text)
@app.route('/')
def home():
    return render_template('first.html')
@app.route('/detection/classic', methods=['GET', 'POST'])
def index():
    result = None
    result_video = None
    filename = None
    model = None
    classify = []
    prompt_text = None
    frame_results = []  # Uloženie výsledkov pre každý frame
    all_classify = []
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = file.filename
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
            else:
                return "Invalid file type. Please upload an image or video."
        else:
            filename = request.form['filename']
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            model = request.form['model']
            prompt_type = request.form['prompt_type']
            prompt_text = request.form['prompt_text']

            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                image = process_image(file_path)
                if model == "paligemma":
                    prompt = f"<image> <bos>{prompt_type}: {prompt_text}"
                    result = run_paligemma(image, prompt)
                    # print()
                    # classify.append((prompt_type, prompt_text, result['content']))
                elif model == "florence2":
                    prompt = f"<{prompt_type}> + {prompt_text}" if prompt_type == "VQA" else f"<{prompt_type}>"
                    result = run_florence(image, prompt)
                    print(result)
                    classify.append(result)
                else:
                    result = "Invalid model selection."

            elif filename.lower().endswith(('.mp4', '.mov', '.avi')):
                frames = process_video(file_path)  # Spracovanie všetkých framov
                for i, frame in enumerate(frames):
                    if model == "paligemma":
                        prompt = f"<image> <bos>{prompt_type}: {prompt_text}"
                        result_video = run_paligemma(frame, prompt)
                        classify.append((prompt_type, prompt_text, result_video['content']))

                    elif model == "florence2":
                        prompt = f"<{prompt_type}> + {prompt_text}" if prompt_type == "VQA" else f"<{prompt_type}>"
                        result_video = run_florence(frame, prompt)
                        classify.append((prompt_type, prompt_text,result_video))
                    else:
                        result_video = "Invalid model selection."

                    frame_results.append((i + 1, result_video))
            if classify:
                result_string = " ".join([f"{prompt_type} {prompt_text} {result_video}" for
                                          prompt_type, prompt_text, result_video in classify])
                all_classify.append(classify_text(result_string))
                print(all_classify)

    return render_template('index.html', filename=filename, result=result, model=model, frame_results=frame_results,
                           prompt_text=prompt_text, all_classify=all_classify)

def check_for_location_tag(outputs):
    if isinstance(outputs, str):  # Ak je to string, urobíme z neho zoznam s jedným prvkom
        outputs = [outputs]

    results = []
    for output in outputs:
        if "<loc" in output:
            results.append("yes")
        else:
            results.append("no")
    return '\n'.join(results)


import ast



def check_for_location_tag_florence(outputs):
    results = []
    print(outputs)
    # for output in outputs:
    #     print(output)
        # print("------------------------------------------------")
    if outputs['<CAPTION_TO_PHRASE_GROUNDING>']['labels'] and outputs['<CAPTION_TO_PHRASE_GROUNDING>']['bboxes']:
        results.append("yes")
    else:
        results.append("no")

    return '\n'.join(results)


def check_VQA_timeline(outputs):
    results = []
    if outputs == "yes":
        results.append("yes")
    else:
        results.append("no")
    # for output in outputs:
    #     try:
    #         output_dict = ast.literal_eval(output) if isinstance(output, str) else output
    #         if isinstance(output_dict, dict) and len(output_dict) == 1:
    #             key, value = next(iter(output_dict.items()))
    #             if key.startswith("<VQA>") and value in ["yes", "no"]:
    #                 results.append(value)
    #             else:
    #                 results.append("no")
    #         else:
    #             results.append("no")
    #     except (ValueError, SyntaxError):
    #         results.append("no")
    return '\n'.join(results)


@app.route('/detection/threat', methods=['GET', 'POST'])
def threat():
    try:
        global avg_scores, final_label, frame_index, times, scores
        result_video = None
        filename = None
        model = None
        frame_results = []  # Výsledky pre každý frame
        prompt_results = []  # Výsledky pre každý prompt
        timeline_images = []  # Zoznam obrázkov časových osí
        all_timelines = []  # Všetky časové osi na spojenie
        combined_timeline = []  # Finálna časová os
        all_classify = []
        classify = []
        result_vqa = []
        timeline_prompts = []
        prompt_type = None
        prompt_text = None
        prompt_type_image = []
        result_image = []
        all_timelines_prompt = []
        frame_results_with_time = []
        times_all = []
        timeline_all=[]
        answers = []
        all_range = []
        image_process = []
        image_process_all = []
        detect = []
        answer = []
        avg_scores = []
        show_button = False
        images = []
        frame_indexes = []
        times_indexes = []
        if request.method == 'POST':
            if 'file' in request.files:
                file = request.files['file']
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(file_path)
                else:
                    return "Invalid file type. Please upload an image or video."
            else:
                filename = request.form.get('filename')
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                model = request.form.get('model')

                prompt_count = 1
                while f'prompt_type-{prompt_count}' in request.form:
                    prompt_type = request.form[f'prompt_type-{prompt_count}']
                    prompt_text = request.form[f'prompt_text-{prompt_count}']
                    time_line = []

                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                        frame = read_image(file_path)
                        if model == "paligemma":
                            prompt = f"{prompt_type}: {prompt_text}"
                            scores = classify_frame(frame)
                            # result = run_paligemma_timeline(image, prompt)
                            if prompt_type == "detect":
                                result = run_paligemma_timeline(frame, prompt)
                                # print("paligemma detect from video")
                                boxes, labels = parse_bbox_and_labels(result)
                                output_path = display_boxes(frame, boxes, labels)
                                images.append(output_path)
                                print(output_path)
                                result_video = check_for_location_tag(result)
                                print(result_video)
                                classify.append((prompt_type, prompt_text, result_video))
                            else:
                                result_video = run_paligemma_timeline(frame, prompt)
                                print(result_video)
                                result_video = result_video.replace("<eos>", "").strip()
                                print(result_video)
                                classify.append((prompt_type, prompt_text, result_video))

                                prompt_type_image.append(prompt_type)
                                result_image.append(result_video)

                        elif model == "florence2":
                            scores = classify_frame(frame)
                            # prompt = f"<{prompt_type}> + {prompt_text}" if prompt_type == "VQA" else f"<{prompt_type}>"
                            # result = run_florence_timeline(frame, prompt)
                            # prompt = f"{prompt_type} + {prompt_text}" if prompt_type == "VQA" and "CAPTION_TO_PHRASE_GROUNDING" else f"{prompt_type}"
                            # scores = classify_frames(frame)
                            # final_label, avg_scores = get_final_label(scores)
                            if prompt_type == "CAPTION_TO_PHRASE_GROUNDING":
                                # result = run_example(frame, prompt_type, prompt_text)
                                result = run_example(frame, '<CAPTION_TO_PHRASE_GROUNDING>', text_input=prompt_text)
                                # print(result)
                                output_path = plot_bbox_florence(frame, result['<CAPTION_TO_PHRASE_GROUNDING>'])
                                print(output_path)
                                images.append(output_path)
                                result_video = check_for_location_tag_florence(result)
                                print(result_video)
                                answers.append((prompt_type, prompt_text, result_video))
                                classify.append((prompt_type, prompt_text, result_video))
                            elif prompt_type == "VQA":
                                result = predict(frame, prompt_text)
                                print(result)
                                # if result == "yes" or result == "no":
                                #     result_video = check_VQA_timeline(result)
                                #     classify.append((prompt_type, prompt_text, result_video))
                                # else:
                                print("som tu VQA")
                                # result_vqa.append(result)
                                prompt_type_image.append(prompt_type)
                                result_image.append(result)
                                classify.append((prompt_type, prompt_text, result))
                            elif prompt_type == "CAPTION":
                                result_video = run_example(frame, '<CAPTION>', text_input=prompt_text)
                                print("som tu caoption")
                                print(result_video)
                                classify.append((prompt_type, prompt_text, result_video))
                                prompt_type_image.append(prompt_type)
                                result_image.append(result_video)
                            elif prompt_type == "MORE_DETAILED_CAPTION":
                                result_video = run_example(frame, '<MORE_DETAILED_CAPTION>', text_input=prompt_text)
                                print("som tu caoption")
                                print(result_video)
                                classify.append((prompt_type, prompt_text, result_video))
                                prompt_type_image.append(prompt_type)
                                result_image.append(result_video)
                        elif model == "paligemmaft":
                            scores = classify_frame(frame)
                            # final_label, avg_scores = get_final_label(scores)
                            prompt = f"<image> <bos>{prompt_type}: {prompt_text}"
                            if prompt_type == "detect":
                                result = run_paligemmaFT_timeline(frame, prompt)
                                boxes, labels = parse_bbox_and_labels(result)
                                output_path = display_boxes(frame, boxes, labels)
                                print(output_path)
                                images.append(output_path)
                                # print("paligemma detect from video")
                                result_video = check_for_location_tag(result)
                                print(result_video)
                                classify.append((prompt_type, prompt_text, result_video))
                            else:
                                result_video = run_paligemmaFT_timeline(frame, prompt)
                                result_video = result_video.replace("<eos>", "").strip()
                                print(result_video)
                                classify.append((prompt_type, prompt_text, result_video))
                        elif model == "florence2ft":
                            # prompt = f"{prompt_type} + {prompt_text}" if prompt_type == "VQA" and "CAPTION_TO_PHRASE_GROUNDING" else f"{prompt_type}"
                            scores = classify_frame(frame)
                            # final_label, avg_scores = get_final_label(scores)
                            if prompt_type == "CAPTION_TO_PHRASE_GROUNDING":
                                # result = run_example(frame, prompt_type, prompt_text)
                                result = run_exampleFT(frame, '<CAPTION_TO_PHRASE_GROUNDING>', text_input=prompt_text)
                                # print(result)
                                output_path = plot_bbox_florence(frame, result['<CAPTION_TO_PHRASE_GROUNDING>'])
                                print(output_path)
                                images.append(output_path)
                                result_video = check_for_location_tag_florence(result)
                                print(result_video)
                                classify.append((prompt_type, prompt_text, result_video))
                            elif prompt_type == "VQA":
                                result = predict(frame, prompt_text)
                                print(result)
                                # if result == "yes" or result == "no":
                                #     result_video = check_VQA_timeline(result)
                                #     classify.append((prompt_type, prompt_text, result_video))
                                # else:
                                print("som tu VQA")

                                result_vqa.append(result)
                                classify.append((prompt_type, prompt_text, result))
                            elif prompt_type == "CAPTION":
                                result_video = run_exampleFT(frame, '<CAPTION>', text_input=prompt_text)
                                print("som tu caoption")
                                print(result_video)
                                classify.append((prompt_type, prompt_text, result_video))
                            elif prompt_type == "MORE_DETAILED_CAPTION":
                                result_video = run_exampleFT(frame, '<MORE_DETAILED_CAPTION>', text_input=prompt_text)
                                print("som tu caoption")
                                print(result_video)
                                classify.append((prompt_type, prompt_text, result_video))
                        else:
                            print("Invalid model selection.")


                        # prompt_results.append(result)

                    elif filename.lower().endswith(('.mp4', '.mov', '.avi')):
                        frames, times, frame_index = process_video(file_path)

                        times_all.append(times * prompt_count)
                        print(times)
                        print(frame_index)
                        for frame in frames:
                            print(frame.size)
                            if model == "paligemma":
                                scores = classify_frames(frames)
                                # final_label, avg_scores = get_final_label(scores)
                                prompt = f"<image> <bos>{prompt_type}: {prompt_text}"
                                if prompt_type == "detect":
                                    result = run_paligemma_timeline(frame, prompt)
                                    # print("paligemma detect from video")
                                    boxes, labels = parse_bbox_and_labels(result)
                                    output_path = display_boxes(frame, boxes, labels)
                                    images.append(output_path)
                                    print(output_path)
                                    result_video = check_for_location_tag(result)
                                    print(result_video)
                                    classify.append((prompt_type,prompt_text,result_video))
                                else:
                                    result_video = run_paligemma_timeline(frame, prompt)
                                    print(result_video)
                                    result_video = result_video.replace("<eos>", "").strip()
                                    print(result_video)
                                    classify.append((prompt_type,prompt_text,result_video))

                            elif model == "florence2":
                                # prompt = f"{prompt_type} + {prompt_text}" if prompt_type == "VQA" and "CAPTION_TO_PHRASE_GROUNDING" else f"{prompt_type}"
                                scores = classify_frames(frames)
                                # final_label, avg_scores = get_final_label(scores)
                                if prompt_type == "CAPTION_TO_PHRASE_GROUNDING":
                                    # result = run_example(frame, prompt_type, prompt_text)
                                    result = run_example(frame, '<CAPTION_TO_PHRASE_GROUNDING>', text_input=prompt_text)
                                    # print(result)
                                    output_path = plot_bbox_florence(frame, result['<CAPTION_TO_PHRASE_GROUNDING>'])
                                    print(output_path)
                                    images.append(output_path)
                                    result_video = check_for_location_tag_florence(result)
                                    print(result_video)
                                    answers.append((prompt_type, prompt_text,result_video))
                                    classify.append((prompt_type, prompt_text, result_video))
                                elif prompt_type == "VQA":
                                    result = predict(frame, prompt_text)
                                    print(result)
                                    if result == "yes" or result == "no":
                                        result_video = check_VQA_timeline(result)
                                        classify.append((prompt_type, prompt_text, result_video))
                                    else:
                                        print("som tu VQA")
                                        print(result_video)
                                        result_vqa.append(result)
                                        classify.append((prompt_type, prompt_text, result))
                                elif prompt_type == "CAPTION":
                                    result_video = run_example(frame, '<CAPTION>', text_input=prompt_text)
                                    print("som tu caoption")
                                    print(result_video)
                                    classify.append((prompt_type, prompt_text, result_video))
                                elif prompt_type == "MORE_DETAILED_CAPTION":
                                    result_video = run_example(frame, '<MORE_DETAILED_CAPTION>', text_input=prompt_text)
                                    print("som tu caoption")
                                    print(result_video)
                                    classify.append((prompt_type, prompt_text, result_video))
                            elif model == "paligemmaft":
                                scores = classify_frames(frames)
                                # final_label, avg_scores = get_final_label(scores)
                                prompt = f"<image> <bos>{prompt_type}: {prompt_text}"
                                if prompt_type == "detect":
                                    result = run_paligemmaFT_timeline(frame, prompt)
                                    boxes, labels = parse_bbox_and_labels(result)
                                    output_path = display_boxes(frame, boxes, labels)
                                    print(output_path)
                                    images.append(output_path)
                                    # print("paligemma detect from video")
                                    result_video = check_for_location_tag(result)
                                    print(result_video)
                                    classify.append((prompt_type, prompt_text, result_video))
                                else:
                                    result_video = run_paligemmaFT_timeline(frame, prompt)
                                    result_video = result_video.replace("<eos>", "").strip()
                                    print(result_video)
                                    classify.append((prompt_type, prompt_text, result_video))
                            elif model == "florence2ft":
                                # prompt = f"{prompt_type} + {prompt_text}" if prompt_type == "VQA" and "CAPTION_TO_PHRASE_GROUNDING" else f"{prompt_type}"
                                scores = classify_frames(frames)
                                # final_label, avg_scores = get_final_label(scores)
                                if prompt_type == "CAPTION_TO_PHRASE_GROUNDING":
                                    # result = run_example(frame, prompt_type, prompt_text)
                                    result = run_exampleFT(frame, '<CAPTION_TO_PHRASE_GROUNDING>', text_input=prompt_text)
                                    # print(result)
                                    output_path = plot_bbox_florence(frame, result['<CAPTION_TO_PHRASE_GROUNDING>'])
                                    print(output_path)
                                    images.append(output_path)
                                    result_video = check_for_location_tag_florence(result)
                                    print(result_video)
                                    classify.append((prompt_type, prompt_text, result_video))
                                elif prompt_type == "VQA":
                                    result = predict(frame, prompt_text)
                                    print(result)
                                    if result == "yes" or result == "no":
                                        result_video = check_VQA_timeline(result)
                                        classify.append((prompt_type, prompt_text, result_video))
                                    else:
                                        print("som tu VQA")
                                        print(result_video)
                                        result_vqa.append(result)
                                        classify.append((prompt_type, prompt_text, result))
                                elif prompt_type == "CAPTION":
                                    result_video = run_exampleFT(frame, '<CAPTION>', text_input=prompt_text)
                                    print("som tu caoption")
                                    print(result_video)
                                    classify.append((prompt_type, prompt_text, result_video))
                                elif prompt_type == "MORE_DETAILED_CAPTION":
                                    result_video = run_exampleFT(frame, '<MORE_DETAILED_CAPTION>', text_input=prompt_text)
                                    print("som tu caoption")
                                    print(result_video)
                                    classify.append((prompt_type, prompt_text, result_video))
                            else:
                                print("Invalid model selection.")

                            if prompt_type.strip().lower() in {"vqa", "caption_to_phrase_grounding", "detect", "answer"}:

                                if result_video is not None:
                                    print("som v if none")
                                    time_line.append(result_video)
                                    # time_line.append(result_video)
                                else:
                                    print("som v else")
                                    frame_results.append(result_video)
                            else:
                                print("som v else")
                                frame_results.append(result_video)
                        print("all frame")
                        print(frame_results)

                    if filename.lower().endswith(('.mp4', '.mov', '.avi')):
                        session["frame_index"] = frame_index
                        session["timestamp"] = times

                        for idx, time in enumerate(times_all):  # Prvý index a čas v zozname times_all

                            # Vytvorí dvojice (frame_result, time) medzi frame_results a aktuálnym časom
                            frame_results_with_time = list(zip(frame_results, time))
                            # print(frame_results_with_time)


                                # time_line.append(result_video)  # Uložíme výsledky pre daný prompt
                            # Po všetkých promptoch (pre obraz alebo video) vygenerujeme text

                    if prompt_type in ["detect", "CAPTION_TO_PHRASE_GROUNDING"]:
                        show_button = True

                    # Ak máme dáta na vykreslenie, vytvoríme obrázok pre tento prompt
                    if time_line:
                        img_filename = visualize_answers(time_line, prompt_count)
                        timeline_images.append(url_for('static', filename=f'results/{img_filename}'))
                        all_timelines.append(time_line)
                        print("som tu")
                        if prompt_type == "VQA":
                            prompt_type = "Answer"
                        elif prompt_type == "CAPTION_TO_PHRASE_GROUNDING":
                            prompt_type = "Detect"
                        timeline_prompts.append((prompt_type, prompt_text))
                        all_timelines_prompt = list(zip(timeline_images, timeline_prompts))

                    prompt_count += 1
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    # Predpokladajme, že image_process obsahuje tuple (prompt_type, result_video)
                    image_process_all = list(zip(prompt_type_image, result_image))

                result_string = " ".join([f"{prompt_type} {prompt_text} {result_video}" for
                                          prompt_type, prompt_text, result_video in classify])
                # print(result_string)  # Môžeš si vypísať, ako vyzerá konečný reťazec
                # classify_text(result_string)  # Uložíme alebo zobrazíme výsledný text
                selected_option = request.form.get("flexRadioDefault")  # Získa hodnotu vybraného tlačidla

                print(selected_option)
                if selected_option == "flexRadioClip":
                    final_label, avg_scores = get_final_label(scores)
                    # Ak je zaškrtnutý, vykonáme nejakú akciu
                    print("CLIP")
                    print("\n🔹 **Globálna predikcia pre video:**")
                    print(avg_scores)
                    avg_scores = dict(sorted(avg_scores.items(), key=lambda item: item[1], reverse=True))
                    print(avg_scores)
                elif selected_option == "flexRadioBart":
                    print("BART")
                    all_classify.append(classify_text(result_string))


            # image_path = "static/uploads/output_detect.png"  # Cesta k obrázku

            session["image_path"] = images  # Uložíme cestu do session


            if all_timelines:
                print("som tu")
                combined_timeline = merge_timelines(all_timelines)  # Spojenie osí
                img_combined = visualize_combined_timeline(combined_timeline)
                timeline_all.append(url_for('static', filename=f'results/{img_combined}'))  # Odošleme na frontend
        return render_template('threat.html',
                               filename=filename,
                               result=prompt_results,
                               model=model,
                               frame_results=frame_results,
                               timeline_images=timeline_images,
                               timeline_prompts=timeline_prompts,
                               timeline_all=timeline_all,
                               all_timelines_prompt=all_timelines_prompt,
                               all_classify=all_classify,
                               result_vqa=result_vqa,
                               times_all=times_all,
                               frame_results_with_time=frame_results_with_time,
                               avg_scores=avg_scores,
                               show_button=show_button,
                               image_process_all=image_process_all,
                               )
    except Exception as e:
        print(f"Error: {e}")
        return render_template('threat.html', error=f"An error occurred while processing the request.{e}")




def merge_timelines(all_timelines):
    """
    Spojí viacero časových osí do jednej. Ak aspoň jeden detekovaný objekt je "yes", výstup bude "yes".
    """
    num_frames = max(len(t) for t in all_timelines)  # Najdlhšia časová os
    combined_timeline = []

    for i in range(num_frames):
        frame_result = "no"  # Predvolená hodnota
        for timeline in all_timelines:
            if i < len(timeline) and timeline[i] == "yes":
                frame_result = "yes"
                break  # Ak nájdeme "yes", nemusíme ďalej kontrolovať
        combined_timeline.append(frame_result)

    # print(combined_timeline)

    return combined_timeline

def visualize_combined_timeline(combined_timeline):
    """
    Vytvorí obrázok finálnej zlúčenej časovej osy.
    """
    timeline_length = len(combined_timeline)
    colors = ["green" if answer == "yes" else "red" for answer in combined_timeline]

    fig, ax = plt.subplots(figsize=(12, 2))
    ax.set_xlim(0, timeline_length)
    ax.set_ylim(0, 1)

    for i, color in enumerate(colors):
        ax.plot([i, i + 1], [0.5, 0.5], color=color, solid_capstyle="butt", lw=6)

    ax.axis("off")

    # Uloženie finálnej časovej osy
    img_filename = "final_threat_timeline.png"
    img_path = os.path.join(app.config['RESULT_FOLDER'], img_filename)
    plt.savefig(img_path, bbox_inches='tight')
    plt.close(fig)

    return img_filename

#TODO: funguje vsetko, zisti na inom videu preco nevyhodnocuje spravne

@app.route("/detection-results")
def detection_results():
    # Získame cestu k obrázku zo session
    image_path = session.get("image_path", [])  # Ak neexistuje, vráti None
    frame_indexes = session.get("frame_index", [])
    time_indexes = session.get("timestamp", [])
    print(image_path, frame_indexes, time_indexes)
    combined_data = zip(image_path, time_indexes, frame_indexes)
    if image_path:
        return render_template("detection_result.html", combined_data=combined_data)  # Pošleme cestu do HTML
    else:
        return "No image found!"


# @app.route('/detection/threat', methods=['GET', 'POST'])
# def threat():
#     result = None
#     result_video = None
#     filename = None
#     model = None
#     prompt_text = None
#     frame_results = []  # Uloženie výsledkov pre každý frame
#     prompt_results = []  # Uloženie výsledkov pre každý prompt
#     time_line = []  # Uloženie časových značiek pre každý prompt
#
#     if request.method == 'POST':
#         if 'file' in request.files:
#             file = request.files['file']
#             if file and allowed_file(file.filename):
#                 filename = file.filename
#                 file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#                 file.save(file_path)
#             else:
#                 return "Invalid file type. Please upload an image or video."
#         else:
#             filename = request.form['filename']
#             file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             model = request.form['model']
#
#             # Handle multiple prompts
#             prompt_count = 1
#             while f'prompt_type-{prompt_count}' in request.form:
#                 prompt_type = request.form[f'prompt_type-{prompt_count}']
#                 prompt_text = request.form[f'prompt_text-{prompt_count}']
#
#                 if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
#                     image = process_image(file_path)
#                     if model == "paligemma":
#                         prompt = f"<image> <bos>{prompt_type}: {prompt_text}"
#                         result = run_paligemma_timeline(image, prompt)
#                         print(result)
#                     elif model == "florence2":
#                         prompt = f"<{prompt_type}> + {prompt_text}" if prompt_type == "VQA" else f"<{prompt_type}>"
#                         result = run_florence_timeline(image, prompt)
#                     else:
#                         result = "Invalid model selection."
#                     prompt_results.append(result)  # Store result for the current prompt
#
#                 elif filename.lower().endswith(('.mp4', '.mov', '.avi')):
#                     frames = process_video(file_path)
#                     for i, frame in enumerate(frames):
#                         if model == "paligemma":
#                             prompt = f"<image> <bos>{prompt_type}: {prompt_text}"
#                             result_video = run_paligemma_timeline(frame, prompt)
#                         elif model == "florence2":
#                             prompt = f"<{prompt_type}> + {prompt_text}" if prompt_type == "VQA" else f"<{prompt_type}>"
#                             result_video = run_florence_timeline(frame, prompt)
#                         else:
#                             result_video = "Invalid model selection."
#                         frame_results.append((result_video))  # Store result for the current frame
#                         time_line.append(result_video)
#                         # print(time_line)
#                     # visualize_answers(time_line)
#                 prompt_count += 1
#
#     return render_template('threat.html', filename=filename, result=prompt_results, model=model, frame_results=frame_results,
#                             prompt_text=prompt_text)






# def threat():
#     result = None
#     result_video = None
#     filename = None
#     model = None
#     prompt_text = None
#     frame_results = []  # Uloženie výsledkov pre každý frame
#
#     if request.method == 'POST':
#         if 'file' in request.files:
#             file = request.files['file']
#             if file and allowed_file(file.filename):
#                 filename = file.filename
#                 file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#                 file.save(file_path)
#             else:
#                 return "Invalid file type. Please upload an image or video."
#         else:
#             filename = request.form['filename']
#             file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             model = request.form['model']
#             prompt_type = request.form['prompt_type']
#             prompt_text = request.form['prompt_text']
#
#             if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
#                 image = process_image(file_path)
#                 if model == "paligemma":
#                     prompt = f"<image> <bos>{prompt_type}: {prompt_text}"
#                     result = run_paligemma(image, prompt)
#                 elif model == "florence2":
#                     prompt = f"<{prompt_type}> + {prompt_text}" if prompt_type == "VQA" else f"<{prompt_type}>"
#                     result = run_florence(image, prompt)
#                 else:
#                     result = "Invalid model selection."
#
#             elif filename.lower().endswith(('.mp4', '.mov', '.avi')):
#                 frames = process_video(file_path)  # Spracovanie všetkých framov
#                 for i, frame in enumerate(frames):
#                     if model == "paligemma":
#                         prompt = f"<image> <bos>{prompt_type}: {prompt_text}"
#                         result_video = run_paligemma(frame, prompt)
#                     elif model == "florence2":
#                         prompt = f"<{prompt_type}> + {prompt_text}" if prompt_type == "VQA" else f"<{prompt_type}>"
#                         result_video = run_florence(frame, prompt)
#                     else:
#                         result_video = "Invalid model selection."
#
#                     frame_results.append((i + 1, result_video))
#                 print(frame_results)
#
#     return render_template('index.html', filename=filename, result=result, model=model, frame_results=frame_results,
#                            prompt_text=prompt_text)


def clear_detect_folder():
    detect_folder = os.path.join("static", "detect")
    if os.path.exists(detect_folder):
        shutil.rmtree(detect_folder)  # Vymaže celý priečinok
    os.makedirs(detect_folder, exist_ok=True)  # Znova ho vytvorí

# Pred spustením aplikácie vymažeme staré súbory
clear_detect_folder()

if __name__ == '__main__':
    app.run(debug=False)

