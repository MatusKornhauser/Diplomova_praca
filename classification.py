import matplotlib
from transformers import CLIPModel, CLIPProcessor
import torch

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
