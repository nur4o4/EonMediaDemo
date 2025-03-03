import cv2
import torch
import open_clip
import numpy as np
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings

# Extract frames from the video using OpenCV
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

# SIFT feature extraction for detecting keypoints in images
def extract_sift_features(frames):
    sift = cv2.SIFT_create()
    keypoints_and_descriptors = []
    
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        keypoints_and_descriptors.append((keypoints, descriptors))
        
    return keypoints_and_descriptors

# CLIP-based logo and face detection
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = open_clip.load('ViT-B/32', pretrained='openai', device=device)

def detect_logo_or_face(image, labels=["Nike logo", "Adidas logo", "Face of famous person"]):
    image_input = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
    text_inputs = torch.cat([open_clip.tokenize(label) for label in labels]).to(device)
    
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        text_features = clip_model.encode_text(text_inputs)
    
    similarities = (image_features @ text_features.T).squeeze(0)
    best_label_idx = similarities.argmax().item()
    
    return labels[best_label_idx]

# Evaluating performance using precision, recall, and F1-score
def evaluate_model(predictions, ground_truth):
    precision = precision_score(ground_truth, predictions, average='weighted', zero_division=0)
    recall = recall_score(ground_truth, predictions, average='weighted', zero_division=0)
    f1 = f1_score(ground_truth, predictions, average='weighted', zero_division=0)
    return precision, recall, f1

# Main Execution Logic
def main(video_path, labels=["Nike logo", "Adidas logo", "Face of famous person"], ground_truth_labels=None):
    frames = extract_frames(video_path)
    
    keypoints_and_descriptors = extract_sift_features(frames)
    
    # Detect logos or faces using CLIP
    predictions = []
    for frame in frames:
        detected_label = detect_logo_or_face(frame, labels)
        predictions.append(detected_label)
    
    if ground_truth_labels is not None:
        precision, recall, f1 = evaluate_model(predictions, ground_truth_labels)
        print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")
    else:
        print("Model predictions (without ground truth evaluation):")
        print(predictions)

if __name__ == "__main__":
    # Provide video file path here
    video_path = "vid_start_again.mp4"  
    
    ground_truth_labels = ["Nike logo", "Adidas logo", "Star shape", "Human face"] 
    
    main(video_path, ground_truth_labels=ground_truth_labels)
