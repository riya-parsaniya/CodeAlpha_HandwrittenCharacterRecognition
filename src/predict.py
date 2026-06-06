"""
predict.py
----------
Run inference on a single handwritten character image.

Usage:
    python src/predict.py --file path/to/image.png --model ResNet_CNN
"""

import sys, os, argparse
import numpy as np
import cv2
sys.path.insert(0, os.path.dirname(__file__))

from config import MODELS_DIR, CLASS_NAMES, IMG_SIZE
from train  import load_saved_model


def preprocess_image(filepath: str) -> np.ndarray:
    """Load any image, convert to 28×28 grayscale, normalise."""
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {filepath}")

    # Invert if background is white (EMNIST uses white-on-black)
    if img.mean() > 127:
        img = 255 - img

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    return img[np.newaxis, :, :, np.newaxis]   # (1, 28, 28, 1)


def predict_character(filepath: str, model_name: str = 'ResNet_CNN') -> dict:
    model  = load_saved_model(model_name)
    X      = preprocess_image(filepath)
    probs  = model.predict(X, verbose=0)[0]
    top3_idx = np.argsort(probs)[::-1][:3]

    predicted = CLASS_NAMES[np.argmax(probs)]
    print(f'\nFile      : {os.path.basename(filepath)}')
    print(f'Predicted : {predicted}  ({probs.max()*100:.1f}%)')
    print('\nTop-3 predictions:')
    for i, idx in enumerate(top3_idx):
        bar = '█' * int(probs[idx] * 40)
        print(f'  {i+1}. {CLASS_NAMES[idx]}  {probs[idx]*100:5.1f}%  {bar}')

    return {'predicted': predicted,
            'confidence': float(probs.max()),
            'top3': [(CLASS_NAMES[i], float(probs[i])) for i in top3_idx]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file',  required=True, help='Path to image file')
    parser.add_argument('--model', default='ResNet_CNN',
                        help='Model name (Simple_CNN / Deep_CNN / ResNet_CNN)')
    args = parser.parse_args()
    predict_character(args.file, args.model)
