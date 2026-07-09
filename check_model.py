from pathlib import Path
from damage_detector import DamageDetector
from config import ACCIDENT_MODEL_PATHS
import cv2

print('model paths:')
for p in ACCIDENT_MODEL_PATHS:
    print(p, 'exists' if p.exists() else 'missing')

img_path = Path('d:/car accident/accident2.jpg')
print('image path', img_path)
print('image exists', img_path.exists())
img = cv2.imread(str(img_path))
print('image shape', None if img is None else img.shape)

d = DamageDetector()
print('detector enabled', d.enabled)
print('detector model path', d.model_path)
print('model file exists', d.model_path.exists())

if img is None:
    print('ERROR: image could not load')
else:
    results = d.detect(img) if d.enabled else []
    print('results count', len(results))
    print(results)
