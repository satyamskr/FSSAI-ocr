import pytesseract
import cv2
from PIL import Image
img_file=("data/sample8.jpeg")
im=Image.open(img_file)
ocr_result=pytesseract.image_to_string(im)
print(ocr_result)