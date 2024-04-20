import numpy as np
import pytesseract
import cv2
from PIL import Image
import re
# img_file=("sample9.jpeg")
# def flatten_curved_image(image_path):
#   """
#   Flattens a curved image using homography estimation and Canny edge detection.
#
#   Args:
#       image_path (str): Path to the curved image.
#
#   Returns:
#       OpenCV image object: The flattened image.
#
#   Raises:
#       ValueError: If the number of contours found is less than 4.
#   """
#
#   # Load the image
#   image = cv2.imread(image_path)
#
#   # Apply Canny edge detection
#   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#   blur = cv2.GaussianBlur(gray, (5, 5), 0)
#   edges = cv2.Canny(blur, low_threshold=50, high_threshold=150)
#
#   # Find contours in the edge image (assuming at least 4 contours represent the image shape)
#   contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#   if len(contours) < 4:
#       raise ValueError("Not enough contours found for reference points (minimum 4 needed)")
#
#   # Select the first 4 largest contours (adjust selection logic as needed)
#   # This assumes the largest contours correspond to the image borders
#   sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:4]
#
#   # Extract corner points from the contours (replace with your selection logic)
#   reference_points_curved = []
#   for cnt in sorted_contours:
#       # Get the bounding rectangle (adjust logic for better point selection)
#       x, y, w, h = cv2.boundingRect(cnt)
#       reference_points_curved.append((x, y))
#       reference_points_curved.append((x + w, y))
#       reference_points_curved.append((x, y + h))
#       reference_points_curved.append((x + w, y + h))
#
#   # Define reference points on the flat surface (adjust coordinates)
#   reference_points_flat = [(50, 0), (200, 0), (50, 400), (200, 400)]
#
#   # Convert points to NumPy arrays
#   src_pts = np.float32(reference_points_curved).reshape(-1, 1, 2)
#   dst_pts = np.float32(reference_points_flat).reshape(-1, 1, 2)
#
#   # Estimate homography matrix
#   m, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
#
#   # Get image size
#   height, width, _ = image.shape
#
#   # Define the destination image size (assuming flat surface has same width)
#   dst_height = height
#
#   # Get the transformation matrix
#   warp_matrix = cv2.getPerspectiveTransform(np.float32([[0, 0], [0, dst_height], [width, dst_height], [width, 0]]), m)
#
#   # Warp the image
#   flattened_image = cv2.warpPerspective(image, warp_matrix, (width, dst_height))
#
#   return flattened_image
#
#
#
#
def preprocess_image(image_flat):
    img = image_flat
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Noise reduction (adjust parameters as needed)
    denoised = cv2.fastNlMeansDenoising(gray)

    # Adaptive thresholding for better segmentation
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    # Sharpening (adjust parameters as needed)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(thresh, -1, kernel)
    return sharpened


def main():
    # img_file = ("sample8.jpeg")
    # im = Image.open(img_file)
    img = cv2.imread("sample5.jpg")
    ocr_result = pytesseract.image_to_string(img)

    fssai_number_regex = r"Lic. No. [\dA-Z]{14}|Lic. No.: [\dA-Z]{14}|Lic.No.[\dA-Z]{14}|cc No [\dA-Z]{14}|License Number[\dA-Z]{14}|icense number[\dA-Z]{14}|ic. no.[\dA-Z]{14}"
    match = re.search(fssai_number_regex, ocr_result)
    if match:
        fssai_number = match.group(0)
        numbers = ''.join(char for char in fssai_number if char.isdigit())
        print(numbers)
    else:
        print("FSSAI number not found in the text.")


if __name__ == "__main__":
    main()

