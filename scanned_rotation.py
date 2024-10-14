import fitz
import numpy as np
import cv2
from PIL import Image
import pytesseract
import pyperclip
from scipy.ndimage import rotate
import matplotlib.pyplot as plt


# Detect skew and rotate image with better quality using OpenCV
def detect_and_rotate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    angles = []
    for line in lines:
        rho, theta = line[0]
        angle = np.degrees(theta) - 90
        if abs(angle) < 45:
            angles.append(angle)

    if len(angles) > 0:
        median_angle = np.median(angles)
        # Get image dimensions
        (h, w) = image.shape[:2]
        # Get the center of the image
        center = (w // 2, h // 2)
        # Calculate the rotation matrix
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        # Rotate the image with higher-quality interpolation
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated
    else:
        # If no rotation is needed, return the original image
        return image


# Function to enhance the image quality
def enhance_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply bilateral filter to smooth image and preserve edges
    smoothed = cv2.bilateralFilter(gray, 9, 75, 75)

    # Sharpen the image using a kernel
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(smoothed, -1, kernel)

    # Convert back to RGB after enhancement
    enhanced_image = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)

    return enhanced_image


# Extract page as an image from a PDF using fitz
def extract_pdf_page(pdf_path, page_number):
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_number - 1)
    pix = page.get_pixmap(dpi=300)  # Use higher DPI for better quality
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return np.array(img)


# Function to remove table lines
def remove_table_lines(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply binary threshold to get binary image
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Remove horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    detected_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    contours, _ = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        cv2.drawContours(image, [c], -1, (255, 255, 255), 3)

    # Remove vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    detected_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    contours, _ = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        cv2.drawContours(image, [c], -1, (255, 255, 255), 3)

    return image


# Function to detect and remove vertical text
def remove_vertical_text(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply binary threshold to get binary image
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Detect vertical lines using morphological operations
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 100))
    detected_vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=1)

    # Find contours in the vertical regions
    contours, _ = cv2.findContours(detected_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = h / float(w)

        # Assume that vertical text has a high aspect ratio
        if aspect_ratio > 2:
            cv2.drawContours(image, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)

    return image


# Main function to remove table lines and vertical text
def process_image(image_path="", image_binary=""):
    if image_path:
    # Read image
        image = cv2.imread(image_path)
    else:
        image = image_binary

    # Step 1: Remove table lines
    image_without_lines = remove_table_lines(image.copy())

    # Step 2: Remove vertical text
    final_image = remove_vertical_text(image_without_lines.copy())

    # Save results
    #cv2.imwrite("image_without_lines.png", image_without_lines)
    #cv2.imwrite("final_image.png", final_image)
    return final_image


# Function to extract text from an image with a bit of blur
def extract_text_from_image(image_path,config="--oem 1 --psm 1",language="eng"):
    # Load the image
    #image = cv2.imread(image_path)
    image = image_path

    # Check if the image is loaded correctly
    if image is None:
        print(f"Failed to load image: {image_path}")
        return None

    # Apply a small amount of Gaussian blur to the image
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    # Convert the image to grayscale (Tesseract works better on grayscale images)
    gray_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)

    # Use Tesseract to extract text
    extracted_text = pytesseract.image_to_string(gray_image,config=config,lang=language)

    return extracted_text

try:
    file_path, language, config, pdf, page_num = pyperclip.paste().split("|")
    page_num = int(page_num)
    print(f"File-{file_path}\nLanguage-{language}\nConfig-{config}\nPDF\IMG - {pdf}")
    # Path to PDF file
    # file_path = r"C:\Users\s5562f\Robot\Attachments\doc15907120240816092734.pdf"
    # config = "--oem 1 --psm 1"
    # language = "lav"
    # pdf = "PDF"

    if pdf == "PDF":
        # Extract and process the PDF page
        image_bin = extract_pdf_page(file_path, page_num)
    else:
        image_bin = cv2.imread(file_path)

    # Enhance the image quality
    enhanced_image = enhance_image(image_bin)

    # Detect skew and rotate the image
    rotated_image = detect_and_rotate(enhanced_image)

    cleaned = process_image(image_binary=rotated_image)

    output_text = extract_text_from_image(cleaned, language=language, config=config)

    print(output_text)
    pyperclip.copy(output_text)
    # Save the original and rotated images
    # cv2.imwrite("extracted.png", cv2.cvtColor(page, cv2.COLOR_RGB2BGR))
    # cv2.imwrite("enhanced.png", cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR))
    # cv2.imwrite("result.png", cv2.cvtColor(cleaned, cv2.COLOR_RGB2BGR))
except Exception as e:
    pyperclip.copy(f"Python error - {str(e)}")


