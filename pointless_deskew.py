import cv2
import math
from math import floor
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray, rgba2rgb
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks
from typing import Optional, Tuple, List, Dict, Any
from PIL import Image
from statistics import median_low
import tempfile
from transformers import BertTokenizer
import os
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import torch
import streamlit as st
import time





ImageType = np.ndarray


def visualize_edges(edges: np.ndarray):
    """
    Displays the edges detected in an image as a grayscale plot.
    
    This function takes a 2D numpy array representing the edges detected in an image, 
    which is the output of the edge detection algorithm "Canny edge detector", 
    and visualizes it using matplotlib. The visualization highlights the edges detected 
    by displaying them in white against a black background, facilitating the assessment 
    of the edge detection process's effectiveness.
    
    Parameters:
    - edges (np.ndarray): A 2D numpy array where the value of each pixel represents the 
      presence (values greater than 0) or absence (value of 0) of an edge at that pixel.
    
    Returns:
    - None: This function does not return a value. It displays the visualization directly.
    
    """
    
    plt.imshow(edges, cmap='gray')
    plt.title("Edges detected")
    plt.show()


def visualize_hough_lines(original_image: np.ndarray, accumulator: np.ndarray, angles: List[float], distances: List[float], angle_peaks, dist_peaks):
    """
    Visualizes the original image alongside the image with detected Hough lines superimposed.
    
    This function takes an image and the results of a Hough transform, including the accumulator,
    the angles, and distances arrays, as well as the peaks in these arrays that correspond to the most 
    significant lines detected in the image. It then plots two images side by side: the original image and 
    the original image with the detected Hough lines superimposed in red. This is useful for 
    understanding the effect of the Hough transform in the line detection task and for verifying the detected 
    lines against the original image.
    
    Parameters:
    - original_image (np.ndarray): The original grayscale image as a 2D numpy array.
    - accumulator (np.ndarray): The Hough accumulator array resulting from the Hough transform.
    - angles (List[float]): A list of angles (in radians) used in the Hough transform.
    - distances (List[float]): A list of distances used in the Hough transform.
    - angle_peaks (np.ndarray): The angles (in radians) corresponding to the most significant lines detected.
    - dist_peaks (np.ndarray): The distances corresponding to the most significant lines detected.
    
    Returns:
    - None: This function does not return a value. It displays the visualization using matplotlib.
    """
    
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    # Original Image
    ax[0].imshow(original_image, cmap='gray')
    ax[0].set_title('Original Image')
    ax[0].set_axis_off()

    # Image with Hough lines
    ax[1].imshow(original_image, cmap='gray')
    for angle, dist in zip(angle_peaks, dist_peaks):
        (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
        ax[1].plot((x0 - 1000 * np.sin(angle), x0 + 1000 * np.sin(angle)),
                   (y0 + 1000 * np.cos(angle), y0 - 1000 * np.cos(angle)), '-r')
    ax[1].set_xlim((0, original_image.shape[1]))
    ax[1].set_ylim((original_image.shape[0], 0))
    ax[1].set_title('Image with Hough lines')
    ax[1].set_axis_off()

    plt.tight_layout()
    plt.show()


def visualize_hough_space(accumulator: np.ndarray, angles: np.ndarray, distances: np.ndarray):
    """
    Enhanced visualization of the Hough space (accumulator space) of the Hough Transform.

    Args:
        accumulator (np.ndarray): The accumulator array from the Hough Transform.
        angles (np.ndarray): The array of angles used in the Hough Transform.
        distances (np.ndarray): The array of distances used in the Hough Transform.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    angle_degrees = np.rad2deg(angles)  # Convert angles to degrees for easier interpretation

    # Using a colormap that highlights peaks more clearly
    cax = ax.imshow(accumulator, cmap='hot', aspect='auto',
                    extent=[angle_degrees[0], angle_degrees[-1], distances[-1], distances[0]])

    # Adding color bar for better understanding of intensity
    fig.colorbar(cax, ax=ax, label='Accumulator Counts')

    ax.set_title('Hough Space')
    ax.set_xlabel('Angles (degrees)')
    ax.set_ylabel('Distances (pixels)')
    
    # Adding grid for easier value reading
    ax.grid(True, color='blue', linestyle='-.', linewidth=0.5, alpha=0.5)

    plt.show()


def visualize_image_and_hough_space(img_gray: np.ndarray, accumulator: np.ndarray, angles: np.ndarray, distances: np.ndarray):
    """
    Visualize the original image and its corresponding Hough space side by side.

    Args:
        img_gray (np.ndarray): The original grayscale image.
        accumulator (np.ndarray): The accumulator array from the Hough Transform.
        angles (np.ndarray): The array of angles used in the Hough Transform.
        distances (np.ndarray): The array of distances used in the Hough Transform.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Original Image
    axes[0].imshow(img_gray, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].set_axis_off()

    # Hough Space
    angle_degrees = np.rad2deg(angles)  # Convert angles to degrees
    cax = axes[1].imshow(accumulator, cmap='hot', aspect='auto', 
                         extent=[angle_degrees[0], angle_degrees[-1], distances[-1], distances[0]])
    axes[1].set_title('Hough Space')
    axes[1].set_xlabel('Angles (degrees)')
    axes[1].set_ylabel('Distances (pixels)')
    fig.colorbar(cax, ax=axes[1], label='Accumulator Counts')
    axes[1].grid(True, color='blue', linestyle='-.', linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    plt.show()


def convert_to_grayscale(image: ImageType) -> ImageType:
    """
    Convert an input image to grayscale.

    This function first checks if the input image is in RGBA format (4 channels). If it is,
    it converts the image to RGB format. Then, it converts the RGB or the original image 
    (if it wasn't RGBA) to grayscale. If the input image is already in grayscale or has less 
    than 3 channels, it returns the image as-is.

    Args:
        image (ImageType): An image array. The image can be in RGBA, RGB, or grayscale format.

    Returns:
        ImageType: The grayscale version of the input image.

    """
    # Convert RGBA to RGB if the image has 4 channels.
    if image.shape[-1] == 4:
        image = rgba2rgb(image)

    # Convert to grayscale.
    img_gray = rgb2gray(image) if image.ndim == 3 else image
    
    return img_gray


def perform_hough_transform(img_gray: np.ndarray, sigma: float, num_angles: int) -> Tuple[np.ndarray, List[float], List[float]]:
    """
    Perform the Hough Transform on a grayscale image to detect lines.

    This function first applies the Canny edge detection algorithm to the input grayscale image. 
    Then, it performs the Hough Line Transform on the detected edges. The Hough Transform is used 
    to detect lines in the image, represented in the Hough space (accumulator array) with corresponding 
    angles and distances.

    Args:
        img_gray (np.ndarray): A grayscale image array.
        sigma (float): The standard deviation of the Gaussian filter used in the Canny edge detector.
        num_angles (int): The number of angles to consider in the Hough Transform. More angles provide 
                          finer angular resolution.

    Returns:
        Tuple[np.ndarray, List[float], List[float]]: A tuple containing:
            - The accumulator array from the Hough Transform.
            - A list of angles (in radians) corresponding to the peaks in the accumulator array.
            - A list of distances (in pixel units) corresponding to the peaks in the accumulator array.
            - Edges

    Visualization Suggestion:
        - To visualize the edges, use `plt.imshow(edges, cmap='gray')`.
        - To visualize the results of the Hough Transform, create a function that overlays the detected 
          lines on the original image or displays them in the Hough space (accumulator array).
    """

    # Apply Canny edge detection to the grayscale image
    edges = canny(img_gray, sigma=sigma)

    # Perform the Hough Line Transform on the detected edges
    accumulator, angles, distances = hough_line(edges, np.linspace(-np.pi / 2, np.pi / 2, num_angles, endpoint=False))

    # Convert angles and distances to lists and return them along with the accumulator
    return accumulator, angles.tolist(), distances.tolist(), edges


def filter_and_correct_angles(angles_peaks: List[float], min_angle: float, max_angle: float) -> List[float]:
    """
    Filters and corrects a list of angle peaks to ensure they fall within a specified range.
    
    This function performs two main operations on the input list of angles: correction and filtering.
    The correction step adjusts each angle by adding π/4, then modulo π/2, and subtracting π/4 again. 
    This effectively rotates the angles within a π/2 range centered around 0. The purpose of this 
    correction is to normalize the angles, ensuring that they are within a standard range for further processing.
    
    After correction, the function filters out any angles that do not fall within the specified minimum 
    and maximum angle range. This step ensures that only angles of interest, as defined by the min_angle 
    and max_angle parameters, are retained for further analysis or use.
    
    Parameters:
    - angles_peaks (List[float]): A list of angles (in radians) to be corrected and filtered.
    - min_angle (float): The minimum allowable angle (in radians) after correction.
    - max_angle (float): The maximum allowable angle (in radians) after correction.

    Returns:
    - List[float]: A list of corrected angles that fall within the specified min and max angle range.
    """
    
    # Correct the angles by normalizing them within a π/2 range centered around 0
    corrected_angles = [((angle + np.pi / 4) % (np.pi / 2) - np.pi / 4) for angle in angles_peaks]
    
    # Filter the corrected angles to retain only those within the specified min and max range
    corrected_angles = [angle for angle in corrected_angles if min_angle <= angle <= max_angle]
    
    return corrected_angles


def calculate_frequency_of_angles(angles_peaks: List[float]) -> Dict[float, int]:
    """
    Calculates the frequency of each unique angle peak in a list.
    
    This function iterates through the list of angle peaks and determines the frequency of each unique angle,
    providing a dictionary where the keys are the unique angles and the values are the counts (frequencies) of
    those angles in the input list. 
    
    Parameters:
    - angles_peaks (List[float]): A list of angles (in radians), where each angle is a floating-point number.
    
    Returns:
    - Dict[float, int]: A dictionary where each key is a unique angle from the input list and each value is the
      frequency of that angle in the list.
    
    Example:
    >>> calculate_frequency_of_angles([0.0, 0.5, 0.0])
    {0.0: 2, 0.5: 1}
    
    Note:
    The function does not distinguish between angles that are numerically close but not exactly equal; each unique
    value in the input list is treated as a distinct angle. Consequently, the precision of the input angles directly
    affects the output frequency distribution.
    """
    
    return {peak: angles_peaks.count(peak) for peak in angles_peaks}


def determine_skew_angle(freqs: Dict[float, int]) -> Optional[float]:
    """
    Determines the most frequent skew angle from a dictionary of angle frequencies.
    
    This function identifies the angle with the highest frequency in a given dictionary where keys represent angle values 
    and values represent the frequency of those angles. The angle with the highest frequency is considered the most 
    frequent skew angle. If the input dictionary is empty, the function returns None, indicating that no skew angle 
    can be determined from an empty dataset.
    
    Parameters:
    - freqs (Dict[float, int]): A dictionary with angles as keys (float) and their frequencies as values (int).
    
    Returns:
    - Optional[float]: The angle with the highest frequency or None if the input dictionary is empty. This return 
      value is of type Optional[float] to accommodate the possibility of an empty input.
    
    Example:
    >>> determine_skew_angle({0.0: 2, 0.5: 3, -0.5: 1})
    0.5
    """

    return max(freqs, key=freqs.get) if freqs else None


def rotate_image(image: Any, angle: float) -> Any:
    """
    Rotates an image by a specified angle, adjusting the image's dimensions to fit the rotated content and filling 
    the corners with white to maintain the aesthetic integrity of the image.
    
    Rotating an image around its center without cropping the rotated image or leaving out any part of it requires 
    calculating the new image dimensions that can fully encompass the rotated image. This process involves:
    - Computing the angle in radians for trigonometric calculations.
    - Determining the new width and height of the image based on the rotation angle to ensure that the entire 
      image content is visible post-rotation.
    - Calculating the center of the original image to set the pivot point for rotation.
    - Creating a rotation matrix that defines the rotation parameters including the center, angle, and scale.
    - Adjusting the translation part of the rotation matrix to ensure the rotated image is centered within the 
      new dimensions.
    - Filling the corners, which do not contain image data after rotation, with white to provide a visually 
      seamless appearance.
    
    This complexity arises because rotating an image is not simply about pivoting its pixels; it involves 
    spatial transformation, requiring adjustments to both the content and the canvas size to ensure the entire 
    image is correctly oriented and fully visible without distortion or unwanted cropping.
    
    Parameters:
    - image (Any): The input image to be rotated. The type here is generic to accommodate various image representations.
    - angle (float): The angle in degrees by which to rotate the image clockwise.
    
    Returns:
    - Any: The rotated image with its dimensions adjusted to fit the entire content and corners filled with white.
    
    """
    
    # Calculate new image dimensions to ensure the whole image is visible after rotation
    old_width, old_height = image.shape[1], image.shape[0]
    angle_radian = math.radians(angle)
    new_width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    new_height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)
    
    # Find the center of the original image to use as the pivot for rotation
    image_center = np.array(image.shape[1::-1]) / 2
    
    # Generate the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(tuple(image_center), angle, 1.0)
    
    # Adjust the translation component of the rotation matrix
    rotation_matrix[1, 2] += (new_width - old_width) / 2
    rotation_matrix[0, 2] += (new_height - old_height) / 2
    
    # Determine the border color based on the image type
    if len(image.shape) == 3 and image.shape[2] == 3:  # Color image
        borderValue = (255, 255, 255)  # White for BGR color images
    else:  # Grayscale image
        borderValue = (255,)  # White for grayscale images
    
    # Convert new size dimensions to integers as expected by warpAffine
    new_size = (int(round(new_width)), int(round(new_height)))
    
    # Perform the rotation
    rotated_image = cv2.warpAffine(image, rotation_matrix, new_size, borderValue=borderValue)
    
    return rotated_image


def determine_skew(image: ImageType, sigma: float = 1.0, num_peaks: int = 20, min_deviation: float = 0.1, plot_visualization: bool = False) -> Tuple[Optional[float], np.ndarray, np.ndarray]:
    """
    Detects the skew angle of an image after performing a Hough transform to find lines in the image.
    
    The function first converts the image to grayscale and then applies a Hough transform to detect lines. 
    It filters these lines to focus on a specific range of angles (-π/8 to π/8 radians) and calculates the 
    frequency of occurrence of each angle within this range. The most frequent angle (skew angle) is considered 
    the dominant line orientation, which can be indicative of the image's skew. Optionally, this function can 
    also plot visualizations of the edges detected, the Hough transform space, and the detected lines on the 
    original image for analysis and debugging purposes.
    
    Parameters:
    - image (Any): The input image on which skew detection will be performed.
    - sigma (float, optional): The standard deviation of the Gaussian filter used in edge detection, defaulting to 2.0.
    - num_peaks (int, optional): The number of peaks to identify in the Hough transform, defaulting to 20.
    - min_deviation (float, optional): The minimum deviation for angle calculations, affecting the granularity of the analysis, defaulting to 1.0 degree.
    - plot_visualization (bool, optional): If True, visualizations of the processing steps will be displayed, defaulting to False.
    
    Returns:
    - Tuple[Optional[float], np.ndarray, np.ndarray]: A tuple containing the detected skew angle in degrees (or None if not determined), 
      the array of angles considered peaks in the Hough transform, and the array of corrected angles within the desired range.
    
    Note:
    The accuracy of the skew detection depends on the quality of the image, the appropriateness of the sigma value for 
    edge detection, and the specified range for considering angle peaks. Adjusting these parameters may be necessary 
    for optimal skew detection in different images.
    """

    # Convert image to grayscale
    img_gray = convert_to_grayscale(image)
    # Determine the number of angles to analyze based on the minimum deviation
    num_angles = round(180 / min_deviation)
    # Perform Hough transform on the grayscale image
    accumulator, angles, distances, edges = perform_hough_transform(img_gray, sigma, num_angles)
    # Find peaks in the Hough transform
    _, angles_peaks, dists_peaks = hough_line_peaks(accumulator, np.array(angles), np.array(distances), num_peaks=num_peaks)
    # Correct and filter angle peaks
    corrected_angles = filter_and_correct_angles(angles_peaks, -np.pi/4, np.pi/4)

    # Optional visualization of the process
    if plot_visualization:
        visualize_edges(edges)  # Visualize detected edges in the image
        visualize_image_and_hough_space(img_gray, accumulator, angles, distances)  # Visualize the Hough space
        visualize_hough_lines(img_gray, accumulator, angles, distances, angles_peaks, dists_peaks)  # Visualize Hough lines on the image
    
    # Calculate the frequency of corrected angles and determine the skew angle
    freqs = calculate_frequency_of_angles(corrected_angles)
    skew_angle = determine_skew_angle(freqs)
    # Convert the skew angle to degrees
    skew_angle_deg = np.rad2deg(skew_angle) if skew_angle is not None else None
    
    return skew_angle_deg, angles_peaks, corrected_angles

 
def process_image(image_path: str, plot_visualization: bool = True, image_scale_factor: float = 0.5):
    """
    Processes an image to detect and correct skew, then saves the corrected image.
    
    This function performs several steps to process an image, including loading, resizing,
    skew detection, skew correction, and optionally visualizing the corrected image. The image is first
    resized based on a given scale factor to optimize the skew detection process. Then, the skew of the
    resized image is detected. If a significant skew is identified, the original image is rotated to correct
    this skew and the corrected image is saved to the specified output path. Optionally, the corrected image
    can be displayed using matplotlib.

    Parameters:
    - image_path (str): The path to the input image to be processed.
    - output_path (str): The path where the corrected image should be saved.
    - plot_visualization (bool, optional): If True, displays the corrected image using matplotlib. Defaults to True.
    - image_scale_factor (float, optional): The factor by which the image should be scaled for processing. Defaults to 0.5.

    Returns:
    - tuple: Returns a tuple containing the detected skew angle (or None if not detected), the angles considered
      as peaks in the Hough transform, and the corrected angles within the desired range.

    """

    # Load the image from the specified path
    image = cv2.imread(image_path)

    # Calculate the new dimensions for resizing
    width = int(image.shape[1] * image_scale_factor)
    height = int(image.shape[0] * image_scale_factor)
    new_dim = (width, height)

    # Resize the image to the new dimensions
    resized_image = cv2.resize(image, new_dim, interpolation=cv2.INTER_AREA)
    
    # Determine the skew of the resized image and get the skew angle and peaks
    skew_angle, angles_peaks, corrected_angles = determine_skew(resized_image, plot_visualization=plot_visualization)
    
    # If a skew angle is detected, correct the skew by rotating the original image
    if skew_angle is not None:
        # Rotate the original image to correct the skew
        #rotated_image = rotate_image(image, skew_angle)
        rotated_image = Image.open(image_path).rotate(skew_angle, expand=True, fillcolor="white") # quality needs to be tested
        
        # Save the corrected image to the specified output path
        #rotated_image.save(output_path)
        #cv2.imwrite(output_path, rotated_image)
        skew_angle = round(skew_angle, 2)
        st.write(f"Hough thinks the angle is {skew_angle}°")
        print(f"Detected angle is {skew_angle}")
        #print(f"Rotated image saved to {output_path}")

        # Optionally, display the corrected image using matplotlib
        if plot_visualization:
            plt.imshow(rotated_image)
            plt.axis('off')
            plt.show()
    else:
        print("No significant skew detected.")
        
    return skew_angle, angles_peaks, corrected_angles, rotated_image


def crop_center_vertically(image, height=200):
    """
    Crops the center portion of an image vertically to a specified height while maintaining the original width.
    This function calculates the vertical center of the image and crops the image to the specified height from
    this center point. The width of the image remains unchanged.

    Args:
        image (PIL.Image.Image): The image to be cropped. This should be an instance of a PIL Image.
        height (int, optional): The height of the crop in pixels. Defaults to 200 pixels. If the specified height
            is greater than the image height, the original image height is used, resulting in no vertical cropping.

    Returns:
        PIL.Image.Image: A new image object representing the vertically cropped image. This image has the same width
            as the original image and a height as specified by the `height` parameter, unless the original image
            is shorter, in which case the original height is preserved.
    """
    # Get the dimensions of the original image
    img_width, img_height = image.size
    
    # Calculate the top coordinate to start cropping from, ensuring it's centered vertically
    top = (img_height - height) // 2
    # Calculate the bottom coordinate by adding the desired height to the top coordinate
    bottom = top + height
    
    # Crop the image from the calculated top to bottom while keeping the full width
    # The crop box is defined as (left, upper, right, lower)
    return image.crop((0, top, img_width, bottom))


def crop_center(image, width=200, height=200):
    """
    Crops the center portion of an image to a specified width and height.
    This function calculates the center of the image and crops the image to the specified width and height
    from this center point.

    Args:
        image (PIL.Image.Image): The image to be cropped. This should be an instance of a PIL Image.
        width (int, optional): The width of the crop in pixels. Defaults to 200 pixels. If the specified width
            is greater than the image width, the original image width is used, resulting in no horizontal cropping.
        height (int, optional): The height of the crop in pixels. Defaults to 200 pixels. If the specified height
            is greater than the image height, the original image height is used, resulting in no vertical cropping.

    Returns:
        PIL.Image.Image: A new image object representing the cropped image. This image has the width and height
            as specified by the `width` and `height` parameters, unless the original image is smaller in either
            dimension, in which case the original dimension is preserved.
    """
    # Get the dimensions of the original image
    img_width, img_height = image.size
    
    # Calculate the center and start coordinates for both vertical and horizontal cropping
    left = max((img_width - width) // 2, 0)
    top = max((img_height - height) // 2, 0)
    
    # Calculate the right and bottom coordinates by adding the desired width and height to the left and top coordinates
    right = left + width
    bottom = top + height
    
    # Ensure the right and bottom do not exceed the image's dimensions
    right = min(right, img_width)
    bottom = min(bottom, img_height)
    
    # Crop the image from the calculated coordinates
    # The crop box is defined as (left, upper, right, lower)
    return image.crop((left, top, right, bottom))


def bert_tokenizer_score_list_of_words(words: List[str], confidences: List[float], tokenizer: BertTokenizer) -> float:
    """
    Computes a score for a list of words based on tokenization metrics using a BERT tokenizer,
    while also considering the confidence of each word and the total number of words.

    Args:
        words (List[str]): The list of words to be scored. Each word is a string.
        confidences (List[float]): The list of confidence scores corresponding to each word.
        tokenizer (BertTokenizer): An instance of BertTokenizer used for tokenizing the words.

    Returns:
        float: An average score for the list of words, where a lower score indicates a higher
            likelihood of the content being meaningful. The score is influenced by the presence
            of unknown tokens, the average length of subtokens, the proportion of known
            subtokens, and the confidence levels of the words.
    """

    if not words:  # Check if the list of words is empty
        return 1000  # Return a high penalty for empty inputs
         
    total_score = 0  # Initialize total score
    total_confidence_penalty = 0  # Initialize total confidence penalty
    total_word_length = 0
    
    for word, confidence in zip(words, confidences):  # Iterate over each word and its confidence
        tokens = tokenizer.tokenize(word)  # Tokenize the current word

        if not tokens:  # Check if the word could not be tokenized at all
            total_score += 100  # Apply a heavy penalty for untokenizable words
            continue

        # Update total word length
        total_word_length += len(word)
        
        # Initialize score for the current word
        score = 0
        # Count known tokens, ignoring '[UNK]' which stands for unknown tokens
        known_token_count = sum(1 for token in tokens if token != '[UNK]')
        # Calculate the proportion of known tokens to total tokens
        known_token_proportion = known_token_count / len(tokens) if tokens else 0
        
        # Calculate average subtoken length, excluding '[UNK]', '[CLS]', and '[SEP]' tokens
        avg_subtoken_length = sum(len(token) for token in tokens if token not in ['[UNK]', '[CLS]', '[SEP]']) / len(tokens) if tokens else 0
        
        # Apply heuristic adjustments based on tokenization results
        if '[UNK]' in tokens:  # Penalize the presence of unknown tokens
            score += 20
        score += (2 - known_token_proportion * 5)  # Reward higher proportions of known tokens
        score += (2 - avg_subtoken_length)  # Reward longer average subtoken lengths
        
        if len(tokens) > 1 and known_token_proportion == 1:
            score -= 2  # Lesser penalty for fully known compound words

        # Adjust score based on word confidence
        confidence_penalty = (1 - confidence) * 10  # Scale confidence penalty
        total_confidence_penalty += confidence_penalty

        total_score += score + confidence_penalty  # Update total score with the score for this word
    
    # Modify scoring to favor more and longer words
    avg_word_length = total_word_length / len(words) if words else 0
    words_score_bonus = len(words) ** 1.5  # Exponential bonus for more words
    length_score_bonus = avg_word_length ** 2  # Exponential bonus for longer average word length

    # Incorporate bonuses into the average score calculation
    adjusted_score = (total_score + total_confidence_penalty - words_score_bonus - length_score_bonus) / max(1, len(words))

    return adjusted_score

def normalize_scores(scores):
    min_score = min(scores.values())
    max_score = max(scores.values())
    normalized_scores = {key: 1 - ((value - min_score) / (max_score - min_score)) for key, value in scores.items()}
    sum_normalized_scores = sum(normalized_scores.values())
    adjusted_scores = {key: value / sum_normalized_scores for key, value in normalized_scores.items()}
    return adjusted_scores
    
def analyze_ocr_results(docs: List[dict], tokenizer: BertTokenizer) -> Tuple[int, dict]:
    """
    Analyzes OCR results from multiple document orientations to determine the best orientation.
    Now also considers the confidence of each word in the OCR results.

    Args:
        docs (List[dict]): List of OCR result documents for different orientations.
        tokenizer (BertTokenizer): An instance of BertTokenizer used for tokenizing the words.

    Returns:
        Tuple[int, dict]: The index of the best orientation and the OCR results document for that orientation.
    """
    scores = {}
    best_score = float('inf')
    best_index = -1

    for index, doc in enumerate(docs):
        list_of_words = []
        confidences = []
        for block in doc['blocks']:
            for line in block['lines']:
                for word_info in line['words']:
                    if len(word_info['value'])>1:
                        list_of_words.append(word_info['value'])
                        confidences.append(word_info['confidence'])
        # Adjust the function call to include confidences
        score = bert_tokenizer_score_list_of_words(list_of_words, confidences, tokenizer)
        scores[f"orientation {index}"] = score
        print(f"Score for orientation {index}: {score}")
        if score < best_score:
            best_score = score
            best_index = index

    # Normalize and adjust scores so their sum equals 1
    adjusted_scores = normalize_scores(scores)
    
    # Return both the index of the best orientation and the OCR results for that orientation
    return best_index, adjusted_scores


def orientation_rotation_estimation(img: Image, predictor: ocr_predictor, tokenizer: BertTokenizer):
    """
    Estimates the orientation of an image by rotating it to several angles, applying OCR,
    and determining the best orientation based on OCR results and tokenization metrics.

    The function rotates the input image to 0, 90, -90, and 180 degrees, crops the center
    vertically for each rotation, and saves these variants as temporary files. It then
    processes each variant with an OCR predictor, analyzes the OCR results to estimate the
    most probable correct orientation of the image, and finally returns the estimated angle
    and the rotated image in this orientation.

    Args:
        img (Image): The input image to estimate orientation for. This should be a PIL Image instance.
        predictor (ocr_predictor): An OCR model predictor capable of processing images to extract text.
        tokenizer (BertTokenizer): A BERT tokenizer used for analyzing OCR results to estimate the best orientation.

    Returns:
        tuple: A tuple containing the estimated angle (as an integer from the set [0, 90, -90, 180])
               and the rotated image in the estimated correct orientation (as a PIL Image).
    """
    # Define potential rotation angles
    angles = [0, 90, -90, 180]
    temp_files = []  # Store paths to temporary files for OCR processing

    start_time = time.time()
    # Rotate, crop, and save each rotated image variant
    for angle in angles:
        cropped_image = img.rotate(angle, expand=True, fillcolor="white")  # Rotate with angle, filling background with white
        #cropped_image = crop_center_vertically(rotated_image, 300)  # Crop the rotated image to focus on the center

        # Convert the cropped image to RGB if it's in RGBA mode to avoid the OSError when saving as JPEG
        if cropped_image.mode == 'RGBA':
            cropped_image = cropped_image.convert("RGB")

        # Save the cropped image to a temporary file for OCR processing
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        cropped_image.save(temp_file.name)
        temp_files.append(temp_file)
    end_time = time.time()
    print(f"Execution time rotate and save: {end_time - start_time} seconds")

    start_time = time.time()
    # Apply OCR on each cropped and rotated image, storing results
    ocr_results = []
    for temp_file in temp_files:
        page = DocumentFile.from_images(temp_file.name)[0]  # Load image for OCR
        result = predictor([page]).export()["pages"][0]  # Process image with OCR and get results
        ocr_results.append(result)
    end_time = time.time()
    print(f"Execution time ocr: {end_time - start_time} seconds")
    
    # Analyze OCR results with tokenizer to find best rotation
    best_index, scores_normalized = analyze_ocr_results(ocr_results, tokenizer)

    # Cleanup temporary files
    for temp_file in temp_files:
        temp_file.close()  # Close the file
        os.unlink(temp_file.name)  # Delete the file

    # Determine and print the estimated best angle for the original image
    estimated_angle = angles[best_index]
    estimated_angle = round(estimated_angle, 2)
    print(f"Estimated angle: {estimated_angle}")
    st.write(f"Robert aka Rotation Bert corrects Houghs estimage by {estimated_angle}°")
    
    # Rotate the original image to the estimated best orientation
    rotated_image = img.rotate(estimated_angle, expand=True, fillcolor="white")

    # Return the estimated angle and the rotated image
    return estimated_angle, rotated_image, ocr_results, scores_normalized
