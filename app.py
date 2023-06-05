import os
from flask import Flask, request, render_template, send_file
import cv2
import numpy as np
from tempfile import NamedTemporaryFile

app = Flask(__name__)

def remove_background(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a binary threshold or any other background removal technique
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours of the objects in the image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask with white pixels on a black background
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

    # Apply the mask to the image
    result = cv2.bitwise_and(image, image, mask=mask)

    # Load the orange background image
    background_path = 'orange_background.jpg'
    background = cv2.imread(background_path)

    # Resize the background image to match the size of the input image
    background = cv2.resize(background, (image.shape[1], image.shape[0]))

    # Apply the inverse mask to the background
    background = cv2.bitwise_and(background, background, mask=cv2.bitwise_not(mask))

    # Combine the image with the background
    final_result = cv2.add(result, background)

    return final_result


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Check if file is included in the request
        if 'file' not in request.files:
            return render_template('index.html', error='No file selected.')

        file = request.files['file']

        # Check if file is selected
        if file.filename == '':
            return render_template('index.html', error='No file selected.')

        # Save the uploaded file temporarily
        temp_file = NamedTemporaryFile(delete=False)
        temp_file_path = temp_file.name
        file.save(temp_file_path)

        # Perform background removal
        output_image = remove_background(temp_file_path)

        # Generate a unique filename for the output image
        output_filename = 'output.jpg'

        # Save the output image
        cv2.imwrite(output_filename, output_image)

        # Delete the temporary file
        os.remove(temp_file_path)

        # Return the download link to the output image
        return render_template('index.html', download_link=output_filename)

    return render_template('index.html')


@app.route('/download/<filename>', methods=['GET'])
def download(filename):
    return send_file(filename, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
