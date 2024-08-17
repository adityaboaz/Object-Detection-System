# YOLO Object Detection Flask Application

This project is a web application that performs real-time object detection using YOLO models with Flask. The application supports streaming from local files, YouTube URLs, and webcam input. It also allows for model selection and configuration through a web interface.

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Setup](#setup)
- [Running the Application](#running-the-application)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Features
- Real-time object detection using YOLOv8.
- Stream video from local files, YouTube URLs, and webcam.
- Configure detection settings (confidence, preview, and flipping).
- View detection results and accuracy metrics.

## Requirements
- Python 3.8+
- Flask
- OpenCV
- NumPy
- YOLOv8 (via `ultralytics`)
- `yt_dlp` for YouTube video extraction
- Flask-SocketIO

You can install the necessary Python packages with:

```bash
pip install -r requirements.txt
Setup
Clone the Repository

bash
Copy code
git clone https://github.com/adityaboaz/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME
Create a Virtual Environment (optional but recommended)

bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
Install Requirements

bash
Copy code
pip install -r requirements.txt
Download YOLO Models
Ensure you have the YOLOv8 model files (yolov8x.pt, yolov8n.pt) in the project directory or modify the paths in app.py to point to the correct locations.

Running the Application
To start the Flask application, run:

bash
Copy code
python app.py
The application will start on http://127.0.0.1:5000/ by default.

Usage
Homepage: Visit http://127.0.0.1:5000/ to access the homepage.
Index Page: Enter a video URL or file path and submit to start streaming.
Video Feed: View real-time video and object detection on http://127.0.0.1:5000/video_feed.
Results Page: Check detection accuracy and statistics on http://127.0.0.1:5000/results.
Contributing
If you have suggestions or improvements, feel free to submit a pull request or open an issue on the GitHub repository.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Feel free to replace placeholders like YOUR_REPOSITORY_NAME and adjust the instructions based on the actual structure and requirements of your project.

markdown
Copy code

### Notes
- **Update Repository Name**: Make sure to replace `YOUR_REPOSITORY_NAME` with your actual repository name.
- **YOLO Model Paths**: If the YOLO models are stored elsewhere, update the paths accordingly.
- **Dependencies**: Make sure your `requirements.txt` file includes all the dependencies your project needs.

This template provides a structured approach to documenting how to set up and run your project, making it easier for others to use and contribute.





