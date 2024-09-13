# 🚗 Real-Time Lane Detection Using OpenCV 🛣️

## Overview

This project implements **real-time lane detection** using Python and OpenCV. The system processes video frames to detect and visualize road lane lines, making it a useful tool for advanced driver assistance systems (ADAS) and autonomous vehicles.

## Features

- **Video Frame Processing**: Handles video streams, processing each frame individually.
- **Noise Reduction**: Applies Gaussian blur to smooth the image.
- **Edge Detection**: Uses Canny edge detection to find lane boundaries.
- **Region of Interest Masking**: Focuses on the road area where lanes are most likely to be.
- **Lane Detection**: Uses Hough Transform to identify lane lines.
- **Real-Time Visualization**: Overlays detected lane lines onto the original video feed.

## Getting Started

### Prerequisites

Make sure you have the following dependencies installed:

- Python 3.x
- OpenCV
- NumPy

You can install the required libraries using:

```bash
pip install opencv-python numpy
```

## Running the Project
**Clone the repository:**

```
git clone https://github.com/your-username/lane-detection-opencv.git
```

**Navigate to the project directory:**

```
cd lane-detection-opencv
```
**Run the script:**


```python lane_detection.py```

Provide your video file: Ensure the path to your video file is correctly set in the script. Replace 'road_video.mp4' with your video file path.

## Explanation
**1. Region of Interest Masking**
The region_of_interest function creates a mask that isolates the area of the image where lane lines are expected. This helps in focusing on the relevant part of the image and reduces noise from other areas.

**2. Edge Detection with Canny**
The Canny edge detector is applied to the blurred grayscale image to highlight the edges where lane lines are likely to be found.

**3. Lane Detection with Hough Transform**
The Hough Transform is used to detect straight lines in the edge-detected image. This helps in identifying the lane markings based on their linear characteristics.

**4. Drawing Detected Lanes**
Detected lane lines are drawn onto a blank image and then overlaid onto the original frame. This visualizes the lane lines in the context of the original video feed.

**5. Real-Time Frame Processing**
The script processes each frame from the video feed, detects lanes, and displays the processed video in real-time. Press q to exit the video display.

## Future Enhancements
**Curved Lane Detection:** Adapt the algorithm to handle curved lanes.
**Lane Departure Warning System:** Implement features to alert the driver if the vehicle drifts out of the lane.
**Object Detection Integration:** Combine lane detection with vehicle and obstacle detection for enhanced autonomous driving capabilities.

