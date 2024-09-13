import cv2
import numpy as np

def draw_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), 10)

    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img


def ROI(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask,vertices,255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def process(img):
    height = img.shape[0]
    width = img.shape[1]

    # Convert image to gray
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    blur = cv2.GaussianBlur(src=gray,ksize=(5,5),sigmaX=0)

    # Detect edges using canny
    canny = cv2.Canny(blur,threshold1=50,threshold2=150)

    # Define ROI
    vertices = np.array([[
        (int(width * 0.1), height),   # Bottom-left corner (10% from the left edge)
        (int(width * 0.55), int(height * 0.55)),  # Peak of the triangle (middle of the image, 60% down the height)
        (width, height)    # Bottom-right corner (10% from the right edge)
    ]], dtype=np.int32)
    cropped_image = ROI(img=canny,vertices=vertices)

    # Detect lines using Hough transform
    lines = cv2.HoughLinesP(image=cropped_image,rho=2,theta=np.pi/180,threshold=100,lines=np.array([]),minLineLength=40,maxLineGap=5)

    # Draw detected lines on the original image
    result = draw_lines(img,lines)

    return result,cropped_image
cap = cv2.VideoCapture("road.mp4")

while cap.isOpened():
    ret,frame = cap.read()
    if ret:
        # Process the frame to detect lanes
        lane_detected,roi = process(frame)

        lane_detected = cv2.resize(lane_detected,(720,480))
        roi = cv2.resize(roi,(720,480))

        # Display the lane detected frame
        cv2.imshow("Lane Detection",lane_detected)
        cv2.imshow("ROI",roi)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()

