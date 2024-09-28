import cv2
import numpy as np

def canny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def region_of_interest(img):
    height, width = img.shape[:2]
    mask = np.zeros_like(img)
    
    # Define a region that is proportional to the frame size
    polygon = np.array([[
        (int(0.1 * width), height),  # Bottom left
        (int(0.9 * width), height),  # Bottom right
        (int(0.6 * width), int(0.6 * height)),  # Top right
        (int(0.4 * width), int(0.6 * height))   # Top left
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def houghLines(img):
    lines = cv2.HoughLinesP(img, rho=2, theta=np.pi/180, threshold=100, minLineLength=40, maxLineGap=5)
    return lines

def display_lines(img, lines):
    line_img = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 10)
    return img

def make_points(img, line):
    slope, intercept = line
    height = img.shape[0]
    y1 = height
    y2 = int(height * 0.6)  # Line ends at 60% of the image height
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [x1, y1, x2, y2]

def average_slope_intercept(img, lines):
    left_fit = []
    right_fit = []
    
    if lines is None:
        return None
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope, intercept = np.polyfit((x1, x2), (y1, y2), 1)
            if slope < 0:  # Left lane
                left_fit.append((slope, intercept))
            else:  # Right lane
                right_fit.append((slope, intercept))
    
    if left_fit:
        left_fit_average = np.average(left_fit, axis=0)
        left_line = make_points(img, left_fit_average)
    else:
        left_line = None
    
    if right_fit:
        right_fit_average = np.average(right_fit, axis=0)
        right_line = make_points(img, right_fit_average)
    else:
        right_line = None
    
    return [left_line, right_line]

def display_lines_average(img, lines):
    line_img = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            if line is not None:
                x1, y1, x2, y2 = line
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 10)
    return img

cap = cv2.VideoCapture('test1.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    
    canny_out = canny(frame)
    masked_output = region_of_interest(canny_out)
    lines = houghLines(masked_output)

    average_lines = average_slope_intercept(frame, lines)
    line_img = display_lines_average(frame, average_lines)

    cv2.imshow("Lane Detection", line_img)
    
    if cv2.waitKey(10) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

