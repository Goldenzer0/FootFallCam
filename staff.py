
import cv2
import numpy as np

image_path = "staff_1.png"
reference_image = cv2.imread(image_path)

reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
reference_gray = cv2.GaussianBlur(reference_gray, (5,5),0)

#create feature detector and descriptor
detector = cv2.ORB_create()

#Extract keypoints and descriptors from the reference image
reference_keypoints, reference_descriptors = detector.detectAndCompute(reference_gray, None)

video = "sample.mp4"
cap = cv2.VideoCapture(video,0)

frame_count = 0  # Variable to keep track of the frame count
match_found = False

while cap.isOpened():
    # Read the current frame
    ret, frame = cap.read()

    # Check if frame reading was successful
    if ret:
        # Convert the frame to grayscale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compare the reference image with the current frame
        result = cv2.matchTemplate(frame_gray, reference_gray, cv2.TM_CCOEFF_NORMED)

        # Define a threshold for the match
        threshold = 0.6

        # Find the locations where the match exceeds the threshold
        locations = np.where(result >= threshold)
        
        # Check if any match is found
        if len(locations[0]) > 0:
            match_found = True

            # Get the coordinates of the matched area
            top_left = (locations[1][0], locations[0][0])
            bottom_right = (top_left[0] + reference_image.shape[1], top_left[1] + reference_image.shape[0])

            # Draw a rectangle around the matched area
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 3)
            print(f"The object appears in frame {frame_count}")
            print(f"The object is located in  {top_left},{bottom_right}")
            cv2.imshow('Matched Frame', frame)
            
        # Exit the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

    frame_count += 1

# Release the video capture object and close any open windows
cap.release()
cv2.destroyAllWindows()
