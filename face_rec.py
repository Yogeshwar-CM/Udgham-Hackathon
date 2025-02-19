import os
import cv2
from deepface import DeepFace

# Get applicant name as user input
applicant_name = input("Enter applicant name: ").strip()

# Define folder path for applicant's images
APPLICANT_FOLDER = f"APPLICANT_PROFILE/{applicant_name}"

# Ensure the folder exists
if not os.path.exists(APPLICANT_FOLDER):
    print(f"❌ No folder found for {applicant_name}! Please add images.")
    exit()
else:
    print(f"✅ Folder found: {APPLICANT_FOLDER}")

# Capture live image from webcam with real-time display
def capture_image():
    cam = cv2.VideoCapture(0)  # Open webcam
    cv2.namedWindow("Live Capture - Press SPACE to capture, ESC to exit")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("❌ Failed to open webcam.")
            break

        cv2.imshow("Live Capture - Press SPACE to capture, ESC to exit", frame)
        key = cv2.waitKey(1)

        if key == 27:  # Press 'Esc' to exit without capturing
            print("❌ Capture cancelled.")
            cam.release()
            cv2.destroyAllWindows()
            return None
        elif key == 32:  # Press 'Space' to capture the image
            image_path = "live_capture.jpg"
            cv2.imwrite(image_path, frame)
            print("✅ Image captured successfully!")
            cam.release()
            cv2.destroyAllWindows()
            return image_path

# Face verification function
def verify_face():
    test_image_path = capture_image()  # Capture a live image

    if test_image_path is None:
        return False

    # Loop through stored images in the applicant folder
    for file in os.listdir(APPLICANT_FOLDER):
        if file.endswith((".jpg", ".jpeg", ".png")):
            stored_image_path = os.path.join(APPLICANT_FOLDER, file)

            try:
                # Compare faces using DeepFace
                result = DeepFace.verify(test_image_path, stored_image_path, model_name='Facenet512', distance_metric='cosine')

                if result["verified"] and result["distance"] < 0.3:
                    print(f"✅ Face Verified! Welcome, {applicant_name}.")
                    return True

            except Exception as e:
                print(f"⚠️ Error comparing images: {e}")

    print("❌ Face NOT Verified!")
    return False
# Run face verification
if verify_face():
    print("✅ You may proceed to the HR interview.")
else:
    print("❌ Verification failed. Access denied.")
