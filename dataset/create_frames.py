import cv2
import os

def video_to_frames(video_path, out_dir, max_frames = 10):


    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    i = 0
    while i < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(f"{out_dir}/video_4_frame_{i+1}.png", frame)
        if i+1%10 == 0:
            print(f"Saving frame {i+1}")
        i+=1
    cap.release()
base_dir = os.getcwd()
joined_path = os.path.join(base_dir, "TestData/video4.mov")


video_to_frames(joined_path,"frames/video4")
print(joined_path)