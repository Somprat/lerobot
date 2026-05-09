from datasets import Dataset
import cv2
import numpy as np
import os


import numpy as np

def make_fake_state_action(num_steps):
    states = []
    actions = []

    for _ in range(num_steps):
        state = np.random.randn(7)
        action = np.random.randn(7)

        states.append(state)
        actions.append(action)

    return states, actions


data = []
num_episodes = 4

for ep in range(num_episodes):
    frames_dir = f"frames/video{ep+1}"
    frames = sorted(os.listdir(frames_dir))

    states, actions = make_fake_state_action(len(frames))

    for t, frame_name in enumerate(frames):
        img = cv2.imread(os.path.join(frames_dir, frame_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        data.append({
            "observation.image.camera_0": img,
            "observation.state": states[t].astype(np.float32),
            "action": actions[t].astype(np.float32),
            "episode_index": ep,
            "frame_index": t
        })
    print("Episode done!")

data = Dataset.from_list(data)
print(data)
data.push_to_hub("SompratPWG/TestLerobot")