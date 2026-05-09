"""Convert local frames into a proper LeRobot-format dataset.

Smoke-test data only: state/action are random — there is no real teleoperation log.
Reads frames from ``dataset/frames/video[1-4]/`` and writes a LeRobot dataset
under ``$HF_LEROBOT_HOME/{REPO_ID}`` (default ``~/.cache/huggingface/lerobot``).
"""

from pathlib import Path

import cv2
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

REPO_ID = "Begench02/TestLerobotSample3cam"
FPS = 10
NUM_EPISODES = 4
STATE_DIM = 6
ACTION_DIM = 6
IMG_H, IMG_W = 256, 256
TASK = "sample task"
CAMERA_KEYS = (
    "observation.images.camera1",
    "observation.images.camera2",
    "observation.images.camera3",
)

FRAMES_ROOT = Path(__file__).parent / "frames"


def natural_sort_key(name: str) -> tuple[int, str]:
    digits = "".join(c for c in name.split("frame_")[-1].split(".")[0] if c.isdigit())
    return (int(digits) if digits else 0, name)


def main() -> None:
    features = {
        cam: {
            "dtype": "video",
            "shape": (IMG_H, IMG_W, 3),
            "names": ["height", "width", "channel"],
        }
        for cam in CAMERA_KEYS
    }
    features |= {
        "observation.state": {
            "dtype": "float32",
            "shape": (STATE_DIM,),
            "names": [f"s{i}" for i in range(STATE_DIM)],
        },
        "action": {
            "dtype": "float32",
            "shape": (ACTION_DIM,),
            "names": [f"a{i}" for i in range(ACTION_DIM)],
        },
    }

    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=FPS,
        features=features,
        robot_type="sample",
        use_videos=True,
    )

    rng = np.random.default_rng(0)

    for ep in range(NUM_EPISODES):
        ep_dir = FRAMES_ROOT / f"video{ep + 1}"
        frame_files = sorted(ep_dir.iterdir(), key=lambda p: natural_sort_key(p.name))
        print(f"Episode {ep}: {len(frame_files)} frames from {ep_dir}")

        for frame_path in frame_files:
            bgr = cv2.imread(str(frame_path))
            if bgr is None:
                raise RuntimeError(f"Failed to read {frame_path}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)

            frame = {cam: rgb for cam in CAMERA_KEYS}
            frame["observation.state"] = rng.standard_normal(STATE_DIM).astype(np.float32)
            frame["action"] = rng.standard_normal(ACTION_DIM).astype(np.float32)
            frame["task"] = TASK
            dataset.add_frame(frame)

        dataset.save_episode()
        print(f"  saved episode {ep}")

    dataset.finalize()
    print(f"\nDone. Dataset written to: {dataset.root}")


if __name__ == "__main__":
    main()
