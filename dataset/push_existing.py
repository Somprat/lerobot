"""Push an already-built local LeRobot dataset to the Hub.

Use this after running ``create_data.py`` if the push step failed (e.g. 401
because no HF token was set). Logs in with whichever token is in the local
HF cache; run ``hf auth login`` first if needed.
"""

from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

REPO_ID = "Begench02/tritonDroids"
LOCAL_ROOT = Path.home() / ".cache" / "huggingface" / "lerobot" / REPO_ID


def main() -> None:
    if not LOCAL_ROOT.exists():
        raise SystemExit(
            f"Local dataset not found at {LOCAL_ROOT}. Run create_data.py first."
        )
    ds = LeRobotDataset(REPO_ID, root=LOCAL_ROOT)
    print(f"Pushing {REPO_ID} from {LOCAL_ROOT}")
    ds.push_to_hub(tags=["lerobot", "smolvla", "sample"])
    print(f"Done. https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
