import pandas as pd
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import os
import requests
df = pd.read_csv("../../datasets/webvid_10M_train.csv")
save_dir = "../../datasets/webvid_videos"
os.makedirs(save_dir, exist_ok=True)

df = df.iloc[:2000]
save_dir = "../../datasets/webvid_videos"

def download_video(entry):
    video_id = entry["videoid"]
    video_url = entry["contentUrl"]
    video_filename = f"{video_id}.mp4"
    video_path = os.path.join(save_dir, video_filename)

    if os.path.exists(video_path):
        print(f"Skipping {video_filename}, already exists.")
        return


    try:
        response = requests.get(video_url, stream=True, timeout=10)
        response.raise_for_status()
        with open(video_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {video_filename}")
    except Exception as e:
        print(f"Failed to download {video_filename}: {e}")
entries = df.to_dict(orient="records")

max_workers = 10
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    executor.map(download_video, entries)

print("Download complete!")