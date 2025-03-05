from datasets import load_dataset
import os
import requests
from huggingface_hub import login

login(token="your_token")
save_dir = "../../datasets/webvid_videos"
os.makedirs(save_dir, exist_ok=True)
train = load_dataset("TempoFunk/webvid-10M", split="train")
train.to_csv("../../datasets/webvid_10M_train.csv")
val = load_dataset("TempoFunk/webvid-10M", split="validation")
val.to_csv("../../datasets/webvid_10M_val.csv")

for entry in train.select(range(2000)):
    video_id = entry["videoid"]  # Get the videoid
    video_url = entry["contentUrl"]  # Get the video URL
    video_filename = f"{video_id}.mp4"
    video_path = os.path.join(save_dir, video_filename)

    print(f"Downloading {video_url} as {video_filename}...")

    response = requests.get(video_url, stream=True)
    with open(video_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    #if os.listdir(save_dir).__len__() >= 5:
     #   break

print("Download complete!")