import cv2
import pandas as pd
import time
import os
import json
from datetime import datetime
import requests
import numpy as np

SLACK1 = "https://hooks.slack.com/services/"
SLACK2 = "T03782HMA/B074WAS3NPK"
SLACK3 = "/Ca7qSLPmFgSAj6mQdT7xuUmS"
SLACK_WEBHOOK_URL = SLACK1 + SLACK2 + SLACK3

def read_config(config_path):
    try:
        with open(config_path, 'r') as config_file:
            return json.load(config_file)
    except Exception as e:
        log_error(str(e))
        return {"default_photo_duration": 60, "media_is_full_screen": False, "shows_time_remaining": False}

def read_playlist(playlist_path):
    try:
        playlist = pd.read_csv(playlist_path)
        if 'filename' not in playlist.columns:
            raise KeyError("The playlist CSV file must contain a 'filename' column.")
        return playlist
    except Exception as e:
        log_error(f"Error reading playlist: {e}")
        return pd.DataFrame()

def log_display(logs_path, asset_name, duration, error=None):
    log_entry = {
        "asset_name": asset_name,
        "display_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "duration": duration,
        "error": error
    }
    if os.path.exists(logs_path):
        logs_df = pd.read_csv(logs_path)
        logs_df = pd.concat([logs_df, pd.DataFrame([log_entry])], ignore_index=True)
    else:
        logs_df = pd.DataFrame([log_entry])
    logs_df.to_csv(logs_path, index=False)
    if error:
        send_slack_notification(error)

def send_slack_notification(message):
    payload = {"text": message}
    try:
        requests.post(SLACK_WEBHOOK_URL, json=payload)
    except requests.RequestException as e:
        print(f"Failed to send Slack notification: {e}")

def display_image(image_path, display_time, full_screen, show_timer):
    img = cv2.imread(image_path)
    if img is not None:
        screen_res = (1920, 1080)
        scale_width = screen_res[0] / img.shape[1]
        scale_height = screen_res[1] / img.shape[0]
        scale = min(scale_width, scale_height)
        window_width = int(img.shape[1] * scale)
        window_height = int(img.shape[0] * scale)
        img = cv2.resize(img, (window_width, window_height))
        y_offset = (screen_res[1] - window_height) // 2
        x_offset = (screen_res[0] - window_width) // 2
        black_screen = np.zeros((screen_res[1], screen_res[0], 3), dtype=np.uint8)
        black_screen[y_offset:y_offset + window_height, x_offset:x_offset + window_width] = img
        
        start_time = time.time()
        while time.time() - start_time < display_time:
            frame = black_screen.copy()
            if show_timer:
                remaining_time = int(display_time - (time.time() - start_time))
                timer_text = f"Next in: {remaining_time}s"
                cv2.putText(frame, timer_text, (screen_res[0] - 200, screen_res[1] - 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Media Display", frame)
            if cv2.waitKey(1000) & 0xFF == ord('q'):
                return False
    else:
        log_error(f"Failed to load image: {image_path}")
    return True

def display_video(video_path, full_screen, show_timer):
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        screen_res = (1920, 1080)
        black_screen = np.zeros((screen_res[1], screen_res[0], 3), dtype=np.uint8)
        
        # Get the total duration of the video
        total_duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            scale_width = screen_res[0] / frame.shape[1]
            scale_height = screen_res[1] / frame.shape[0]
            scale = min(scale_width, scale_height)
            window_width = int(frame.shape[1] * scale)
            window_height = int(frame.shape[0] * scale)
            frame = cv2.resize(frame, (window_width, window_height))
            y_offset = (screen_res[1] - window_height) // 2
            x_offset = (screen_res[0] - window_width) // 2
            black_screen.fill(0)
            black_screen[y_offset:y_offset + window_height, x_offset:x_offset + window_width] = frame

            if show_timer:
                # Calculate the remaining time
                current_time = int(cap.get(cv2.CAP_PROP_POS_FRAMES) / cap.get(cv2.CAP_PROP_FPS))
                remaining_time = total_duration - current_time
                timer_text = f"Next in: {remaining_time}s"
                cv2.putText(black_screen, timer_text, (screen_res[0] - 200, screen_res[1] - 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Media Display", black_screen)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return False
        cap.release()
    else:
        log_error(f"Failed to load video: {video_path}")
    return True

def log_error(message):
    print(f"Error: {message}")
    log_display('logs.csv', 'Error', 0, error=message)

def main():
    config_path = 'config.json'
    config = read_config(config_path)
    playlist_path = 'playlist.csv'
    logs_path = 'logs.csv'
    assets_dir = 'assets'
    current_index = 0

    default_photo_duration = config.get("default_photo_duration", 60)
    media_is_full_screen = config.get("media_is_full_screen", False)
    shows_time_remaining = config.get("shows_time_remaining", False)

    cv2.namedWindow("Media Display", cv2.WND_PROP_FULLSCREEN if media_is_full_screen else cv2.WINDOW_NORMAL)
    if media_is_full_screen:
        cv2.setWindowProperty("Media Display", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        playlist = read_playlist(playlist_path)
        if playlist.empty or current_index >= len(playlist):
            current_index = 0

        try:
            asset_name = playlist.iloc[current_index]['filename']
        except KeyError as e:
            log_error(f"Playlist key error: {e}")
            break

        asset_path = os.path.join(os.getcwd(), assets_dir, asset_name)

        start_time = time.time()

        if os.path.exists(asset_path):
            if asset_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
                if not display_image(asset_path, default_photo_duration, media_is_full_screen, shows_time_remaining):
                    break
                duration = default_photo_duration
            elif asset_name.lower().endswith(('.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv')):
                if not display_video(asset_path, media_is_full_screen, shows_time_remaining):
                    break
                duration = time.time() - start_time
            else:
                log_error(f"Unsupported file type: {asset_name}")
                duration = 0
        else:
            log_error(f"File not found: {asset_name}")
            duration = 0

        log_display(logs_path, asset_name, duration)
        current_index += 1

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
