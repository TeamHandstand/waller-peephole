import cv2
import pandas as pd
import time
import os
import json
from datetime import datetime
import requests
import numpy as np
import ctypes

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

def log_state(logs_path, message):
    print(f"State: {message}")
    log_display(logs_path, 'Health Check', 0, message=message)

def log_display(logs_path, asset_name, duration, message=None):
    log_entry = {
        "asset_name": asset_name,
        "display_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "duration": duration,
        "message": message
    }
    if os.path.exists(logs_path):
        logs_df = pd.read_csv(logs_path)
        logs_df = pd.concat([logs_df, pd.DataFrame([log_entry])], ignore_index=True)
    else:
        logs_df = pd.DataFrame([log_entry])
    logs_df.to_csv(logs_path, index=False)

def send_slack_notification(message):
    payload = {"text": message}
    try:
        requests.post(SLACK_WEBHOOK_URL, json=payload)
    except requests.RequestException as e:
        print(f"Failed to send Slack notification: {e}")

def display_image(image_path, display_time, full_screen, show_timer):
    img = cv2.imread(image_path)
    if img is not None:
        screen_res = 1280, 720
        scale_width = screen_res[0] / img.shape[1]
        scale_height = screen_res[1] / img.shape[0]
        scale = min(scale_width, scale_height)
        window_width = int(img.shape[1] * scale)
        window_height = int(img.shape[0] * scale)

        cv2.namedWindow("Media Display", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Media Display", window_width, window_height)

        start_time = time.time()
        while time.time() - start_time < display_time:
            cv2.imshow("Media Display", img)
            if show_timer:
                elapsed_time = time.time() - start_time
                remaining_time = display_time - elapsed_time
                cv2.putText(img, f"Time left: {int(remaining_time)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return False
        return True
    else:
        log_error(f"Failed to load image: {image_path}")
        return False

def display_video(video_path, full_screen, show_timer):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log_error(f"Failed to open video: {video_path}")
        return False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Media Display", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            return False
    cap.release()
    return True

def log_error(message):
    print(f"Error: {message}")
    send_slack_notification(f"Error: {message}")

def main():


    base_path = "G:\My Drive\Waller Peephole"
    config_path = os.path.join(base_path, 'config.json')
    
    config = read_config(config_path)
    playlist_path = os.path.join(base_path, 'playlist.csv')
    logs_path = os.path.join(base_path, 'logs.csv')
    assets_dir = os.path.join(base_path, 'assets')
    current_index = 0




    # config_path = "config.json"
    # playlist_path = "playlist.csv"
    # logs_path = "logs.csv"
    # assets_dir = "assets"
    
    config = read_config(config_path)
    default_photo_duration = config.get("default_photo_duration", 60)
    media_is_full_screen = config.get("media_is_full_screen", False)
    shows_time_remaining = config.get("shows_time_remaining", False)

    log_state(logs_path, f"⚡️ System booted!")

    cv2.namedWindow("Media Display", cv2.WND_PROP_FULLSCREEN if media_is_full_screen else cv2.WINDOW_NORMAL)
    if media_is_full_screen:
        cv2.setWindowProperty("Media Display", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        hide_cursor()
    else:
        show_cursor()

    current_index = 0

    while True:
        try:
            playlist = read_playlist(playlist_path)
            if playlist.empty or current_index >= len(playlist):
                current_index = 0

            asset_name = playlist.iloc[current_index]['filename']
            asset_path = os.path.join(assets_dir, asset_name)

            start_time = time.time()

            if os.path.exists(asset_path):
                if asset_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
                    log_state(logs_path, f"📸 Beginning photo display: {asset_name}")
                    if not display_image(asset_path, default_photo_duration, media_is_full_screen, shows_time_remaining):
                        break
                    duration = default_photo_duration
                elif asset_name.lower().endswith(('.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv')):
                    log_state(logs_path, f"🎥 Beginning video display: {asset_name}")
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
        except Exception as e:
            log_error(f"Unexpected error in main loop: {e}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()