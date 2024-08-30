import cv2
import csv
import threading
import requests
import numpy as np
import ctypes
import pandas as pd
import time
import os
import json
import textwrap
from datetime import datetime
from pubnub.pnconfiguration import PNConfiguration
from pubnub.pubnub import PubNub
from pubnub.callbacks import SubscribeCallback
from pubnub.enums import PNStatusCategory, PNOperationType
from pubnub.exceptions import PubNubException

SLACK1 = "https://hooks.slack.com/services/"
SLACK2 = "T03782HMA/B074WAS3NPK"
SLACK3 = "/Ca7qSLPmFgSAj6mQdT7xuUmS"
SLACK_WEBHOOK_URL = SLACK1 + SLACK2 + SLACK3

# PubNub configuration
pnconfig = PNConfiguration()
pnconfig.subscribe_key = 'sub-c-dbb21446-f1f7-11e5-8679-02ee2ddab7fe'
pnconfig.publish_key = 'pub-c-833536ab-bc83-48ed-8270-9a96fefed6a4'
pnconfig.uuid = 'waller-peephole'

pubnub = PubNub(pnconfig)

# Global variable to store overlay message
overlay_message = None
overlay_duration = 0
overlay_start_time = 0

# Function to show overlay message
def show_overlay(frame, message):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)
    line_type = 2

    # Create an overlay with the same dimensions as the frame
    overlay = frame.copy()

    # Calculate the coordinates for the background rectangle
    top_left_corner = (0, frame.shape[0] // 6)
    bottom_right_corner = (frame.shape[1], 5 * frame.shape[0] // 6)

    # Draw black rectangle as background on the overlay
    cv2.rectangle(overlay, top_left_corner, bottom_right_corner, (0, 0, 0), -1)

    # Blend the overlay with the frame using addWeighted to get 80% opacity
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    # Calculate the maximum width for text
    max_text_width = int(frame.shape[1] * 0.8)

    # Wrap text
    wrapped_text = textwrap.fill(message, width=40)  # Adjust the width as needed

    # Calculate the size and position for each line of text
    y0, dy = (frame.shape[0] // 3), 30  # Starting y position and line spacing
    lines = wrapped_text.split('\n')
    for i, line in enumerate(lines):
        text_size, _ = cv2.getTextSize(line, font, font_scale, line_type)
        text_width, text_height = text_size
        text_x = (frame.shape[1] - text_width) // 2
        text_y = y0 + i * dy
        # Put each line of text on the frame
        cv2.putText(frame, line, (text_x, text_y), font, font_scale, font_color, line_type)

    return frame

# PubNub callback class
class MySubscribeCallback(SubscribeCallback):
    def message(self, pubnub, message):
        global overlay_message, overlay_duration, overlay_start_time
        overlay_message = message.message['message']
        overlay_duration = message.message['duration']
        overlay_start_time = time.time()

# Subscribe to the channel
pubnub.add_listener(MySubscribeCallback())
pubnub.subscribe().channels('waller-peephole').execute()

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

def log_state(logs_path,message):
    print(f"State: {message}")
    log_display(logs_path, 'Health Check', 0, message=message)

def log_display(logs_path, asset_name, duration, message=None):
    log_entry = {
        "asset_name": asset_name,
        "display_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "duration": duration,
        "message": message
    }

    # if os.path.exists(logs_path):
    #     logs_df = pd.read_csv(logs_path)
    #     logs_df = pd.concat([logs_df, pd.DataFrame([log_entry])], ignore_index=True)
    # else:
    #     logs_df = pd.DataFrame([log_entry])
    # logs_df.to_csv(logs_path, index=False)

    
    log_exists = os.path.exists(logs_path)
    
    with open(logs_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=log_entry.keys())
        if not log_exists:
            writer.writeheader()
        writer.writerow(log_entry)

def log_health_check(log_file_path, active_name):
    log_display(log_file_path, asset_name="Health Check", duration=0, message=f"❤️ Alive and well. Currently showing: {active_name}")
    threading.Timer(300, log_health_check, args=(log_file_path, active_name)).start()

def send_slack_notification(message):
    payload = {"text": message}
    try:
        requests.post(SLACK_WEBHOOK_URL, json=payload)
    except requests.RequestException as e:
        print(f"Failed to send Slack notification: {e}")

# Display image function with overlay
def display_image(asset_path, duration, full_screen, show_timer):
    global overlay_message, overlay_duration, overlay_start_time
    img = cv2.imread(asset_path)
    start_time = time.time()

    # Set up full-screen display
    cv2.namedWindow('Media Display', cv2.WND_PROP_FULLSCREEN)
    if full_screen:
        cv2.setWindowProperty('Media Display', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        elapsed_time = time.time() - start_time
        if elapsed_time > duration:
            break
        if overlay_message and time.time() - overlay_start_time < overlay_duration:
            img = show_overlay(img, overlay_message)
        cv2.imshow('Media Display', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False

    return True

# def display_image(image_path, display_time, full_screen, show_timer):
#     img = cv2.imread(image_path)
#     if img is not None:
#         screen_res = (1920, 1080)
#         scale_width = screen_res[0] / img.shape[1]
#         scale_height = screen_res[1] / img.shape[0]
#         scale = min(scale_width, scale_height)
#         window_width = int(img.shape[1] * scale)
#         window_height = int(img.shape[0] * scale)
#         img = cv2.resize(img, (window_width, window_height))
#         y_offset = (screen_res[1] - window_height) // 2
#         x_offset = (screen_res[0] - window_width) // 2
#         black_screen = np.zeros((screen_res[1], screen_res[0], 3), dtype=np.uint8)
#         black_screen[y_offset:y_offset + window_height, x_offset:x_offset + window_width] = img
        
#         start_time = time.time()
#         while time.time() - start_time < display_time:
#             frame = black_screen.copy()
#             if show_timer:
#                 remaining_time = int(display_time - (time.time() - start_time))
#                 timer_text = f"Next in: {remaining_time}s"
#                 cv2.putText(frame, timer_text, (screen_res[0] - 200, screen_res[1] - 30), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
#             cv2.imshow("Media Display", frame)
#             if cv2.waitKey(1000) & 0xFF == ord('q'):
#                 return False
#     else:
#         log_error(f"Failed to load image: {image_path}")
#     return True

def display_video(asset_path, full_screen, show_timer):
    global overlay_message, overlay_duration, overlay_start_time
    cap = cv2.VideoCapture(asset_path)

    # Set up full-screen display
    cv2.namedWindow('Media Display', cv2.WND_PROP_FULLSCREEN)
    if full_screen:
        cv2.setWindowProperty('Media Display', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if overlay_message and time.time() - overlay_start_time < overlay_duration:
                frame = show_overlay(frame, overlay_message)
            cv2.imshow('Media Display', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                return False
        else:
            break

    cap.release()
    return True

# def display_video(video_path, full_screen, show_timer):
#     cap = cv2.VideoCapture(video_path)
#     if cap.isOpened():
#         screen_res = (1920, 1080)
#         black_screen = np.zeros((screen_res[1], screen_res[0], 3), dtype=np.uint8)
        
#         # Get the total duration of the video
#         total_duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))
        
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             scale_width = screen_res[0] / frame.shape[1]
#             scale_height = screen_res[1] / frame.shape[0]
#             scale = min(scale_width, scale_height)
#             window_width = int(frame.shape[1] * scale)
#             window_height = int(frame.shape[0] * scale)
#             frame = cv2.resize(frame, (window_width, window_height))
#             y_offset = (screen_res[1] - window_height) // 2
#             x_offset = (screen_res[0] - window_width) // 2
#             black_screen.fill(0)
#             black_screen[y_offset:y_offset + window_height, x_offset:x_offset + window_width] = frame

#             if show_timer:
#                 # Calculate the remaining time
#                 current_time = int(cap.get(cv2.CAP_PROP_POS_FRAMES) / cap.get(cv2.CAP_PROP_FPS))
#                 remaining_time = total_duration - current_time
#                 timer_text = f"Next in: {remaining_time}s"
#                 cv2.putText(black_screen, timer_text, (screen_res[0] - 200, screen_res[1] - 30), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
#             cv2.imshow("Media Display", black_screen)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 return False
#         cap.release()
#     else:
#         log_error(f"Failed to load video: {video_path}")
#     return True

def log_error(message):
    print(f"Error: {message}")
    log_display('logs.csv', 'Error', 0, message=message)
    send_slack_notification(message)

def hide_cursor():
    # ctypes.windll.user32.ShowCursor(False)
    print('hiding')

def show_cursor():
    # ctypes.windll.user32.ShowCursor(True)
    print('showing')

def main():
    base_path = os.path.dirname(os.path.abspath(__file__))
    # base_path = "G:\My Drive\Waller Peephole"
    config_path = os.path.join(base_path, 'config.json')
    config = read_config(config_path)
    playlist_path = os.path.join(base_path, 'playlist.csv')
    logs_path = os.path.join(base_path, 'logs.csv')
    assets_dir = os.path.join(base_path, 'assets')
    current_index = 0

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

    playlist = read_playlist(playlist_path)
    if not playlist.empty:
        asset_friendly_name = playlist.iloc[0]['title']
    else:
        asset_friendly_name = "No assets"

    log_health_check(logs_path, asset_friendly_name)

    while True:
        playlist = read_playlist(playlist_path)
        if playlist.empty or current_index >= len(playlist):
            current_index = 0

        try:
            asset_name = playlist.iloc[current_index]['filename']
            asset_friendly_name = playlist.iloc[current_index]['title']
        except KeyError as e:
            log_error(f"Playlist key error: {e}")
            break

        asset_path = os.path.join(assets_dir, asset_name)

        start_time = time.time()

        if os.path.exists(asset_path):
            if asset_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
                log_state(logs_path, f"📸 Beginning photo display: {asset_friendly_name}")
                if not display_image(asset_path, default_photo_duration, media_is_full_screen, shows_time_remaining):
                    break
            elif asset_name.lower().endswith(('.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv')):
                log_state(logs_path, f"🎥 Beginning video display: {asset_friendly_name}")
                if not display_video(asset_path, media_is_full_screen, shows_time_remaining):
                    break
            else:
                log_error(f"Unsupported file type: {asset_name}")
        else:
            log_error(f"File not found: {asset_name}")

        current_index += 1

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()