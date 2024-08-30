import subprocess
import os
import pandas as pd

def read_playlist(playlist_path):
    try:
        playlist = pd.read_csv(playlist_path)
        if 'filename' not in playlist.columns:
            raise KeyError("The playlist CSV file must contain a 'filename' column.")
        return playlist
    except Exception as e:
        log_error(f"Error reading playlist: {e}")
        return pd.DataFrame()

def log_error(message):
    print(f"Error: {message}")
    # You can add your logging functionality here.

def play_video_vlc(asset_path):
    # Path to VLC executable
    vlc_path = r"C:\Program Files\VideoLAN\VLC\vlc.exe"

    # VLC command to play the video on an infinite loop
    command = [vlc_path, asset_path, "--loop"]

    # Open VLC and play the video, wait for it to finish
    subprocess.run(command)

def main():
    base_path = os.path.dirname(os.path.abspath(__file__))
    playlist_path = os.path.join(base_path, 'playlist.csv')
    assets_dir = os.path.join(base_path, 'assets')

    playlist = read_playlist(playlist_path)

    if not playlist.empty:
        try:
            asset_name = playlist.iloc[0]['filename']  # Only the first file
            asset_friendly_name = playlist.iloc[0]['title']
        except KeyError as e:
            log_error(f"Playlist key error: {e}")
            return

        asset_path = os.path.join(assets_dir, asset_name)

        if os.path.exists(asset_path):
            if asset_name.lower().endswith(('.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv')):
                log_state(f"🎥 Beginning infinite loop video display: {asset_friendly_name}")
                play_video_vlc(asset_path)
            else:
                log_error(f"Unsupported file type: {asset_name}")
        else:
            log_error(f"File not found: {asset_name}")
    else:
        log_error("Playlist is empty or not found.")

def log_state(message):
    print(f"State: {message}")
    # You can add your state logging functionality here.

if __name__ == "__main__":
    main() 