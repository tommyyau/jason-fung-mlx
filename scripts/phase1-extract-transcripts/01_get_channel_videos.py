#!/usr/bin/env python3
"""
Extract all video URLs/IDs from a YouTube channel.

This script uses yt-dlp to extract all video information from a YouTube channel
and saves it to a JSON file for later processing.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict

try:
    import yt_dlp
except ImportError:
    print("Error: yt-dlp is not installed. Install it with: pip install yt-dlp")
    sys.exit(1)


def get_channel_videos(channel_url: str, output_path: str) -> List[Dict]:
    """
    Extract all video information from a YouTube channel.
    
    Args:
        channel_url: URL of the YouTube channel videos page
        output_path: Path to save the video list JSON file
    
    Returns:
        List of video dictionaries with video_id, url, and title
    """
    print(f"Fetching videos from channel: {channel_url}")
    
    # Configure yt-dlp options
    ydl_opts = {
        'quiet': False,
        'no_warnings': False,
        'extract_flat': True,  # Don't download, just get metadata
        'ignoreerrors': True,  # Continue on errors
    }
    
    videos = []
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract info about all videos in the channel
            info = ydl.extract_info(channel_url, download=False)
            
            if 'entries' in info:
                for entry in info['entries']:
                    if entry is None:
                        continue
                    
                    video_id = entry.get('id')
                    url = entry.get('url') or f"https://www.youtube.com/watch?v={video_id}"
                    title = entry.get('title', 'Unknown Title')
                    
                    video_info = {
                        'video_id': video_id,
                        'url': url,
                        'title': title
                    }
                    videos.append(video_info)
                    
                    print(f"Found: {title} ({video_id})")
            
            print(f"\nTotal videos found: {len(videos)}")
            
    except Exception as e:
        print(f"Error extracting videos: {e}")
        sys.exit(1)
    
    # Save to JSON file
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(videos, f, indent=2, ensure_ascii=False)
    
    print(f"\nVideo list saved to: {output_path}")
    return videos


def main():
    """Main function to run the script."""
    # Load configuration
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    config_path = project_root / 'config' / 'config.yaml'
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Warning: Could not load config.yaml: {e}")
        print("Using default values...")
        config = {
            'youtube': {'channel_url': 'https://www.youtube.com/@DrJasonFung/videos'},
            'paths': {'video_list': 'data/raw/video_list.json'}
        }
    
    channel_url = config.get('youtube', {}).get('channel_url', 'https://www.youtube.com/@DrJasonFung/videos')
    output_path = config.get('paths', {}).get('video_list', 'data/raw/video_list.json')
    
    # Make path absolute
    if not os.path.isabs(output_path):
        output_path = str(project_root / output_path)
    
    # Get videos
    videos = get_channel_videos(channel_url, output_path)
    
    print(f"\n✓ Successfully extracted {len(videos)} videos")
    return videos


if __name__ == '__main__':
    main()

