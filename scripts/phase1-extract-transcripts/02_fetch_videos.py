#!/usr/bin/env python3
"""
Fetch video transcripts and metadata from Supadata API.

This script reads video URLs from the video list and fetches transcripts
and metadata using the Supadata API with rate limiting and retry logic.
"""

import json
import os
import sys
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def load_config() -> Dict:
    """Load configuration from config.yaml"""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    config_path = project_root / 'config' / 'config.yaml'
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Warning: Could not load config.yaml: {e}")
        return {}


def fetch_transcript(api_key: str, video_url: str, base_url: str) -> tuple[Optional[str], Optional[Dict]]:
    """
    Fetch transcript for a video using Supadata API.
    
    Args:
        api_key: Supadata API key
        video_url: YouTube video URL
        base_url: Supadata API base URL
    
    Returns:
        Tuple of (transcript_text, metadata_dict) or (None, None) if failed
    """
    url = f"{base_url}/transcript"
    headers = {
        'x-api-key': api_key
    }
    
    params = {
        'url': video_url
    }
    
    try:
        # Use 8 second timeout - shorter for faster failure on membership videos
        response = requests.get(url, params=params, headers=headers, timeout=(3, 8))  # (connect timeout, read timeout)
        
        # Check for membership/private video before processing
        if is_membership_or_private_video(response):
            return None, None
        
        response.raise_for_status()
        data = response.json()
        
        # Extract metadata
        metadata = {}
        if 'lang' in data:
            metadata['language'] = data['lang']
        if 'availableLangs' in data:
            metadata['available_languages'] = data['availableLangs']
        
        # Extract transcript text
        transcript_text = None
        if 'content' in data and isinstance(data['content'], list):
            # Combine all text segments
            transcript_parts = []
            for segment in data['content']:
                if 'text' in segment:
                    transcript_parts.append(segment['text'])
            transcript_text = ' '.join(transcript_parts)
            
            # Calculate duration from transcript segments
            if len(data['content']) > 0:
                last_segment = data['content'][-1]
                if 'offset' in last_segment and 'duration' in last_segment:
                    metadata['duration'] = last_segment['offset'] + last_segment['duration']
        elif 'transcript' in data:
            transcript_text = data['transcript']
        elif 'text' in data:
            transcript_text = data['text']
        elif isinstance(data, str):
            transcript_text = data
        
        if transcript_text is None:
            return None, None  # Will trigger retry
        
        return transcript_text, metadata if metadata else None
            
    except (requests.exceptions.Timeout, requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout):
        # Timeout - will trigger retry in fetch_video_data
        return None, None
    except requests.exceptions.RequestException:
        # Other request errors - will trigger retry
        return None, None
    except Exception:
        # Any other errors - will trigger retry
        return None, None


def fetch_metadata(api_key: str, video_url: str, base_url: str) -> Optional[Dict]:
    """
    Fetch metadata for a YouTube video using Supadata API.
    Note: Supadata doesn't have a separate metadata endpoint, so we extract
    what we can from the transcript response or return None.
    
    Args:
        api_key: Supadata API key
        video_url: YouTube video URL
        base_url: Supadata API base URL
    
    Returns:
        Metadata dictionary or None if failed
    """
    # Try to get additional info from transcript endpoint
    url = f"{base_url}/transcript"
    headers = {
        'x-api-key': api_key
    }
    
    params = {
        'url': video_url
    }
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Extract metadata from transcript response
        metadata = {}
        if 'lang' in data:
            metadata['language'] = data['lang']
        if 'availableLangs' in data:
            metadata['available_languages'] = data['availableLangs']
        
        # Calculate duration from transcript segments if available
        if 'content' in data and isinstance(data['content'], list) and len(data['content']) > 0:
            last_segment = data['content'][-1]
            if 'offset' in last_segment and 'duration' in last_segment:
                metadata['duration'] = last_segment['offset'] + last_segment['duration']
        
        return metadata if metadata else None
        
    except requests.exceptions.RequestException as e:
        # Metadata is optional, so we don't fail if this doesn't work
        return None


def is_membership_or_private_video(response) -> bool:
    """
    Check if video is membership-only or private based on API response.
    
    Args:
        response: requests.Response object
    
    Returns:
        True if video appears to be membership/private, False otherwise
    """
    if response.status_code != 200:
        # Check error messages for membership/private indicators
        try:
            error_data = response.json()
            error_text = str(error_data).lower()
            if any(keyword in error_text for keyword in ['membership', 'private', 'unavailable', 'restricted', 'premium']):
                return True
        except:
            pass
        
        # Check response text
        response_text = response.text.lower()
        if any(keyword in response_text for keyword in ['membership', 'private', 'unavailable', 'restricted', 'premium']):
            return True
    
    # Check if response is empty or has no content (membership videos might return empty)
    try:
        data = response.json()
        if 'content' in data and (not data['content'] or len(data['content']) == 0):
            # Empty content might indicate membership video
            return True
    except:
        pass
    
    return False


def fetch_video_data(
    api_key: str,
    video_info: Dict,
    base_url: str,
    rate_limit_delay: float,
    max_retries: int = 2,  # Reduced to 2 attempts for membership videos
    retry_wait: int = 10
) -> Optional[Dict]:
    """
    Fetch both transcript and metadata for a video with retry logic.
    Optimized to make only one API call per video.
    
    Args:
        api_key: Supadata API key
        video_info: Dictionary with video_id, url, title
        base_url: Supadata API base URL
        rate_limit_delay: Seconds to wait between requests (not used here, kept for compatibility)
        max_retries: Maximum number of retry attempts (default: 3)
        retry_wait: Seconds to wait between retries (default: 10)
    
    Returns:
        Dictionary with video data including transcript and metadata, or None if failed
    """
    video_url = video_info['url']
    video_id = video_info['video_id']
    
    for attempt in range(max_retries):
        try:
            # Fetch transcript and metadata in one API call
            transcript, metadata = fetch_transcript(api_key, video_url, base_url)
            
            if transcript is None:
                # Check if this might be a membership/private video by making a direct check
                if attempt == 0:  # Only check on first attempt
                    try:
                        check_url = f"{base_url}/transcript"
                        check_headers = {'x-api-key': api_key}
                        check_params = {'url': video_url}
                        check_response = requests.get(check_url, params=check_params, headers=check_headers, timeout=10)
                        
                        if is_membership_or_private_video(check_response):
                            print(f"   🔒 Detected membership/private video. Skipping immediately.")
                            return None
                    except:
                        pass  # If check fails, continue with normal retry
                
                if attempt < max_retries - 1:
                    print(f"   ⚠️  Attempt {attempt + 1}/{max_retries} failed. Retrying in {retry_wait}s...")
                    time.sleep(retry_wait)
                    continue
                else:
                    print(f"   ❌ Failed after {max_retries} attempts. Skipping this video.")
                    return None
            
            # Success! Combine data
            result = {
                'video_id': video_id,
                'url': video_url,
                'title': video_info.get('title', 'Unknown Title'),
                'transcript': transcript,
                'fetched_at': datetime.utcnow().isoformat() + 'Z'
            }
            
            # Add metadata fields if available
            if metadata:
                result.update({
                    'language': metadata.get('language', 'en'),
                    'available_languages': metadata.get('available_languages', []),
                    'duration': metadata.get('duration', 0),
                    'metadata': metadata  # Keep full metadata object
                })
            
            return result
            
        except requests.exceptions.Timeout:
            # Timeout might indicate membership video - try once more then skip
            if attempt < max_retries - 1:
                print(f"   ⚠️  Timeout on attempt {attempt + 1}/{max_retries}. Retrying in {retry_wait}s...")
                time.sleep(retry_wait)
                continue
            else:
                print(f"   ⏭️  Timeout after {max_retries} attempts (may be membership video). Skipping.")
                return None
        except Exception as e:
            error_msg = str(e).lower()
            # Check if error indicates membership/private video
            if any(keyword in error_msg for keyword in ['membership', 'private', 'unavailable', 'restricted', '403', '404']):
                print(f"   🔒 Detected membership/private video from error. Skipping immediately.")
                return None
            
            if attempt < max_retries - 1:
                print(f"   ⚠️  Error on attempt {attempt + 1}/{max_retries}: {str(e)[:50]}... Retrying in {retry_wait}s...")
                time.sleep(retry_wait)
                continue
            else:
                print(f"   ⏭️  Error after {max_retries} attempts. Skipping this video.")
                return None
    
    return None


def load_existing_transcripts(output_path: str) -> tuple[set, int]:
    """
    Load existing video IDs from transcripts file to skip already fetched videos.
    Also detects and removes incomplete/broken transcripts.
    
    Returns:
        Tuple of (set of valid video IDs, count of removed broken entries)
    """
    existing_ids = set()
    broken_entries = []
    valid_entries = []
    removed_count = 0
    
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    
                    try:
                        data = json.loads(line)
                        video_id = data.get('video_id')
                        transcript = data.get('transcript', '')
                        
                        # Check if transcript is valid (not empty, not None, has reasonable length)
                        if not video_id:
                            broken_entries.append(line_num)
                            continue
                        
                        if not transcript or len(transcript.strip()) < 50:
                            # Transcript is missing or too short (likely incomplete)
                            broken_entries.append(line_num)
                            removed_count += 1
                            print(f"   ⚠️  Found incomplete transcript for {video_id} (will re-fetch)")
                            continue
                        
                        # Valid entry
                        existing_ids.add(video_id)
                        valid_entries.append(line)
                        
                    except json.JSONDecodeError:
                        # Invalid JSON line
                        broken_entries.append(line_num)
                        removed_count += 1
                        print(f"   ⚠️  Found broken JSON on line {line_num} (will remove)")
            
            # If we found broken entries, rewrite the file with only valid entries
            if broken_entries:
                print(f"\n   🔧 Removing {removed_count} incomplete/broken entries...")
                with open(output_path, 'w', encoding='utf-8') as f:
                    for entry in valid_entries:
                        f.write(entry)
                print(f"   ✅ Cleaned transcripts file: {len(valid_entries)} valid entries retained")
            
        except Exception as e:
            print(f"Warning: Could not read existing transcripts: {e}")
    
    return existing_ids, removed_count


def main():
    """Main function to fetch all video transcripts and metadata."""
    # Load configuration
    config = load_config()
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Get API key
    api_key = os.getenv('SUPADATA_API_KEY')
    if not api_key:
        print("Error: SUPADATA_API_KEY not found in environment variables.")
        print("Please create a .env file with your Supadata API key.")
        sys.exit(1)
    
    # Get paths from config
    video_list_path = config.get('paths', {}).get('video_list', 'data/raw/video_list.json')
    transcripts_path = config.get('paths', {}).get('transcripts', 'data/transcripts/transcripts.jsonl')
    
    # Make paths absolute
    if not os.path.isabs(video_list_path):
        video_list_path = str(project_root / video_list_path)
    if not os.path.isabs(transcripts_path):
        transcripts_path = str(project_root / transcripts_path)
    
    # Get API settings
    base_url = config.get('supadata', {}).get('base_url', 'https://api.supadata.ai/v1')
    rate_limit_delay = config.get('supadata', {}).get('rate_limit_delay', 0.1)
    max_retries = config.get('supadata', {}).get('max_retries', 3)
    retry_wait = config.get('supadata', {}).get('retry_wait', 10)  # Wait 10 seconds between retries
    
    # Load video list
    if not os.path.exists(video_list_path):
        print(f"Error: Video list not found at {video_list_path}")
        print("Please run scripts/get_channel_videos.py first to generate the video list.")
        sys.exit(1)
    
    with open(video_list_path, 'r', encoding='utf-8') as f:
        video_list = json.load(f)
    
    print(f"Loaded {len(video_list)} videos from {video_list_path}")
    
    # Load existing transcripts to skip already fetched videos
    # This also cleans up any incomplete/broken transcripts
    print("Checking for existing transcripts...")
    existing_ids, removed_count = load_existing_transcripts(transcripts_path)
    print(f"Found {len(existing_ids)} valid transcripts")
    if removed_count > 0:
        print(f"Removed {removed_count} incomplete/broken transcripts (will re-fetch)")
    
    # Create output directory
    os.makedirs(os.path.dirname(transcripts_path), exist_ok=True)
    
    # Open output file in append mode
    mode = 'a' if existing_ids else 'w'
    f_out = open(transcripts_path, mode, encoding='utf-8')
    
    # Process each video
    fetched_count = 0
    skipped_count = 0
    failed_count = 0
    total_to_fetch = len(video_list) - len(existing_ids)
    
    print(f"\n{'='*70}")
    print(f"Starting to fetch transcripts...")
    print(f"Total videos: {len(video_list)}")
    print(f"Already fetched: {len(existing_ids)}")
    print(f"Remaining to fetch: {total_to_fetch}")
    print(f"Rate limit: {1/rate_limit_delay:.1f} requests/second")
    print(f"{'='*70}\n")
    
    try:
        for i, video_info in enumerate(video_list, 1):
            video_id = video_info['video_id']
            video_title = video_info.get('title', 'Unknown Title')
            
            # Skip if already fetched
            if video_id in existing_ids:
                skipped_count += 1
                if skipped_count % 10 == 0 or skipped_count == 1:
                    print(f"[{i}/{len(video_list)}] ⏭️  Skipping {video_id} (already fetched) | Progress: {i}/{len(video_list)} ({100*i/len(video_list):.1f}%)")
                continue
            
            # Calculate progress
            current_fetch_num = i - skipped_count
            progress_pct = 100 * current_fetch_num / total_to_fetch if total_to_fetch > 0 else 0
            
            print(f"\n[{i}/{len(video_list)}] 📹 Fetching video {current_fetch_num}/{total_to_fetch} ({progress_pct:.1f}%)")
            print(f"   Video ID: {video_id}")
            print(f"   Title: {video_title[:80]}{'...' if len(video_title) > 80 else ''}")
            sys.stdout.flush()
            
            # Fetch video data
            start_time = time.time()
            video_data = fetch_video_data(
                api_key,
                video_info,
                base_url,
                rate_limit_delay,
                max_retries,
                retry_wait
            )
            elapsed_time = time.time() - start_time
            
            if video_data:
                # Save to JSONL file
                f_out.write(json.dumps(video_data, ensure_ascii=False) + '\n')
                f_out.flush()  # Ensure data is written immediately
                fetched_count += 1
                transcript_len = len(video_data.get('transcript', ''))
                print(f"   ✅ Success! Transcript: {transcript_len:,} chars | Time: {elapsed_time:.1f}s")
                print(f"   📊 Overall: {fetched_count} fetched, {failed_count} failed, {skipped_count} skipped")
            else:
                failed_count += 1
                print(f"   ⏭️  Skipping video after {elapsed_time:.1f}s (will continue to next)")
                print(f"   📊 Overall: {fetched_count} fetched, {failed_count} failed, {skipped_count} skipped")
            
            sys.stdout.flush()
            
            # Rate limiting - wait between requests
            if i < len(video_list):
                time.sleep(rate_limit_delay)
        
    finally:
        f_out.close()
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"📊 FETCH SUMMARY")
    print(f"{'='*70}")
    print(f"  Total videos in list: {len(video_list)}")
    print(f"  Already fetched (skipped): {skipped_count}")
    print(f"  Newly fetched this run: {fetched_count}")
    print(f"  Failed this run: {failed_count}")
    print(f"  Total valid transcripts: {len(existing_ids) + fetched_count}")
    print(f"\n  📁 Transcripts saved to: {transcripts_path}")
    
    # Show progress
    total_completed = len(existing_ids) + fetched_count
    remaining = len(video_list) - total_completed
    completion_pct = 100 * total_completed / len(video_list) if len(video_list) > 0 else 0
    
    print(f"\n  📈 Progress: {total_completed}/{len(video_list)} videos ({completion_pct:.1f}%)")
    print(f"  ⏳ Remaining: {remaining} videos")
    
    if remaining > 0:
        print(f"\n  💡 To resume: Run this script again and it will continue from where it left off")
        print(f"     The script automatically skips already-fetched videos and removes incomplete ones")
    
    print(f"{'='*70}")


if __name__ == '__main__':
    main()

