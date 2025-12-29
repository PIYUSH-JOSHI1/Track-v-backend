import cv2
import os
from urllib.parse import urlparse
import subprocess
import re

class YouTubeVideoHandler:
    """Handler for YouTube and other streaming videos using yt-dlp"""
    
    def __init__(self):
        """Initialize YouTube handler"""
        self.stream_url_cache = {}
        print("YouTube Video Handler Initialized")
    
    def extract_youtube_url(self, youtube_url):
        """Extract actual streaming URL from YouTube using yt-dlp"""
        try:
            # Try to use yt-dlp if available
            try:
                import yt_dlp
                with yt_dlp.YoutubeDL({'quiet': True, 'no_warnings': True}) as ydl:
                    info = ydl.extract_info(youtube_url, download=False)
                    return info['url']
            except ImportError:
                # Fallback: use direct command-line yt-dlp
                cmd = ['yt-dlp', '-f', 'worst', '-g', youtube_url]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    return result.stdout.strip().split('\n')[0]
            
            print(f"⚠ Could not extract YouTube URL: {youtube_url}")
            return None
        except Exception as e:
            print(f"Error extracting YouTube URL: {e}")
            return None
    
    def get_video_stream(self, video_url, resolution='480p'):
        """
        Get video stream from YouTube or streaming URL
        
        Args:
            video_url: YouTube URL or direct streaming URL
            resolution: Target resolution ('360p', '480p', '720p')
        
        Returns:
            cv2.VideoCapture object or None
        """
        try:
            # Check if it's a YouTube URL
            if 'youtube.com' in video_url or 'youtu.be' in video_url:
                actual_url = self.extract_youtube_url(video_url)
                if not actual_url:
                    return None
            else:
                actual_url = video_url
            
            # Open stream
            cap = cv2.VideoCapture(actual_url)
            if cap.isOpened():
                print(f"✓ Successfully opened video stream")
                # Set buffer size to prevent lag
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return cap
            else:
                print(f"✗ Failed to open video stream")
                return None
        except Exception as e:
            print(f"Error opening video stream: {e}")
            return None


class GoogleDriveVideoHandler:
    """Handler for streaming videos from Google Drive"""
    
    def __init__(self):
        """Initialize with Google Drive video URLs from environment or defaults"""
        # Get URLs from environment variables with fallback to defaults
        self.videos = {
            'cam1': os.getenv('GOOGLE_DRIVE_CAM1', 'https://drive.google.com/uc?export=download&id=1E-O-0h2CNzRwQIdoVhZjbDC0EaB43lVu'),
            'cam2': os.getenv('GOOGLE_DRIVE_CAM2', 'https://drive.google.com/uc?export=download&id=1ojHTFncpSeu5cVNC9ki1b_siYevmCtZo'),
            'cam3': os.getenv('GOOGLE_DRIVE_CAM3', 'https://drive.google.com/uc?export=download&id=1NMqrX0z0PLWE-F7em-38URA1qpfjDMqc'),
            'cam4': os.getenv('GOOGLE_DRIVE_CAM4', 'https://drive.google.com/uc?export=download&id=1NMqrX0z0PLWE-F7em-38URA1qpfjDMqc')
        }
        
        print("Google Drive Video Handler Initialized")
        print("Videos configured:")
        for cam_name, url in self.videos.items():
            status = "✓" if "YOUR_" not in url else "⚠ (needs file ID)"
            print(f"  {cam_name}: {status}")
    
    def get_video_stream(self, camera_index=0):
        """
        Get video stream from Google Drive by camera index
        
        Args:
            camera_index: 0=cam1, 1=cam2, 2=cam3, 3=cam4
        
        Returns:
            cv2.VideoCapture object
        """
        cam_names = ['cam1', 'cam2', 'cam3', 'cam4']
        
        if 0 <= camera_index < len(cam_names):
            cam_name = cam_names[camera_index]
            video_url = self.videos.get(cam_name)
            
            if video_url and "YOUR_" not in video_url:
                try:
                    print(f"Loading {cam_name} from Google Drive...")
                    return cv2.VideoCapture(video_url)
                except Exception as e:
                    print(f"Error loading {cam_name}: {e}")
                    return None
            else:
                print(f"⚠ {cam_name} URL not configured")
                return None
        
        return None
    
    def get_video_stream_by_name(self, camera_name):
        """Get video stream by camera name (cam1, cam2, etc)"""
        video_url = self.videos.get(camera_name.lower())
        
        if video_url and "YOUR_" not in video_url:
            try:
                return cv2.VideoCapture(video_url)
            except Exception as e:
                print(f"Error loading {camera_name}: {e}")
                return None
        
        return None


# Legacy compatibility - keep simple public video handler as fallback
class PublicVideoHandler:
    """Fallback handler for demo/sample videos"""
    
    def __init__(self):
        self.sample_videos = [
            "https://sample-videos.com/zip/10/mp4/720/SampleVideo_720x480_1mb.mp4",
            "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4",
        ]
    
    def get_sample_video_stream(self, index=0):
        """Get demo video stream"""
        if index < len(self.sample_videos):
            return cv2.VideoCapture(self.sample_videos[index])
        return None


# Placeholder classes for backward compatibility
class CloudVideoHandler:
    """Legacy AWS S3 handler - not used in current implementation"""
    def __init__(self):
        print("Note: AWS S3 handler not configured. Using Google Drive instead.")
    
    def get_video_stream_from_s3(self, video_key):
        return None


class LiveStreamHandler:
    """IP Camera handler - can be configured later"""
    def __init__(self):
        self.ip_cameras = []
    
    def get_live_stream(self, camera_index=0):
        """Fallback to webcam if IP cameras not configured"""
        return cv2.VideoCapture(0)
