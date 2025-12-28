import boto3
import requests
from flask import Flask, Response
import cv2
import os
from urllib.parse import urlparse

class CloudVideoHandler:
    def __init__(self):
        # AWS S3 Configuration
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION', 'us-east-1')
        )
        self.bucket_name = os.getenv('S3_BUCKET_NAME')
        
    def get_video_stream_from_s3(self, video_key):
        """Stream video from S3 bucket"""
        try:
            # Generate presigned URL for video
            presigned_url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': video_key},
                ExpiresIn=3600  # 1 hour
            )
            return cv2.VideoCapture(presigned_url)
        except Exception as e:
            print(f"Error accessing S3 video: {e}")
            return None
    
    def get_video_stream_from_url(self, video_url):
        """Stream video from any URL"""
        try:
            return cv2.VideoCapture(video_url)
        except Exception as e:
            print(f"Error accessing video URL: {e}")
            return None
    
    def upload_video_to_s3(self, file_path, s3_key):
        """Upload video to S3 bucket"""
        try:
            self.s3_client.upload_file(file_path, self.bucket_name, s3_key)
            return True
        except Exception as e:
            print(f"Error uploading to S3: {e}")
            return False

# Alternative: Use YouTube or public video URLs
class PublicVideoHandler:
    def __init__(self):
        # Sample traffic videos (replace with your own)
        self.sample_videos = [
            "https://sample-videos.com/zip/10/mp4/720/SampleVideo_720x480_1mb.mp4",
            "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4",
            # Add more public video URLs
        ]
    
    def get_sample_video_stream(self, index=0):
        """Get video stream from public URLs"""
        if index < len(self.sample_videos):
            return cv2.VideoCapture(self.sample_videos[index])
        return None

# IP Camera / Live Stream Handler
class LiveStreamHandler:
    def __init__(self):
        self.ip_cameras = [
            "http://your-ip-camera-1/stream",
            "http://your-ip-camera-2/stream",
            # Add IP camera URLs
        ]
    
    def get_live_stream(self, camera_index=0):
        """Connect to live IP camera"""
        if camera_index < len(self.ip_cameras):
            return cv2.VideoCapture(self.ip_cameras[camera_index])
        return cv2.VideoCapture(0)  # Default to webcam

# Google Drive Configuration
GOOGLE_DRIVE_VIDEOS = {
    'cam1': 'https://drive.google.com/file/d/1E-O-0h2CNzRwQIdoVhZjbDC0EaB43lVu/view?usp=sharing',
    'cam2': 'https://drive.google.com/file/d/1ojHTFncpSeu5cVNC9ki1b_siYevmCtZo/view?usp=sharing', 
    'cam3': 'https://drive.google.com/file/d/1NMqrX0z0PLWE-F7em-38URA1qpfjDMqc/view?usp=sharing',
    'cam4': 'https://drive.google.com/uc?export=download&id=YOUR_CAM4_FILE_ID'
}

def get_video_source(camera_name, use_google_drive=True):
    if use_google_drive and camera_name in GOOGLE_DRIVE_VIDEOS:
        return GOOGLE_DRIVE_VIDEOS[camera_name]
    # Fallback to demo videos
    return demo_videos.get(camera_name, 0)

