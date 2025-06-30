import os
import io
import uuid
import base64
import subprocess
import cv2
import numpy as np
import traceback
import requests
import tempfile
import shutil
import runpod
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.modules.rp_logger import RunPodLogger
from typing import Union
from PIL import Image

# Configuration
TMP_PATH = '/tmp/facefusion'
FACEFUSION_PATH = '/facefusion'
logger = RunPodLogger()

# Input validation schema (adapted from inswapper)
INPUT_SCHEMA = {
    'chat_id': {
        'type': str,
        'required': True
    },
    'source_base64': {
        'type': str,
        'required': True
    },
    'target_base64': {
        'type': str,
        'required': False
    },
    'target_url': {
        'type': str,
        'required': False
    },
    'is_video': {
        'type': bool,
        'required': False,
        'default': False
    },
    'face_swapper_model': {
        'type': str,
        'required': False,
        'default': 'inswapper_128'
    },
    'output_quality': {
        'type': int,
        'required': False,
        'default': 90
    }
}

def send_telegram_message(bot_token: str, chat_id: str, text: str):
    """Send text message to Telegram"""
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        data = {'chat_id': chat_id, 'text': text}
        response = requests.post(url, data=data, timeout=10)
        response.raise_for_status()
        return True
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}")
        return False

def send_telegram_media(bot_token: str, chat_id: str, file_path: str, is_video: bool, caption: str = "✅ Face swap completed!"):
    """Send photo or video to Telegram"""
    try:
        if is_video:
            url = f"https://api.telegram.org/bot{bot_token}/sendVideo"
            with open(file_path, 'rb') as video_file:
                files = {'video': video_file}
                data = {
                    'chat_id': chat_id,
                    'caption': caption,
                    'supports_streaming': True
                }
                response = requests.post(url, files=files, data=data, timeout=300)
        else:
            url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
            with open(file_path, 'rb') as photo_file:
                files = {'photo': photo_file}
                data = {
                    'chat_id': chat_id,
                    'caption': caption
                }
                response = requests.post(url, files=files, data=data, timeout=120)
        
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Failed to send media to Telegram: {e}")
        raise

def determine_file_extension(image_data: str) -> str:
    """Determine file extension from base64 data"""
    try:
        if image_data.startswith('/9j/'):
            return '.jpg'
        elif image_data.startswith('iVBORw0Kg'):
            return '.png'
        elif image_data.startswith('UklGR'):  # WebP
            return '.webp'
        else:
            return '.jpg'  # Default
    except Exception:
        return '.jpg'

def clean_up_temporary_files(*file_paths):
    """Clean up temporary files"""
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Cleaned up: {file_path}")
        except Exception as e:
            logger.error(f"Failed to clean up {file_path}: {e}")

def download_target_file(target_url: str, target_path: str) -> bool:
    """Download target file from URL"""
    try:
        logger.info(f"Downloading target from: {target_url}")
        response = requests.get(target_url, timeout=120, stream=True)
        response.raise_for_status()
        
        with open(target_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Target downloaded successfully: {os.path.getsize(target_path)} bytes")
        return True
    except Exception as e:
        logger.error(f"Failed to download target: {e}")
        return False

def run_facefusion_swap(job_id: str, source_path: str, target_path: str, output_path: str, 
                       face_swapper_model: str, is_video: bool, output_quality: int) -> bool:
    """Execute FaceFusion face swap"""
    try:
        # Build FaceFusion command[1]
        cmd = [
            'python', 'facefusion.py',
            'headless-run',
            '--source', source_path,
            '--target', target_path,
            '--output', output_path,
            '--execution-providers', 'cuda',
            '--face-swapper-model', face_swapper_model,
            '--frame-processors', 'face_swapper'
        ]
        
        # Add video-specific parameters
        if is_video:
            cmd.extend([
                '--output-video-quality', str(output_quality),
                '--output-video-encoder', 'libx264',
                '--output-video-preset', 'medium'
            ])
        
        logger.info(f"Executing FaceFusion: {' '.join(cmd)}", job_id)
        
        # Set timeout based on content type
        timeout = 600 if is_video else 180  # 10 mins for video, 3 mins for image
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=FACEFUSION_PATH,
            timeout=timeout
        )
        
        if result.returncode != 0:
            logger.error(f"FaceFusion failed: {result.stderr}", job_id)
            return False
        
        logger.info("FaceFusion completed successfully", job_id)
        return True
        
    except subprocess.TimeoutExpired:
        logger.error(f"FaceFusion timed out after {timeout} seconds", job_id)
        return False
    except Exception as e:
        logger.error(f"FaceFusion execution error: {e}", job_id)
        return False

def facefusion_swap_api(job_id: str, job_input: dict):
    """Main face swap processing function"""
    
    # Get bot token
    bot_token = os.environ.get('RUNPOD_SECRET_TELEGRAM_BOT_TOKEN')
    if not bot_token:
        return {'error': 'Bot token not configured'}
    
    chat_id = job_input['chat_id']
    
    # Send processing notification
    send_telegram_message(bot_token, chat_id, "🔄 Starting face swap processing...")
    
    # Create temp directory
    if not os.path.exists(TMP_PATH):
        os.makedirs(TMP_PATH)
    
    unique_id = uuid.uuid4()
    temp_files = []
    
    try:
        # Process source image
        logger.info("Processing source image", job_id)
        source_data = base64.b64decode(job_input['source_base64'])
        source_ext = determine_file_extension(job_input['source_base64'])
        source_path = f'{TMP_PATH}/source_{unique_id}{source_ext}'
        
        with open(source_path, 'wb') as f:
            f.write(source_data)
        temp_files.append(source_path)
        logger.info(f"Source saved: {len(source_data)} bytes", job_id)
        
        # Process target file
        is_video = job_input.get('is_video', False)
        target_ext = '.mp4' if is_video else '.jpg'
        target_path = f'{TMP_PATH}/target_{unique_id}{target_ext}'
        temp_files.append(target_path)
        
        if 'target_url' in job_input:
            if not download_target_file(job_input['target_url'], target_path):
                send_telegram_message(bot_token, chat_id, "❌ Failed to download target file")
                return {'error': 'Failed to download target file'}
        else:
            logger.info("Processing target from base64", job_id)
            target_data = base64.b64decode(job_input['target_base64'])
            with open(target_path, 'wb') as f:
                f.write(target_data)
            logger.info(f"Target saved: {len(target_data)} bytes", job_id)
        
        # Set output path
        output_ext = target_ext
        output_path = f'{TMP_PATH}/output_{unique_id}{output_ext}'
        temp_files.append(output_path)
        
        # Extract parameters
        face_swapper_model = job_input.get('face_swapper_model', 'inswapper_128')
        output_quality = job_input.get('output_quality', 90)
        
        logger.info(f"Parameters - Model: {face_swapper_model}, Quality: {output_quality}, Video: {is_video}", job_id)
        
        # Send processing update
        send_telegram_message(bot_token, chat_id, "⚡ Running FaceFusion...")
        
        # Execute face swap
        success = run_facefusion_swap(
            job_id, source_path, target_path, output_path,
            face_swapper_model, is_video, output_quality
        )
        
        if not success:
            send_telegram_message(bot_token, chat_id, "❌ Face swap processing failed")
            return {'error': 'FaceFusion processing failed'}
        
        # Check output file
        if not os.path.exists(output_path):
            send_telegram_message(bot_token, chat_id, "❌ Output file was not generated")
            return {'error': 'Output file not generated'}
        
        file_size = os.path.getsize(output_path)
        max_size = 50 * 1024 * 1024  # 50MB Telegram limit
        
        if file_size > max_size:
            error_msg = f"Output too large: {file_size/1024/1024:.1f}MB"
            send_telegram_message(bot_token, chat_id, f"❌ {error_msg}")
            return {'error': error_msg}
        
        logger.info(f"Output file ready: {file_size} bytes", job_id)
        
        # Send to Telegram
        try:
            telegram_response = send_telegram_media(bot_token, chat_id, output_path, is_video)
            logger.info("Successfully sent result to Telegram", job_id)
            
            return {
                'success': True,
                'chat_id': chat_id,
                'file_size': file_size,
                'telegram_message_id': telegram_response.get('result', {}).get('message_id'),
                'processing_type': 'video' if is_video else 'image'
            }
            
        except Exception as e:
            logger.error(f"Failed to send to Telegram: {e}", job_id)
            return {'error': f'Failed to send to Telegram: {str(e)}'}
    
    except Exception as e:
        logger.error(f"Processing error: {e}", job_id)
        send_telegram_message(bot_token, chat_id, "❌ An unexpected error occurred")
        return {
            'error': str(e),
            'traceback': traceback.format_exc(),
            'refresh_worker': True
        }
    
    finally:
        # Clean up all temporary files
        clean_up_temporary_files(*temp_files)

# ---------------------------------------------------------------------------- #
# RunPod Handler                                                               #
# ---------------------------------------------------------------------------- #
def handler(event):
    """Main RunPod handler function"""
    job_id = event['id']
    logger.info(f"=== JOB RECEIVED: {job_id} ===")
    
    # Validate input using the schema
    validated_input = validate(event['input'], INPUT_SCHEMA)
    
    if 'errors' in validated_input:
        logger.error(f"Input validation failed: {validated_input['errors']}", job_id)
        return {'error': validated_input['errors']}
    
    return facefusion_swap_api(job_id, validated_input['validated_input'])

if __name__ == '__main__':
    logger.info("=== STARTING FACEFUSION RUNPOD WORKER ===")
    logger.info(f"FaceFusion path: {FACEFUSION_PATH}")
    logger.info(f"Temp path: {TMP_PATH}")
    
    # Start the RunPod serverless worker
    runpod.serverless.start({'handler': handler})
