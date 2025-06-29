import runpod
import subprocess
import os
import tempfile
import base64
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def send_video_to_telegram(bot_token, chat_id, video_path, caption="✅ Face swap completed!"):
    """Send video directly to Telegram chat[4]"""
    url = f"https://api.telegram.org/bot{bot_token}/sendVideo"
    
    try:
        with open(video_path, 'rb') as video_file:
            files = {'video': video_file}
            data = {
                'chat_id': chat_id,
                'caption': caption,
                'supports_streaming': True  # Enable streaming[4]
            }
            
            response = requests.post(url, files=files, data=data, timeout=60)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Failed to send video to Telegram: {e}")
        raise

def send_photo_to_telegram(bot_token, chat_id, photo_path, caption="✅ Face swap completed!"):
    """Send photo directly to Telegram chat"""
    url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
    
    try:
        with open(photo_path, 'rb') as photo_file:
            files = {'photo': photo_file}
            data = {
                'chat_id': chat_id,
                'caption': caption
            }
            
            response = requests.post(url, files=files, data=data, timeout=30)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Failed to send photo to Telegram: {e}")
        raise

def send_error_message(bot_token, chat_id, error_msg):
    """Send error message to user[6]"""
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    
    try:
        data = {
            'chat_id': chat_id,
            'text': f"❌ Face swap failed: {error_msg}",
            'parse_mode': 'HTML'
        }
        requests.post(url, data=data, timeout=10)
    except Exception as e:
        logger.error(f"Failed to send error message: {e}")

def handler(job):
    """RunPod handler with direct Telegram integration"""
    
    # Get bot token from RunPod secrets[2]
    bot_token = os.environ.get('RUNPOD_SECRET_TELEGRAM_BOT_TOKEN')
    if not bot_token:
        return {"error": "Bot token not found in secrets"}
    
    try:
        job_input = job['input']
        chat_id = job_input['chat_id']
        is_video = job_input.get('is_video', False)
        
        # Create temp directory
        temp_dir = tempfile.mkdtemp()
        
        # Handle source image
        source_path = os.path.join(temp_dir, 'source.jpg')
        with open(source_path, 'wb') as f:
            f.write(base64.b64decode(job_input['source_base64']))
        
        # Handle target
        target_ext = '.mp4' if is_video else '.jpg'
        target_path = os.path.join(temp_dir, f'target{target_ext}')
        output_path = os.path.join(temp_dir, f'output{target_ext}')
        
        if 'target_url' in job_input:
            response = requests.get(job_input['target_url'])
            with open(target_path, 'wb') as f:
                f.write(response.content)
        else:
            with open(target_path, 'wb') as f:
                f.write(base64.b64decode(job_input['target_base64']))
        
        # Run FaceFusion
        cmd = [
            'python', 'facefusion.py',
            'headless-run',
            '--source', source_path,
            '--target', target_path,
            '--output', output_path,
            '--execution-providers', 'cuda'
        ]
        
        logger.info(f"Processing face swap for chat_id: {chat_id}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd='/facefusion', timeout=300)
        
        if result.returncode != 0:
            error_msg = f"Processing failed: {result.stderr[:100]}..."
            send_error_message(bot_token, chat_id, error_msg)
            return {"error": error_msg, "chat_id": chat_id}
        
        # Check output file exists and size[5]
        if not os.path.exists(output_path):
            error_msg = "Output file was not generated"
            send_error_message(bot_token, chat_id, error_msg)
            return {"error": error_msg, "chat_id": chat_id}
        
        file_size = os.path.getsize(output_path)
        max_size = 50 * 1024 * 1024  # 50MB Telegram limit[5]
        
        if file_size > max_size:
            error_msg = f"Output file too large ({file_size/1024/1024:.1f}MB). Please use a smaller input file."
            send_error_message(bot_token, chat_id, error_msg)
            return {"error": error_msg, "chat_id": chat_id}
        
        # Send directly to Telegram[4]
        if is_video:
            telegram_response = send_video_to_telegram(bot_token, chat_id, output_path)
        else:
            telegram_response = send_photo_to_telegram(bot_token, chat_id, output_path)
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
        
        return {
            "success": True,
            "chat_id": chat_id,
            "file_size": file_size,
            "telegram_message_id": telegram_response.get('result', {}).get('message_id'),
            "direct_delivery": True
        }
        
    except subprocess.TimeoutExpired:
        send_error_message(bot_token, chat_id, "Processing timed out. Please try with a smaller file.")
        return {"error": "Processing timeout", "chat_id": chat_id}
    except Exception as e:
        logger.error(f"Handler error: {e}")
        send_error_message(bot_token, chat_id, f"Unexpected error: {str(e)[:50]}...")
        return {"error": str(e), "chat_id": chat_id}

runpod.serverless.start({"handler": handler})
