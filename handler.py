import runpod
import subprocess
import os
import tempfile
import base64
import requests
import logging
import time

# Enhanced logging configuration[9]
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def send_telegram_message(bot_token, chat_id, text):
    """Send simple text message to Telegram"""
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        data = {'chat_id': chat_id, 'text': text}
        requests.post(url, data=data, timeout=10)
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}")

def handler(job):
    """Enhanced handler with better error handling"""
    
    bot_token = os.environ.get('RUNPOD_SECRET_TELEGRAM_BOT_TOKEN')
    if not bot_token:
        logger.error("Bot token not found in secrets")
        return {"error": "Bot token not configured"}
    
    job_input = job.get('input', {})
    chat_id = job_input.get('chat_id')
    
    if not chat_id:
        logger.error("No chat_id provided in job input")
        return {"error": "chat_id is required"}
    
    logger.info(f"Processing job for chat_id: {chat_id}")
    
    try:
        # Send processing notification
        send_telegram_message(bot_token, chat_id, "🔄 Starting face swap processing...")
        
        # Create temp directory
        temp_dir = tempfile.mkdtemp()
        logger.info(f"Created temp directory: {temp_dir}")
        
        # Process files
        is_video = job_input.get('is_video', False)
        
        # Handle source image
        source_path = os.path.join(temp_dir, 'source.jpg')
        try:
            with open(source_path, 'wb') as f:
                f.write(base64.b64decode(job_input['source_base64']))
            logger.info("Source image processed successfully")
        except Exception as e:
            error_msg = f"Failed to process source image: {e}"
            logger.error(error_msg)
            send_telegram_message(bot_token, chat_id, f"❌ {error_msg}")
            return {"error": error_msg}
        
        # Handle target file
        target_ext = '.mp4' if is_video else '.jpg'
        target_path = os.path.join(temp_dir, f'target{target_ext}')
        output_path = os.path.join(temp_dir, f'output{target_ext}')
        
        try:
            if 'target_url' in job_input:
                response = requests.get(job_input['target_url'], timeout=60)
                response.raise_for_status()
                with open(target_path, 'wb') as f:
                    f.write(response.content)
            else:
                with open(target_path, 'wb') as f:
                    f.write(base64.b64decode(job_input['target_base64']))
            logger.info("Target file processed successfully")
        except Exception as e:
            error_msg = f"Failed to process target file: {e}"
            logger.error(error_msg)
            send_telegram_message(bot_token, chat_id, f"❌ {error_msg}")
            return {"error": error_msg}
        
        # Progress update[10]
        runpod.serverless.progress_update(job, "Files processed, starting FaceFusion...")
        
        # Run FaceFusion with timeout
        cmd = [
            'python', 'facefusion.py',
            'headless-run',
            '--source', source_path,
            '--target', target_path,
            '--output', output_path,
            '--execution-providers', 'cuda'
        ]
        
        logger.info(f"Executing FaceFusion command: {' '.join(cmd)}")
        
        # Send progress update
        send_telegram_message(bot_token, chat_id, "⚡ FaceFusion processing started...")
        
        # Execute with proper timeout
        timeout = 600 if is_video else 180  # 10 mins for video, 3 mins for image
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            cwd='/facefusion',
            timeout=timeout
        )
        
        if result.returncode != 0:
            error_msg = f"FaceFusion failed: {result.stderr[:200]}..."
            logger.error(error_msg)
            send_telegram_message(bot_token, chat_id, f"❌ Processing failed: {result.stderr[:100]}...")
            return {"error": error_msg}
        
        # Check output file
        if not os.path.exists(output_path):
            error_msg = "Output file was not generated"
            logger.error(error_msg)
            send_telegram_message(bot_token, chat_id, f"❌ {error_msg}")
            return {"error": error_msg}
        
        file_size = os.path.getsize(output_path)
        max_size = 50 * 1024 * 1024  # 50MB limit
        
        if file_size > max_size:
            error_msg = f"Output too large: {file_size/1024/1024:.1f}MB"
            logger.error(error_msg)
            send_telegram_message(bot_token, chat_id, f"❌ {error_msg}")
            return {"error": error_msg}
        
        # Send to Telegram
        try:
            if is_video:
                url = f"https://api.telegram.org/bot{bot_token}/sendVideo"
                with open(output_path, 'rb') as video_file:
                    files = {'video': video_file}
                    data = {
                        'chat_id': chat_id,
                        'caption': '✅ Face swap completed!',
                        'supports_streaming': True
                    }
                    response = requests.post(url, files=files, data=data, timeout=120)
            else:
                url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
                with open(output_path, 'rb') as photo_file:
                    files = {'photo': photo_file}
                    data = {
                        'chat_id': chat_id,
                        'caption': '✅ Face swap completed!'
                    }
                    response = requests.post(url, files=files, data=data, timeout=60)
            
            response.raise_for_status()
            logger.info(f"Successfully sent result to Telegram for chat {chat_id}")
            
        except Exception as e:
            error_msg = f"Failed to send to Telegram: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
        
        return {
            "success": True,
            "chat_id": chat_id,
            "file_size": file_size,
            "processing_time": time.time() - time.time(),  # You'd track this properly
            "refresh_worker": True  # Clean state for next job[10]
        }
        
    except subprocess.TimeoutExpired:
        error_msg = f"Processing timed out after {timeout} seconds"
        logger.error(error_msg)
        send_telegram_message(bot_token, chat_id, "❌ Processing timed out. Please try a smaller file.")
        return {"error": error_msg}
        
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg)
        send_telegram_message(bot_token, chat_id, f"❌ Unexpected error occurred")
        return {"error": error_msg}

# Start the serverless worker
runpod.serverless.start({"handler": handler})
