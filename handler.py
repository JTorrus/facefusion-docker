import runpod
from runpod import RunpodLogger  # Use RunPod's logger[4]
import subprocess
import os
import tempfile
import base64
import requests
import time

# Use RunPod's logger instead of standard logging[4]
log = RunpodLogger()

def send_telegram_message(bot_token, chat_id, text):
    """Send message to Telegram with error handling"""
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        data = {'chat_id': chat_id, 'text': text}
        response = requests.post(url, data=data, timeout=10)
        log.info(f"Telegram message sent successfully to {chat_id}")
        return True
    except Exception as e:
        log.error(f"Failed to send Telegram message: {e}")
        return False

def handler(job):
    """Handler with proper RunPod logging[4]"""
    
    # Log immediately to confirm handler is running
    log.info("=== HANDLER STARTED ===")
    log.info(f"Received job: {job}")
    
    # Get bot token from environment
    bot_token = os.environ.get('RUNPOD_SECRET_TELEGRAM_BOT_TOKEN')
    if not bot_token:
        log.error("CRITICAL: Bot token not found in environment")
        return {"error": "Bot token not configured"}
    
    log.info("Bot token found successfully")
    
    # Extract job input
    job_input = job.get('input', {})
    chat_id = job_input.get('chat_id')
    
    if not chat_id:
        log.error("CRITICAL: No chat_id provided")
        return {"error": "chat_id is required"}
    
    log.info(f"Processing for chat_id: {chat_id}")
    
    try:
        # Send immediate confirmation
        send_telegram_message(bot_token, chat_id, "🔄 Job received and processing started!")
        
        # Validate required inputs
        if 'source_base64' not in job_input:
            error_msg = "source_base64 is required"
            log.error(error_msg)
            send_telegram_message(bot_token, chat_id, f"❌ {error_msg}")
            return {"error": error_msg}
        
        # Create temp directory
        temp_dir = tempfile.mkdtemp()
        log.info(f"Created temp directory: {temp_dir}")
        
        # Process source image
        source_path = os.path.join(temp_dir, 'source.jpg')
        try:
            source_data = base64.b64decode(job_input['source_base64'])
            with open(source_path, 'wb') as f:
                f.write(source_data)
            log.info(f"Source image saved: {len(source_data)} bytes")
        except Exception as e:
            error_msg = f"Failed to decode source image: {e}"
            log.error(error_msg)
            send_telegram_message(bot_token, chat_id, f"❌ {error_msg}")
            return {"error": error_msg}
        
        # Process target file
        is_video = job_input.get('is_video', False)
        target_ext = '.mp4' if is_video else '.jpg'
        target_path = os.path.join(temp_dir, f'target{target_ext}')
        output_path = os.path.join(temp_dir, f'output{target_ext}')
        
        try:
            if 'target_url' in job_input:
                log.info(f"Downloading target from URL: {job_input['target_url']}")
                response = requests.get(job_input['target_url'], timeout=120)
                response.raise_for_status()
                with open(target_path, 'wb') as f:
                    f.write(response.content)
                log.info(f"Target downloaded: {len(response.content)} bytes")
            else:
                target_data = base64.b64decode(job_input['target_base64'])
                with open(target_path, 'wb') as f:
                    f.write(target_data)
                log.info(f"Target decoded: {len(target_data)} bytes")
        except Exception as e:
            error_msg = f"Failed to process target: {e}"
            log.error(error_msg)
            send_telegram_message(bot_token, chat_id, f"❌ {error_msg}")
            return {"error": error_msg}
        
        # Send processing update
        send_telegram_message(bot_token, chat_id, "⚡ Running FaceFusion...")
        
        # Execute FaceFusion
        cmd = [
            'python', 'facefusion.py',
            'headless-run',
            '--source', source_path,
            '--target', target_path,
            '--output', output_path,
            '--execution-providers', 'cuda'
        ]
        
        log.info(f"Executing command: {' '.join(cmd)}")
        
        # Run with timeout
        timeout = 600 if is_video else 180
        start_time = time.time()
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd='/facefusion',
            timeout=timeout
        )
        
        execution_time = time.time() - start_time
        log.info(f"FaceFusion completed in {execution_time:.2f} seconds")
        
        if result.returncode != 0:
            error_msg = f"FaceFusion failed: {result.stderr[:500]}"
            log.error(error_msg)
            send_telegram_message(bot_token, chat_id, f"❌ Processing failed")
            return {"error": error_msg}
        
        # Check output
        if not os.path.exists(output_path):
            error_msg = "Output file not generated"
            log.error(error_msg)
            send_telegram_message(bot_token, chat_id, f"❌ {error_msg}")
            return {"error": error_msg}
        
        file_size = os.path.getsize(output_path)
        log.info(f"Output file size: {file_size} bytes")
        
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
                    response = requests.post(url, files=files, data=data, timeout=300)
            else:
                url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
                with open(output_path, 'rb') as photo_file:
                    files = {'photo': photo_file}
                    data = {
                        'chat_id': chat_id,
                        'caption': '✅ Face swap completed!'
                    }
                    response = requests.post(url, files=files, data=data, timeout=120)
            
            response.raise_for_status()
            log.info("Successfully sent result to Telegram")
            
        except Exception as e:
            error_msg = f"Failed to send to Telegram: {e}"
            log.error(error_msg)
            return {"error": error_msg}
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
        log.info("Cleanup completed")
        
        return {
            "success": True,
            "chat_id": chat_id,
            "file_size": file_size,
            "execution_time": execution_time
        }
        
    except Exception as e:
        error_msg = f"Handler error: {str(e)}"
        log.error(error_msg)
        send_telegram_message(bot_token, chat_id, "❌ Unexpected error occurred")
        return {"error": error_msg}

# Critical: Proper startup[4]
if __name__ == "__main__":
    log.info("=== STARTING RUNPOD SERVERLESS HANDLER ===")
    runpod.serverless.start({"handler": handler})
