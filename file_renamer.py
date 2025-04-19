#!/usr/bin/env python3

import os
import sys
import subprocess
import logging
import platform
from datetime import datetime
from typing import Optional

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add error handler
error_handler = logging.StreamHandler(sys.stderr)
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(error_handler)

def ensure_dependencies():
    """Ensure all required packages are installed."""
    # Check if dependencies have already been installed
    deps_flag_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.deps_installed')
    if os.path.exists(deps_flag_file):
        return

    required_packages = [
        'pikepdf',
        'pillow',
        'pymupdf',  # Added PyMuPDF for better PDF handling
        'opencv-python',
        'python-dotenv',
        'pyyaml',
        'watchdog',
        'openai'
    ]
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            logger.info(f"Installing required package: {package}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    # Create flag file to indicate dependencies have been installed
    with open(deps_flag_file, 'w') as f:
        f.write('Dependencies installed on: ' + datetime.now().isoformat())

# Ensure dependencies are installed
ensure_dependencies()

# Now import all required packages
import yaml
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import openai
from datetime import datetime
import glob
from pathlib import Path
import shutil
import re
from dotenv import load_dotenv
import cv2
import tempfile
import pikepdf
from PIL import Image
import io

# Load environment variables from .env file
load_dotenv()

class FileRenamer:
    def __init__(self, config_path='config.yaml'):
        self.config = self._load_config(config_path)
        # Load API key from environment variable
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it in your .env file.")
        self.openai_client = openai.OpenAI(api_key=api_key)
        self.observers = []
        self.temp_dir = tempfile.mkdtemp(prefix='file_renamer_')

    def __del__(self):
        """Cleanup temporary directory on object destruction."""
        try:
            shutil.rmtree(self.temp_dir)
        except Exception as e:
            logger.warning(f"Error cleaning up temporary directory: {e}")

    def _load_config(self, config_path):
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise

    def get_file_created_date(self, file_path):
        """Get file creation date in different formats."""
        try:
            if platform.system() == 'Windows':
                created = os.path.getctime(file_path)
            else:  # macOS and Linux
                stat = os.stat(file_path)
                try:
                    created = stat.st_birthtime  # macOS
                except AttributeError:
                    created = stat.st_mtime  # Linux fallback

            timestamp = datetime.fromtimestamp(created)
            return {
                'iso': timestamp.isoformat(),
                'date': timestamp.strftime('%Y%m%d'),
                'datetime': timestamp.strftime('%Y%m%d-%H%M%S'),
                'datetime_short': timestamp.strftime('%Y%m%d-%H%M'),
                'year': timestamp.strftime('%Y'),
                'month': timestamp.strftime('%m'),
                'day': timestamp.strftime('%d'),
                'time': timestamp.strftime('%H%M%S')
            }
        except Exception as e:
            logger.error(f"Error getting file creation date: {e}")
            return None

    def is_video_file(self, file_path):
        """Check if the file is a video file based on its extension."""
        video_extensions = {'.mp4', '.mov', '.avi', '.wmv', '.flv', '.mkv', '.webm'}
        return os.path.splitext(file_path)[1].lower() in video_extensions

    def extract_video_frame(self, video_path):
        """Extract a frame from the video for analysis."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Error opening video file: {video_path}")
                return None

            # Get the middle frame of the video
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
            
            ret, frame = cap.read()
            cap.release()

            if not ret:
                logger.error(f"Error reading frame from video: {video_path}")
                return None

            # Create a temporary file to save the frame
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, f"frame_{os.path.basename(video_path)}.jpg")
            cv2.imwrite(temp_path, frame)
            return temp_path

        except Exception as e:
            logger.error(f"Error processing video file {video_path}: {e}")
            return None

    def convert_pdf_to_image(self, pdf_path):
        """Convert the first page of a PDF to an image for analysis."""
        try:
            # Open the PDF
            pdf = pikepdf.Pdf.open(pdf_path)
            
            # Get the first page
            page = pdf.pages[0]
            
            # Create a temporary file for the PDF page
            temp_pdf_path = os.path.join(self.temp_dir, f"temp_page_{os.path.basename(pdf_path)}")
            
            # Create a new PDF with just the first page
            new_pdf = pikepdf.Pdf.new()
            new_pdf.pages.append(page)
            new_pdf.save(temp_pdf_path)
            
            # Use Pillow to convert PDF to image
            from PIL import Image
            import fitz  # PyMuPDF

            # Open the PDF with PyMuPDF
            doc = fitz.open(pdf_path)
            page = doc.load_page(0)  # Load the first page
            
            # Convert page to image with higher resolution
            pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
            
            # Save as JPEG
            temp_path = os.path.join(self.temp_dir, f"pdf_page_{os.path.basename(pdf_path)}.jpg")
            pix.save(temp_path)
            
            # Clean up temporary PDF
            doc.close()
            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)
            
            return temp_path

        except Exception as e:
            logger.error(f"Error converting PDF to image {pdf_path}: {e}")
            return None

    def get_new_filename(self, file_path, prompt, folder_config):
        """Get new filename suggestion from OpenAI using vision analysis."""
        try:
            file_name = os.path.basename(file_path)
            file_ext = os.path.splitext(file_name)[1].lower()

            if folder_config.get('debug', False):
                logger.info("\n=== OpenAI Analysis Debug Info ===")
                logger.info(f"Analyzing file: {file_name}")

            # Process the image/PDF
            if file_ext.lower() == '.pdf':
                image_path = self.convert_pdf_to_image(file_path)
            else:
                image_path = file_path

            if not image_path:
                logger.error("Failed to process file for image analysis")
                return None

            # Read the image file as base64
            try:
                with open(image_path, "rb") as image_file:
                    import base64
                    image_data = base64.b64encode(image_file.read()).decode('utf-8')
            except Exception as e:
                logger.error(f"Error reading image file: {e}")
                return None

            # Clean up temporary file if it was created
            if file_ext.lower() == '.pdf' and image_path != file_path:
                try:
                    os.remove(image_path)
                except Exception as e:
                    logger.warning(f"Error removing temporary file: {e}")

            created_dates = self.get_file_created_date(file_path)
            if not created_dates:
                return None

            formatted_prompt = prompt.format(
                created_date=created_dates['date'],
                created_datetime=created_dates['datetime'],
                created_datetime_short=created_dates['datetime_short'],
                created_iso=created_dates['iso'],
                created_year=created_dates['year'],
                created_month=created_dates['month'],
                created_day=created_dates['day'],
                created_time=created_dates['time']
            )

            messages = [
                {
                    "role": "system",
                    "content": """You are a file naming assistant. Analyze the document and return a JSON object with{} properties:
                    {}'filename': The new filename following the specified rules{}
                    Example format: {{"filename": "YYYYMMDD-HHMM__TYPE__COMPANY__AMOUNT.ext"{}}}""".format(
                        " two" if folder_config.get('debug', False) else " one",
                        "'analysis': Detailed explanation of how you determined the filename components,\n                    " if folder_config.get('debug', False) else "",
                        "" if folder_config.get('debug', False) else "",
                        ",\n                    'analysis': \"Found date XX/XX/XX, classified as STATEMENT...\"" if folder_config.get('debug', False) else "",
                    )
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": formatted_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            }
                        }
                    ]
                }
            ]

            if folder_config.get('debug', False):
                logger.info("Sending request to OpenAI...")

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )

            # Parse the JSON response
            try:
                import json
                response_text = response.choices[0].message.content.strip()
                
                # Try to fix common JSON issues
                if response_text.startswith('```json'):
                    response_text = response_text.split('```json')[1]
                if response_text.endswith('```'):
                    response_text = response_text.split('```')[0]
                
                # Remove any trailing commas before closing braces
                response_text = re.sub(r',(\s*})', r'\1', response_text)
                
                # Attempt to parse the JSON
                try:
                    result = json.loads(response_text)
                except json.JSONDecodeError:
                    # If parsing fails, try to extract filename directly from response
                    if folder_config.get('debug', False):
                        logger.warning("Failed to parse JSON response, attempting to extract filename directly")
                    
                    # Look for filename pattern in the response
                    filename_match = re.search(r'"filename":\s*"([^"]+)"', response_text)
                    if filename_match:
                        result = {
                            'filename': filename_match.group(1),
                            'analysis': 'Analysis not available due to JSON parsing error' if folder_config.get('debug', False) else None
                        }
                    else:
                        logger.error("Could not extract filename from response")
                        return None
                
                if folder_config.get('debug', False):
                    logger.info("\n=== Analysis of Document ===")
                    logger.info(result.get('analysis', 'No analysis available'))
                    logger.info("\n=== Generated Filename ===")
                    logger.info(result.get('filename', 'No filename available'))

                new_name = result['filename']
                
                # Ensure the extension is preserved
                if not new_name.endswith(file_ext):
                    new_name += file_ext

                if folder_config.get('debug', False):
                    logger.info(f"Final filename with extension: {new_name}")
                    logger.info("=== End OpenAI Analysis ===\n")

                return new_name

            except json.JSONDecodeError as e:
                if folder_config.get('debug', False):
                    logger.error(f"Failed to parse OpenAI response as JSON: {response.choices[0].message.content}")
                    logger.error(f"JSON Error: {e}")
                return None

        except Exception as e:
            if folder_config.get('debug', False):
                logger.error(f"Error getting new filename from OpenAI: {e}")
            return None

    def matches_pattern(self, filename, pattern):
        """Check if filename matches the given pattern using regex."""
        try:
            return bool(re.search(pattern, filename, re.IGNORECASE))
        except Exception as e:
            logger.error(f"Error matching pattern: {e}")
            return False

    def rename_file(self, file_path, folder_config):
        """Rename a single file using OpenAI suggestions."""
        try:
            if folder_config.get('debug', False):
                logger.info(f"\n\n\n\n=== Starting File Rename Process ===")
                logger.info(f"Processing file: {file_path}")

            # Check if file matches pattern
            if folder_config.get('file_pattern'):
                filename = os.path.basename(file_path)
                patterns = [p.strip() for p in folder_config['file_pattern'].split(',')]
                if folder_config.get('debug', False):
                    logger.info(f"Checking patterns: {patterns}")
                    logger.info(f"Against filename: {filename}")
                if not any(self.matches_pattern(filename, pattern) for pattern in patterns):
                    if folder_config.get('debug', False):
                        logger.info("File did not match pattern - skipping")
                    return

            # Get original file stats
            original_stats = os.stat(file_path)

            # Pass folder_config to get_new_filename
            new_name = self.get_new_filename(file_path, folder_config['prompt'], folder_config)
            if not new_name:
                return

            # Create new path
            dir_path = os.path.dirname(file_path)
            new_path = os.path.join(dir_path, new_name)

            if folder_config.get('debug', False):
                logger.info(f"Renaming file to: {new_path}")

            # Rename file
            shutil.move(file_path, new_path)

            # Preserve timestamps
            os.utime(new_path, (original_stats.st_atime, original_stats.st_mtime))

            if folder_config.get('debug', False):
                logger.info("=== End File Rename Process ===\n")

            logger.info(f"Renamed '{file_path}' to '{new_path}'")

        except Exception as e:
            logger.error(f"Error renaming file {file_path}: {e}")

    def process_existing_files(self, folder_config):
        """Process all existing files in a folder."""
        path = folder_config['path']
        
        try:
            # Get all files in the directory
            for filename in os.listdir(path):
                file_path = os.path.join(path, filename)
                if os.path.isfile(file_path):
                    self.rename_file(file_path, folder_config)
        except Exception as e:
            logger.error(f"Error processing directory {path}: {e}")

    def start_watching(self):
        """Start watching configured folders."""
        for folder_config in self.config['folders']:
            if folder_config.get('watch', False):
                path = folder_config['path']
                event_handler = FileChangeHandler(self, folder_config)
                observer = Observer()
                observer.schedule(event_handler, path, recursive=False)
                observer.start()
                self.observers.append(observer)
                logger.info(f"Started watching folder: {path}")

    def stop_watching(self):
        """Stop all folder observers."""
        for observer in self.observers:
            observer.stop()
        for observer in self.observers:
            observer.join()

class FileChangeHandler(FileSystemEventHandler):
    def __init__(self, renamer, folder_config):
        self.renamer = renamer
        self.folder_config = folder_config

    def on_created(self, event):
        if not event.is_directory:
            self.renamer.rename_file(event.src_path, self.folder_config)

class RenameResult:
    def __init__(self, new_name: str, analysis: Optional[str] = None):
        self.new_name = new_name
        self.analysis = analysis

async def get_folder_config(file_path: str, config: dict) -> Optional[dict]:
    """Get the configuration for the folder containing the given file."""
    file_path = os.path.abspath(file_path)
    for folder_path, folder_config in config.items():
        if not isinstance(folder_config, dict):  # Skip non-folder configurations
            continue
        abs_folder_path = os.path.abspath(folder_path)
        if file_path.startswith(abs_folder_path):
            return {
                "path": abs_folder_path,
                "prompt": folder_config["prompt"],
                "debug": folder_config.get("debug", False)  # Default to False if not specified
            }
    return None

async def get_new_filename(content: str, current_name: str, folder_config: dict) -> RenameResult:
    debug_mode = folder_config.get('debug', False)
    print(f"\nDebug mode status: {debug_mode}")  # Debug print
    
    if debug_mode:
        analysis_prompt = "\n\nProvide your response in exactly this format:\nFILENAME\n\nANALYSIS: Explain in detail why you chose this filename."
    else:
        analysis_prompt = ""
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant that suggests better file names based on file content."},
        {"role": "user", "content": f"{folder_config['prompt']}\n\nPlease suggest a better name for this file based on its contents.{analysis_prompt}\n\nCurrent name: {current_name}\n\nContent:\n{content}"}
    ]

    response = await openai.ChatCompletion.create(
        model=config["model"],
        messages=messages,
        temperature=config["temperature"]
    )

    response_text = response.choices[0].message.content.strip()
    print(f"\nRaw AI response:\n{response_text}")  # Debug print
    
    if debug_mode:
        parts = response_text.split('\n\n', 1)
        new_name = parts[0].strip()
        analysis = parts[1] if len(parts) > 1 else None
        return RenameResult(new_name, analysis)
    
    return RenameResult(response_text)

async def rename_file(file_path: str, config: dict) -> None:
    dirname = os.path.dirname(file_path)
    folder_config = config[dirname]
    
    print(f"\nProcessing file in directory: {dirname}")  # Debug print
    print(f"Folder config: {folder_config}")  # Debug print
    
    # ... existing file reading code ...
    
    result = await get_new_filename(content, current_name, folder_config)
    
    if folder_config.get('debug'):
        print("\n=== AI Analysis of Rename Decision ===")
        if result.analysis:
            print(result.analysis)
        else:
            print("No analysis was provided by the AI")
        print("=====================================\n")

    new_path = os.path.join(dirname, result.new_name)
    os.rename(file_path, new_path)
    logger.info(f"Renamed '{file_path}' to '{new_path}'")

def main():
    renamer = FileRenamer()
    
    # Process existing files for all folders
    for folder_config in renamer.config['folders']:
        renamer.process_existing_files(folder_config)
    
    # Start watching folders that have watch enabled
    renamer.start_watching()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        renamer.stop_watching()

if __name__ == "__main__":
    main() 