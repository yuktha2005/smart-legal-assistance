from __future__ import annotations

import os
import re
import json
import time
import hashlib
import logging
import subprocess
from typing import Dict, Any, Optional

logger = logging.getLogger("tts_engine")
logger.setLevel(logging.INFO)

# Consolidate logs into logs/app.log
os.makedirs("logs", exist_ok=True)
if not logger.handlers:
    fh = logging.FileHandler(os.path.join("logs", "app.log"))
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)


class TTSManager:
    """Manages Piper TTS configuration, text cleaning, audio caching, and execution."""

    def __init__(self, config_path: str = "config/tts_config.json"):
        self.repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.config_path = os.path.join(self.repo_root, config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Loads configuration settings from the JSON file."""
        defaults = {
            "enable_tts": True,
            "voice_model": "piper/models/en_US-lessac-medium.onnx",
            "speaking_rate": 1.0,
            "output_directory": "generated_audio",
            "auto_play": False,
            "read_sources": False
        }
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    user_config = json.load(f)
                    defaults.update(user_config)
            except Exception as e:
                logger.error(f"Failed to read config file {self.config_path}: {e}")
        return defaults

    def save_config(self) -> bool:
        """Saves current configuration settings back to the JSON file."""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2)
            logger.info("Saved TTS configuration successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to write config file {self.config_path}: {e}")
            return False

    def get_output_dir(self) -> str:
        """Resolves absolute path of the output directory."""
        out_dir = self.config.get("output_directory", "generated_audio")
        if not os.path.isabs(out_dir):
            out_dir = os.path.join(self.repo_root, out_dir)
        return out_dir

    def preprocess_text(self, text: str) -> str:
        """Cleans and preprocesses response text for natural TTS output."""
        if not text:
            return ""

        clean_text = text

        # 1. Remove sources citation section if enabled configuration is false
        if not self.config.get("read_sources", False):
            # Split by common sources headings and discard the tail
            for pattern in [r"(?i)###\s*Sources", r"(?i)\*\*Sources:\*\*", r"(?i)\bSources\b"]:
                parts = re.split(pattern, clean_text)
                if parts:
                    clean_text = parts[0]

        # 2. Remove markdown symbols
        clean_text = re.sub(r"\*\*|__|\*|_|`", "", clean_text)
        # Remove markdown heading prefixes
        clean_text = re.sub(r"^#+\s*", "", clean_text, flags=re.MULTILINE)
        # Remove markdown/bullet formatting symbols at line starts
        clean_text = re.sub(r"^\s*[•\-*]\s*", "", clean_text, flags=re.MULTILINE)

        # 3. Remove similarity scores or confidence values if present
        clean_text = re.sub(r"(?i)(?:similarity|confidence)\s*(?:score|value)?s*:\s*\d+(?:\.\d+)?%?", "", clean_text)

        # 4. Expand abbreviations naturally
        clean_text = re.sub(r"\bIPC\b", "Indian Penal Code", clean_text)
        clean_text = re.sub(r"\bBNS\b", "Bharatiya Nyaya Sanhita", clean_text)
        clean_text = re.sub(r"\bCrPC\b", "Code of Criminal Procedure", clean_text)
        
        # Expand section symbols and abbreviations
        clean_text = re.sub(r"§§", "Sections", clean_text)
        clean_text = re.sub(r"§", "Section", clean_text)
        clean_text = re.sub(r"\bSec\b\.?", "Section", clean_text, flags=re.IGNORECASE)

        # Convert currency symbols
        clean_text = re.sub(r"₹\s*([\d,]+)", r"\1 rupees", clean_text)
        clean_text = re.sub(r"Rs\b\.?\s*([\d,]+)", r"\1 rupees", clean_text, flags=re.IGNORECASE)

        # Clean special chars, replace newlines/extra whitespace
        clean_text = re.sub(r"\s+", " ", clean_text).strip()

        return clean_text

    def generate_speech(self, text: str) -> Optional[str]:
        """Generates audio for input text. Returns path of generated WAV file."""
        if not self.config.get("enable_tts", True):
            logger.warning("TTS is disabled in configuration.")
            return None

        # Clean text
        clean_text = self.preprocess_text(text)
        if not clean_text:
            return None

        # Check local installation files
        piper_exe = os.path.join(self.repo_root, "piper", "piper.exe")
        voice_model_path = self.config.get("voice_model", "piper/models/en_US-lessac-medium.onnx")
        if not os.path.isabs(voice_model_path):
            voice_model_path = os.path.join(self.repo_root, voice_model_path)

        if not os.path.exists(piper_exe):
            err_msg = f"Piper executable not found at: {piper_exe}"
            logger.error(err_msg)
            raise FileNotFoundError(err_msg)

        if not os.path.exists(voice_model_path):
            err_msg = f"Voice model ONNX file not found at: {voice_model_path}"
            logger.error(err_msg)
            raise FileNotFoundError(err_msg)

        # Build cache key based on settings and text
        rate = self.config.get("speaking_rate", 1.0)
        cache_str = f"{clean_text}||{voice_model_path}||{rate}"
        text_hash = hashlib.md5(cache_str.encode("utf-8")).hexdigest()

        output_dir = self.get_output_dir()
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{text_hash}.wav")

        # Use cached audio if it already exists
        if os.path.exists(output_file):
            logger.info(f"Using cached audio path: {output_file}")
            return output_file

        logger.info(f"TTS Text Received: '{clean_text[:60]}...'")
        logger.info(f"Voice Model: {voice_model_path}")
        logger.info(f"Speaking Rate (Length Scale): {rate}")

        # Run Piper executable subprocess
        start_time = time.time()
        try:
            # Speaking rate corresponds to phoneme length scale: a smaller number is faster.
            # We map user rate to phoneme length scale.
            # rate = 1.0 (default length scale = 1.0)
            # rate = 1.5 (faster -> length scale = 1.0 / 1.5 = 0.67)
            # rate = 0.5 (slower -> length scale = 1.0 / 0.5 = 2.0)
            length_scale = 1.0
            if rate > 0:
                length_scale = round(1.0 / rate, 2)

            process = subprocess.Popen(
                [
                    piper_exe,
                    "-m", voice_model_path,
                    "-f", output_file,
                    "--length_scale", str(length_scale)
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8"
            )
            stdout, stderr = process.communicate(input=clean_text)

            if process.returncode != 0:
                err_log = f"Piper execution failed: {stderr}"
                logger.error(err_log)
                raise RuntimeError(err_log)

            generation_time = time.time() - start_time
            logger.info(f"TTS Audio file generated successfully at: {output_file}")
            logger.info(f"Generation Time: {generation_time:.3f} seconds")

            # Clean up old audio files to stay within space limit
            self._clean_old_audio_files()

            return output_file
        except Exception as e:
            logger.error(f"Error during TTS generation: {e}")
            raise e

    def _clean_old_audio_files(self):
        """Removes older temporary audio files to keep directory under size limit."""
        try:
            output_dir = self.get_output_dir()
            if not os.path.exists(output_dir):
                return

            # Get list of files
            files = []
            for f in os.listdir(output_dir):
                fp = os.path.join(output_dir, f)
                if os.path.isfile(fp) and f.endswith(".wav"):
                    stat = os.stat(fp)
                    files.append({
                        "path": fp,
                        "mtime": stat.st_mtime,
                        "size": stat.st_size
                    })

            # Sort oldest first
            files.sort(key=lambda x: x["mtime"])

            max_size_bytes = 50 * 1024 * 1024  # 50 MB
            max_files = 100

            total_size = sum(f["size"] for f in files)
            total_files = len(files)

            while (total_size > max_size_bytes or total_files > max_files) and files:
                oldest = files.pop(0)
                try:
                    os.remove(oldest["path"])
                    total_size -= oldest["size"]
                    total_files -= 1
                    logger.info(f"Cleaned up old cached audio: {oldest['path']}")
                except Exception as e:
                    logger.error(f"Failed to remove old file {oldest['path']}: {e}")
        except Exception as e:
            logger.error(f"Error during audio folder cleanup: {e}")
