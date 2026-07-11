import os
import shutil
import pytest
from src.tts import TTSManager


@pytest.fixture
def temp_tts_manager(tmp_path):
    # Create a temporary config path and manager
    config_file = tmp_path / "tts_config.json"
    manager = TTSManager(config_path=str(config_file))
    
    # Override output directory to temporary directory
    manager.config["output_directory"] = str(tmp_path / "generated_audio")
    manager.save_config()
    
    yield manager


def test_preprocess_text_basic(temp_tts_manager):
    # Test markdown stripping and basic cleaning
    input_text = "### Definition\nThis is **important** text with *bullets*:\n- bullet 1\n• bullet 2"
    expected = "Definition This is important text with bullets: bullet 1 bullet 2"
    processed = temp_tts_manager.preprocess_text(input_text)
    assert processed == expected


def test_preprocess_text_sources_omitted(temp_tts_manager):
    # Test that sources citations section is stripped by default
    input_text = "This is a legal answer.\n### Sources\n• document1.pdf (Page 5)\n• document2.pdf (Page 12)"
    expected = "This is a legal answer."
    processed = temp_tts_manager.preprocess_text(input_text)
    assert processed.strip() == expected


def test_preprocess_text_sources_kept(temp_tts_manager):
    # Enable reading sources in config
    temp_tts_manager.config["read_sources"] = True
    input_text = "This is a legal answer.\n### Sources\n• document1.pdf (Page 5)"
    # Markdown symbols should still be cleaned
    expected = "This is a legal answer. Sources document1.pdf (Page 5)"
    processed = temp_tts_manager.preprocess_text(input_text)
    assert processed.strip() == expected


def test_preprocess_abbreviations(temp_tts_manager):
    # Test abbreviation expansions: IPC, BNS, Sec.
    input_text = "Under IPC 376 and BNS 64, Sec. 34 is joint liability."
    expected = "Under Indian Penal Code 376 and Bharatiya Nyaya Sanhita 64, Section 34 is joint liability."
    processed = temp_tts_manager.preprocess_text(input_text)
    assert processed == expected


def test_preprocess_special_characters(temp_tts_manager):
    # Test currency and section signs
    input_text = "The penalty is §§ 303–305 fine of ₹25,000."
    expected = "The penalty is Sections 303–305 fine of 25,000 rupees."
    processed = temp_tts_manager.preprocess_text(input_text)
    assert processed == expected


def test_config_save_load(temp_tts_manager):
    # Test saving config modifies the file, and loading reads it
    temp_tts_manager.config["speaking_rate"] = 1.25
    temp_tts_manager.config["auto_play"] = True
    assert temp_tts_manager.save_config() is True
    
    # Load in new manager instance from same file path
    new_manager = TTSManager(config_path=temp_tts_manager.config_path)
    assert new_manager.config["speaking_rate"] == 1.25
    assert new_manager.config["auto_play"] is True


def test_output_dir_resolution(temp_tts_manager):
    # Test that relative output directories resolve cleanly relative to repository root
    temp_tts_manager.config["output_directory"] = "custom_audio_dir"
    expected_path = os.path.join(temp_tts_manager.repo_root, "custom_audio_dir")
    assert temp_tts_manager.get_output_dir() == expected_path
