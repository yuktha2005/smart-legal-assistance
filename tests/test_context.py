from src.context_manager import ContextManager


def test_context_extracts_section_from_history():
    cm = ContextManager()
    history = [
        {"user": "What is section 376DA?", "assistant": "x"},
    ]
    slots = cm.extract_slots_from_history(history)
    assert slots.current_section == "376DA"

