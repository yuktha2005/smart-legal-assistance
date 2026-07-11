from src.memory import ConversationMemory
from src.entity_tracker import EntityTracker

class MemoryManager:
    def __init__(self, memory: ConversationMemory):
        self.memory = memory
        self.entity_tracker = EntityTracker()
        
    def get_state(self) -> dict:
        return self.memory.state
        
    def update_state(self, user_query: str, answer: str, sources: list):
        # Track entities from user query
        user_entities = self.entity_tracker.extract_entities(user_query)
        if 'section_number' in user_entities:
            self.memory.state['current_section'] = user_entities['section_number']
        if 'document_name' in user_entities:
            self.memory.state['current_document'] = user_entities['document_name']
            
        # Extract general topics using LegalTermMapper
        from src.legal_term_mapper import LegalTermMapper
        mapper = LegalTermMapper()
        topic_keywords = list(mapper.mapping.keys())
        q_lower = user_query.lower()
        for topic in topic_keywords:
            if topic in q_lower:
                self.memory.state['current_topic'] = topic.title()
                break
            
        # Track from retrieval sources
        if sources:
            # Assumes sources is a list of strings
            self.memory.state['current_document'] = sources[0]
            
        # Update last answer
        self.memory.state['last_answer'] = answer
        
        self.memory.save()
