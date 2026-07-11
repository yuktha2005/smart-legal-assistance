import re

class FollowupResolver:
    """Resolves follow-up questions by replacing pronouns with tracked context."""
    
    # Pronouns that indicate a follow-up
    PRONOUNS = {'it', 'its', 'this', 'that', 'they', 'them', 'those'}
    
    def resolve(self, query: str, state: dict) -> str:
        q_lower = query.lower().strip()
        current_topic = state.get("current_topic")
        current_section = state.get("current_section")
        
        # Check if query contains any pronoun
        query_words = set(re.findall(r'\b\w+\b', q_lower))
        has_pronoun = bool(query_words & self.PRONOUNS)
        
        if not has_pronoun:
            return query
        
        # We have a pronoun — resolve it
        # Priority: current_topic > current_section
        
        if current_topic:
            # Replace pronouns with the topic
            resolved = query
            for pronoun in self.PRONOUNS:
                # Case-insensitive whole-word replacement
                resolved = re.sub(
                    r'\b' + pronoun + r'\b',
                    current_topic,
                    resolved,
                    flags=re.IGNORECASE
                )
            return resolved
        
        if current_section:
            resolved = query
            for pronoun in self.PRONOUNS:
                resolved = re.sub(
                    r'\b' + pronoun + r'\b',
                    f"section {current_section}",
                    resolved,
                    flags=re.IGNORECASE
                )
            return resolved
        
        return query
