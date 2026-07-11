import re

class IntentResult(str):
    def __new__(cls, val, sub_intent=None):
        obj = str.__new__(cls, val)
        obj._sub_intent = sub_intent or val
        return obj

    @property
    def intent(self) -> str:
        return str(self)

    @property
    def sub_intent(self) -> str:
        return self._sub_intent


class IntentClassifier:
    def __init__(self):
        self.intent_patterns = {
            'greeting': [
                r'^(hi|hello|hey|good morning|good afternoon|good evening|greetings)\b'
            ],
            'punishment': [
                r'\b(punishment|penalty|sentence|prison|jail|fine|how many years|what happens if|prescribed for)\b'
            ],
            'definition': [
                r'\b(what is|define|meaning of|explain|what does it mean)\b'
            ],
            'section_lookup': [
                r'\b(which section|what section|under what section|section for)\b'
            ],
            'comparison': [
                r'\b(difference between|versus|vs|compared to|compare)\b'
            ],
            'procedure': [
                r'\b(how to file|procedure|FIR|bail|cognizable|compoundable)\b'
            ],
            'scenario': [
                r'\b(if someone|a person|suppose|if i|someone forged|someone dishonestly|someone threatens)\b'
            ],
            'followup': [
                r'\b(what about|and for|tell me more|is that all|where is it defined)\b'
            ],
            'bns_mapping': [
                r'\b(bns|bharatiya nyaya sanhita|new section for)\b'
            ]
        }
        
    def get_intent(self, query: str) -> IntentResult:
        query_lower = query.lower().strip()
        
        # Check greetings first
        for pattern in self.intent_patterns['greeting']:
            if re.search(pattern, query_lower):
                return IntentResult('GREETING', 'Greeting Query')
                
        # Check summarize / simplify
        if "summarize" in query_lower:
            return IntentResult('SUMMARIZE', 'Summarize Query')
        if "simplify" in query_lower:
            return IntentResult('SIMPLIFY', 'Simplify Query')
            
        # Check scenario
        for pattern in self.intent_patterns['scenario']:
            if re.search(pattern, query_lower):
                return IntentResult('LEGAL_QUERY', 'Scenario Query')
                
        # Check punishment
        for pattern in self.intent_patterns['punishment']:
            if re.search(pattern, query_lower):
                return IntentResult('LEGAL_QUERY', 'Punishment Query')
                
        # Check definition
        for pattern in self.intent_patterns['definition']:
            if re.search(pattern, query_lower):
                return IntentResult('LEGAL_QUERY', 'Definition Query')
                
        # Check other legal intents
        for intent in ['section_lookup', 'comparison', 'procedure', 'bns_mapping']:
            for pattern in self.intent_patterns[intent]:
                if re.search(pattern, query_lower):
                    return IntentResult('LEGAL_QUERY', intent.replace('_', ' ').title() + ' Query')
                    
        # Check followup
        for pattern in self.intent_patterns['followup']:
            if re.search(pattern, query_lower):
                return IntentResult('FOLLOWUP', 'Followup Query')
                
        # If it looks like a section query, e.g. contains section numbers (like 376, 467)
        if re.search(r'\b(section|sec|ipc|bns)\b', query_lower) or re.search(r'\b\d+[A-Z]*\b', query_lower):
            return IntentResult('LEGAL_QUERY', 'General Legal Query')
            
        # Check if query is completely gibberish/unknown
        if len(query_lower.split()) < 3 and not re.search(r'\b(theft|rape|murder|hurt|kidnap|fraud|forg|steal)\b', query_lower):
            return IntentResult('UNKNOWN', 'Unknown')
            
        return IntentResult('LEGAL_QUERY', 'General Legal Query')
