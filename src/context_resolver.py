from src.query_rewriter import QueryRewriter
from src.followup_resolver import FollowupResolver

class ContextResolver:
    def __init__(self):
        self.query_rewriter = QueryRewriter()
        self.followup_resolver = FollowupResolver()
        
    def resolve(self, user_query: str, state: dict) -> str:
        # First check strict follow-ups
        rewritten = self.followup_resolver.resolve(user_query, state)
        
        # If it was rewritten by strict rules, return it
        if rewritten != user_query:
            return rewritten
            
        # Otherwise fallback to general query rewriting rules
        res = self.query_rewriter.rewrite(user_query, state)
        return res.rewritten_query
