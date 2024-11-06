"""
Exception classes for Judgeval
"""


class JudgmentAPIError(Exception):
    """
    Exception raised when an error occurs while executing a Judgment API request
    """
    
    def __init__(self, message: str):
        super().__init__()
        self.message = message

