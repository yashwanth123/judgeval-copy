import os
import re
from typing import List, Dict, Tuple, Optional
import openai
import json

class DocumentAnonymizer:
    """
    A class to replace named entities in documents with believable alternatives
    using OpenAI API. Ensures consistent replacement across multiple documents.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        """
        Initialize the DocumentAnonymizer.
        
        Args:
            api_key: OpenAI API key. If None, will try to use environment variable.
            model: OpenAI model to use for entity replacement.
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set as OPENAI_API_KEY environment variable")
        
        self.model = model
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=self.api_key)
    
    def replace_entities(self, documents: List[str]) -> Tuple[List[str], Dict[str, str]]:
        """
        Replace named entities with believable alternatives across multiple documents in a single API call.
        
        Args:
            documents: List of document strings to process.
            
        Returns:
            Tuple containing:
                - List of documents with replaced entities
                - Dictionary mapping original entities to their replacements
        """
        # Combine documents with clear separators for the API call
        combined_docs = "\n\n---DOCUMENT SEPARATOR---\n\n".join(documents)
        
        # Make a single API call to replace all entities
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": (
                    "You are an expert at replacing named entities in documents with believable alternatives. "
                    "Your task is to identify all named entities (people, organizations, locations, case numbers, dates, etc.) "
                    "and replace them with distinct but believable alternatives. For example, replace 'John Smith' with "
                    "'James Doe', 'Acme Corp' with 'Zenith Industries', etc. Ensure consistency across all documents."
                )},
                {"role": "user", "content": (
                    "Replace all named entities in the following documents with believable alternatives. "
                    "Ensure that the same entity gets the same replacement across all documents. "
                    "Return a JSON object with these keys:\n"
                    "1. 'processed_documents': An array of documents with replaced entities\n"
                    "2. 'entity_mapping': An object mapping original entities to their replacements\n\n"
                    f"Documents to process:\n{combined_docs}"
                )}
            ],
            response_format={"type": "json_object"}
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
            processed_docs = result.get("processed_documents", [])
            entity_mapping = result.get("entity_mapping", {})
            
            # Ensure we have the right number of documents
            if len(processed_docs) != len(documents):
                raise ValueError(f"Expected {len(documents)} processed documents, got {len(processed_docs)}")
                
            return processed_docs, entity_mapping
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing API response: {e}")
            print(f"Response content: {response.choices[0].message.content}")
            raise RuntimeError("Failed to process documents") from e


def replace_named_entities(documents: List[str], api_key: Optional[str] = None) -> Tuple[List[str], Dict[str, str]]:
    """
    Convenience function to replace named entities in a list of documents with believable alternatives.
    
    Args:
        documents: List of document strings to process.
        api_key: OpenAI API key. If None, will try to use environment variable.
        
    Returns:
        Tuple containing:
            - List of documents with replaced entities
            - Dictionary mapping original entities to their replacements
    """
    anonymizer = DocumentAnonymizer(api_key=api_key)
    return anonymizer.replace_entities(documents)


if __name__ == "__main__":
    # Example usage
    sample_docs = [
        "John Smith from Acme Corp filed a lawsuit against Jane Doe in Case No. 2019-0572-TMR on July 25, 2019.",
        "The Delaware Court of Chancery received the complaint from John Smith regarding Acme Corp's actions."
    ]
    
    processed_docs, entity_mapping = replace_named_entities(sample_docs)
    
    print("Processed Documents:")
    for doc in processed_docs:
        print(f"- {doc}")
    
    print("\nEntity Mapping:")
    for original, replacement in entity_mapping.items():
        print(f"- {original} → {replacement}")


