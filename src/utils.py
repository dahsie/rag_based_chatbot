
import re
import json
from typing import Dict, List
from langchain_core.documents import Document


def format_docs(docs: Document) -> str:
    """
    Formats a list of documents by joining their content.

    Args:
        docs (list): A list of document objects, each containing a `page_content` attribute.

    Returns:
        str: A single string with the content of each document joined by two newlines.
    """
    
    return "\n\n".join(doc.page_content for doc in docs)


def parse_web_search_result(text: str) -> List[Dict]:
    """
    Parses a text string to extract segments starting with 'snippet', followed by 'title' and 'link'.
    
    Each extracted segment is converted into a dictionary containing:
    - "snippet": the snippet text,
    - "title": the title of the snippet,
    - "link": the URL link associated with the snippet.

    Args:
        text (str): The input text containing multiple segments in the format
                    'snippet: ..., title: ..., link: URL,'.

    Returns:
        list of dict: A list of dictionaries, each containing keys "snippet", "title", and "link",
                      representing each segment found in the text.
    
    Example:
        text = 'snippet: Sample text, title: Example Title, link: https://example.com,'
        result = parse_snippets(text)
        # Result: [{'snippet': 'Sample text', 'title': 'Example Title', 'link': 'https://example.com'}]
    """

    pattern = re.compile(
        r"snippet: (.*?),\s*title: (.*?),\s*link: (https?://[^\s,]+),"
    )
    
    # Trouver toutes les correspondances dans le texte
    matches = pattern.findall(text)
    
    # Construire une liste de dictionnaires à partir des correspondances
    snippets = []
    for snippet, title, link in matches:
        snippets.append({
            "snippet": snippet.strip(),
            "title": title.strip(),
            "link": link.strip()
        })
    
    return snippets


def extract_dict(text: str) -> Dict:
    """
    Extracts a dictionary from a given text string.

    This function searches for a JSON-like dictionary format within the text
    and returns it as a Python dictionary.

    Args:
        text (str): The input text containing a dictionary in JSON format.

    Returns:
        dict: The extracted dictionary. If no dictionary is found, returns an empty dictionary.
    
    Example:
        text = 'score: Based on the provided document: {"score": "yes"}'
        result = extract_dict(text)
        # Result: {'score': 'yes'}
    """
    match = re.search(r'{.*}', text)
    if match:
        return json.loads(match.group())
    else:
        return {}
