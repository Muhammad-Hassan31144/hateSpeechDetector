import re
import string

def preprocess_text(text):
    # Validate input type
    if not isinstance(text, str):
        raise ValueError("Input text must be a string.")
    
    # Validate input length
    max_length = 10000  # Set a reasonable limit for text length
    if len(text) > max_length:
        raise ValueError(f"Input text is too long. Maximum length is {max_length} characters.")

    # Check for harmful patterns
    harmful_patterns = [
        r'<script.*?>.*?</script.*?>',  # Script tags
        r'(?:\b(?:SELECT|INSERT|UPDATE|DELETE|DROP|ALTER)\b\s+)',  # SQL keywords
        r'(?:<[^>]*>)'  # HTML tags
    ]
    for pattern in harmful_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            raise ValueError("Input text contains potentially harmful content.")

    # Lowercase the text
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())

    return text
