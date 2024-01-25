#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys


def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'koushikroy.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample test cases
test_cases = [
    "Verify user login with valid credentials.",
    "Check if the system handles incorrect password gracefully during login.",
    "Test the functionality of the 'Forgot Password' link.",
    "Ensure the system properly displays error messages for invalid inputs.",
    "Verify that user sessions are maintained after successful login."
]

# Function to calculate similarity index
def calculate_similarity(test_cases):
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Transform test cases into TF-IDF feature vectors
    tfidf_matrix = vectorizer.fit_transform(test_cases)

    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return similarity_matrix

# Display similarity matrix
similarity_matrix = calculate_similarity(test_cases)
print("Similarity Matrix:")
print(similarity_matrix)


