import re
from typing import List, Tuple
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from langdetect import detect
import time

# Download required NLTK data
try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("punkt")
    nltk.download("stopwords")


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extract key terms from text using frequency analysis"""
    # Tokenize and clean text
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word.isalnum() and word not in stop_words]

    # Calculate word frequencies
    freq_dist = nltk.FreqDist(words)
    return [word for word, _ in freq_dist.most_common(max_keywords)]


def detect_language(text: str) -> str:
    """Detect the primary language of the text"""
    try:
        return detect(text)
    except:
        return "unknown"


def count_words(text: str) -> int:
    """Count the number of words in the text"""
    return len(word_tokenize(text))


def chunk_text(
    text: str, chunk_size: int = 1000, overlap: int = 100
) -> List[Tuple[str, int]]:
    """Split text into overlapping chunks for processing"""
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_size = 0
    chunk_number = 0

    for sentence in sentences:
        sentence_size = len(sentence.split())

        if current_size + sentence_size > chunk_size and current_chunk:
            # Store current chunk
            chunk_text = " ".join(current_chunk)
            chunks.append((chunk_text, chunk_number))
            chunk_number += 1

            # Start new chunk with overlap
            overlap_size = 0
            overlap_chunk = []

            for prev_sentence in reversed(current_chunk):
                if overlap_size + len(prev_sentence.split()) > overlap:
                    break
                overlap_chunk.insert(0, prev_sentence)
                overlap_size += len(prev_sentence.split())

            current_chunk = overlap_chunk
            current_size = overlap_size

        current_chunk.append(sentence)
        current_size += sentence_size

    # Add the last chunk if it exists
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunks.append((chunk_text, chunk_number))

    return chunks


def clean_text(text: str) -> str:
    """Clean and normalize text"""
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove special characters but keep punctuation
    text = re.sub(r"[^\w\s.,!?-]", "", text)

    # Normalize whitespace around punctuation
    text = re.sub(r"\s*([.,!?-])\s*", r"\1 ", text)

    return text.strip()


def measure_processing_time(func):
    """Decorator to measure processing time"""

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        processing_time = end_time - start_time
        if isinstance(result, tuple):
            return (*result, processing_time)
        return result, processing_time

    return wrapper
