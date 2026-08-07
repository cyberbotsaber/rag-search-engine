import string

from nltk.stem import PorterStemmer


PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)
stemmer = PorterStemmer()


def tokenize_text(text: str) -> list[str]:
    cleaned_text = text.lower().translate(PUNCTUATION_TABLE)

    return [
        token
        for token in cleaned_text.split()
        if token
    ]


def load_stopwords() -> set[str]:
    with open("data/stopwords.txt", "r", encoding="utf-8") as file:
        stopword_lines = file.read().splitlines()

    stopwords: set[str] = set()

    for line in stopword_lines:
        stopwords.update(tokenize_text(line))

    return stopwords


STOPWORDS = load_stopwords()


def preprocess_text(text: str) -> list[str]:
    tokens = tokenize_text(text)

    filtered_tokens = [
        token
        for token in tokens
        if token not in STOPWORDS
    ]

    return [
        stemmer.stem(token)
        for token in filtered_tokens
    ]

def tokenize_term(term: str) -> str:
    tokens = preprocess_text(term)

    if len(tokens) != 1:
        raise ValueError("Term must produce exactly one token")

    return tokens[0]
