import re
import string
import json
import contractions
import emoji
import spacy
import nltk
from sklearn.base import BaseEstimator, TransformerMixin

# Ensure spaCy model is available
try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except OSError:
    import subprocess, sys
    print("[INFO] spaCy model not found. Downloading 'en_core_web_sm'...")
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# Ensure NLTK stopwords are available
try:
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    print("[INFO] NLTK stopwords not found. Downloading...")
    nltk.download("stopwords")
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words("english"))

URL_RE = re.compile(r'http\S+|www\.\S+')
MENTION_RE = re.compile(r'@\w+')
HASHTAG_RE = re.compile(r'#(\w+)')
ELONG_RE = re.compile(r'(.)\1{2,}')

def remove_urls(txt: str) -> str:
    return URL_RE.sub('', txt)

def remove_mentions(txt: str) -> str:
    return MENTION_RE.sub('', txt)

def handle_hashtags(txt: str) -> str:
    return HASHTAG_RE.sub(r'\1', txt)

def remove_emoji(txt: str) -> str:
    try:
        return emoji.replace_emoji(txt, replace='')
    except Exception:
        return emoji.demojize(txt).replace(":", " ")

def normalize_elongation(word: str) -> str:
    return ELONG_RE.sub(r'\1\1', word)

def expand_contractions(txt: str) -> str:
    return contractions.fix(txt)

def remove_punctuation(txt: str) -> str:
    return txt.translate(str.maketrans('', '', string.punctuation))

def tokenize_lemmatize(txt: str) -> str:
    doc = nlp(txt)
    tokens = []
    for token in doc:
        w = token.lemma_.lower().strip()
        if w and w not in STOPWORDS and not token.is_punct and not token.is_space:
            tokens.append(w)
    return " ".join(tokens)  

class TextNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self, slang_path: str = None, keep_emojis: bool = False, lowercase: bool = True):
        self.slang_path = slang_path
        self.keep_emojis = keep_emojis
        self.lowercase = lowercase
        self.slang_map = {}
        if slang_path:
            try:
                with open(slang_path, 'r', encoding='utf-8') as f:
                    self.slang_map = json.load(f)
            except FileNotFoundError:
                print(f"[WARN] Slang file {slang_path} not found. Continuing without slang replacement.")

    def fit(self, X, y=None):
        return self

    def replace_slang(self, txt: str) -> str:
        toks = txt.split()
        fixed_words = []
        for t in toks:
            key = t.lower()
            if key in self.slang_map:
                fixed_words.append(self.slang_map[key])
            else:
                fixed_words.append(t)
        return " ".join(fixed_words)

    def transform(self, X, y=None):
        output = []
        for raw_txt in X:
            if not isinstance(raw_txt, str):
                raw_txt = str(raw_txt)
            txt = raw_txt.strip()
            if self.lowercase:
                txt = txt.lower()
            txt = expand_contractions(txt)
            txt = remove_urls(txt)
            txt = remove_mentions(txt)
            txt = handle_hashtags(txt)
            if not self.keep_emojis:
                txt = remove_emoji(txt)
            txt = re.sub(r'\s+', ' ', txt)
            txt = self.replace_slang(txt)
            txt = " ".join(normalize_elongation(w) for w in txt.split())
            txt = remove_punctuation(txt)
            txt = tokenize_lemmatize(txt)
            output.append(txt)
        return output