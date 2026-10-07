import re
import numpy as np
from collections import defaultdict, Counter

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

def full_preprocess(text, remove_stopwords=True, use_lemmatization=True):
    text = text.lower()

    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)

    tokens = word_tokenize(text)

    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]

    if use_lemmatization:
        tokens = [LEMMATIZER.lemmatize(t) for t in tokens]

    return tokens


def build_ngram_model(tokens, n):
    model = defaultdict(Counter)

    for i in range(len(tokens) - n + 1):
        history = tuple(tokens[i:i + n - 1])
        next_word = tokens[i + n - 1]
        model[history][next_word] += 1


    prob_model = defaultdict(dict)
    for history, counter in model.items():
        total = sum(counter.values())
        for word, count in counter.items():
            prob_model[history][word] = count / total  

    return prob_model



def shannons_predictor(history_phrase, model, n, use_lemmatization=False):
    processed_history = full_preprocess(
        history_phrase, remove_stopwords=False, use_lemmatization=use_lemmatization
    )

    key = tuple(processed_history[-(n - 1):])

    if key in model:
        candidates = model[key]
        sorted_predictions = sorted(
            candidates.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_predictions
    else:
        return []


def generate_text(model, seed_phrase, max_words, n, use_lemmatization=False):
    output_tokens = full_preprocess(
        seed_phrase, remove_stopwords=False, use_lemmatization=use_lemmatization
    )

    for _ in range(max_words):
        key = tuple(output_tokens[-(n - 1):])

        if key in model:
            candidates = model[key]
            choices = list(candidates.keys())
            probabilities = list(candidates.values())
            next_word = np.random.choice(choices, p=probabilities)
            output_tokens.append(next_word)
        else:
            break  

    return ' '.join(output_tokens)



with open('scan_bohemia_corpus.txt', 'r', encoding='utf-8') as f:
    raw_text = f.read()

corpus_tokens = full_preprocess(raw_text, remove_stopwords=False, use_lemmatization=False)

print(f"Total tokens in corpus: {len(corpus_tokens)}")
print(f"Unique vocabulary size: {len(set(corpus_tokens))}")

bigram_model = build_ngram_model(corpus_tokens, n=2)
trigram_model = build_ngram_model(corpus_tokens, n=3)
quadgram_model= build_ngram_model(corpus_tokens, n=4)

print(f"Number of distinct bigram histories:  {len(bigram_model)}")
print(f"Number of distinct trigram histories: {len(trigram_model)}")


test_phrases = ["sherlock", "not a", "irene adler", "she was still", "dog eye"]

print("\n=== BIGRAM predictions (n=2) ===")
for phrase in test_phrases:
    preds = shannons_predictor(phrase, bigram_model, n=2)
    print(f"\nHistory: {phrase!r}")
    for word, prob in preds[:5]:
        print(f"   {word:<15} {prob:.3f}")

print("\n=== TRIGRAM predictions (n=3) ===")
for phrase in test_phrases:
    preds = shannons_predictor(phrase, trigram_model, n=3)
    print(f"\nHistory: {phrase!r}")
    for word, prob in preds[:5]:
        print(f"   {word:<15} {prob:.3f}")

print("\n=== QUADGRAM predictions (n=4) ===")
for phrase in test_phrases:
    preds = shannons_predictor(phrase, trigram_model, n=3)
    print(f"\nHistory: {phrase!r}")
    for word, prob in preds[:5]:
        print(f"   {word:<15} {prob:.3f}")



seeds = input("\nEnter seeds :").split(',')

print("\n=== BIGRAM generation (n=2) ===")
for seed in seeds:
    print(f"\nSeed: {seed!r}")
    print("  ->", generate_text(bigram_model, seed, max_words=25, n=2))

print("\n=== TRIGRAM generation (n=3) ===")
for seed in seeds:
    print(f"\nSeed: {seed!r}")
    print("  ->", generate_text(trigram_model, seed, max_words=25, n=3))

print("\n=== QUADGRAM generation (n=4) ===")
for seed in seeds:
    print(f"\nSeed: {seed!r}")
    print("  ->", generate_text(quadgram_model, seed, max_words=25, n=4))