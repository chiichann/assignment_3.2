##  Cherilyn Marie Deocampo 
##  Assignment 3.2

from nltk import bigrams
from nltk.tokenize import word_tokenize
from collections import Counter

# Extracting bi-grams from the text
text = "I love machine learning and artificial intelligence."
tokens = word_tokenize(text.lower())  # Tokenizing words
print("Tokens:", tokens)

# Generating bigram tokens
bigram_list = list(bigrams(tokens))  # Generating bigrams
print("Bigrams:", bigram_list)

# Bigram Model with Laplace Smoothing on probability calculation
def bigram_probabilities(text):
    tokens = word_tokenize(text.lower()) 
    bigram_counts = Counter(bigrams(tokens))
    unigram_counts = Counter(tokens)
    
    Vocab = len(unigram_counts)  # Size of Vocabulary
    
    # Computing probabilities using Laplace Smoothing
    bigram_probs = {bigram: (count + 1) / (unigram_counts[bigram[0]] + V)
                    for bigram, count in bigram_counts.items()}
    
    return bigram_probs, Vocab

text = "the dog barks. the dog runs. the cat meows."

# Compute bigram probabilities with Laplace smoothing
bigram_probs, Vocab = bigram_probabilities(text)

# Print bigram probabilities
print("\nBigram Probabilities with Laplace Smoothing:")
for bigram, prob in bigram_probs.items():
    print(f"P({bigram[1]} | {bigram[0]}) = {prob:.4f}")

# Text Prediction
def predict_next_word(bigram_probs, current_word):
    candidates = {k[1]: v for k, v in bigram_probs.items() if k[0] == current_word}
    return max(candidates, key=candidates.get) if candidates else None

# Predict the next word
predicted_word = predict_next_word(bigram_probs, "barks")
print(f"\nPredicted next word after 'barks': {predicted_word}")

# Text Generation
def generate_bigram_text(bigram_probs, start_word, length=10):
    current_word = start_word.lower()
    generated_text = [current_word]

    for _ in range(length):
        candidates = {k[1]: v for k, v in bigram_probs.items() if k[0] == current_word}
        if not candidates:
            break
        current_word = max(candidates, key=candidates.get)
        generated_text.append(current_word)
    
    return " ".join(generated_text)

# Generate text using the bigram model
print("\nGenerated Text:", generate_bigram_text(bigram_probs, "the", 10))
