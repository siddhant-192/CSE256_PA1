"""
Byte Pair Encoding (BPE) Implementation
Based on Sennrich et al. (2016) "Neural Machine Translation of Rare Words with Subword Units"

This module implements BPE for subword tokenization.
"""

import re
from collections import Counter, defaultdict
import pickle


class BytePairEncoding:
    """
    Byte Pair Encoding tokenizer for subword segmentation.
    
    BPE iteratively merges the most frequent pair of characters or character sequences
    to build a vocabulary of subwords.
    """
    
    def __init__(self, num_merges=1000):
        """
        Initialize BPE tokenizer.
        
        Args:
            num_merges: Number of merge operations to perform (controls vocab size)
        """
        self.num_merges = num_merges
        self.merges = []  # List of (pair, merged) tuples in order of learning
        self.vocab = set()  # Final vocabulary
        self.word_to_tokens = {}  # Cache for tokenized words
        
    def get_stats(self, word_freq):
        """
        Count frequency of adjacent character pairs across all words.
        
        Args:
            word_freq: Dict mapping word (as tuple of chars/subwords) to frequency
            
        Returns:
            Counter of (char1, char2) pairs to their frequencies
        """
        pairs = defaultdict(int)
        for word, freq in word_freq.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs
    
    def merge_pair(self, pair, word_freq):
        """
        Merge all occurrences of a character pair in the vocabulary.
        
        Args:
            pair: Tuple of (char1, char2) to merge
            word_freq: Dict mapping word to frequency
            
        Returns:
            Updated word_freq with merged pairs
        """
        new_word_freq = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        
        for word, freq in word_freq.items():
            # Replace all occurrences of the pair
            new_word = word.replace(bigram, replacement)
            new_word_freq[new_word] = freq
            
        return new_word_freq
    
    def learn_bpe(self, texts):
        """
        Learn BPE merges from training texts.
        
        Args:
            texts: List of text strings (sentences)
        """
        print(f"\nLearning BPE with {self.num_merges} merges...")
        
        # Step 1: Initialize vocabulary with characters
        # Each word is represented as space-separated characters with </w> end marker
        word_freq = Counter()
        for text in texts:
            # Split into words and count frequencies
            words = text.lower().split()
            for word in words:
                # Add </w> marker to indicate end of word
                word_with_marker = ' '.join(list(word)) + ' </w>'
                word_freq[word_with_marker] += 1
        
        print(f"Initial vocabulary size (characters): {len(set(''.join(word_freq.keys()).split()))}")
        print(f"Number of unique words: {len(word_freq)}")
        
        # Step 2: Iteratively merge most frequent pairs
        for i in range(self.num_merges):
            # Get frequency of all adjacent pairs
            pairs = self.get_stats(word_freq)
            
            if not pairs:
                print(f"No more pairs to merge. Stopping at {i} merges.")
                break
            
            # Find most frequent pair
            best_pair = max(pairs, key=pairs.get)
            
            # Merge this pair in all words
            word_freq = self.merge_pair(best_pair, word_freq)
            
            # Record this merge operation
            self.merges.append(best_pair)
            
            if (i + 1) % 100 == 0:
                print(f"  Merge {i + 1}/{self.num_merges}: {best_pair} (freq: {pairs[best_pair]})")
        
        # Step 3: Build final vocabulary
        self.vocab = set()
        for word in word_freq.keys():
            self.vocab.update(word.split())
        
        print(f"Final vocabulary size: {len(self.vocab)}")
        print(f"Learned {len(self.merges)} merge operations")
        
    def tokenize_word(self, word):
        """
        Tokenize a single word using learned BPE merges.
        
        Args:
            word: String word to tokenize
            
        Returns:
            List of subword tokens
        """
        # Check cache first
        if word in self.word_to_tokens:
            return self.word_to_tokens[word]
        
        # Start with character-level representation
        word = word.lower()
        word_chars = ' '.join(list(word)) + ' </w>'
        
        # Apply merges in order
        for pair in self.merges:
            bigram = ' '.join(pair)
            if bigram in word_chars:
                word_chars = word_chars.replace(bigram, ''.join(pair))
        
        tokens = word_chars.split()
        
        # Cache result
        self.word_to_tokens[word] = tokens
        
        return tokens
    
    def tokenize(self, text):
        """
        Tokenize a text into subword units.
        
        Args:
            text: Input text string
            
        Returns:
            List of subword tokens
        """
        words = text.split()
        tokens = []
        for word in words:
            tokens.extend(self.tokenize_word(word))
        return tokens
    
    def get_vocab(self):
        """Get the learned vocabulary."""
        return sorted(list(self.vocab))
    
    def save(self, filepath):
        """Save BPE model to file."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'num_merges': self.num_merges,
                'merges': self.merges,
                'vocab': self.vocab,
                'word_to_tokens': self.word_to_tokens
            }, f)
        print(f"BPE model saved to {filepath}")
    
    def load(self, filepath):
        """Load BPE model from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.num_merges = data['num_merges']
            self.merges = data['merges']
            self.vocab = data['vocab']
            self.word_to_tokens = data['word_to_tokens']
        print(f"BPE model loaded from {filepath}")
        print(f"Vocabulary size: {len(self.vocab)}")


class BPEVocabulary:
    """
    Vocabulary for BPE tokens with special tokens.
    """
    
    def __init__(self, bpe_vocab):
        """
        Initialize vocabulary from BPE tokens.
        
        Args:
            bpe_vocab: List of BPE tokens
        """
        # Add special tokens
        self.special_tokens = ['<PAD>', '<UNK>']
        
        # Build vocabulary: special tokens first, then BPE tokens
        self.token_to_idx = {}
        self.idx_to_token = {}
        
        idx = 0
        for token in self.special_tokens:
            self.token_to_idx[token] = idx
            self.idx_to_token[idx] = token
            idx += 1
        
        for token in sorted(bpe_vocab):
            if token not in self.token_to_idx:
                self.token_to_idx[token] = idx
                self.idx_to_token[idx] = token
                idx += 1
        
        self.pad_idx = self.token_to_idx['<PAD>']
        self.unk_idx = self.token_to_idx['<UNK>']
    
    def __len__(self):
        return len(self.token_to_idx)
    
    def __getitem__(self, token):
        """Get index for token, return UNK if not found."""
        return self.token_to_idx.get(token, self.unk_idx)
    
    def get_token(self, idx):
        """Get token for index."""
        return self.idx_to_token.get(idx, '<UNK>')


def train_bpe_from_file(filepath, num_merges=1000, save_path=None):
    """
    Train BPE model from a text file.
    
    Args:
        filepath: Path to training text file
        num_merges: Number of merge operations
        save_path: Optional path to save trained model
        
    Returns:
        Trained BytePairEncoding object
    """
    print(f"\nTraining BPE from {filepath}")
    print(f"Target merges: {num_merges}")
    
    # Read training texts
    texts = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            # Extract sentence part (remove label)
            parts = line.strip().split('\t')
            if len(parts) == 2:
                texts.append(parts[1])
    
    print(f"Loaded {len(texts)} training sentences")
    
    # Train BPE
    bpe = BytePairEncoding(num_merges=num_merges)
    bpe.learn_bpe(texts)
    
    # Save if path provided
    if save_path:
        bpe.save(save_path)
    
    return bpe


# Test code commented out - uncomment to run tests
# if __name__ == "__main__":
#     # Test BPE implementation
#     print("Testing BPE Implementation")
#     print("=" * 60)
#     
#     # Small test
#     test_texts = [
#         "low lower lowest",
#         "new newer newest",
#         "wide wider widest"
#     ]
#     
#     bpe = BytePairEncoding(num_merges=10)
#     bpe.learn_bpe(test_texts)
#     
#     print("\nVocabulary:", bpe.get_vocab())
#     
#     print("\nTokenization examples:")
#     for text in ["lowest", "newer", "wider", "low", "unknown"]:
#         tokens = bpe.tokenize_word(text)
#         print(f"  '{text}' -> {tokens}")
#     
#     # Train on actual data
#     print("\n" + "=" * 60)
#     print("Training on actual sentiment data...")
#     bpe_model = train_bpe_from_file(
#         'data/train.txt',
#         num_merges=500,
#         save_path='data/bpe_500.pkl'
#     )
#     
#     print("\nSample tokenizations:")
#     test_words = ["excellent", "terrible", "amazing", "disappointing", "wonderful"]
#     for word in test_words:
#         tokens = bpe_model.tokenize_word(word)
#         print(f"  '{word}' -> {tokens}")
