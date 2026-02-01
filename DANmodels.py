# DANmodels.py

import torch
from torch import nn
import torch.nn.functional as F
from sentiment_data import read_sentiment_examples
from torch.utils.data import Dataset
from utils import Indexer


# Custom collate function to handle variable-length sentences
def collate_fn(batch):
    """Pads variable-length sentences to the same length in a batch."""
    sentences, labels = zip(*batch)
    # Pad sentences to same length in batch
    max_len = max(len(s) for s in sentences)
    if max_len == 0:
        max_len = 1  # Handle empty sentences

    padded = []
    lengths = []
    for s in sentences:
        length = len(s) if len(s) > 0 else 1
        lengths.append(length)
        if len(s) == 0:
            padded.append([1])  # UNK index
        elif len(s) < max_len:
            padded.append(s + [0] * (max_len - len(s)))  # Pad with PAD token
        else:
            padded.append(s)

    return torch.tensor(padded, dtype=torch.long), torch.tensor(lengths, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class SentimentDatasetDAN(Dataset):
    """
    Dataset class for DAN model that converts sentences to sequences of word indices.
    Now returns (sentence, length, label) for proper padding handling.
    """
    def __init__(self, infile, word_embeddings):
        """
        Args:
            infile: Path to the data file
            word_embeddings: WordEmbeddings object containing the word indexer
        """
        # Read the sentiment examples from the input file
        self.examples = read_sentiment_examples(infile)
        self.word_indexer = word_embeddings.word_indexer
        
        # Convert words to indices
        self.indexed_sentences = []
        for ex in self.examples:
            indices = []
            for word in ex.words:
                # Get the index of the word, or UNK if not found
                idx = self.word_indexer.index_of(word)
                if idx == -1:
                    idx = self.word_indexer.index_of("UNK")
                indices.append(idx)
            self.indexed_sentences.append(indices)
        
        # Extract labels
        self.labels = torch.tensor([ex.label for ex in self.examples], dtype=torch.long)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # Return list of indices (collate_fn will handle padding)
        return self.indexed_sentences[idx], self.labels[idx]


class DAN(nn.Module):
    """
    Deep Averaging Network for sentiment classification.
    Averages word embeddings and passes through a feedforward network.
    """
    def __init__(self, word_embeddings, hidden_size, num_classes=2, num_layers=2, 
                 dropout_prob=0.3, dropout_position="embedding", use_pretrained=True):
        """
        Args:
            word_embeddings: WordEmbeddings object
            hidden_size: Size of hidden layer(s)
            num_classes: Number of output classes (2 for binary sentiment)
            num_layers: Number of hidden layers (2 or 3)
            dropout_prob: Dropout probability
            dropout_position: Where to apply dropout - "embedding" or "hidden"
            use_pretrained: If True, use GloVe embeddings; if False, use random initialization
        """
        super(DAN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout_position = dropout_position
        
        # Get embedding dimension
        self.embedding_dim = word_embeddings.get_embedding_length()
        
        # Initialize embedding layer
        if use_pretrained:
            # Use pretrained GloVe embeddings (trainable)
            self.embedding = word_embeddings.get_initialized_embedding_layer(frozen=False)
        else:
            # Use random embeddings (will be trained from scratch)
            vocab_size = len(word_embeddings.word_indexer)
            self.embedding = nn.Embedding(vocab_size, self.embedding_dim, padding_idx=0)
            # Initialize with small random values
            nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
            # Make sure PAD embedding is zero
            self.embedding.weight.data[0].fill_(0)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_prob)
        
        # Build feedforward layers based on num_layers
        if num_layers == 2:
            self.fc1 = nn.Linear(self.embedding_dim, hidden_size)
            self.fc2 = nn.Linear(hidden_size, num_classes)
        elif num_layers == 3:
            self.fc1 = nn.Linear(self.embedding_dim, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, num_classes)
        
        # Log softmax for output (used with NLLLoss)
        self.log_softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x, lengths):
        """
        Forward pass of the DAN model.
        
        Args:
            x: Tensor of word indices, shape (batch_size, seq_length)
            lengths: Tensor of actual lengths, shape (batch_size,)
        
        Returns:
            Log probabilities for each class, shape (batch_size, num_classes)
        """
        device = x.device
        
        # Step 1: Embed the words
        # Shape: (batch_size, seq_length, embedding_dim)
        embedded = self.embedding(x)
        
        # Step 2: Create mask for padding tokens
        batch_size, seq_len = x.shape
        mask = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        mask = mask < lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1).float()  # (batch_size, seq_len, 1)
        
        # Step 3: Average the embeddings, ignoring padding
        masked_embedded = embedded * mask
        summed = masked_embedded.sum(dim=1)  # (batch_size, embedding_dim)
        averaged = summed / lengths.unsqueeze(1).float()  # Average over actual length
        
        # Step 4: Apply dropout and feedforward network
        if self.dropout_position == "embedding":
            x = self.dropout(averaged)
            x = F.relu(self.fc1(x))
        else:
            x = F.relu(self.fc1(averaged))
        
        # Dropout after hidden layer
        if self.dropout_position == "hidden":
            x = self.dropout(x)
        
        if self.num_layers == 2:
            x = self.fc2(x)
        elif self.num_layers == 3:
            x = F.relu(self.fc2(x))
            if self.dropout_position == "hidden":
                x = self.dropout(x)
            x = self.fc3(x)
        
        return self.log_softmax(x)


# Dataset class for DAN with randomly initialized embeddings (Part 1b)
class SentimentDatasetDANRandom(Dataset):
    """
    Dataset for DAN with random embeddings - builds vocabulary from scratch.
    """
    def __init__(self, infile, word_indexer=None):
        # Read examples
        self.examples = read_sentiment_examples(infile)
        
        # Build or use existing word indexer
        if word_indexer is None:
            self.word_indexer = Indexer()
            self.word_indexer.add_and_get_index("PAD")  # Index 0
            self.word_indexer.add_and_get_index("UNK")  # Index 1
            
            # Add all words from training data
            for ex in self.examples:
                for word in ex.words:
                    self.word_indexer.add_and_get_index(word)
        else:
            self.word_indexer = word_indexer
        
        # Convert sentences to indices
        self.indexed_sentences = []
        for ex in self.examples:
            indices = []
            for word in ex.words:
                idx = self.word_indexer.index_of(word)
                if idx == -1:
                    idx = self.word_indexer.index_of("UNK")
                indices.append(idx)
            self.indexed_sentences.append(indices)
        
        # Extract labels
        self.labels = torch.tensor([ex.label for ex in self.examples], dtype=torch.long)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.indexed_sentences[idx], self.labels[idx]


# DAN with random embeddings
class DANRandomEmb(nn.Module):
    """DAN model with randomly initialized embeddings."""
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes=2,
                 dropout_prob=0.3, num_layers=2, dropout_position="embedding"):
        super(DANRandomEmb, self).__init__()
        
        self.num_layers = num_layers
        self.dropout_position = dropout_position
        
        # Random embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        self.embedding.weight.data[0].fill_(0)  # PAD embedding is zero
        
        self.dropout = nn.Dropout(dropout_prob)
        
        # Feedforward network
        if num_layers == 2:
            self.fc1 = nn.Linear(embedding_dim, hidden_size)
            self.fc2 = nn.Linear(hidden_size, num_classes)
        elif num_layers == 3:
            self.fc1 = nn.Linear(embedding_dim, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, num_classes)
        
        self.log_softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x, lengths):
        device = x.device
        embedded = self.embedding(x)
        
        # Mask padding
        batch_size, seq_len = x.shape
        mask = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        mask = mask < lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1).float()
        
        # Average
        masked_embedded = embedded * mask
        summed = masked_embedded.sum(dim=1)
        averaged = summed / lengths.unsqueeze(1).float()
        
        # Feedforward with dropout
        if self.dropout_position == "embedding":
            x = self.dropout(averaged)
            x = F.relu(self.fc1(x))
        else:
            x = F.relu(self.fc1(averaged))
        
        if self.dropout_position == "hidden":
            x = self.dropout(x)
        
        if self.num_layers == 2:
            x = self.fc2(x)
        elif self.num_layers == 3:
            x = F.relu(self.fc2(x))
            if self.dropout_position == "hidden":
                x = self.dropout(x)
            x = self.fc3(x)
        
        return self.log_softmax(x)
