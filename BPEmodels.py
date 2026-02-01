"""
BPE-based Dataset and Model for DAN
"""

import torch
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from BPE import BytePairEncoding, BPEVocabulary


class SentimentDatasetBPE(Dataset):
    """Dataset for sentiment analysis with BPE tokenization."""
    
    def __init__(self, filepath, bpe_model, vocab):
        """
        Initialize dataset with BPE tokenization.
        
        Args:
            filepath: Path to data file
            bpe_model: Trained BytePairEncoding object
            vocab: BPEVocabulary object
        """
        self.bpe = bpe_model
        self.vocab = vocab
        self.examples = []
        
        # Load and tokenize data
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    label = int(parts[0])
                    text = parts[1]
                    
                    # Tokenize with BPE
                    tokens = self.bpe.tokenize(text)
                    
                    # Convert to indices
                    indices = [self.vocab[token] for token in tokens]
                    
                    self.examples.append((indices, label))
        
        print(f"Loaded {len(self.examples)} examples")
        
        # Compute statistics
        lengths = [len(ex[0]) for ex in self.examples]
        print(f"  Token sequence lengths: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.1f}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        indices, label = self.examples[idx]
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def collate_fn_bpe(batch):
    """
    Collate function for BPE tokenized data with padding.
    
    Args:
        batch: List of (token_indices, label) tuples
        
    Returns:
        Tuple of (padded_sequences, lengths, labels)
    """
    # Separate sequences and labels
    sequences = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    
    # Get lengths
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    
    # Pad sequences
    max_len = lengths.max().item()
    padded = torch.zeros(len(sequences), max_len, dtype=torch.long)
    
    for i, seq in enumerate(sequences):
        padded[i, :len(seq)] = seq
    
    return padded, lengths, labels


class DANBPE(nn.Module):
    """
    Deep Averaging Network with BPE subword tokenization.
    
    Since BPE uses subwords, we cannot use pretrained word embeddings.
    Embeddings are initialized randomly and trained from scratch.
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes=2, 
                 num_layers=2, dropout_prob=0.3, dropout_position="embedding", 
                 pad_idx=0):
        """
        Initialize BPE-based DAN.
        
        Args:
            vocab_size: Size of BPE vocabulary
            embedding_dim: Dimension of subword embeddings
            hidden_size: Size of hidden layers
            num_classes: Number of output classes (2 for binary)
            num_layers: Number of feedforward layers (2 or 3)
            dropout_prob: Dropout probability
            dropout_position: Where to apply dropout ("embedding" or "hidden")
            pad_idx: Index of padding token
        """
        super(DANBPE, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_position = dropout_position
        self.pad_idx = pad_idx
        
        # Randomly initialized embeddings (no pretrained available for BPE)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        # Feedforward network
        self.fc1 = nn.Linear(embedding_dim, hidden_size)
        
        if num_layers == 2:
            self.fc2 = nn.Linear(hidden_size, num_classes)
        elif num_layers == 3:
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_prob)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        
        if self.num_layers == 2:
            nn.init.xavier_uniform_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)
        elif self.num_layers == 3:
            nn.init.xavier_uniform_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)
            nn.init.xavier_uniform_(self.fc3.weight)
            nn.init.zeros_(self.fc3.bias)
    
    def forward(self, x, lengths):
        """
        Forward pass.
        
        Args:
            x: Padded token indices [batch_size, max_len]
            lengths: Actual lengths of sequences [batch_size]
            
        Returns:
            Log probabilities [batch_size, num_classes]
        """
        device = x.device
        batch_size, seq_len = x.shape
        
        # Embed tokens
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        
        # Create mask for padding
        mask = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        mask = mask < lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1).float()  # [batch_size, seq_len, 1]
        
        # Average embeddings (ignoring padding)
        masked_embedded = embedded * mask
        summed = masked_embedded.sum(dim=1)  # [batch_size, embedding_dim]
        averaged = summed / lengths.unsqueeze(1).float()  # [batch_size, embedding_dim]
        
        # Feedforward with dropout
        if self.dropout_position == "embedding":
            x = self.dropout(averaged)
            x = F.relu(self.fc1(x))
        else:
            x = F.relu(self.fc1(averaged))
        
        if self.dropout_position == "hidden":
            x = self.dropout(x)
        
        # Output layers
        if self.num_layers == 2:
            x = self.fc2(x)
        elif self.num_layers == 3:
            x = F.relu(self.fc2(x))
            if self.dropout_position == "hidden":
                x = self.dropout(x)
            x = self.fc3(x)
        
        return F.log_softmax(x, dim=1)
    
    def get_num_params(self):
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


# Test code commented out - uncomment to run tests
# if __name__ == "__main__":
#     # Test BPE dataset and model
#     print("Testing BPE Dataset and Model")
#     print("=" * 60)
#     
#     from BPE import train_bpe_from_file
#     
#     # Train small BPE model
#     bpe = train_bpe_from_file('data/train.txt', num_merges=100)
#     vocab = BPEVocabulary(bpe.get_vocab())
#     
#     print(f"\nBPE Vocabulary size: {len(vocab)}")
#     
#     # Create dataset
#     train_data = SentimentDatasetBPE('data/train.txt', bpe, vocab)
#     
#     # Test collate function
#     from torch.utils.data import DataLoader
#     train_loader = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=collate_fn_bpe)
#     
#     print("\nTesting dataloader:")
#     for X, lengths, y in train_loader:
#         print(f"  Batch shape: {X.shape}, Lengths: {lengths}, Labels: {y}")
#         break
#     
#     # Test model
#     print("\nTesting model:")
#     model = DANBPE(
#         vocab_size=len(vocab),
#         embedding_dim=128,
#         hidden_size=128,
#         num_layers=2,
#         dropout_position="embedding"
#     )
#     
#     print(f"  Model parameters: {model.get_num_params():,}")
#     
#     # Forward pass
#     output = model(X, lengths)
#     print(f"  Output shape: {output.shape}")
#     print(f"  Output (log probs): {output[0]}")
