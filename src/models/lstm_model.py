import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, n_layers=1, dropout=0.2):
        super(LSTMClassifier, self).__init__()
        
        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 2. LSTM Layer
        # dropout is added between layers if n_layers > 1
        self.lstm = nn.Linear(embed_dim, hidden_dim) # Placeholder for architecture
        self.lstm = nn.LSTM(embed_dim, 
                            hidden_dim, 
                            num_layers=n_layers, 
                            batch_first=True, 
                            dropout=dropout if n_layers > 1 else 0)
        
        # Final Linear Layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        # text shape: [batch_size, seq_len]
        
        embedded = self.dropout(self.embedding(text))
        # embedded shape: [batch_size, seq_len, embed_dim]
        
        # LSTM returns: output, (hidden, cell)
        output, (hidden, cell) = self.lstm(embedded)
        
        # We take the final hidden state of the last layer
        # hidden shape: [n_layers, batch_size, hidden_dim]
        last_hidden = hidden[-1, :, :]
        
        return self.fc(last_hidden)