import torch
import torch.nn as nn

class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(RNNClassifier, self).__init__()
        
        # 1. Embedding Layer: Turns word indices into dense vectors
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 2. RNN Layer: Processes the sequence step-by-step
        # batch_first=True means input shape is (batch, seq, feature)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        
        # 3. Fully Connected Layer: Maps hidden state to sentiment (0 or 1)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text):
        # text shape: [batch_size, seq_len]
        
        embedded = self.embedding(text) 
        # embedded shape: [batch_size, seq_len, embed_dim]
        
        output, hidden = self.rnn(embedded)
        # output shape: [batch_size, seq_len, hidden_dim]
        # hidden shape: [1, batch_size, hidden_dim]
        
        # We only care about the last hidden state for classification
        # 'hidden' contains the final state after the last word
        last_hidden = hidden.squeeze(0)
        
        return self.fc(last_hidden)