import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

def get_dataloaders(model_name="bert-base-uncased", batch_size=32, max_len=256):
    """
    Loads IMDb dataset and returns PyTorch DataLoaders.
    For Phase 1 (Uniform Preprocessing), we use a consistent tokenizer.
    """
    # Load the dataset from Hugging Face
    dataset = load_dataset("imdb")
    
    # Load a tokenizer
    # We use a standard one to keep the baseline fair across RNN/Transformer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=max_len
        )

    # Preprocess the data
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # Set format for PyTorch
    # We only need input_ids and label for our basic comparison
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets.set_format("torch")

    # Create DataLoaders
    train_loader = DataLoader(
        tokenized_datasets["train"], 
        shuffle=True, 
        batch_size=batch_size
    )
    test_loader = DataLoader(
        tokenized_datasets["test"], 
        batch_size=batch_size
    )

    return train_loader, test_loader, tokenizer.vocab_size

if __name__ == "__main__":
    # Quick test to verify shapes
    train_loader, _, vocab_size = get_dataloaders(batch_size=8)
    batch = next(iter(train_loader))
    print(f"Input IDs shape: {batch['input_ids'].shape}")
    print(f"Labels shape: {batch['label'].shape}")
    print(f"Vocab size: {vocab_size}")