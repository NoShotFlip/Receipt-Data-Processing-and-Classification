import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    BertForTokenClassification,
    BertTokenizerFast,
    AdamW,
    DataCollatorForTokenClassification,
)
import json


# Custom Dataset class for tokenized data
class TokenizedDataset(Dataset):
    def __init__(self, tokenized_data):
        self.tokenized_data = tokenized_data

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val) for key, val in self.tokenized_data[idx].items()}
        return item


# Load your tokenized data from the JSON file
with open("tokenized_dataset.json", "r") as f:
    tokenized_data = json.load(f)

# Ensure each item has the correct keys
for i in range(len(tokenized_data)):
    assert "input_ids" in tokenized_data[i]
    assert "attention_mask" in tokenized_data[i]
    assert "labels" in tokenized_data[i]
    # Make sure that labels and input_ids have the same length
    assert len(tokenized_data[i]["input_ids"]) == len(tokenized_data[i]["labels"])

# Create dataset
dataset = TokenizedDataset(tokenized_data)

# Initialize the tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# Initialize the data collator with padding for token classification
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# Create data loader
data_loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=data_collator)

# Load the model
model = BertForTokenClassification.from_pretrained(
    "bert-base-uncased", num_labels=21
)  # Adjust num_labels if necessary

# Move the model to the GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Set up the optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
epochs = 1  # Set to 1 epoch for quick testing
for epoch in range(epochs):
    model.train()
    for batch in data_loader:
        # Move batch to the device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Ensure batch has correct keys
        assert "input_ids" in batch and "attention_mask" in batch and "labels" in batch

        # Forward pass
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

# Save the model
model.save_pretrained("./bert-token-classifier")
