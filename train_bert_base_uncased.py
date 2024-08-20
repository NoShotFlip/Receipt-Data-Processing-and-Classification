import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    BertForTokenClassification,
    BertTokenizerFast,
    AdamW,
    DataCollatorForTokenClassification,
)
import json

label_map = {
    "menu_item": 1,
    "sub_total": 2,
    "total": 3,
    "address": 4,
    "date": 5,
    "time": 6,
    "receipt_id": 7,
    "payment_method": 8,
    "currency": 9,
    "quantity": 10,
    "unit_price": 11,
    "discount": 12,
    "tax": 13,
    "change_due": 14,
    "phone_number": 15,
    "website": 16,
    "item_group": 17,
    "roi": 18,
    "repeating_symbol": 19,
    "is_key": 20,
    "other": 99,
    "O": 0,
}


# Custom Dataset class for tokenized data with a validation check
class TokenizedDataset(Dataset):
    def __init__(self, tokenized_data):
        self.tokenized_data = tokenized_data
        self.skipped_count = 0  # Initialize a counter for skipped examples

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        try:
            item = {
                key: torch.tensor(val) for key, val in self.tokenized_data[idx].items()
            }

            # Validation check: ensure input_ids and labels have the same length
            if len(item["input_ids"]) != len(item["labels"]):
                raise ValueError(
                    f"Length mismatch between input_ids and labels for index {idx}"
                )
            return item
        except Exception as e:
            print(f"Skipping index {idx} due to error: {e}")
            self.skipped_count += 1  # Increment the counter for each skipped example
            return None


# Load your tokenized data from the JSON files
with open("training_dataset_tokenized_800_examples.json", "r") as f:
    training_data = json.load(f)

with open("testing_dataset_tokenized_200_examples.json", "r") as f:
    testing_data = json.load(f)

# Filter out any None values (skipped examples) from the dataset
training_data = [ex for ex in training_data if ex is not None]
testing_data = [ex for ex in testing_data if ex is not None]

# Create datasets
train_dataset = TokenizedDataset(training_data)
test_dataset = TokenizedDataset(testing_data)

# Initialize the data collator with padding for token classification
data_collator = DataCollatorForTokenClassification(
    tokenizer=BertTokenizerFast.from_pretrained("bert-base-uncased")
)


# Create a safe collate_fn that skips empty batches
def safe_collate_fn(batch):
    batch = [ex for ex in batch if ex is not None]
    if len(batch) == 0:
        return None
    return data_collator(batch)


# Create data loaders
train_loader = DataLoader(
    train_dataset, batch_size=8, shuffle=True, collate_fn=safe_collate_fn
)
test_loader = DataLoader(
    test_dataset, batch_size=8, shuffle=False, collate_fn=safe_collate_fn
)

# Load the model
model = BertForTokenClassification.from_pretrained(
    "bert-base-uncased", num_labels=len(label_map)
)

# Move the model to the GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Save the model
model.save_pretrained("./bert-token-classifier")

# Set up the optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
epochs = 3
for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        if batch is None:
            continue  # Skip the batch if it is None (empty after filtering)

        # Move batch to the device
        batch = {k: v.to(device) for k, v in batch.items()}

        try:
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        except Exception as e:
            print(f"Error during training loop: {e}")
            continue  # Skip this batch if an error occurs

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

# Save the model
model.save_pretrained("./bert-token-classifier")

# Evaluation
model.eval()
correct, total = 0, 0
for batch in test_loader:
    if batch is None:
        continue  # Skip the batch if it is None (empty after filtering)

    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        try:
            outputs = model(**batch)
        except Exception as e:
            print(f"Error during evaluation loop: {e}")
            continue  # Skip this batch if an error occurs

    predictions = outputs.logits.argmax(dim=-1)
    labels = batch["labels"]

    # Calculate accuracy
    mask = labels != -100  # Ignore the padding labels
    correct += (predictions[mask] == labels[mask]).sum().item()
    total += mask.sum().item()

print(f"Test Accuracy: {correct / total:.2f}")

# Report the number of skipped examples
print(f"Skipped {train_dataset.skipped_count} problematic examples during training.")
print(f"Skipped {test_dataset.skipped_count} problematic examples during testing.")
