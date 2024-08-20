from transformers import BertTokenizerFast
from datasets import load_dataset
import torch
import json

# Load the CORD dataset and extract only the ground truth list
# training set only
# ds = load_dataset("naver-clova-ix/cord-v2")
# ds = ds["train"]["ground_truth"]  # List of JSON strings

# testing set only, combined "dev" and "test sets to just make a single set
ds = load_dataset("naver-clova-ix/cord-v2")
ds_dev = ds["validation"]["ground_truth"].copy()
ds_test = ds["test"]["ground_truth"].copy()
ds_combined = ds_dev + ds_test
ds = ds_combined

# for testing only
# ds = load_dataset("naver-clova-ix/cord-v2")["train"][:5]
# ds = ds["ground_truth"]

# Initialize the BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# Initialize the BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# Define a comprehensive mapping from NER tags to numerical labels
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
    "O": 0,  # 'O' typically represents tokens that are not part of any entity
}


def parse_ground_truth(json_string):
    ground_truth = json.loads(json_string)
    tokens = []
    labels = []

    # Ensure "gt_parse" and "text_data" keys exist
    if "gt_parse" in ground_truth and "text_data" in ground_truth["gt_parse"]:
        for entity in ground_truth["gt_parse"]["text_data"]:
            entity_text = entity.get("text", "")
            entity_type = entity.get("label", "O")

            entity_tokens = tokenizer.tokenize(entity_text)
            tokens.extend(entity_tokens)

            # Assign label to each token
            labels.extend([label_map.get(entity_type, 0)] * len(entity_tokens))
    else:
        print(
            f"Warning: 'gt_parse' or 'text_data' not found in ground_truth: {json_string}"
        )

    return tokens, labels


def tokenize_and_align_labels(tokens, labels):
    tokenized_inputs = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    aligned_labels = []
    word_ids = tokenized_inputs.word_ids()
    previous_word_idx = None
    label_ids = []

    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)  # Padding label
        elif word_idx != previous_word_idx:
            label_ids.append(labels[word_idx])  # Assign label to the first subword
        else:
            label_ids.append(-100)  # Ignore subsequent subwords
        previous_word_idx = word_idx

    tokenized_inputs["labels"] = torch.tensor(label_ids)
    return tokenized_inputs


tokenized_dataset = []

for idx, ex in enumerate(ds):
    tokens, labels = parse_ground_truth(ex)
    tokenized_result = tokenize_and_align_labels(tokens, labels)
    tokenized_dataset.append(
        {
            "input_ids": tokenized_result["input_ids"].tolist(),
            "attention_mask": tokenized_result["attention_mask"].tolist(),
            "labels": tokenized_result["labels"].tolist(),
        }
    )

# Save the tokenized dataset to a JSON file
with open("tokenized_dataset.json", "w") as f:
    json.dump(tokenized_dataset, f)

print("Preprocessing completed and saved to tokenized_dataset.json")
