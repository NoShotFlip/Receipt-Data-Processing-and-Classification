from pprint import pprint
from datasets import load_dataset

# Load the CORD dataset using the datasets library
ds = load_dataset("naver-clova-ix/cord-v2")

# Print the structure of the dataset
print("Dataset Structure:")
print(ds)

# Inspect a few examples from the training set
print("\nInspecting a few examples from the training set:")

# Inspect the first few examples
for i in range(7):
    print(f"\nExample {i+1}:")
    pprint(ds["train"][i])

    # Print keys and the length of the lists/tensors for input_ids and labels (if available)
    if "input_ids" in ds["train"][i]:
        print(f"Input IDs Length: {len(ds['train'][i]['input_ids'])}")
    if "labels" in ds["train"][i]:
        print(f"Labels Length: {len(ds['train'][i]['labels'])}")

    # Check if there are any other important fields
    for key in ds["train"][i].keys():
        print(
            f"Field: {key}, Length: {len(ds['train'][i][key]) if isinstance(ds['train'][i][key], list) else 'N/A'}"
        )
