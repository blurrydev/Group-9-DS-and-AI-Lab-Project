import json
import os

def validate_dataset(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
        
    print(f"Analyzing {filepath}...")
    
    total_original_tokens = 0
    total_preserved_tokens = 0
    total_examples = 0
    answers_preserved = 0

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            total_examples += 1
            
            original_len = len(data["tokens"])
            preserved_len = sum(data["labels"])
            
            total_original_tokens += original_len
            total_preserved_tokens += preserved_len
            
            # Reconstruct the text marked as preserved
            compressed_words = [word for idx, word in enumerate(data["tokens"]) if data["labels"][idx] == 1]
            compressed_sentence = " ".join(compressed_words)
            
            # Verify if the answer substring is safely preserved
            if data["original_answer"].lower() in compressed_sentence.lower():
                answers_preserved += 1

    if total_examples == 0:
        print("No data items found in file.")
        return

    compression_ratio = 100 - ((total_preserved_tokens / total_original_tokens) * 100)
    retention_accuracy = (answers_preserved / total_examples) * 100

    print("\n=== DATASET METRICS ===")
    print(f"Total Validation Examples: {total_examples}")
    print(f"Average Original Length:   {total_original_tokens / total_examples:.0f} tokens")
    print(f"Average Compressed Length: {total_preserved_tokens / total_examples:.0f} tokens")
    print(f"Total Compression:       {compression_ratio:.1f}% reduction in prompt size")
    print(f"Answer Retention Rate:   {retention_accuracy:.1f}%\n")

if __name__ == "__main__":
    validate_dataset("data/train_indicqa.jsonl")
    validate_dataset("data/val_indicqa.jsonl")
