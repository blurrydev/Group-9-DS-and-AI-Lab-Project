import os
import json
import time
from openai import OpenAI

def compress_context(context, question, client):
    system_prompt = """You are a highly efficient text compressor for Hindi text. 
    Given a [Question] and a [Context], aggressively remove all words from the [Context] 
    that are irrelevant to the [Question]. 
    CRITICAL RULES:
    1. Output ONLY the compressed Hindi text. No introductions, no explanations.
    2. Do NOT change the grammar, do NOT translate to English, and do NOT add new words.
    3. Ensure the exact information needed to answer the question remains intact in the output."""
    
    user_prompt = f"[Question]: {question}\n\n[Context]: {context}"
    
    completion = client.chat.completions.create(
        model="qwen/qwen3.5-397b-a17b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1, 
        top_p=0.95,
        max_tokens=2048,
        stream=False
    )
    
    content = completion.choices[0].message.content
    if content is None:
        raise ValueError("API returned None (Possible safety filter trigger).")
        
    return content.strip()

def align_tokens(original_text, compressed_text):
    # Basic whitespace tokenization for aligning arrays
    orig_tokens = original_text.split()
    comp_tokens = set(compressed_text.split())
    
    # Generate binary labels (1 if preserved, 0 if dropped)
    labels = [1 if token in comp_tokens else 0 for token in orig_tokens]
    return orig_tokens, labels

def run_pipeline():
    # 1. Load the variables from the .env file
    load_dotenv()
    
    api_key = os.getenv("API_KEY")
    base_url = os.getenv("BASE_URL")
    
    if not api_key or not base_url:
        raise ValueError("ERROR: Missing API_KEY or BASE_URL in .env file!")

    # 2. Inject the variables into the client
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=60.0
    )

    raw_data_path = "data/raw_hindi_qa.json"
    output_path = "data/master_processed_data.jsonl"

    if not os.path.exists(raw_data_path):
        print(f"Error: {raw_data_path} not found. Run fetch_data.py first!")
        return

    with open(raw_data_path, "r", encoding="utf-8") as f:
        raw_items = json.load(f)

    # State Management: Check for existing progress
    processed_count = 0
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            processed_count = sum(1 for _ in f)
        print(f"Found existing data! Resuming from row {processed_count}...")
    else:
        print("Starting a fresh distillation pipeline...")

    max_retries = 3
    base_wait_time = 10

    with open(output_path, "a", encoding="utf-8") as f:
        for idx in range(processed_count, len(raw_items)):
            item = raw_items[idx]
            print(f"Processing {idx + 1}/{len(raw_items)}...", flush=True)
            
            success = False
            for attempt in range(max_retries):
                try:
                    compressed_text = compress_context(item["context"], item["question"], client)
                    tokens, labels = align_tokens(item["context"], compressed_text)
                    
                    output_record = {
                        "tokens": tokens,
                        "labels": labels,
                        "question": item["question"],
                        "original_answer": item["original_answer"]
                    }
                    
                    f.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                    f.flush() # Forces immediate write to disk
                    success = True
                    break
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    if any(msg in error_msg for msg in ["429", "too many requests", "timed out", "timeout"]):
                        wait_time = base_wait_time * (2 ** attempt)
                        print(f"Network/Rate limit hiccup! Sleeping for {wait_time}s... (Attempt {attempt + 1}/{max_retries})", flush=True)
                        time.sleep(wait_time)
                    else:
                        print(f"Skipping row {idx + 1} due to unrecoverable error: {e}", flush=True)
                        break
            
            if not success and idx not in range(processed_count, len(raw_items)):
                # If it completely failed all retries, log it and keep moving
                print(f"Skipping row {idx + 1} permanently after max retries.")

    print("Pipeline run complete!")

if __name__ == "__main__":
    run_pipeline()
