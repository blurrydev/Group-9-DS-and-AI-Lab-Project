import pandas as pd
import json
import os

def fetch_and_prepare_data():
    print("Fetching raw Hindi IndicQA data from Hugging Face Parquet storage...")
    
    # URL to the validation split of IndicQA Hindi subset
    parquet_url = "https://huggingface.co/datasets/ai4bharat/IndicQA/resolve/main/indicqa-hi/validation-00000-of-00001.parquet"
    
    try:
        df = pd.read_parquet(parquet_url)
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    raw_records = []
    print(f"Processing {len(df)} raw rows...")
    
    for idx, row in df.iterrows():
        context = row["context"]
        question = row["question"]
        
        # Extract ground truth answer
        answers = row["answers"]
        answer_text = ""
        if answers and len(answers.get("text", [])) > 0:
            answer_text = answers["text"][0]
            
        raw_records.append({
            "context": context,
            "question": question,
            "original_answer": answer_text
        })
        
    # Save raw records temporarily or for reference
    output_path = "data/raw_hindi_qa.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(raw_records, f, ensure_ascii=False, indent=4)
        
    print(f"Success! Raw data compiled and saved to {output_path}")

if __name__ == "__main__":
    fetch_and_prepare_data()
