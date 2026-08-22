import torch
import gradio as gr
import spaces

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
)


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "nnnhitesh/xlm-roberta-prompt-compressor"
MAX_LENGTH = 512


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True
)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading model...")

model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME
)

model.to(device)
model.eval()

print("Model loaded successfully.")
print("Labels:", model.config.id2label)
print("Label mapping:", model.config.label2id)


# ============================================================
# PREDICT FUNCTION
# ============================================================
@spaces.GPU
def predict(question, context):

    if not question or not question.strip():
        raise gr.Error("Question cannot be empty.")

    if not context or not context.strip():
        raise gr.Error("Context cannot be empty.")

    # EXACT SAME FORMAT AS YOUR WORKING FUNCTION
    tokens = context.split()

    encoding = tokenizer(
        question.split(),
        tokens,
        is_split_into_words=True,
        truncation="only_second",
        max_length=512,
        return_tensors="pt"
    )

    word_ids = encoding.word_ids(batch_index=0)
    sequence_ids = encoding.sequence_ids(batch_index=0)

    encoding = {
        k: v.to(device)
        for k, v in encoding.items()
    }

    with torch.no_grad():
        outputs = model(**encoding)

    # Predictions
    pred_labels = (
        outputs.logits
        .argmax(dim=-1)
        .squeeze()
        .cpu()
        .tolist()
    )

    compressed_tokens = []
    previous_word = None

    keep_count = 0
    remove_count = 0

    for pred, word_id, seq_id in zip(
        pred_labels,
        word_ids,
        sequence_ids
    ):

        if word_id is None:
            continue

        if seq_id != 1:
            continue

        if word_id == previous_word:
            continue

        if pred == 1:
            compressed_tokens.append(tokens[word_id])
            keep_count += 1
        else:
            remove_count += 1

        previous_word = word_id

    compressed_context = " ".join(compressed_tokens)

    # DEBUG
    print("\n" + "=" * 80)
    print("QUESTION:")
    print(question)

    print("\nNUMBER OF INPUT WORDS:")
    print(len(tokens))

    print("\nNUMBER OF KEPT WORDS:")
    print(keep_count)

    print("\nNUMBER OF REMOVED WORDS:")
    print(remove_count)

    print("\nCOMPRESSION RATIO:")
    print(
        f"{keep_count}/{len(tokens)} = "
        f"{keep_count / len(tokens):.2%}"
    )

    print("\nPREDICTIONS:")
    print(pred_labels)

    print("\nCOMPRESSED:")
    print(compressed_context)

    print("=" * 80)

    return compressed_context
# ============================================================
# GRADIO INTERFACE
# ============================================================

demo = gr.Interface(

    fn=predict,

    inputs=[

        gr.Textbox(
            label="Question",
            placeholder="Enter your question...",
            lines=2
        ),

        gr.Textbox(
            label="Retrieved Context",
            placeholder="Paste the retrieved context here...",
            lines=15
        ),

    ],

    outputs=gr.Textbox(
        label="Compressed Context",
        lines=15
    ),

    title="XLM-R Prompt Compressor",

    description=(
        "Query-aware context compression using "
        "XLM-RoBERTa token classification."
    ),

    api_name="compress"
)


# ============================================================
# LAUNCH
# ============================================================

if __name__ == "__main__":
    demo.launch()