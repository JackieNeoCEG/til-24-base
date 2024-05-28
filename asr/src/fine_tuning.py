import os
os.environ["USE_TORCH_XLA"] = "0"

from transformers import WhisperProcessor, WhisperForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset
from pathlib import Path
import jsonlines
import torch
import torchaudio

# Define paths
#data_dir = Path("../../advanced")

# Read data from a jsonl file and reformat it
data = {'key': [], 'audio': [], 'transcript': []}
# with jsonlines.open(data_dir / "asr.jsonl") as reader:
with jsonlines.open("../../splittedfiles3/split_1.jsonl") as reader:
    for obj in reader:
        data['key'].append(obj['key'])
        data['audio'].append(obj['audio'])
        data['transcript'].append(obj['transcript'])

# Convert to a Hugging Face dataset
dataset = Dataset.from_dict(data)

# Shuffle the dataset
dataset = dataset.shuffle(seed=42)

# Split the dataset into training, validation, and test sets
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset = dataset.select(range(train_size))
val_dataset = dataset.select(range(train_size, train_size + val_size))
test_dataset = dataset.select(range(train_size + val_size, train_size + val_size + test_size))

# Initialize processor and model
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

audio_dir = Path("../../wavfilefolder2")

# Function to load and preprocess audio
def preprocess_data(examples, max_audio_length=3000):
    input_features = []
    labels = []

    for audio_path, transcript in zip(examples['audio'], examples['transcript']):
        #speech_array, sampling_rate = torchaudio.load(data_dir / "audio" / audio_path)
        speech_array, sampling_rate = torchaudio.load(audio_dir / audio_path)
        # Process audio to get mel features
        processed = processor(speech_array.squeeze(0), sampling_rate=sampling_rate, return_tensors="pt", padding=True)

        # Pad or truncate the mel features to the desired length
        input_feature = processed.input_features.squeeze(0)
        if input_feature.shape[-1] < max_audio_length:
            # Pad
            padding = torch.zeros((input_feature.shape[0], max_audio_length - input_feature.shape[-1]))
            input_feature = torch.cat((input_feature, padding), dim=-1)
        else:
            # Truncate
            input_feature = input_feature[:, :max_audio_length]

        # Debug: Print the length of input_feature
        print(f"Input feature length: {input_feature.shape[-1]}")

        # Append input features
        input_features.append(input_feature)

        # Process transcript as labels and ensure they match the input length
        label = processor.tokenizer(transcript, return_tensors="pt", padding="max_length", truncation=True, max_length= 448).input_ids.squeeze(0)

        # Debug: Print the length of label
        print(f"Label length: {label.shape[-1]}")

        labels.append(label)

    examples['input_features'] = torch.stack(input_features)
    examples['labels'] = torch.stack(labels)
    
    return examples

# Apply preprocessing
train_dataset = train_dataset.map(preprocess_data, batched=True, batch_size=1, remove_columns=train_dataset.column_names)
val_dataset = val_dataset.map(preprocess_data, batched=True, batch_size=1, remove_columns=val_dataset.column_names)
test_dataset = test_dataset.map(preprocess_data, batched=True, batch_size=1, remove_columns=test_dataset.column_names)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    learning_rate=1e-4,
    per_device_train_batch_size=1,
    num_train_epochs=10,
    weight_decay=0.005,
    save_steps=500,
    eval_steps=500,
    logging_steps=10,
    load_best_model_at_end=True
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=processor.feature_extractor
)

# Train the model
trainer.train()

# Save the model and processor
model.save_pretrained("./fine_tuned_whisper")
processor.save_pretrained("workspace/fine_tuned_whisper")
