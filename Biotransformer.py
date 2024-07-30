from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig
import evaluate
import torch
import numpy as np

# Load the dataset
dataset = load_dataset("khairi/uniprot-swissprot")

# Define the tokenizer
tokenizer = AutoTokenizer.from_pretrained("dotan1111/BioTokenizer-BFD-BPE-100")

# Define the model checkpoint
model_checkpoint = "dmis-lab/biobert-base-cased-v1.2"

# Define label maps
labels = dataset['train'].features['label'].names
id2label = {i: label for i, label in enumerate(labels)}
label2id = {label: i for i, label in enumerate(labels)}

# Generate classification model from model_checkpoint
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, num_labels=len(labels), id2label=id2label, label2id=label2id
)

# Display architecture
print(model)

# Create tokenize function
def tokenize_function(examples):
    # Extract text
    text = examples["sequence"]
    
    # Tokenize and truncate text
    tokenized_inputs = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        max_length=512
    )
    
    return tokenized_inputs

# Tokenize training and validation datasets
tokenized_dataset = dataset.map(tokenize_function, batched=True)
print(tokenized_dataset)

# Create data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Import accuracy evaluation metric
accuracy = evaluate.load("accuracy")

# Define an evaluation function to pass into trainer later
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    
    return {"accuracy": accuracy.compute(predictions=predictions, references=labels)}

# Define list of examples
text_list = ["MVLSPADKTNVKAAW", "GAGGAGAAGGTGTGGCG", "MGDVEKGKKIFIMKCSQCHTVEKGGKHKTGPNEKAGA"]

print("Untrained model predictions:")
print("----------------------------")
for text in text_list:
    # Tokenize text
    inputs = tokenizer.encode(text, return_tensors="pt")
    # Compute logits
    logits = model(inputs).logits
    # Convert logits to label
    predictions = torch.argmax(logits)
    
    print(text + " - " + id2label[predictions.tolist()])

peft_config = LoraConfig(task_type="SEQ_CLS",
                         r=4,
                         lora_alpha=32,
                         lora_dropout=0.01,
                         target_modules=['q_lin'])

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Hyperparameters
lr = 1e-3
batch_size = 4
num_epochs = 10

# Define training arguments
training_args = TrainingArguments(
    output_dir=model_checkpoint + "-lora-text-classification",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Create trainer object
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator, # this will dynamically pad examples in each batch to be equal length
    compute_metrics=compute_metrics,
)

# Train model
trainer.train()

model.to('cpu') # moving to cpu

print("Trained model predictions:")
print("--------------------------")
for text in text_list:
    inputs = tokenizer.encode(text, return_tensors="pt").to("cpu")
    
    logits = model(inputs).logits
    predictions = torch.argmax(logits, dim=1)
    
    print(text + " - " + id2label[predictions.tolist()[0]])
