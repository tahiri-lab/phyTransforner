from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig
import evaluate
import torch
import numpy as np

# Charger le jeu de données
dataset = load_dataset("jonghyunlee/UniProt_function_text_descriptions")

# Diviser les données en train, validation et test
dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
test_valid = dataset["test"].train_test_split(test_size=0.5, seed=42)

dataset = DatasetDict({
    'train': dataset['train'],
    'test': test_valid['test'],
    'validation': test_valid['train']
})

# Vérifier les clés du dataset divisé
print(dataset)

# Définir le point de contrôle du modèle
model_checkpoint = "dmis-lab/biobert-base-cased-v1.2"

# Créer une liste unique des noms de protéines (labels)
all_protein_names = dataset['train']['protein_name'] + dataset['validation']['protein_name'] + dataset['test']['protein_name']
label_names = list(set(all_protein_names))
id2label = {i: label for i, label in enumerate(label_names)}
label2id = {label: i for i, label in enumerate(label_names)}

# Ajouter les labels au jeu de données
def map_labels(example):
    try:
        example['labels'] = label2id[example['protein_name']]
    except KeyError as e:
        print(f"Label not found in label2id: {e}")
        example['labels'] = -1  # Ou une autre valeur par défaut pour gérer les erreurs
    return example

dataset = dataset.map(map_labels)

# Charger le tokenizer et le modèle
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, num_labels=len(label_names), id2label=id2label, label2id=label2id
)

# Inspecter les modules du modèle
for name, module in model.named_modules():
    print(name)

# Exemple de configuration PEFT ajustée avec LoraConfig
peft_config = LoraConfig(
    task_type="SEQUENCE_CLASSIFICATION",
    target_modules=[
        "bert.encoder.layer.0.attention.self.query",
        "bert.encoder.layer.0.attention.self.key",
        "bert.encoder.layer.0.attention.self.value",
        "bert.encoder.layer.0.output.dense",
        "bert.encoder.layer.1.attention.self.query",
        "bert.encoder.layer.1.attention.self.key",
        "bert.encoder.layer.1.attention.self.value",
        "bert.encoder.layer.1.output.dense",
        "bert.encoder.layer.2.attention.self.query",
        "bert.encoder.layer.2.attention.self.key",
        "bert.encoder.layer.2.attention.self.value",
        "bert.encoder.layer.2.output.dense",
        "bert.encoder.layer.3.attention.self.query",
        "bert.encoder.layer.3.attention.self.key",
        "bert.encoder.layer.3.attention.self.value",
        "bert.encoder.layer.3.output.dense",
        # Continuez pour les autres couches si nécessaire
    ],
    r=8,  # Configuration spécifique à LoRA
    lora_alpha=32,
    lora_dropout=0.1,
)

# Appliquer le modèle PEFT
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Préparer les données pour l'entraînement
def tokenize_function(examples):
    return tokenizer(examples['sequence'], truncation=True, padding=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Vérifier les clés du dataset tokenisé
print(tokenized_dataset)

# Créer le data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Importer la métrique d'évaluation de précision
accuracy = evaluate.load("accuracy")

# Définir une fonction d'évaluation pour l'entraîneur
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": accuracy.compute(predictions=predictions, references=labels)}

# Définir les arguments d'entraînement
training_args = TrainingArguments(
    output_dir=model_checkpoint + "-lora-text-classification",
    learning_rate=1e-3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=10,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Créer l'objet Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,  # Ceci va dynamiquement remplir les exemples dans chaque batch pour être de longueur égale
    compute_metrics=compute_metrics,
)

# Entraîner le modèle
trainer.train()
