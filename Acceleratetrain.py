from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler
from accelerate import Accelerator
from peft import PeftConfig, get_peft_model, LoraConfig
import evaluate
from tqdm.auto import tqdm
import torch

# Charger le jeu de données
raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Fonction de tokenisation
def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True, padding="max_length")

# Tokeniser le jeu de données
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# Créer les DataLoader
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
train_dataloader = torch.utils.data.DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = torch.utils.data.DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)

# Initialiser Accelerator
accelerator = Accelerator()

# Charger le modèle de base
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

# Inspecter les modules du modèle pour trouver les cibles appropriées pour LoRA
for name, module in model.named_modules():
    print(name)

# Configurer PEFT avec LoRA en utilisant les modules trouvés
peft_config = LoraConfig(
    task_type="SEQ_CLS",
    r=4,
    lora_alpha=32,
    lora_dropout=0.01,
    target_modules=[
        "bert.encoder.layer.0.attention.self.query",
        "bert.encoder.layer.0.attention.self.key",
        "bert.encoder.layer.0.attention.self.value",
        "bert.encoder.layer.1.attention.self.query",
        "bert.encoder.layer.1.attention.self.key",
        "bert.encoder.layer.1.attention.self.value",
        # Ajoutez d'autres modules selon vos besoins
    ]
)

# Appliquer le modèle PEFT
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Configurer l'optimiseur
optimizer = AdamW(model.parameters(), lr=3e-5)

# Préparer les objets avec Accelerator
train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
    train_dataloader, eval_dataloader, model, optimizer
)

# Définir le scheduler de taux d'apprentissage
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

# Barre de progression
progress_bar = tqdm(range(num_training_steps))

# Entraînement du modèle
model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(accelerator.device) for k, v in batch.items()}  # Ajouter cette ligne pour déplacer les données sur le bon appareil
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

# Évaluation du modèle (facultatif)
metric = evaluate.load("glue", "mrpc")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(accelerator.device) for k, v in batch.items()}  # Ajouter cette ligne pour déplacer les données sur le bon appareil
    with torch.no_grad():
        outputs = model(**batch)

    predictions = outputs.logits.argmax(dim=-1)
    metric.add_batch(predictions=accelerator.gather(predictions), references=accelerator.gather(batch["labels"]))

eval_metric = metric.compute()
print(f"Evaluation metric: {eval_metric}")
