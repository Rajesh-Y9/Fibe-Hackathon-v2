import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
import wandb
from transformers import AdamW, get_cosine_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler



class TransformerDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_encoder = LabelEncoder()
        if labels is not None:
            self.labels = self.label_encoder.fit_transform(labels)
        else:
            self.labels = None

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }
        
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return item



def train_transformer_model(model_name, train_dataset, val_dataset, num_labels, num_epochs, batch_size, fp16):
    print(f"Training {model_name}...")
    
    if dist.get_rank() == 0:
        wandb.init(project="transformer_training_v3", name=model_name)

    # Set seed for reproducibility
    torch.manual_seed(42)
    
    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    
    # Reinitialize some layers
    model.classifier.apply(model._init_weights)
    
    model = model.to(dist.get_rank())
    model = DDP(model, device_ids=[dist.get_rank()])

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)

    if len(train_loader) == 0 or len(val_loader) == 0:
        raise ValueError(f"Empty data loader for {model_name}. Check dataset or batch size.")

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    
    # Learning rate scheduler
    num_training_steps = len(train_loader) * num_epochs
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    scaler = GradScaler() if fp16 else None

    best_val_loss = float('inf')
    output_dir = f'./results_{model_name}'
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", disable=dist.get_rank() != 0)

        for batch in progress_bar:
            batch = {k: v.to(dist.get_rank()) for k, v in batch.items()}

            optimizer.zero_grad()

            if fp16:
                with autocast():
                    outputs = model(**batch)
                    loss = outputs.loss
                scaler.scale(loss).backward()
                
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()

            scheduler.step()
            total_loss += loss.item()

            preds = outputs.logits.argmax(dim=-1)
            correct_predictions += (preds == batch['labels']).sum().item()
            total_predictions += len(batch['labels'])

            accuracy = correct_predictions / total_predictions
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}", 'accuracy': f"{accuracy:.4f}", 'lr': scheduler.get_last_lr()[0]})

            if dist.get_rank() == 0:
                wandb.log({
                    'train_loss': loss.item(),
                    'train_accuracy': accuracy,
                    'learning_rate': scheduler.get_last_lr()[0]
                })

        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = correct_predictions / total_predictions

        # Validation
        model.eval()
        total_eval_loss = 0
        correct_eval_predictions = 0
        total_eval_predictions = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(dist.get_rank()) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                total_eval_loss += loss.item()

                preds = outputs.logits.argmax(dim=-1)
                correct_eval_predictions += (preds == batch['labels']).sum().item()
                total_eval_predictions += len(batch['labels'])

        avg_eval_loss = total_eval_loss / len(val_loader)
        val_accuracy = correct_eval_predictions / total_eval_predictions

        if dist.get_rank() == 0:
            wandb.log({
                'epoch': epoch + 1,
                'val_loss': avg_eval_loss,
                'val_accuracy': val_accuracy
            })

            print(f"Epoch {epoch+1}/{num_epochs}, Validation loss: {avg_eval_loss:.4f}, Validation accuracy: {val_accuracy:.4f}")

        if avg_eval_loss < best_val_loss:
            best_val_loss = avg_eval_loss
            if dist.get_rank() == 0:
                print(f"New best model found! Saving model with validation loss: {best_val_loss:.4f}")
                model_to_save = model.module if hasattr(model, 'module') else model
                model_to_save.save_pretrained(os.path.join(output_dir, 'best_model'))
                print(f"Best model saved to {os.path.join(output_dir, 'best_model')}")

                wandb.log({
                    'best_val_loss': best_val_loss,
                    'best_val_accuracy': val_accuracy
                })

    if dist.get_rank() == 0:
        wandb.finish()

    return model




def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {'accuracy': (preds == labels).mean()}

def predict_transformer(model, dataset, device):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            outputs = model(**inputs)
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            predictions.extend(preds)
    return predictions



def main():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    # Load data and set parameters
    print("Loading dataset")
    print("Loading and processing data...")
    train_df = pd.read_csv('/workspace/train.csv', encoding='ISO-8859-1')
    test_df = pd.read_csv('/workspace/test.csv', encoding='ISO-8859-1')
    print("Dataset Loaded")
    
    max_len = 512
    num_epochs = 5
    batch_size = 200
    fp16 = True

    train_texts, val_texts, train_labels, val_labels = train_test_split(train_df['text'].values, train_df['target'].values, test_size=0.2, random_state=42)
    test_texts = test_df['text'].values

    transformer_models = {
        'distilbert': 'distilbert-base-uncased',
    }

    trained_transformer_models = {}
    label_encoder = LabelEncoder()
    label_encoder.fit(train_labels)  # Fit the LabelEncoder on all train labels

    for model_name, model_path in transformer_models.items():
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        train_dataset = TransformerDataset(train_texts, train_labels, tokenizer, max_len)
        val_dataset = TransformerDataset(val_texts, val_labels, tokenizer, max_len)
        num_labels = len(np.unique(train_labels))

        model = train_transformer_model(model_path, train_dataset, val_dataset, num_labels, num_epochs, batch_size, fp16)
        trained_transformer_models[model_name] = (model, tokenizer)

        if dist.get_rank() == 0:
            model.save_pretrained(f'./saved_models/{model_name}')
            tokenizer.save_pretrained(f'./saved_models/{model_name}')

    if dist.get_rank() == 0:
        # Make predictions on the test set
        test_predictions = {}
        for model_name, (model, tokenizer) in trained_transformer_models.items():
            test_dataset = TransformerDataset(test_texts, [0] * len(test_texts), tokenizer, max_len)
            test_predictions[model_name] = predict_transformer(model, test_dataset, local_rank)

        # Ensemble test predictions
        final_test_predictions = []
        for i in range(len(test_texts)):
            votes = [test_predictions[model_name][i] for model_name in test_predictions]
            final_test_predictions.append(max(set(votes), key=votes.count))

        # Convert numerical predictions to text labels
        final_test_predictions_text = label_encoder.inverse_transform(final_test_predictions)

        # Create submission file with text labels
        submission_df = pd.DataFrame({
            'index': test_df['index'],
            'target': final_test_predictions_text
        })
        submission_df.to_csv('submission.csv', index=False)
        print("Submission file created: submission.csv")

        # Evaluate on validation set
        val_predictions = {}
        for model_name, (model, tokenizer) in trained_transformer_models.items():
            val_dataset = TransformerDataset(val_texts, val_labels, tokenizer, max_len)
            val_predictions[model_name] = predict_transformer(model, val_dataset, local_rank)

        # Ensemble validation predictions
        final_val_predictions = []
        for i in range(len(val_texts)):
            votes = [val_predictions[model_name][i] for model_name in val_predictions]
            final_val_predictions.append(max(set(votes), key=votes.count))

        # Convert numerical predictions to text labels for validation
        final_val_predictions_text = label_encoder.inverse_transform(final_val_predictions)
        val_labels_text = label_encoder.inverse_transform(val_labels)

        print("Validation Set Performance:")
        print(classification_report(val_labels_text, final_val_predictions_text))
        print("Confusion Matrix (Validation Set):")
        cm = confusion_matrix(val_labels_text, final_val_predictions_text)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix (Validation Set)')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('validation_confusion_matrix.png')
        plt.close()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()