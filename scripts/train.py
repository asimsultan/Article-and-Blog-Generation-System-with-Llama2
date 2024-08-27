
import os
import torch
import argparse
import pandas as pd
from transformers import LlamaTokenizer, LlamaForConditionalGeneration, AdamW, get_scheduler
from sklearn.model_selection import train_test_split
from utils import get_device, preprocess_text, ArticleDataset

def main(data_path):
    # Parameters
    model_name = 'Llama2-base'
    max_length = 512
    batch_size = 4
    epochs = 3
    learning_rate = 5e-5

    # Load Dataset
    dataset = pd.read_csv(data_path)
    train_texts, val_texts = train_test_split(dataset['title'], test_size=0.1)

    # Tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(model_name)

    # Preprocess Data
    train_encodings = preprocess_text(tokenizer, train_texts.tolist(), max_length)
    val_encodings = preprocess_text(tokenizer, val_texts.tolist(), max_length)

    # DataLoader
    train_dataset = ArticleDataset(train_encodings)
    val_dataset = ArticleDataset(val_encodings)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model
    device = get_device()
    model = LlamaForConditionalGeneration.from_pretrained(model_name)
    model.to(device)

    # Optimizer and Scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = len(train_loader) * epochs
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    # Training Function
    def train_epoch(model, data_loader, optimizer, device, scheduler):
        model.train()
        total_loss = 0

        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['input_ids'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(data_loader)
        return avg_loss

    # Evaluation Function
    def evaluate(model, data_loader, device):
        model.eval()
        total_loss = 0
        total_samples = 0

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['input_ids'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
                total_samples += labels.size(0)

        avg_loss = total_loss / total_samples
        return avg_loss

    # Training Loop
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, lr_scheduler)
        val_loss = evaluate(model, val_loader, device)
        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Train Loss: {train_loss}')
        print(f'Validation Loss: {val_loss}')

    # Save Model
    model_dir = './models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to the CSV file containing article titles')
    args = parser.parse_args()
    main(args.data_path)
