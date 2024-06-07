import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import BertTokenizerFast, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm.notebook import tqdm
from sklearn.preprocessing import LabelEncoder
import os

def encode_text(sent, tokenizer, max_len):
    '''
    Encode text(arg:sent) by tokenizer. Padding and truncation is true.
    '''
    encoded = tokenizer.encode_plus(
        sent,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        return_attention_mask=True,
        truncation=True
    )
    return encoded['input_ids'], encoded['attention_mask']

def bert_encode(data, max_len):
    '''
    Define BertTokenizer and encode data(arg:data) with.
    '''
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    input_ids = []
    attention_masks = []

    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(encode_text, 
                                        data, 
                                        [tokenizer]*len(data), 
                                        [max_len]*len(data)), 
                                        total=len(data), 
                                        desc="Encoding"))
        for ids, masks in results:
            input_ids.append(ids)
            attention_masks.append(masks)

        input_ids = torch.tensor(input_ids)
        attention_masks = torch.tensor(attention_masks)

        return input_ids, attention_masks

def train(model, train_dataloader, optimizer):
    '''
    Train model.
    '''
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_loss = 0

    for batch in tqdm(train_dataloader, desc="Training"):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        model.zero_grad()
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()

        optimizer.step()

    avg_loss = total_loss / len(train_dataloader)

    return avg_loss

def extract_dimension_labels(encoded_labels, label_encoder):
    '''
    Evaluate accuracy by type.
    '''
    decoded_labels = label_encoder.inverse_transform(encoded_labels)
    dimensions = {
        'EI': [1 if label[0] == 'E' else 0 for label in decoded_labels],
        'NS': [1 if label[1] == 'N' else 0 for label in decoded_labels],
        'FT': [1 if label[2] == 'F' else 0 for label in decoded_labels],
        'JP': [1 if label[3] == 'J' else 0 for label in decoded_labels]
    }
    return dimensions

def evaluate(model, validation_dataloader, label_encoder):
    '''
    Evaluate model. This func output is accuracy, precision, recall, f1 score, accuracy by type.
    '''
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions, true_labels = [], []

    for batch in tqdm(validation_dataloader, desc="Evaluating"):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        
        logits = outputs.logits
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        predictions.extend(np.argmax(logits, axis=1).flatten())
        true_labels.extend(label_ids.flatten())

    overall_accuracy = accuracy_score(true_labels, predictions)
    overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')

    true_dimensions = extract_dimension_labels(true_labels, label_encoder)
    pred_dimensions = extract_dimension_labels(predictions, label_encoder)
    dimension_accuracies = {}

    for dimension, true_dim_labels in true_dimensions.items():
        dimension_accuracies[dimension] = accuracy_score(true_dim_labels, pred_dimensions[dimension])

    return overall_accuracy, overall_precision, overall_recall, overall_f1, dimension_accuracies

    
if __name__ == '__main__':
    data_upsampled = pd.read_csv('filtered_dataset.csv')

    # Define label encoder.
    label_encoder = LabelEncoder()
    data_upsampled['encoded_label'] = label_encoder.fit_transform(data_upsampled['type'])

    # Encode data.
    input_ids, attention_masks = bert_encode(data_upsampled['filtered_posts'], max_len=64)
    labels = torch.tensor(data_upsampled['encoded_labels'].values)

    # Split dataset into train, val, test.
    train_inputs, temp_inputs, train_labels, temp_labels = train_test_split(input_ids, labels, random_state=2018, test_size=0.2)
    validation_inputs, test_inputs, validation_labels, test_labels = train_test_split(temp_inputs, temp_labels, random_state=2018, test_size=0.5)
    train_masks, temp_masks, _, _ = train_test_split(attention_masks, labels, random_state=2018, test_size=0.2)
    validation_masks, test_masks, _, _ = train_test_split(temp_masks, temp_labels, random_state=2018, test_size=0.5)

    # Define dataset and then data loader.(GROUPING)
    train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
    validation_dataset = TensorDataset(validation_inputs, validation_masks, validation_labels)
    test_dataset = TensorDataset(test_inputs, test_masks, test_labels)

    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=32)
    validation_dataloader = DataLoader(validation_dataset, sampler=SequentialSampler(validation_dataset), batch_size=32)
    test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=32)

    # Define bert_mbti's base model.
    model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(label_encoder.classes_),
    output_attentions=False,
    output_hidden_states=False,
    )

    # Ready to train. (and set hyperparams)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

    model_save_path = 'models'
    os.makedirs(model_save_path, exist_ok=True)

    epochs = 20 # up to 15 is ideal.
    best_accuracy = 0.0
    no_improve_epochs = 0
    early_stopping_patience = 3

    # Train and test(val) model.
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        print('-' * 10)

        train_loss = train(model, train_dataloader, optimizer)
        print(f'Training loss: {train_loss}')

        accuracy, precision, recall, f1, dimension_accuracies = evaluate(model, validation_dataloader, label_encoder)
        print(f'Accuracy: {accuracy}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1 Score: {f1}')

        for dim, acc in dimension_accuracies.items():
            print(f'{dim} Accuracy: {acc}')

        if accuracy > best_accuracy: # improve.
            best_accuracy = accuracy
            no_improve_epochs = 0
            torch.save(model.state_dict(), os.path.join(model_save_path, 'best_model.pth'))
            print("Improved validation accuracy. Model saved.")
        else: # no improve.
            no_improve_epochs += 1
            print("No improvement in validation accuracy.")
            if no_improve_epochs >= early_stopping_patience:
                print("Stopping early due to no improvement.")
                break

    # After this is testing model with test data loader. We can do this in another file and it's ideal.
    '''
    best_model_path = os.path.join(model_save_path, 'best_model.pth')
    model.load_state_dict(torch.load(best_model_path))

    # 测试集评估
    test_accuracy, test_precision, test_recall, test_f1, dimension_accuracies = evaluate(model, test_dataloader, label_encoder)

    print(f'Test Accuracy: {test_accuracy}')
    print(f'Test Precision: {test_precision}')
    print(f'Test Recall: {test_recall}')
    print(f'Test F1 Score: {test_f1}')

    # 打印每个MBTI维度的准确率
    print("MBTI Dimension Accuracies:")
    for dimension, accuracy in dimension_accuracies.items():
        print(f"{dimension} Accuracy: {accuracy}")
    '''

