import torchtext
import string
import nltk
import re
import html
import subprocess
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, recall_score, f1_score, confusion_matrix

# Download NLTK data if not available
def download_nltk_data():
    try:
        nltk.data.find('punkt.zip')
    except:
        nltk.download('punkt', download_dir='/kaggle/working/')
        subprocess.run("unzip /usr/share/nltk_data/corpora/punkt.zip -d /usr/share/nltk_data/corpora/".split())
        nltk.data.path.append('/kaggle/input')

    try:
        nltk.data.find('stopwords.zip')
    except:
        nltk.download('stopwords', download_dir='/kaggle/working/')
        subprocess.run("unzip /usr/share/nltk_data/corpora/stopwords.zip -d /usr/share/nltk_data/corpora/".split())
        nltk.data.path.append('/kaggle/input')

    try:
        nltk.data.find('wordnet.zip')
    except:
        nltk.download('wordnet', download_dir='/kaggle/working/')
        subprocess.run("unzip /usr/share/nltk_data/corpora/wordnet.zip -d /usr/share/nltk_data/corpora/".split())
        nltk.data.path.append('/kaggle/input')

# Clean text
def clean_text(text):
    text = re.sub(r"([a-zA-Z]+)n[\'’]t", r"\1 not", text)
    text = re.sub(r"([iI])[\'’]m", r"\1 am", text)
    text = re.sub(r"([iI])[\'’]ll", r"\1 will", text)
    text = re.sub(r"[^a-zA-Z0-9\:\$\-\,\%\.\?\!]+", " ", text)
    text = html.unescape(text)
    text = re.sub(r"_(.*?)_", r"\1", text)
    text = re.sub(r'(\w+)-(\w+)', r'\1\2', text)
    return text

# Tokenize text
def tokenize_text(text, stop_words, lemmatizer):
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens if token.isalpha()]  
    tokens = [token for token in tokens if token not in stop_words]  
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens

# Prepare datasets
def prepare_datasets(dataset):
    stop_words = set(stopwords.words('english'))
    symbols = sorted(['.', ',', '?', '-', ':', ';', '$', '%', '!', 's'])
    stop_words.update(symbols)
    lemmatizer = WordNetLemmatizer()

    train_texts, val_texts, test_texts = [], [], []
    lnt_train, lnt_val, lnt_test = [], [], []

    for example in dataset["train"]:
        text = example["text"]
        tokens = tokenize_text(clean_text(text), stop_words, lemmatizer)
        train_texts.append({'text': tokens, 'label': example['label']})
        lnt_train.append(len(tokens))

    for example in dataset["test"]:
        text = example["text"]
        tokens = tokenize_text(clean_text(text), stop_words, lemmatizer)
        test_texts.append({'text': tokens, 'label': example['label']})
        lnt_test.append(len(tokens))

    for example in dataset["validation"]:
        text = example["text"]
        tokens = tokenize_text(clean_text(text), stop_words, lemmatizer)
        val_texts.append({'text': tokens, 'label': example['label']})
        lnt_val.append(len(tokens))

    tokens = [d['text'] for d in train_texts]  
    labels = [d['label'] for d in train_texts]  

    train_tokens, val_tokens, train_labels, val_labels = train_test_split(tokens, labels, test_size=0.2, stratify=labels)

    train, val, test = [], [], []
    lnt = lnt_train + lnt_val + lnt_test
    MAX_LENGTH = max(lnt)

    for tokenL, label in zip(train_tokens, train_labels):
        train.append({'text': tokenL, 'label': label})

    for tokenL, label in zip(val_tokens, val_labels):
        val.append({'text': tokenL, 'label': label})

    for tokenL, label in zip(test_texts, test_labels):
        test.append({'text': tokenL, 'label': label})

    return train, val, test, MAX_LENGTH

# Create vocabulary
def create_vocab(tokenized_dataset, freq_threshold=4):
    word_counts = {}

    for example in tokenized_dataset:
        for word in example['text']:
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1

    vocab = {'<PAD>': 0, '<UNK>': 1}
    index = 2
    for word, count in word_counts.items():
        if count >= freq_threshold:
            vocab[word] = index
            index += 1

    return vocab

# Convert tokenized datasets to indexed datasets
def token2index_dataset(tokenized_dataset, vocab):
    indexed_dataset = []
    for example in tokenized_dataset:
        indexed_text = [vocab.get(word, 1) for word in example['text']]
        indexed_dataset.append({'text': indexed_text, 'label': example['label']})

    return indexed_dataset

# Define ELMo model
class ELMo(nn.Module):
    def __init__(self, vocab_size, weights_matrix, hidden_size=100, num_layers=2, dropout=0.5):
        super(ELMo, self).__init__()

        self.embedding = nn.Embedding(vocab_size, 100)
        self.embedding.weight.data.copy_(torch.FloatTensor(weights_matrix))
        self.embedding.weight.requires_grad = True

        self.lstm_forward = nn.LSTM(100, hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.lstm_backward = nn.LSTM(100, hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)

        self.dropout = nn.Dropout(p=dropout)

        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x_forward, x_backward):
        x_forward = self.embedding(x_forward)
        x_backward = self.embedding(x_backward)

        out_forward, _ = self.lstm_forward(x_backward)
        out_backward, _ = self.lstm_backward(out_forward)

        out = self.fc(out_backward)

        return out

# Define Sentimental Analysis model
class SentimentalAnalysis(nn.Module):
    def __init__(self, elmo_embeddings, embedding_dim=100, hidden_size=100, dropout=0.5, output_dim=4):
        super(SentimentalAnalysis, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(elmo_embeddings))
        self.embedding.weight.requires_grad = True
        self.weightage = nn.Parameter(torch.FloatTensor([0.333, 0.333, 0.333]))
        self.lstm1 = elmo_lstmf
        self.lstm2 = elmo_lstmb
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(embedding_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        embed1 = self.fc1(x)
        embed2, _ = self.lstm1(x)
        embed3, _ = self.lstm2(embed2)
        embed = torch.stack([embed1, embed2, embed3], dim=1)
        embed = torch.sum(embed * self.weightage.view(1, -1, 1, 1), dim=1)
        out = torch.max(embed, dim=1)[0]
        out = self.dropout(out)
        out = self.fc2(out)
        return out

# Training loop for ELMo
def train_loop_EE(model, criterion, optimizer, train_loader, valid_loader, num_epochs, vocab):
    train_losses, valid_losses = [], []
    train_accuracies, valid_accuracies = [], []
    model.to(device)

    best_valid_acc = 0.0 
    best_model_path = 'best_bilstm.pt' 

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_total_accuracy = 0.0, 0.0

        for x, x_forward, x_backward, labels in tqdm(train_loader):
            x_forward, x_backward = x_forward.to(device), x_backward.to(device)
            optimizer.zero_grad()
            outputs = model(x_forward, x_backward).view(-1, len(vocab))
            target = x_forward.view(-1).to(device)
            loss = criterion(outputs, target)
            acc = accuracy(outputs, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_total_accuracy += acc.item()

        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_total_accuracy / len(train_loader))

        model.eval()
        valid_loss, valid_total_accuracy = 0.0, 0.0
        with torch.no_grad():
            for x, x_forward, x_backward, labels in tqdm(valid_loader):
                x_forward, x_backward = x_forward.to(device), x_backward.to(device)
                outputs = model(x_forward, x_backward).view(-1, len(vocab))
                target = x_forward.view(-1).to(device)
                loss = criterion(outputs, target)
                acc = accuracy(outputs, target)
                valid_loss += loss.item()
                valid_total_accuracy += acc.item()

        valid_losses.append(valid_loss / len(valid_loader))
        valid_accuracies.append(valid_total_accuracy / len(valid_loader))

        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_losses[-1]}, Validation Loss: {valid_losses[-1]}, Training Accuracy: {train_accuracies[-1]}, Validation Accuracy: {valid_accuracies[-1]}')

        if valid_accuracies[-1] > best_valid_acc:
            best_valid_acc = valid_accuracies[-1]
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model with validation accuracy: {best_valid_acc}")

    return train_losses, valid_losses, train_accuracies, valid_accuracies

# Downstream task training loop
def train_loop_SA(model, criterion, optimizer, train_loader, valid_loader, num_epochs, modelName):
    train_losses = []
    valid_losses = []
    train_acc = []
    valid_acc = []
    max_valid_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        for x, x_forward, x_backward, labels in tqdm(train_loader):
            optimizer.zero_grad()
            outputs = model(x.to(device)).to(device)
            labels = labels.to(device)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_correct += (torch.max(outputs, 1)[1] == labels).sum().item()

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        train_acc_percentage = train_correct / len(train_loader.dataset)
        train_acc.append(train_acc_percentage)

        model.eval()
        valid_loss = 0.0
        valid_correct = 0
        with torch.no_grad():
            for x, x_forward, x_backward, labels in tqdm(valid_loader):
                outputs = model(x.to(device)).to(device)
                labels = labels.to(device)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                valid_correct += (torch.max(outputs, 1)[1] == labels).sum().item()

        valid_loss /= len(valid_loader.dataset)
        valid_losses.append(valid_loss)
        valid_acc_percentage = valid_correct / len(valid_loader.dataset)
        valid_acc.append(valid_acc_percentage)

        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss}, Validation Loss: {valid_loss}, Training Accuracy: {train_acc_percentage}, Validation Accuracy: {valid_acc_percentage}')

        if valid_acc_percentage > max_valid_acc:
            max_valid_acc = valid_acc_percentage
            torch.save(model.state_dict(), f'best_{modelName}.pt')
            print(f"Saved new best model with Validation Accuracy: {max_valid_acc}")

    return train_losses, valid_losses, train_acc, valid_acc

# Evaluate the downstream task model
def evaluate_model(model, DatasetType, data_loader):
    correct_pred, num_examples = 0, 0
    true_labels = []
    predicted_labels = []
    model.eval()
    for x, _, _, labels in tqdm(data_loader):
        outputs = model(x.to(device)).to(device)
        labels = labels.to(device)
        _, predicted = torch.max(outputs, 1)
        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())
        num_examples += labels.size(0)
        correct_pred += (predicted == labels).sum()
    test_accuracy = correct_pred.float() / num_examples
    accuracy = accuracy_score(true_labels, predicted_labels)

    print(f'Accuracy on the {DatasetType} set: {accuracy}')

    print("Classification Report:")
    print(classification_report(true_labels, predicted_labels))

    print("Confusion Matrix:")
    print(confusion_matrix(true_labels, predicted_labels))

    micro_recall = recall_score(true_labels, predicted_labels, average='micro')
    macro_recall = recall_score(true_labels, predicted_labels, average='macro')
    print(f'Micro Recall: {micro_recall}')
    print(f'Macro Recall: {macro_recall}')

    micro_f1 = f1_score(true_labels, predicted_labels, average='micro')
    macro_f1 = f1_score(true_labels, predicted_labels, average='macro')
    print(f'Micro F1 Score: {micro_f1}')
    print(f'Macro F1 Score: {macro_f1}')
    print()

# Define Downstream Task function
def DownStreamTask(model, criterion, optimizer, train_loader, valid_loader, test_loader, num_epochs, modelName):
    train_losses, valid_losses, train_acc, valid_acc = train_loop_SA(model, criterion, optimizer, train_loader, valid_loader, num_epochs, modelName)
    print()
    plt.plot(train_losses, label='Training loss')
    plt.plot(valid_losses, label='Validation loss')
    plt.legend()
    plt.show()

    plt.plot(train_acc, label='Training accuracy')
    plt.plot(valid_acc, label='Validation accuracy')
    plt.legend()
    plt.show()

    model.load_state_dict(torch.load(f'best_{modelName}.pt'))
    print()
    evaluate_model(model, 'Validation', valid_loader)
    print()
    evaluate_model(model, 'Test', test_loader)

# Load dataset
dataset = load_dataset('amazon_polarity')

# Prepare datasets
train, valid, test, MAX_LENGTH = prepare_datasets(dataset)

# Create vocabulary
vocab = create_vocab(train + valid + test)

# Convert tokenized datasets to indexed datasets
train_dataset = token2index_dataset(train, vocab)
valid_dataset = token2index_dataset(valid, vocab)
test_dataset = token2index_dataset(test, vocab)

# Define PyTorch DataLoader
batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# Initialize the ELMo model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights_matrix = torch.randn(len(vocab), 100)
elmo_model = ELMo(len(vocab), weights_matrix).to(device)

# Define the loss function and optimizer for ELMo model
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(elmo_model.parameters(), lr=0.001)

# Train the ELMo model
num_epochs = 3
train_losses, valid_losses, train_accuracies, valid_accuracies = train_loop_EE(elmo_model, criterion, optimizer, train_loader, valid_loader, num_epochs, vocab)

# Plot the training and validation loss for ELMo model
plt.plot(train_losses, label='Training loss')
plt.plot(valid_losses, label='Validation loss')
plt.legend()
plt.show()

# Plot the training and validation accuracy for ELMo model
plt.plot(train_accuracies, label='Training accuracy')
plt.plot(valid_accuracies, label='Validation accuracy')
plt.legend()
plt.show()

# Load the best ELMo model
elmo_model.load_state_dict(torch.load('best_bilstm.pt'))

# Get ELMo embeddings
def get_elmo_embeddings(model, data_loader):
    elmo_embeddings = []
    model.eval()
    with torch.no_grad():
        for x, _, _, _ in tqdm(data_loader):
            x_forward, x_backward = x.to(device), x.to(device)
            embeddings = model(x_forward, x_backward)
            elmo_embeddings.extend(embeddings.cpu().numpy())
    return elmo_embeddings

# Get ELMo embeddings for downstream task
elmo_embeddings_train = get_elmo_embeddings(elmo_model, train_loader)
elmo_embeddings_valid = get_elmo_embeddings(elmo_model, valid_loader)
elmo_embeddings_test = get_elmo_embeddings(elmo_model, test_loader)

# Initialize the Sentimental Analysis model
sentimental_model = SentimentalAnalysis(elmo_embeddings_train).to(device)

# Define the loss function and optimizer for Sentimental Analysis model
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(sentimental_model.parameters(), lr=0.001)

# Train the Sentimental Analysis model
DownStreamTask(sentimental_model, criterion, optimizer, train_loader, valid_loader, test_loader, num_epochs, 'sentimental_model')

