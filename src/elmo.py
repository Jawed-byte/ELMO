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
import matplotlib.pyplot as plt
from datasets import load_dataset
from tqdm import tqdm

# Download NLTK data if not already downloaded
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

# Text preprocessing functions
def clean_text(text):
    text = re.sub(r"([a-zA-Z]+)n[\'’]t", r"\1 not", text)
    text = re.sub(r"([iI])[\'’]m", r"\1 am", text)
    text = re.sub(r"([iI])[\'’]ll", r"\1 will", text)
    text = re.sub(r"[^a-zA-Z0-9\:\$\-\,\%\.\?\!]+", " ", text)
    text = html.unescape(text)
    text = re.sub(r"_(.*?)_", r"\1", text)
    text = re.sub(r'(\w+)-(\w+)', r'\1\2', text)
    return text

def tokenize_text(text, stop_words, lemmatizer):
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens if token.isalpha()]  
    tokens = [token for token in tokens if token not in stop_words]  
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens

# Dataset preparation functions
def prepare_dataset(dataset):
    stop_words = set(stopwords.words('english'))
    symbols = sorted(['.', ',', '?', '-', ':', ';', '$', '%', '!', 's'])
    stop_words.update(symbols)
    lemmatizer = WordNetLemmatizer()
    
    data = []
    lengths = []
    for example in dataset:
        text = example["text"]
        tokens = tokenize_text(clean_text(text), stop_words, lemmatizer)
        data.append({"text": tokens, "label": example["label"]})
        lengths.append(len(tokens))
    return data, max(lengths)

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

def token2index_dataset(tokenized_dataset, vocab):
    indexed_dataset = []
    for example in tokenized_dataset:
        indexed_text = [vocab.get(word, 1) for word in example['text']]
        indexed_dataset.append({'text': indexed_text, 'label': example['label']})

    return indexed_dataset

# Model definition
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

# Training function
def train_ELMo(elmo_model, train_loader, valid_loader, criterion, optimizer, num_epochs, vocab):
    train_losses, valid_losses = [], []
    train_accuracies, valid_accuracies = [], []
    elmo_model.to(device)

    best_valid_acc = 0.0 
    best_model_path = 'best_bilstm.pt' 

    for epoch in range(num_epochs):
        elmo_model.train()
        train_loss, train_total_accuracy = 0.0, 0.0

        for x, x_forward, x_backward, labels in tqdm(train_loader):
            x_forward, x_backward = x_forward.to(device), x_backward.to(device)
            optimizer.zero_grad()
            outputs = elmo_model(x_forward, x_backward).view(-1, len(vocab))
            target = x_forward.view(-1).to(device)
            loss = criterion(outputs, target)
            acc = accuracy(outputs, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_total_accuracy += acc.item()

        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_total_accuracy / len(train_loader))

        elmo_model.eval()
        valid_loss, valid_total_accuracy = 0.0, 0.0
        with torch.no_grad():
            for x, x_forward, x_backward, labels in tqdm(valid_loader):
                x_forward, x_backward = x_forward.to(device), x_backward.to(device)
                outputs = elmo_model(x_forward, x_backward).view(-1, len(vocab))
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
            torch.save(elmo_model.state_dict(), best_model_path)
            print(f"Saved new best model with validation accuracy: {best_valid_acc}")

    return train_losses, valid_losses, train_accuracies, valid_accuracies

# Plotting functions
def plot_metrics(train_losses, valid_losses, train_accuracies, valid_accuracies):
    plt.plot(train_losses, label='Training loss')
    plt.plot(valid_losses, label='Validation loss')
    plt.legend()
    plt.show()

    plt.plot(train_accuracies, label='Training accuracy')
    plt.plot(valid_accuracies, label='Validation accuracy')
    plt.legend()
    plt.show()

# Main function
def main():
    download_nltk_data()

    # Load dataset
    dataset = load_dataset("ag_news", trust_remote_code=True)

    # Prepare datasets
    train_data, val_data, test_data, MAX_LENGTH = prepare_datasets(dataset)

    # Create vocabulary
    vocab = create_vocab(train_data)

    # Convert tokenized datasets to indexed datasets
    train_data_indexed = token2index_dataset(train_data, vocab)
    val_data_indexed = token2index_dataset(val_data, vocab)
    test_data_indexed = token2index_dataset(test_data, vocab)

    # Prepare DataLoader
    batch_size = 32
    train_loader = DataLoader(train_data_indexed, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(val_data_indexed, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data_indexed, batch_size=batch_size, shuffle=True)

    # Load pre-trained GloVe vectors
    glove_vectors = torchtext.vocab.Vectors(name='glove.6B.100d.txt')

    # Create weight matrix for embedding layer
    weights_matrix = torch.zeros((len(vocab), 100))
    for i, word in enumerate(vocab.keys()):
        try:
            weights_matrix[i] = glove_vectors[word]
        except KeyError:
            weights_matrix[i] = torch.zeros(100)

    # Define ELMo model
    elmo_model = ELMo(len(vocab), weights_matrix).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(elmo_model.parameters(), lr=0.01)

    # Train the model
    num_epochs = 3
    train_losses, valid_losses, train_accuracies, valid_accuracies = train_ELMo(elmo_model, train_loader, valid_loader, criterion, optimizer, num_epochs, vocab)

    # Plot training and validation metrics
    plot_metrics(train_losses, valid_losses, train_accuracies, valid_accuracies)

    # Load best model
    elmo_model.load_state_dict(torch.load('best_bilstm.pt'))

    # Save ELMo embeddings
    elmo_embeddings = list(elmo_model.parameters())[0]
    torch.save(elmo_embeddings, 'bilstm_elmo_embeddings.pt')

    # Extract LSTM layers
    elmo_lstmf = elmo_model.lstm_forward
    elmo_lstmb = elmo_model.lstm_backward


