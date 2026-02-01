# models.py

import torch
from torch import nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer
from sentiment_data import read_sentiment_examples, read_word_embeddings
from torch.utils.data import Dataset, DataLoader
import time
import argparse
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from BOWmodels import SentimentDatasetBOW, NN2BOW, NN3BOW
from DANmodels import SentimentDatasetDAN, DAN, collate_fn, SentimentDatasetDANRandom, DANRandomEmb
from BPEmodels import SentimentDatasetBPE, DANBPE, collate_fn_bpe
from BPE import train_bpe_from_file, BPEVocabulary, BytePairEncoding
import os
import pickle

# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Training function for BOW models
def train_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.train()
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(data_loader):
        # Don't convert to float for DAN (needs long for embedding indices)
        if X.dtype != torch.long:
            X = X.float()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_train_loss = train_loss / num_batches
    accuracy = correct / size
    return accuracy, average_train_loss


# Evaluation function for BOW models
def eval_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.eval()
    eval_loss = 0
    correct = 0
    for batch, (X, y) in enumerate(data_loader):
        # Don't convert to float for DAN (needs long for embedding indices)
        if X.dtype != torch.long:
            X = X.float()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        eval_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    average_eval_loss = eval_loss / num_batches
    accuracy = correct / size
    return accuracy, average_eval_loss


# Training function for DAN models (handles variable-length input)
def train_epoch_dan(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.train()
    train_loss, correct = 0, 0
    for batch, (X, lengths, y) in enumerate(data_loader):
        # Move to device
        X = X.to(device)
        lengths = lengths.to(device)
        y = y.to(device)

        # Compute prediction error
        pred = model(X, lengths)
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_train_loss = train_loss / num_batches
    accuracy = correct / size
    return accuracy, average_train_loss


# Evaluation function for DAN models
def eval_epoch_dan(data_loader, model, loss_fn):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.eval()
    eval_loss = 0
    correct = 0
    with torch.no_grad():
        for batch, (X, lengths, y) in enumerate(data_loader):
            # Move to device
            X = X.to(device)
            lengths = lengths.to(device)
            y = y.to(device)

            # Compute prediction error
            pred = model(X, lengths)
            loss = loss_fn(pred, y)
            eval_loss += loss.item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    average_eval_loss = eval_loss / num_batches
    accuracy = correct / size
    return accuracy, average_eval_loss


# Experiment function to run training and evaluation for multiple epochs (BOW)
def experiment(model, train_loader, test_loader):
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    all_train_accuracy = []
    all_test_accuracy = []
    for epoch in range(100):
        train_accuracy, train_loss = train_epoch(train_loader, model, loss_fn, optimizer)
        all_train_accuracy.append(train_accuracy)

        test_accuracy, test_loss = eval_epoch(test_loader, model, loss_fn, optimizer)
        all_test_accuracy.append(test_accuracy)

        if epoch % 10 == 9:
            print(f'Epoch #{epoch + 1}: train accuracy {train_accuracy:.3f}, dev accuracy {test_accuracy:.3f}')
    
    return all_train_accuracy, all_test_accuracy


# Experiment function for DAN models
def experiment_dan(model, train_loader, test_loader, num_epochs=100, lr=0.001):
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    all_train_accuracy = []
    all_test_accuracy = []
    all_train_loss = []
    all_test_loss = []
    best_dev_acc = 0.0
    
    for epoch in range(num_epochs):
        train_accuracy, train_loss = train_epoch_dan(train_loader, model, loss_fn, optimizer)
        all_train_accuracy.append(train_accuracy)
        all_train_loss.append(train_loss)

        test_accuracy, test_loss = eval_epoch_dan(test_loader, model, loss_fn)
        all_test_accuracy.append(test_accuracy)
        all_test_loss.append(test_loss)
        
        if test_accuracy > best_dev_acc:
            best_dev_acc = test_accuracy

        if epoch % 10 == 9:
            print(f'Epoch #{epoch + 1}: train acc {train_accuracy:.3f}, dev acc {test_accuracy:.3f}, train loss {train_loss:.3f}, dev loss {test_loss:.3f}')
    
    return all_train_accuracy, all_test_accuracy, all_train_loss, all_test_loss, best_dev_acc


def main():

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run model training based on specified model type')
    parser.add_argument('--model', type=str, required=True, help='Model type to train (e.g., BOW, DAN, DANRANDOM, BPE)')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Load dataset only for BOW model
    if args.model == "BOW":
        start_time = time.time()

        train_data = SentimentDatasetBOW("data/train.txt")
        dev_data = SentimentDatasetBOW("data/dev.txt", vectorizer=train_data.vectorizer, train=False)
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        test_loader = DataLoader(dev_data, batch_size=16, shuffle=False)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Data loaded in : {elapsed_time} seconds")
    else:
        # For DAN and BPE models, data loading is handled within their respective sections
        print(f"Data loaded in : 0.0 seconds")  # Placeholder for consistent output format


    # Check if the model type is "BOW"
    if args.model == "BOW":
        # Train and evaluate NN2
        start_time = time.time()
        print('\n2 layers:')
        nn2_train_accuracy, nn2_test_accuracy = experiment(NN2BOW(input_size=512, hidden_size=100), train_loader, test_loader)

        # Train and evaluate NN3
        print('\n3 layers:')
        nn3_train_accuracy, nn3_test_accuracy = experiment(NN3BOW(input_size=512, hidden_size=100), train_loader, test_loader)

        # Plot the training accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(nn2_train_accuracy, label='2 layers')
        plt.plot(nn3_train_accuracy, label='3 layers')
        plt.xlabel('Epochs')
        plt.ylabel('Training Accuracy')
        plt.title('Training Accuracy for 2, 3 Layer Networks')
        plt.legend()
        plt.grid()

        # Save the training accuracy figure
        training_accuracy_file = 'train_accuracy.png'
        plt.savefig(training_accuracy_file)
        print(f"\n\nTraining accuracy plot saved as {training_accuracy_file}")

        # Plot the testing accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(nn2_test_accuracy, label='2 layers')
        plt.plot(nn3_test_accuracy, label='3 layers')
        plt.xlabel('Epochs')
        plt.ylabel('Dev Accuracy')
        plt.title('Dev Accuracy for 2 and 3 Layer Networks')
        plt.legend()
        plt.grid()

        # Save the testing accuracy figure
        testing_accuracy_file = 'dev_accuracy.png'
        plt.savefig(testing_accuracy_file)
        print(f"Dev accuracy plot saved as {testing_accuracy_file}\n\n")

        # plt.show()

    elif args.model == "DAN":
        print(f"Using device: {device}")

        # Grid search parameters
        embedding_files = {
            "50d": "data/glove.6B.50d-relativized.txt",
            "300d": "data/glove.6B.300d-relativized.txt"
        }
        hidden_sizes = [100, 256]
        num_layers_options = [2, 3]
        dropout_positions = ["embedding", "hidden"]
        dropout_prob = 0.3

        # Store results for all experiments
        all_results = {}

        # Run grid search
        for emb_name, emb_file in embedding_files.items():
            print("\n" + "=" * 60)
            print(f"Loading {emb_name} embeddings...")
            print("=" * 60)

            start_time = time.time()
            word_embeddings = read_word_embeddings(emb_file)
            print(f"Embeddings loaded in: {time.time() - start_time:.2f} seconds")

            # Load datasets for this embedding
            train_data = SentimentDatasetDAN("data/train.txt", word_embeddings)
            dev_data = SentimentDatasetDAN("data/dev.txt", word_embeddings)
            train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_fn)
            test_loader = DataLoader(dev_data, batch_size=32, shuffle=False, collate_fn=collate_fn)

            for hidden_size in hidden_sizes:
                for num_layers in num_layers_options:
                    for drop_pos in dropout_positions:
                        drop_label = "EmbDropout" if drop_pos == "embedding" else "HidDropout"
                        config_name = f"{emb_name.upper()}_{hidden_size}H_{num_layers}Layers_{drop_label}"
                        print(f"\n--- Config: {config_name} ---")
                        print(f"Embedding: {emb_name}, Hidden: {hidden_size}, Layers: {num_layers}, Dropout: {dropout_prob} ({drop_pos})")

                        # Create model and move to device
                        model = DAN(word_embeddings, hidden_size=hidden_size,
                                   dropout_prob=dropout_prob, num_layers=num_layers,
                                   dropout_position=drop_pos)
                        model = model.to(device)

                        # Train and evaluate
                        train_acc, test_acc, train_loss, test_loss, best_dev = experiment_dan(
                            model, train_loader, test_loader, num_epochs=100, lr=0.001
                        )

                        all_results[config_name] = {
                            "train_acc": train_acc,
                            "test_acc": test_acc,
                            "train_loss": train_loss,
                            "test_loss": test_loss,
                            "best_dev": best_dev,
                            "final_dev": test_acc[-1],
                            "emb": emb_name,
                            "hidden": hidden_size,
                            "layers": num_layers,
                            "dropout_pos": drop_pos
                        }

                        print(f"Best dev accuracy: {best_dev:.3f}")

        # Print summary of all results
        print("\n" + "=" * 70)
        print("GRID SEARCH RESULTS SUMMARY")
        print("=" * 70)
        print(f"{'Config':<30} {'Best Dev Acc':<15} {'Final Dev Acc':<15}")
        print("-" * 60)

        best_config = None
        best_acc = 0.0
        for config_name, results in all_results.items():
            print(f"{config_name:<30} {results['best_dev']:<15.3f} {results['final_dev']:<15.3f}")
            if results['best_dev'] > best_acc:
                best_acc = results['best_dev']
                best_config = config_name

        print("-" * 60)
        print(f"Best configuration: {best_config} with dev accuracy: {best_acc:.3f}")

        # Separate results by embedding dimension
        results_50d = {k: v for k, v in all_results.items() if v['emb'] == '50d'}
        results_300d = {k: v for k, v in all_results.items() if v['emb'] == '300d'}

        # Plot 1: 50d Training Accuracy
        plt.figure(figsize=(12, 6))
        for config_name, results in results_50d.items():
            label = f"{results['hidden']}H_{results['layers']}L_{results['dropout_pos'][:3].upper()}"
            plt.plot(results["train_acc"], label=label)
        plt.xlabel('Epochs')
        plt.ylabel('Training Accuracy')
        plt.title('DAN Training Accuracy - 50d Embeddings')
        plt.legend(loc='lower right', fontsize=8)
        plt.grid()
        plt.tight_layout()
        plt.savefig('results/dan_train_accuracy_50d.png')
        print(f"\n50d training accuracy plot saved as results/dan_train_accuracy_50d.png")

        # Plot 2: 50d Dev Accuracy
        plt.figure(figsize=(12, 6))
        for config_name, results in results_50d.items():
            label = f"{results['hidden']}H_{results['layers']}L_{results['dropout_pos'][:3].upper()}"
            plt.plot(results["test_acc"], label=label)
        plt.xlabel('Epochs')
        plt.ylabel('Dev Accuracy')
        plt.title('DAN Dev Accuracy - 50d Embeddings')
        plt.legend(loc='lower right', fontsize=8)
        plt.grid()
        plt.tight_layout()
        plt.savefig('results/dan_dev_accuracy_50d.png')
        print(f"50d dev accuracy plot saved as results/dan_dev_accuracy_50d.png")

        # Plot 3: 300d Training Accuracy
        plt.figure(figsize=(12, 6))
        for config_name, results in results_300d.items():
            label = f"{results['hidden']}H_{results['layers']}L_{results['dropout_pos'][:3].upper()}"
            plt.plot(results["train_acc"], label=label)
        plt.xlabel('Epochs')
        plt.ylabel('Training Accuracy')
        plt.title('DAN Training Accuracy - 300d Embeddings')
        plt.legend(loc='lower right', fontsize=8)
        plt.grid()
        plt.tight_layout()
        plt.savefig('results/dan_train_accuracy_300d.png')
        print(f"300d training accuracy plot saved as results/dan_train_accuracy_300d.png")

        # Plot 4: 300d Dev Accuracy
        plt.figure(figsize=(12, 6))
        for config_name, results in results_300d.items():
            label = f"{results['hidden']}H_{results['layers']}L_{results['dropout_pos'][:3].upper()}"
            plt.plot(results["test_acc"], label=label)
        plt.xlabel('Epochs')
        plt.ylabel('Dev Accuracy')
        plt.title('DAN Dev Accuracy - 300d Embeddings')
        plt.legend(loc='lower right', fontsize=8)
        plt.grid()
        plt.tight_layout()
        plt.savefig('results/dan_dev_accuracy_300d.png')
        print(f"300d dev accuracy plot saved as results/dan_dev_accuracy_300d.png")

        # Plot 5: 50d vs 300d comparison
        plt.figure(figsize=(14, 6))
        config_groups = {}
        for config_name, results in all_results.items():
            base_config = f"{results['hidden']}H_{results['layers']}L_{results['dropout_pos'][:3].upper()}"
            if base_config not in config_groups:
                config_groups[base_config] = {}
            config_groups[base_config][results['emb']] = results

        colors_50d = plt.cm.Blues(np.linspace(0.4, 0.9, len(config_groups)))
        colors_300d = plt.cm.Oranges(np.linspace(0.4, 0.9, len(config_groups)))
        for i, (base_config, emb_results) in enumerate(config_groups.items()):
            if '50d' in emb_results:
                plt.plot(emb_results['50d']['test_acc'], label=f'50d_{base_config}',
                        color=colors_50d[i], linestyle='--')
            if '300d' in emb_results:
                plt.plot(emb_results['300d']['test_acc'], label=f'300d_{base_config}',
                        color=colors_300d[i], linestyle='-')
        plt.xlabel('Epochs')
        plt.ylabel('Dev Accuracy')
        plt.title('DAN Dev Accuracy: 50d (dashed) vs 300d (solid) Embeddings')
        plt.legend(loc='lower right', fontsize=6, ncol=2)
        plt.grid()
        plt.tight_layout()
        plt.savefig('results/dan_50d_vs_300d_comparison.png')
        print(f"50d vs 300d comparison plot saved as results/dan_50d_vs_300d_comparison.png")

        # Plot 6: Bar chart of best dev accuracy
        plt.figure(figsize=(16, 6))
        configs = list(all_results.keys())
        best_accs = [all_results[c]["best_dev"] for c in configs]
        colors = ['#2ecc71' if c == best_config else '#3498db' for c in configs]
        bars = plt.bar(configs, best_accs, color=colors)
        plt.xlabel('Configuration')
        plt.ylabel('Best Dev Accuracy')
        plt.title('Best Dev Accuracy for Each Configuration')
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.ylim(0.6, 0.90)
        for bar, acc in zip(bars, best_accs):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=7)
        plt.tight_layout()
        plt.savefig('results/dan_best_dev_accuracy.png')
        print(f"Best dev accuracy bar chart saved as results/dan_best_dev_accuracy.png")

        # Plot 7: Loss curves for 50d embeddings
        plt.figure(figsize=(14, 6))
        
        plt.subplot(1, 2, 1)
        for config_name, results in results_50d.items():
            label = f"{results['hidden']}H_{results['layers']}L_{results['dropout_pos'][:3].upper()}"
            plt.plot(results["train_loss"], label=label, alpha=0.7)
        plt.xlabel('Epochs')
        plt.ylabel('Training Loss')
        plt.title('Training Loss - 50d Embeddings')
        plt.legend(loc='upper right', fontsize=7)
        plt.grid(alpha=0.3)
        
        plt.subplot(1, 2, 2)
        for config_name, results in results_50d.items():
            label = f"{results['hidden']}H_{results['layers']}L_{results['dropout_pos'][:3].upper()}"
            plt.plot(results["test_loss"], label=label, alpha=0.7)
        plt.xlabel('Epochs')
        plt.ylabel('Dev Loss')
        plt.title('Dev Loss - 50d Embeddings')
        plt.legend(loc='upper right', fontsize=7)
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/dan_loss_curves_50d.png', dpi=150)
        print(f"50d loss curves saved as results/dan_loss_curves_50d.png")

        # Plot 8: Loss curves for 300d embeddings
        plt.figure(figsize=(14, 6))
        
        plt.subplot(1, 2, 1)
        for config_name, results in results_300d.items():
            label = f"{results['hidden']}H_{results['layers']}L_{results['dropout_pos'][:3].upper()}"
            plt.plot(results["train_loss"], label=label, alpha=0.7)
        plt.xlabel('Epochs')
        plt.ylabel('Training Loss')
        plt.title('Training Loss - 300d Embeddings')
        plt.legend(loc='upper right', fontsize=7)
        plt.grid(alpha=0.3)
        
        plt.subplot(1, 2, 2)
        for config_name, results in results_300d.items():
            label = f"{results['hidden']}H_{results['layers']}L_{results['dropout_pos'][:3].upper()}"
            plt.plot(results["test_loss"], label=label, alpha=0.7)
        plt.xlabel('Epochs')
        plt.ylabel('Dev Loss')
        plt.title('Dev Loss - 300d Embeddings')
        plt.legend(loc='upper right', fontsize=7)
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/dan_loss_curves_300d.png', dpi=150)
        print(f"300d loss curves saved as results/dan_loss_curves_300d.png")

        # Plot 9: Best configuration detailed analysis (accuracy + loss)
        best_config_results = all_results[best_config]
        plt.figure(figsize=(14, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(best_config_results["train_acc"], label='Train Accuracy', linewidth=2, color='#3498db')
        plt.plot(best_config_results["test_acc"], label='Dev Accuracy', linewidth=2, color='#e74c3c')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title(f'Best Configuration: {best_config}\nAccuracy Over Time')
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(best_config_results["train_loss"], label='Train Loss', linewidth=2, color='#3498db')
        plt.plot(best_config_results["test_loss"], label='Dev Loss', linewidth=2, color='#e74c3c')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Best Configuration: {best_config}\nLoss Over Time')
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/dan_best_config_detailed.png', dpi=150)
        print(f"Best config detailed analysis saved as results/dan_best_config_detailed.png")

    elif args.model == "DANRANDOM":
        # Part 1b: DAN with randomly initialized embeddings
        print(f"Using device: {device}")
        print("\n" + "=" * 60)
        print("Part 1b: DAN with Randomly Initialized Embeddings")
        print("Config: Embedding: 300d, Hidden: 256, Layers: 2, Dropout: 0.3 (embedding)")
        print("=" * 60)

        # Load datasets with random embeddings
        start_time = time.time()
        train_data = SentimentDatasetDANRandom("data/train.txt")
        dev_data = SentimentDatasetDANRandom("data/dev.txt", word_indexer=train_data.word_indexer)
        print(f"Vocabulary size: {len(train_data.word_indexer)}")
        print(f"Data loaded in: {time.time() - start_time:.2f} seconds")

        train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(dev_data, batch_size=32, shuffle=False, collate_fn=collate_fn)

        # Create model with specified config
        model = DANRandomEmb(
            vocab_size=len(train_data.word_indexer),
            embedding_dim=300,
            hidden_size=256,
            num_classes=2,
            dropout_prob=0.3,
            num_layers=2,
            dropout_position="embedding"
        )
        model = model.to(device)

        # Train and evaluate
        train_acc, test_acc, train_loss, test_loss, best_dev = experiment_dan(
            model, train_loader, test_loader, num_epochs=100, lr=0.001
        )

        print(f"\nBest dev accuracy: {best_dev:.3f}")
        print(f"Final dev accuracy: {test_acc[-1]:.3f}")

        # Plot training curves with loss
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Accuracy
        axes[0, 0].plot(train_acc, label='Train Accuracy', linewidth=2, color='#3498db')
        axes[0, 0].plot(test_acc, label='Dev Accuracy', linewidth=2, color='#e74c3c')
        axes[0, 0].set_xlabel('Epochs')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Accuracy Over Time')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Loss
        axes[0, 1].plot(train_loss, label='Train Loss', linewidth=2, color='#3498db')
        axes[0, 1].plot(test_loss, label='Dev Loss', linewidth=2, color='#e74c3c')
        axes[0, 1].set_xlabel('Epochs')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Loss Over Time')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # Overfitting gap (train - dev accuracy)
        overfitting_gap = [train_acc[i] - test_acc[i] for i in range(len(train_acc))]
        axes[1, 0].plot(overfitting_gap, linewidth=2, color='#9b59b6')
        axes[1, 0].set_xlabel('Epochs')
        axes[1, 0].set_ylabel('Accuracy Gap')
        axes[1, 0].set_title('Overfitting Gap (Train - Dev Accuracy)')
        axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.3)
        axes[1, 0].grid(alpha=0.3)
        
        # Loss difference (dev - train loss)
        loss_diff = [test_loss[i] - train_loss[i] for i in range(len(train_loss))]
        axes[1, 1].plot(loss_diff, linewidth=2, color='#e67e22')
        axes[1, 1].set_xlabel('Epochs')
        axes[1, 1].set_ylabel('Loss Difference')
        axes[1, 1].set_title('Loss Difference (Dev - Train Loss)')
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
        axes[1, 1].grid(alpha=0.3)
        
        plt.suptitle('DAN with Random Embeddings - Comprehensive Analysis\n(300d, Hidden: 256, Layers: 2, Dropout: 0.3)', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('results/dan_random_emb_comprehensive.png', dpi=150)
        print(f"\nComprehensive analysis plot saved as results/dan_random_emb_comprehensive.png")
    
    # Part 2: BPE Experiments
    elif args.model == 'BPE':
        print("\n" + "="*80)
        print("PART 2: BYTE PAIR ENCODING (BPE) EXPERIMENTS")
        print("="*80)
        
        # Experiment with different vocabulary sizes
        vocab_sizes = [500, 1000, 2000, 5000, 10000]
        
        results = []
        
        for num_merges in vocab_sizes:
            print(f"\n{'='*80}")
            print(f"TRAINING BPE WITH {num_merges} MERGES (APPROX VOCAB SIZE: {num_merges + 100})")
            print(f"{'='*80}")
            
            # Train or load BPE model
            bpe_file = f'results/bpe_{num_merges}.pkl'
            if os.path.exists(bpe_file):
                print(f"\nLoading existing BPE model from {bpe_file}")
                with open(bpe_file, 'rb') as f:
                    bpe = pickle.load(f)
            else:
                print(f"\nTraining BPE on train.txt with {num_merges} merges...")
                bpe = train_bpe_from_file('data/train.txt', num_merges=num_merges)
                print(f"Saving BPE model to {bpe_file}")
                with open(bpe_file, 'wb') as f:
                    pickle.dump(bpe, f)
            
            # Create vocabulary
            vocab = BPEVocabulary(bpe.get_vocab())
            actual_vocab_size = len(vocab)
            print(f"Actual vocabulary size: {actual_vocab_size}")
            
            # Create datasets
            print("\nCreating datasets with BPE tokenization...")
            train_data = SentimentDatasetBPE('data/train.txt', bpe, vocab)
            dev_data = SentimentDatasetBPE('data/dev.txt', bpe, vocab)
            
            # Create dataloaders
            train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_fn_bpe)
            dev_loader = DataLoader(dev_data, batch_size=32, shuffle=False, collate_fn=collate_fn_bpe)
            
            # Use best architecture from Part 1
            # (300d embeddings, 256 hidden, 2 layers, dropout at embedding)
            # But adapt for BPE with random embeddings
            embedding_dim = 300
            hidden_size = 256
            num_layers = 2
            dropout_prob = 0.3
            dropout_position = "embedding"
            
            print(f"\nInitializing DANBPE model:")
            print(f"  Vocab size: {actual_vocab_size}")
            print(f"  Embedding dim: {embedding_dim}")
            print(f"  Hidden size: {hidden_size}")
            print(f"  Num layers: {num_layers}")
            print(f"  Dropout: {dropout_prob} at {dropout_position}")
            
            model = DANBPE(
                vocab_size=actual_vocab_size,
                embedding_dim=embedding_dim,
                hidden_size=hidden_size,
                num_classes=2,
                num_layers=num_layers,
                dropout_prob=dropout_prob,
                dropout_position=dropout_position,
                pad_idx=vocab.pad_idx
            )
            
            print(f"  Total parameters: {model.get_num_params():,}")
            
            # Training configuration
            loss_fn = nn.NLLLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            epochs = 20
            
            print(f"\nTraining for {epochs} epochs...")
            
            train_losses = []
            dev_losses = []
            train_accs = []
            dev_accs = []
            
            best_dev_acc = 0
            start_time = time.time()
            
            for epoch in range(epochs):
                # Training
                train_acc, train_loss = train_epoch_dan(train_loader, model, loss_fn, optimizer)
                train_losses.append(train_loss)
                train_accs.append(train_acc)
                
                # Evaluation
                dev_acc, dev_loss = eval_epoch_dan(dev_loader, model, loss_fn)
                dev_losses.append(dev_loss)
                dev_accs.append(dev_acc)
                
                if dev_acc > best_dev_acc:
                    best_dev_acc = dev_acc
                
                if (epoch + 1) % 5 == 0:
                    print(f"Epoch {epoch+1:2d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                          f"Dev Loss: {dev_loss:.4f}, Dev Acc: {dev_acc:.4f}")
            
            total_time = time.time() - start_time
            
            print(f"\n{'-'*80}")
            print(f"BPE {num_merges} MERGES - FINAL RESULTS:")
            print(f"  Best Dev Accuracy: {best_dev_acc:.4f}")
            print(f"  Final Train Accuracy: {train_accs[-1]:.4f}")
            print(f"  Final Dev Accuracy: {dev_accs[-1]:.4f}")
            print(f"  Training Time: {total_time:.2f}s")
            print(f"  Vocabulary Size: {actual_vocab_size}")
            print(f"{'-'*80}")
            
            results.append({
                'num_merges': num_merges,
                'vocab_size': actual_vocab_size,
                'best_dev_acc': best_dev_acc,
                'final_train_acc': train_accs[-1],
                'final_dev_acc': dev_accs[-1],
                'train_losses': train_losses,
                'dev_losses': dev_losses,
                'train_accs': train_accs,
                'dev_accs': dev_accs,
                'training_time': total_time
            })
        
        # Print summary table
        print("\n" + "="*80)
        print("PART 2a: BPE VOCABULARY SIZE COMPARISON SUMMARY")
        print("="*80)
        print(f"{'Num Merges':<12} {'Vocab Size':<12} {'Best Dev Acc':<14} {'Final Dev Acc':<14} {'Train Time (s)':<14}")
        print("-"*80)
        for r in results:
            print(f"{r['num_merges']:<12} {r['vocab_size']:<12} {r['best_dev_acc']:<14.4f} {r['final_dev_acc']:<14.4f} {r['training_time']:<14.2f}")
        print("="*80)
        
        # Save results
        with open('results/bpe_results.pkl', 'wb') as f:
            pickle.dump(results, f)
        print("\nResults saved to results/bpe_results.pkl")
        
        # Create visualizations
        print("\nGenerating visualizations...")
        
        # 1. Performance vs Vocabulary Size
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        vocab_sizes_plot = [r['vocab_size'] for r in results]
        best_dev_accs = [r['best_dev_acc'] for r in results]
        final_dev_accs = [r['final_dev_acc'] for r in results]
        train_times = [r['training_time'] for r in results]
        
        # Plot 1: Dev Accuracy vs Vocab Size
        axes[0, 0].plot(vocab_sizes_plot, best_dev_accs, 'o-', linewidth=2, markersize=8, label='Best Dev Acc')
        axes[0, 0].plot(vocab_sizes_plot, final_dev_accs, 's--', linewidth=2, markersize=8, label='Final Dev Acc')
        axes[0, 0].set_xlabel('Vocabulary Size', fontsize=11)
        axes[0, 0].set_ylabel('Accuracy', fontsize=11)
        axes[0, 0].set_title('Dev Accuracy vs Vocabulary Size', fontsize=12, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Plot 2: Training Time vs Vocab Size
        axes[0, 1].plot(vocab_sizes_plot, train_times, 'o-', linewidth=2, markersize=8, color='orange')
        axes[0, 1].set_xlabel('Vocabulary Size', fontsize=11)
        axes[0, 1].set_ylabel('Training Time (s)', fontsize=11)
        axes[0, 1].set_title('Training Time vs Vocabulary Size', fontsize=12, fontweight='bold')
        axes[0, 1].grid(alpha=0.3)
        
        # Plot 3: Learning Curves for Best Config
        best_idx = best_dev_accs.index(max(best_dev_accs))
        best_result = results[best_idx]
        epochs_range = range(1, len(best_result['train_accs']) + 1)
        
        axes[1, 0].plot(epochs_range, best_result['train_accs'], 'b-', linewidth=2, label='Train Accuracy')
        axes[1, 0].plot(epochs_range, best_result['dev_accs'], 'r-', linewidth=2, label='Dev Accuracy')
        axes[1, 0].set_xlabel('Epoch', fontsize=11)
        axes[1, 0].set_ylabel('Accuracy', fontsize=11)
        axes[1, 0].set_title(f'Learning Curves (Best: {best_result["num_merges"]} merges, {best_result["vocab_size"]} vocab)', 
                            fontsize=12, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # Plot 4: Loss Curves for Best Config
        axes[1, 1].plot(epochs_range, best_result['train_losses'], 'b-', linewidth=2, label='Train Loss')
        axes[1, 1].plot(epochs_range, best_result['dev_losses'], 'r-', linewidth=2, label='Dev Loss')
        axes[1, 1].set_xlabel('Epoch', fontsize=11)
        axes[1, 1].set_ylabel('Loss', fontsize=11)
        axes[1, 1].set_title(f'Loss Curves (Best: {best_result["num_merges"]} merges, {best_result["vocab_size"]} vocab)', 
                            fontsize=12, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        
        plt.suptitle('Part 2a: BPE Vocabulary Size Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('results/bpe_vocab_analysis.png', dpi=150)
        print("Saved results/bpe_vocab_analysis.png")
        
        # 2. Individual plots for each configuration
        fig, axes = plt.subplots(len(results), 2, figsize=(14, 4*len(results)))
        
        for i, r in enumerate(results):
            epochs_range = range(1, len(r['train_accs']) + 1)
            
            # Accuracy plot
            axes[i, 0].plot(epochs_range, r['train_accs'], 'b-', linewidth=2, label='Train')
            axes[i, 0].plot(epochs_range, r['dev_accs'], 'r-', linewidth=2, label='Dev')
            axes[i, 0].set_xlabel('Epoch', fontsize=10)
            axes[i, 0].set_ylabel('Accuracy', fontsize=10)
            axes[i, 0].set_title(f'{r["num_merges"]} merges ({r["vocab_size"]} vocab) - Accuracy', fontsize=11, fontweight='bold')
            axes[i, 0].legend()
            axes[i, 0].grid(alpha=0.3)
            
            # Loss plot
            axes[i, 1].plot(epochs_range, r['train_losses'], 'b-', linewidth=2, label='Train')
            axes[i, 1].plot(epochs_range, r['dev_losses'], 'r-', linewidth=2, label='Dev')
            axes[i, 1].set_xlabel('Epoch', fontsize=10)
            axes[i, 1].set_ylabel('Loss', fontsize=10)
            axes[i, 1].set_title(f'{r["num_merges"]} merges ({r["vocab_size"]} vocab) - Loss', fontsize=11, fontweight='bold')
            axes[i, 1].legend()
            axes[i, 1].grid(alpha=0.3)
        
        plt.suptitle('Part 2a: Detailed Learning Curves for All BPE Configurations', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('results/bpe_all_configs.png', dpi=150)
        print("Saved results/bpe_all_configs.png")
        
        print("\n" + "="*80)
        print("PART 2a BPE EXPERIMENTS COMPLETE!")
        print(f"Best configuration: {results[best_idx]['num_merges']} merges, vocab size {results[best_idx]['vocab_size']}")
        print(f"Best dev accuracy: {results[best_idx]['best_dev_acc']:.4f}")
        print("="*80)

if __name__ == "__main__":
    main()
