import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tensorflow as tf
from sklearn.model_selection import train_test_split
from snorkel.classification.data import DictDataset, DictDataLoader


def load_imdb_dataset(load_train_labels: bool = False, split_dev_valid: bool = False):
    """
    Load IMDB movie review dataset for sentiment analysis.
    
    Returns:
        df_train: Training dataframe (unlabeled by default)
        df_test: Test dataframe (labeled)
    """
    try:
        # Try to load from local files first
        if os.path.exists("data/IMDB_Dataset.csv"):
            df = pd.read_csv("data/IMDB_Dataset.csv")
        else:
            # Download from Kaggle or use tensorflow datasets
            print("Downloading IMDB dataset...")
            import tensorflow_datasets as tfds
            
            # Load dataset
            train_data, test_data = tfds.load(
                'imdb_reviews', 
                split=['train', 'test'], 
                as_supervised=True,
                batch_size=-1
            )
            
            # Convert to pandas
            train_reviews, train_labels = tfds.as_numpy(train_data)
            test_reviews, test_labels = tfds.as_numpy(test_data)
            
            # Create dataframes
            df_train = pd.DataFrame({
                'text': [review.decode('utf-8') for review in train_reviews],
                'label': train_labels
            })
            
            df_test_full = pd.DataFrame({
                'text': [review.decode('utf-8') for review in test_reviews],
                'label': test_labels
            })
            
            # Sample to make it manageable
            df_train = df_train.sample(n=5000, random_state=123).reset_index(drop=True)
            df_test_full = df_test_full.sample(n=1000, random_state=123).reset_index(drop=True)
            
            # Save for future use
            os.makedirs("data", exist_ok=True)
            combined = pd.concat([df_train, df_test_full])
            combined.to_csv("data/IMDB_Dataset.csv", index=False)
            
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Using a small sample dataset instead...")
        
        # Fallback: create a small sample dataset
        sample_positive = [
            "This movie was absolutely fantastic! I loved every minute of it.",
            "One of the best films I've ever seen. Incredible performances.",
            "Amazing cinematography and brilliant storytelling. Highly recommend!",
            "Wonderful movie with great acting and a compelling plot.",
            "Excellent film! The director did an outstanding job.",
        ] * 200
        
        sample_negative = [
            "Terrible movie. Complete waste of time and money.",
            "One of the worst films I've ever watched. Boring and predictable.",
            "Awful acting and a nonsensical plot. Would not recommend.",
            "Horrible movie. I walked out of the theater halfway through.",
            "Disappointing film with poor execution. Very boring.",
        ] * 200
        
        df_train = pd.DataFrame({
            'text': sample_positive[:800] + sample_negative[:800],
            'label': [1]*800 + [0]*800
        })
        
        df_test_full = pd.DataFrame({
            'text': sample_positive[800:] + sample_negative[800:],
            'label': [1]*200 + [0]*200
        })
        
        df_train = df_train.sample(frac=1, random_state=123).reset_index(drop=True)
        df_test_full = df_test_full.sample(frac=1, random_state=123).reset_index(drop=True)
    
    # Create dev set
    df_dev = df_train.sample(100, random_state=123)
    
    # Remove labels from training set unless specified
    if not load_train_labels:
        df_train["label"] = np.ones(len(df_train["label"])) * -1
    
    # Split test into validation and test
    df_valid, df_test = train_test_split(
        df_test_full, test_size=250, random_state=123, stratify=df_test_full.label
    )
    
    if split_dev_valid:
        return df_train, df_dev, df_valid, df_test
    else:
        return df_train, df_test


def get_keras_logreg(input_dim, output_dim=2):
    model = tf.keras.Sequential()
    if output_dim == 1:
        loss = "binary_crossentropy"
        activation = tf.nn.sigmoid
    else:
        loss = "categorical_crossentropy"
        activation = tf.nn.softmax
    dense = tf.keras.layers.Dense(
        units=output_dim,
        input_dim=input_dim,
        activation=activation,
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
    )
    model.add(dense)
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
    return model


def get_keras_lstm(num_buckets, embed_dim=16, rnn_state_size=64):
    lstm_model = tf.keras.Sequential()
    lstm_model.add(tf.keras.layers.Embedding(num_buckets, embed_dim))
    lstm_model.add(tf.keras.layers.LSTM(rnn_state_size, activation=tf.nn.relu))
    lstm_model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))
    lstm_model.compile("Adagrad", "binary_crossentropy", metrics=["accuracy"])
    return lstm_model


def get_keras_early_stopping(patience=10, monitor="val_accuracy"):
    """Stops training if monitor value doesn't exceed the current max value after patience num of epochs"""
    return tf.keras.callbacks.EarlyStopping(
        monitor=monitor, patience=patience, verbose=1, restore_best_weights=True
    )


def map_pad_or_truncate(string, max_length=30, num_buckets=30000):
    """Tokenize text, pad or truncate to get max_length, and hash tokens."""
    ids = tf.keras.preprocessing.text.hashing_trick(
        string, n=num_buckets, hash_function="md5"
    )
    return ids[:max_length] + [0] * (max_length - len(ids))


def featurize_df_tokens(df):
    return np.array(list(map(map_pad_or_truncate, df.text)))


def df_to_features(vectorizer, df, split):
    """Convert pandas DataFrame containing text data to bag-of-words PyTorch features."""
    words = [row.text for i, row in df.iterrows()]

    if split == "train":
        feats = vectorizer.fit_transform(words)
    else:
        feats = vectorizer.transform(words)
    X = feats.todense()
    Y = df["label"].values
    return X, Y


def create_dict_dataloader(X, Y, split, **kwargs):
    """Create a DictDataLoader for bag-of-words features."""
    ds = DictDataset.from_tensors(torch.FloatTensor(X), torch.LongTensor(Y), split)
    return DictDataLoader(ds, **kwargs)


def get_pytorch_mlp(hidden_dim, num_layers):
    layers = []
    for _ in range(num_layers):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
    return nn.Sequential(*layers)