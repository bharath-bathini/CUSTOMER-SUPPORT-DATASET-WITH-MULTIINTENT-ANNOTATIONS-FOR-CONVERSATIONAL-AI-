
import os
import pickle
import joblib
import numpy as np
import pandas as pd
from collections import Counter
from scipy.special import expit  # sigmoid

# ===============================
# Data Visualization
# ===============================
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# ===============================
# NLP: NLTK & Gensim
# ===============================
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.util import ngrams

# ===============================
# Scikit-learn: Preprocessing, Models, Metrics
# ===============================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE

# ===============================
# TensorFlow / Keras
# ===============================
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Embedding, Conv1D, GlobalMaxPooling1D, LSTM, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# ===============================
# Transformers (Hugging Face)
# ===============================
from transformers import AutoTokenizer, AutoModel
import torch
from torch.optim import AdamW

# ===============================
# Custom Modules (Assumed to exist)
# ===============================




class CustomerServiceClassifier:
    def __init__(self, model_dir="model"):
        """Initialize the classifier with a model directory."""
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.label_encoders = {}
        self.metrics_calculators = {}
        self.graph_plotters = {}
        self.features_dict = {}
        self.labels_dict = {}
        self.models = {}

    def upload_dataset(self, file_path):
        """Load the dataset from a CSV file."""
        return pd.read_csv(file_path)

    def preprocess_data(self, df, save_path=None, target_cols=None):
        """Preprocess the dataset and encode target labels."""
        if save_path and os.path.exists(save_path):
            print(f"Loading existing preprocessed file: {save_path}")
            df = pd.read_csv(save_path)
        else:
            print("Preprocessing data" + (f" and saving to: {save_path}" if save_path else " (no saving)"))
            lemmatizer = WordNetLemmatizer()
            stop_words = set(stopwords.words('english'))

            def clean_text(text):
                text = str(text).lower()
                tokens = word_tokenize(text)
                tokens = [lemmatizer.lemmatize(t) for t in tokens if t.isalnum() and t not in stop_words]
                return ' '.join(tokens)

            # Separate target columns
            target_df = None
            if target_cols:
                existing_targets = [col for col in target_cols if col in df.columns]
                target_df = df[existing_targets].copy()
                df = df.drop(columns=existing_targets)

            # Process text columns
            text_columns = df.select_dtypes(include='object').columns
            for col in text_columns:
                df[f'processed_{col}'] = df[col].apply(clean_text)

            # Drop original text columns
            df.drop(columns=text_columns, inplace=True)

            # Reattach target columns
            if target_df is not None:
                for col in target_df.columns:
                    df[col] = target_df[col]

            # Save only if path is specified
            if save_path:
                df.to_csv(save_path, index=False)

        # Select processed and numerical columns
        processed_text_cols = [col for col in df.columns if col.startswith('processed_')]
        non_text_cols = [col for col in df.columns if col not in processed_text_cols + (target_cols if target_cols else [])]

        # Join processed text columns into one string
        X_text = df[processed_text_cols].astype(str).agg(' '.join, axis=1)

        # Combine with numerical columns if any
        X_numeric = df[non_text_cols].values if non_text_cols else None
        if X_numeric is not None and len(X_numeric) > 0:
            X = [f"{text} {' '.join(map(str, numeric))}" for text, numeric in zip(X_text, X_numeric)]
        else:
            X = X_text.tolist()

        # Encode multiple target columns
        Y_dict = {}
        if target_cols:
            for col in target_cols:
                if col in df.columns:
                    le = LabelEncoder()
                    Y_dict[col] = le.fit_transform(df[col])
                    self.label_encoders[col] = le

        return X, Y_dict

    def eda_nlp_analysis(self, X_text, num_words=100, top_n_words=20):
        """Perform NLP EDA: WordCloud, Top N words, Document length, POS tags, Bigrams."""
        print("Generating NLP EDA Visualizations...")

        # Flatten all tokens
        all_tokens = [word for doc in X_text for word in word_tokenize(doc)]
        word_freq = Counter(all_tokens)

        # WordCloud
        wc = WordCloud(width=800, height=400, max_words=num_words, background_color='white').generate_from_frequencies(word_freq)
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"Top {num_words} Words - WordCloud")
        plt.show()

        # Top-N Frequent Words
        common_words = word_freq.most_common(top_n_words)
        words, counts = zip(*common_words)
        plt.figure(figsize=(10, 5))
        sns.barplot(x=list(counts), y=list(words), palette="viridis")
        plt.title(f"Top {top_n_words} Most Frequent Words")
        plt.xlabel("Count")
        plt.ylabel("Word")
        plt.show()

        # Document Length Histogram
        doc_lengths = [len(word_tokenize(doc)) for doc in X_text]
        plt.figure(figsize=(10, 5))
        sns.histplot(doc_lengths, bins=20, kde=True, color='teal')
        plt.title("Distribution of Document Lengths (in words)")
        plt.xlabel("Number of Words per Document")
        plt.ylabel("Frequency")
        plt.show()

        # POS Tag Frequency
        all_pos = [tag for _, tag in pos_tag(all_tokens)]
        pos_counts = Counter(all_pos).most_common()
        pos_tags, pos_freqs = zip(*pos_counts)
        plt.figure(figsize=(10, 5))
        sns.barplot(x=list(pos_tags), y=list(pos_freqs), palette="coolwarm")
        plt.title("Part of Speech (POS) Tag Frequency")
        plt.xlabel("POS Tag")
        plt.ylabel("Frequency")
        plt.xticks(rotation=45)
        plt.show()

        # Bigram Frequency
        bigrams = list(ngrams(all_tokens, 2))
        bigram_freq = Counter(bigrams).most_common(top_n_words)
        bigram_labels = [' '.join(b) for b, _ in bigram_freq]
        bigram_counts = [count for _, count in bigram_freq]
        plt.figure(figsize=(10, 5))
        sns.barplot(x=bigram_counts, y=bigram_labels, palette="magma")
        plt.title(f"Top {top_n_words} Bigrams")
        plt.xlabel("Count")
        plt.ylabel("Bigram")
        plt.show()

    def plot_target_distributions(self, Y_dict):
        """Plot class distributions for each target column."""
        y_df = pd.DataFrame(Y_dict)
        for col in y_df.columns:
            plt.figure(figsize=(6, 4))
            sns.countplot(x=y_df[col])
            plt.title(f'Class Distribution: {col}')
            plt.xlabel('Class')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.show()

    def minilm_feature_extraction(self, texts, model_name='microsoft/MiniLM-L12-H384-uncased', batch_size=32, pooling='mean'):
        """Extract MiniLM features from texts with tqdm progress bar."""
        from tqdm import tqdm
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()

        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting MiniLM embeddings"):
            batch_texts = texts[i:i+batch_size]
            encoded_input = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt')
            with torch.no_grad():
                model_output = model(**encoded_input)
            token_embeddings = model_output.last_hidden_state
            attention_mask = encoded_input['attention_mask']
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

            if pooling == 'mean':
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
                sum_mask = input_mask_expanded.sum(dim=1)
                embeddings = sum_embeddings / sum_mask
            elif pooling == 'cls':
                embeddings = token_embeddings[:, 0, :]
            else:
                raise ValueError("Pooling must be 'mean' or 'cls'")

            all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)

    def feature_extraction(self, X_text, method='MiniLM-WE', is_train=True):
        """Extract features using MiniLM or other methods."""
        x_file = os.path.join(self.model_dir, f'X_{method}.pkl')
        print(f"[INFO] Feature extraction method: {method}, Train mode: {is_train}")
        model_name = 'microsoft/MiniLM-L12-H384-uncased'

        if is_train and os.path.exists(x_file):
            print(f"[INFO] Loading cached MiniLM features from {x_file}")
            X = joblib.load(x_file)
        else:
            print("[INFO] Computing MiniLM features...")
            X = self.minilm_feature_extraction(X_text, model_name=model_name, pooling='mean')
            if is_train:
                os.makedirs(self.model_dir, exist_ok=True)
                joblib.dump(X, x_file)

        return X

    def balance_data(self, features, Y_dict):
        """Balance the dataset using SMOTE."""
        smote = SMOTE(random_state=42)
        features_smoted = {}
        labels_smoted = {}

        for i, (key, y) in enumerate(Y_dict.items(), start=1):
            X_resampled, y_resampled = smote.fit_resample(features, y)
            features_smoted[key] = X_resampled
            labels_smoted[key] = y_resampled
            print(f"Balanced '{key}' class: {key}.shape = {X_resampled.shape}, {key}.shape = {y_resampled.shape}")

        return features_smoted, labels_smoted

    def train_single_ml_model(self, algorithm_prefix, features_dict, Y_dict, algorithm):
        """Train a single machine learning model for each target."""
        model_mapping = {
            "DTC": DecisionTreeClassifier,
            "KNN": KNeighborsClassifier,
            "NBC": GaussianNB
        }

        if algorithm not in model_mapping:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        ml_models = {}
        for target_name, y_encoded in Y_dict.items():
            X = features_dict[target_name]
            model_instance = model_mapping[algorithm]()
            model_path = os.path.join(self.model_dir, f"{algorithm_prefix}_{target_name}_{algorithm}_model.pkl")
            algo_name = f"{algorithm_prefix} {algorithm} [{target_name}]"

            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )

            if os.path.exists(model_path):
                print(f"Loading existing {algorithm} model for {target_name}...")
                mdl = joblib.load(model_path)
            else:
                print(f"Training {algorithm} for target: {target_name}...")
                mdl = model_instance
                mdl.fit(X_train, y_train)
                joblib.dump(mdl, model_path)

            y_pred = mdl.predict(X_test)
            try:
                y_score = mdl.predict_proba(X_test)
            except AttributeError:
                y_score = None

    
            ml_models[f"{target_name}_{algorithm}"] = mdl

        return ml_models

    def train_feature_selector(self, features_dict, labels_dict, model_prefix="SBERT"):
        """Train a dense neural network for feature selection."""
        extracted_features_dict = {}

        for target_name, y in labels_dict.items():
            X = features_dict[target_name]
            model_path = os.path.join(self.model_dir, f"{model_prefix}_{target_name}_dense_model.h5")
            feature_path = os.path.join(self.model_dir, f"{model_prefix}_{target_name}_features.npy")
            history_path = os.path.join(self.model_dir, f"{model_prefix}_{target_name}_history.npy")

            if all(os.path.exists(p) for p in [model_path, feature_path, history_path]):
                print(f"✅ [LOAD] {target_name}: Model and features found. Loading...")
                model = load_model(model_path)
                features = np.load(feature_path)
                extracted_features_dict[target_name] = features
                continue

            print(f"🚀 [TRAIN] {target_name}: No saved model found. Training new Dense model...")
            num_classes = len(np.unique(y))
            y_categorical = to_categorical(y, num_classes=num_classes)

            input_layer = Input(shape=(X.shape[1],))
            x = Dense(512, activation='relu')(input_layer)
            x = Dense(256, activation='relu')(x)
            x = Dense(128, activation='relu', name='feature_layer')(x)
            x = Dense(64, activation='relu')(x)
            output_layer = Dense(num_classes, activation='softmax')(x)

            model = Model(inputs=input_layer, outputs=output_layer)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            history = model.fit(
                X, y_categorical,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                verbose=1
            )

            feature_extractor = Model(inputs=model.input, outputs=model.get_layer('feature_layer').output)
            features = feature_extractor.predict(X)

            model.save(model_path)
            np.save(feature_path, features)
            np.save(history_path, history.history)
            print(f"💾 [SAVED] {target_name}: Model, features, and training history saved.")
            extracted_features_dict[target_name] = features

        return extracted_features_dict

    def train_multioutput_knn(self, algorithm_prefix, features_dict, labels_dict):
        """Train KNN models for multiple outputs."""
        model_results = {}

        for target_name, y_encoded in labels_dict.items():
            X = features_dict[target_name]
            model_path = os.path.join(self.model_dir, f"{algorithm_prefix}_{target_name}_KNN_model.pkl")
            algo_name = f"{algorithm_prefix} KNN [{target_name}]"

            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )

            if os.path.exists(model_path):
                print(f"Loading existing KNN model for {target_name}...")
                model = joblib.load(model_path)
            else:
                print(f"Training KNN for target: {target_name}...")
                model = KNeighborsClassifier()
                model.fit(X_train, y_train)
                joblib.dump(model, model_path)

            y_pred = model.predict(X_test)
            try:
                y_score = model.predict_proba(X_test)
            except AttributeError:
                y_score = None

         
            model_results[target_name] = model

        return model_results

    def plot_training_history(self, history, title=None):
        """Plot training and validation accuracy/loss."""
        if history is None or not isinstance(history, dict):
            print("[ERROR] Invalid history format.")
            return

        plt.figure(figsize=(12, 5))
        if 'accuracy' in history:
            plt.subplot(1, 2, 1)
            plt.plot(history['accuracy'], label='Train Acc')
            if 'val_accuracy' in history:
                plt.plot(history['val_accuracy'], label='Val Acc')
            plt.title(f"{title} Accuracy" if title else "Accuracy")
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()

        if 'loss' in history:
            plt.subplot(1, 2, 2)
            plt.plot(history['loss'], label='Train Loss')
            if 'val_loss' in history:
                plt.plot(history['val_loss'], label='Val Loss')
            plt.title(f"{title} Loss" if title else "Loss")
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_all_histories_from_folder(self):
        """Plot all training histories from saved files."""
        history_files = [f for f in os.listdir(self.model_dir) if f.endswith("_history.npy")]
        if history_files:
            print(history_files)

        for file in history_files:
            full_path = os.path.join(self.model_dir, file)
            try:
                history = np.load(full_path, allow_pickle=True).item()
                self.plot_training_history(history, title=file.replace("_history.npy", ""))
            except Exception as e:
                print(f"[ERROR] Could not load {file}: {e}")

    def plot_metrics(self):
        """Plot metrics for all targets using GraphPlotter."""
        for target_name, metrics_calculator in self.metrics_calculators.items():
            self.graph_plotters[target_name] = GraphPlotter(metrics_calculator.metrics_df, metrics_calculator.class_performance_dfs)
            self.graph_plotters[target_name].plot_all()
            melted_df = metrics_calculator.metrics_df[['Algorithm', 'Accuracy', 'Precision', 'Recall', 'F1-Score']].copy()
            melted_df = melted_df.round(3)
            print(f"Metrics for {target_name}:\n", melted_df)

    def test_DNN_feature_selector(self, features_test, model_base_name="MiniLM-WE DNN"):
        """Extract features from test data using trained DNN models."""
        feature_outputs = {}
        targets = ["intent", "category"]

        for target in targets:
            model_filename = f"{model_base_name}_{target}_dense_model.h5"
            model_path = os.path.join(self.model_dir, model_filename)

            if not os.path.exists(model_path):
                print(f"❌ Model not found: {model_path}")
                continue

            print(f"✅ Loading model for {target}: {model_path}")
            model = load_model(model_path)

            try:
                feature_extractor = Model(inputs=model.input, outputs=model.get_layer("feature_layer").output)
            except:
                raise ValueError(f"Model does not contain a layer named 'feature_layer': {model_path}")

            print(f"🚀 Extracting features for {target}...")
            extracted_features = feature_extractor.predict(features_test)
            feature_outputs[target] = extracted_features

        return feature_outputs

    def predict(self, test_path, model_base_name="MiniLM-WE DNN"):
        """Predict on test data and return results with original labels."""
        df_test = self.upload_dataset(test_path)
        df_result = df_test.copy()
        X_test, _ = self.preprocess_data(df_test)
        features_test = self.feature_extraction(X_test, method='MiniLM-WE', is_train=False)
        feature_outputs_dict = self.test_DNN_feature_selector(features_test, model_base_name)

        for target in ["intent", "category"]:
            target_features = feature_outputs_dict.get(target)
            if target_features is not None:
                y_pred = self.models[target].predict(target_features)
                le = self.label_encoders[target]
                mapped_labels = le.inverse_transform(y_pred)
                df_result[f'Predicted_{target}'] = mapped_labels

        return df_result

    def fit(self, train_path, target_cols=["intent", "category"]):
        """Train the entire pipeline on the provided training data."""
        # Load and preprocess data
        df = self.upload_dataset(train_path)
        X, Y_dict = self.preprocess_data(df, save_path=os.path.join(self.model_dir, "cleaned_data.csv"), target_cols=target_cols)

        # Perform EDA
        self.eda_nlp_analysis(X)
        self.plot_target_distributions(Y_dict)

        # Extract features
        features = self.feature_extraction(X, method='MiniLM-WE', is_train=True)

        # Balance data
        self.features_dict, self.labels_dict = self.balance_data(features, Y_dict)


        # Train feature selector (DNN)
        extracted_features_dict = self.train_feature_selector(self.features_dict, self.labels_dict, model_prefix="MiniLM-WE DNN")

        # Train final KNN models
        self.models = self.train_multioutput_knn("MiniLM-WE DNN", extracted_features_dict, self.labels_dict)


