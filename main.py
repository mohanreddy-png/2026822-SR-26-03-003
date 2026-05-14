"""
Emotion Classification using Hybrid Deep Sequential Attention Network (HDSAN)
==============================================================================
Architecture (matches diagram):
  CSV Datasets (train/val/test)
    → Information Gathering
    → Text Pre-Processing  (Tokenization | POS Tagging | Stop-Word Removal | Stemming)
    → Data Split           (pre-split by Kaggle: 16k / 2k / 2k)
    → Word Embedding       (Word2Vec skip-gram, dim=100)
    → HDSAN                (CNN + BiLSTM-GRU + Multi-Head Attention)
    → Classification Results  (Sadness | Joy | Love | Anger | Fear | Surprise)

Dataset: https://www.kaggle.com/datasets/parulpandey/emotion-dataset
  training.csv   – 16,000 samples
  validation.csv –  2,000 samples
  test.csv       –  2,000 samples
"""

import os, re, sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ─── paths ────────────────────────────────────────────────────────────────────
DATA_DIR   = "data"
TRAIN_CSV  = os.path.join(DATA_DIR, "training.csv")
VAL_CSV    = os.path.join(DATA_DIR, "validation.csv")
TEST_CSV   = os.path.join(DATA_DIR, "test.csv")

# ─── hyper-parameters ─────────────────────────────────────────────────────────
EMBEDDING_DIM = 100
MAX_SEQ_LEN   = 40
VOCAB_LIMIT   = 30_000
NUM_CLASSES   = 6
BATCH_SIZE    = 64
EPOCHS        = 30

EMOTION_LABELS = {0: "sadness", 1: "joy", 2: "love",
                  3: "anger",   4: "fear", 5: "surprise"}


# ══════════════════════════════════════════════════════════════════════════════
# 1. INFORMATION GATHERING
# ══════════════════════════════════════════════════════════════════════════════
def load_datasets():
    print("\n" + "="*65)
    print("STAGE 1 | INFORMATION GATHERING")
    print("="*65)
    train = pd.read_csv(TRAIN_CSV)
    val   = pd.read_csv(VAL_CSV)
    test  = pd.read_csv(TEST_CSV)
    for name, df in [("training.csv", train), ("validation.csv", val), ("test.csv", test)]:
        print(f"  {name:<18}: {len(df):>6,} rows  |  cols: {df.columns.tolist()}")
    print(f"\n  {'Label':<12}  {'Index':>5}  {'Train':>7}  {'Val':>6}  {'Test':>6}")
    print(f"  {'-'*45}")
    for idx, name in EMOTION_LABELS.items():
        t  = (train.label == idx).sum()
        v  = (val.label   == idx).sum()
        te = (test.label  == idx).sum()
        print(f"  {name:<12}  {idx:>5}  {t:>7,}  {v:>6,}  {te:>6,}")
    return train, val, test


# ══════════════════════════════════════════════════════════════════════════════
# 2. TEXT PRE-PROCESSING
# ══════════════════════════════════════════════════════════════════════════════
ENGLISH_SW = set("""
a about above after again against all am an and any are aren't as at be because
been before being below between both but by can't cannot could couldn't did didn't
do does doesn't doing don't down during each few for from further get got had
hadn't has hasn't have haven't having he he'd he'll he's her here here's hers
herself him himself his how how's i i'd i'll i'm i've if in into is isn't it it's
its itself let's me more most mustn't my myself no nor not of off on once only or
other ought our ours ourselves out over own same shan't she she'd she'll she's
should shouldn't so some such than that that's the their theirs them themselves
then there there's these they they'd they'll they're they've this those through
to too under until up very was wasn't we we'd we'll we're we've were weren't what
what's when when's where where's which while who who's whom why why's will with
won't would wouldn't you you'd you'll you're you've your yours yourself yourselves
""".split())


def word_tokenize(text: str):
    return re.findall(r"[a-z']+", text.lower())


def pos_tag(tokens):
    tagged = []
    for t in tokens:
        if t.endswith(("ing", "ed", "ize", "ise", "ate")):            tag = "VB"
        elif t.endswith(("ly",)):                                      tag = "RB"
        elif t.endswith(("ful","ous","ive","al","ible","able","ic")): tag = "JJ"
        elif t.endswith(("tion","ness","ment","ity","ism","age")):    tag = "NN"
        else:                                                          tag = "NN"
        tagged.append((t, tag))
    return tagged


class PorterStemmer:
    _rules = [
        ("ational","ate"),("tional","tion"),("enci","ence"),("anci","ance"),
        ("izer","ize"),("ising","ise"),("izing","ize"),("alism","al"),
        ("iness","i"),("fulness","ful"),("ousness","ous"),("iveness","ive"),
        ("ization","ize"),("isation","ise"),("ness",""),("ment",""),
        ("tion","t"),("sses","ss"),("ies","i"),("ing",""),("ed",""),
        ("er",""),("ly",""),("es",""),("s",""),
    ]
    def stem(self, word):
        if len(word) <= 3:
            return word
        for suffix, rep in self._rules:
            if word.endswith(suffix) and len(word) - len(suffix) >= 2:
                return word[:-len(suffix)] + rep
        return word


class TextPreprocessor:
    def __init__(self):
        self.stop_words = ENGLISH_SW
        self.stemmer    = PorterStemmer()

    def preprocess(self, text: str):
        tokens  = word_tokenize(str(text))
        tagged  = pos_tag(tokens)
        clean   = [t for t in tokens if t not in self.stop_words and len(t) > 1]
        stemmed = [self.stemmer.stem(t) for t in clean]
        return stemmed, tagged

    def preprocess_corpus(self, texts, label=""):
        print(f"\n{'='*65}")
        print(f"STAGE 2 | TEXT PRE-PROCESSING  [{label}]")
        print(f"{'='*65}")
        processed, tagged_all = [], []
        for text in texts:
            s, t = self.preprocess(text)
            processed.append(s)
            tagged_all.append(t)
        print(f"  Tokenization     : ok  (e.g. {processed[0][:6]})")
        print(f"  POS Tagging      : ok  (e.g. {tagged_all[0][:4]})")
        print(f"  Stop-Word Removal: ok")
        print(f"  Stemming         : ok  (e.g. {processed[0][:6]})")
        print(f"  Corpus size      : {len(processed):,} documents processed")
        return processed


# ══════════════════════════════════════════════════════════════════════════════
# 3. WORD EMBEDDING PROCESS  (Word2Vec)
# ══════════════════════════════════════════════════════════════════════════════
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.sequence import pad_sequences


class Word2VecEmbedder:
    def __init__(self, dim=EMBEDDING_DIM):
        self.dim        = dim
        self.w2v        = None
        self.word2idx   = {"<PAD>": 0, "<UNK>": 1}
        self.emb_matrix = None

    def train(self, corpus):
        print(f"\n{'='*65}")
        print("STAGE 4 | WORD EMBEDDING PROCESS  (Word2Vec skip-gram)")
        print(f"{'='*65}")
        self.w2v = Word2Vec(
            sentences=corpus,
            vector_size=self.dim,
            window=5,
            min_count=2,
            workers=4,
            epochs=15,
            sg=1,
        )
        for word in self.w2v.wv.index_to_key:
            if len(self.word2idx) >= VOCAB_LIMIT:
                break
            self.word2idx[word] = len(self.word2idx)

        V = len(self.word2idx)
        self.emb_matrix = np.zeros((V, self.dim), dtype="float32")
        hits = 0
        for word, idx in self.word2idx.items():
            if word in self.w2v.wv:
                self.emb_matrix[idx] = self.w2v.wv[word]
                hits += 1
        print(f"  Algorithm        : Skip-gram  |  vector_size={self.dim}  |  window=5  |  min_count=2")
        print(f"  Vocabulary size  : {V:,}")
        print(f"  Embedding matrix : {self.emb_matrix.shape}  ({hits:,} word vectors loaded)")
        return self

    def encode(self, corpus):
        seqs = [[self.word2idx.get(t, 1) for t in tokens] for tokens in corpus]
        return pad_sequences(seqs, maxlen=MAX_SEQ_LEN, padding="post", truncating="post")


# ══════════════════════════════════════════════════════════════════════════════
# 4. HYBRID DEEP SEQUENTIAL ATTENTION NETWORK
# ══════════════════════════════════════════════════════════════════════════════
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, MaxPooling1D,
    Bidirectional, LSTM, GRU, Dense, Dropout,
    LayerNormalization, GlobalAveragePooling1D,
    Concatenate, MultiHeadAttention, Add,
    Reshape, Lambda, SpatialDropout1D,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical


def build_hdsan(vocab_size: int, emb_matrix: np.ndarray) -> Model:
    inp = Input(shape=(MAX_SEQ_LEN,), name="token_input")

    emb = Embedding(
        input_dim=vocab_size,
        output_dim=EMBEDDING_DIM,
        weights=[emb_matrix],
        input_length=MAX_SEQ_LEN,
        trainable=True,
        name="word2vec_embedding",
    )(inp)
    emb = SpatialDropout1D(0.2, name="spatial_dropout")(emb)

    # CNN branch
    cnn = Conv1D(128, kernel_size=3, activation="relu", padding="same", name="cnn1")(emb)
    cnn = MaxPooling1D(pool_size=2, name="pool1")(cnn)
    cnn = Conv1D(64,  kernel_size=3, activation="relu", padding="same", name="cnn2")(cnn)
    cnn = MaxPooling1D(pool_size=2, name="pool2")(cnn)
    cnn = GlobalAveragePooling1D(name="cnn_gap")(cnn)
    cnn = Reshape((1, 64), name="cnn_reshape")(cnn)
    cnn = Lambda(lambda x: tf.tile(x, [1, MAX_SEQ_LEN, 1]), name="cnn_tile")(cnn)

    # BiLSTM-GRU branch
    rnn = Bidirectional(
        LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.1),
        name="bilstm"
    )(emb)
    rnn = GRU(64, return_sequences=True, dropout=0.2, name="gru")(rnn)

    # Merge
    merged = Concatenate(axis=-1, name="merge")([rnn, cnn])

    # Multi-Head Self-Attention + residual + LayerNorm
    attn = MultiHeadAttention(num_heads=4, key_dim=32, dropout=0.1, name="mhsa")(
        query=merged, value=merged, key=merged
    )
    x = Add(name="residual")([merged, attn])
    x = LayerNormalization(name="layernorm")(x)

    # Classification head
    x = GlobalAveragePooling1D(name="gap")(x)
    x = Dense(128, activation="relu", name="fc1")(x)
    x = Dropout(0.4, name="drop1")(x)
    x = Dense(64,  activation="relu", name="fc2")(x)
    x = Dropout(0.3, name="drop2")(x)
    out = Dense(NUM_CLASSES, activation="softmax", name="emotion_output")(x)

    model = Model(inputs=inp, outputs=out, name="HDSAN_EmotionClassifier")
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 5. TRAINING MODULE
# ══════════════════════════════════════════════════════════════════════════════
def training_module(model, X_train, y_train, X_val, y_val):
    print(f"\n{'='*65}")
    print("STAGE 5 | TRAINING MODULE")
    print(f"{'='*65}")
    model.summary(print_fn=lambda s: print("  " + s))

    y_train_cat = to_categorical(y_train, NUM_CLASSES)
    y_val_cat   = to_categorical(y_val,   NUM_CLASSES)

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=6,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3,
                          min_lr=1e-5, verbose=1),
        ModelCheckpoint("/home/claude/best_hdsan.keras",
                        monitor="val_accuracy", save_best_only=True, verbose=0),
    ]

    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )
    return history


# ══════════════════════════════════════════════════════════════════════════════
# 6. TESTING MODULE + CLASSIFICATION RESULTS
# ══════════════════════════════════════════════════════════════════════════════
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def evaluate(model, X_enc, y_true, split_name="TEST"):
    print(f"\n{'='*65}")
    print(f"STAGE 6 | {split_name} MODULE  --  CLASSIFICATION RESULTS")
    print(f"{'='*65}")
    probs  = model.predict(X_enc, batch_size=128, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    labels = [EMOTION_LABELS[i] for i in range(NUM_CLASSES)]
    print(classification_report(y_true, y_pred, target_names=labels, digits=4))
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred), index=labels, columns=labels)
    print("Confusion Matrix:")
    print(cm.to_string())
    print(f"\n  Overall accuracy: {accuracy_score(y_true, y_pred):.4f}")
    return y_pred, probs


# ══════════════════════════════════════════════════════════════════════════════
# 7. INFERENCE
# ══════════════════════════════════════════════════════════════════════════════
def predict_emotion(model, embedder, preprocessor, text: str):
    tokens, _ = preprocessor.preprocess(text)
    enc   = embedder.encode([tokens])
    probs = model.predict(enc, verbose=0)[0]
    top   = int(np.argmax(probs))
    print(f"\n  Input   : \"{text}\"")
    print(f"  Emotion : {EMOTION_LABELS[top].upper()}  (confidence: {probs[top]:.2%})")
    for i, name in EMOTION_LABELS.items():
        bar = "x" * int(probs[i] * 30)
        print(f"    {name:<10}: {bar:<30}  {probs[i]:.2%}")
    return EMOTION_LABELS[top], probs


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("\n+================================================================+")
    print("|  EMOTION CLASSIFICATION  |  HDSAN  |  Kaggle Emotion Dataset  |")
    print("+================================================================+")

    train_df, val_df, test_df = load_datasets()

    preprocessor = TextPreprocessor()
    train_tok = preprocessor.preprocess_corpus(train_df["text"].tolist(), "TRAIN 16k")
    val_tok   = preprocessor.preprocess_corpus(val_df["text"].tolist(),   "VAL 2k")
    test_tok  = preprocessor.preprocess_corpus(test_df["text"].tolist(),  "TEST 2k")

    print(f"\n{'='*65}")
    print("STAGE 3 | DATA SPLIT  (pre-split by Kaggle)")
    print(f"{'='*65}")
    y_train = train_df["label"].values
    y_val   = val_df["label"].values
    y_test  = test_df["label"].values
    print(f"  Train: {len(y_train):,}  |  Val: {len(y_val):,}  |  Test: {len(y_test):,}")

    embedder = Word2VecEmbedder(dim=EMBEDDING_DIM)
    embedder.train(train_tok)

    X_train = embedder.encode(train_tok)
    X_val   = embedder.encode(val_tok)
    X_test  = embedder.encode(test_tok)
    print(f"\n  X_train: {X_train.shape}  X_val: {X_val.shape}  X_test: {X_test.shape}")

    model = build_hdsan(len(embedder.word2idx), embedder.emb_matrix)
    training_module(model, X_train, y_train, X_val, y_val)

    evaluate(model, X_train, y_train, "TRAINING")
    evaluate(model, X_val,   y_val,   "VALIDATION")
    evaluate(model, X_test,  y_test,  "TESTING")

    print(f"\n{'='*65}")
    print("DEMO: REAL-TIME EMOTION PREDICTIONS")
    print(f"{'='*65}")
    demos = [
        "I am so happy and excited about my new job offer!",
        "I feel completely devastated and broken inside.",
        "I love you with all my heart forever.",
        "This injustice makes me absolutely furious!",
        "The dark hallway filled me with terror.",
        "I never expected this amazing surprise party!",
    ]
    for s in demos:
        predict_emotion(model, embedder, preprocessor, s)

    model.save("/home/claude/hdsan_emotion_model.keras")
    embedder.w2v.save("/home/claude/word2vec_emotion.model")
    print(f"\n{'='*65}")
    print("Model saved -> /home/claude/hdsan_emotion_model.keras")
    print("Word2Vec   saved -> /home/claude/word2vec_emotion.model")


if __name__ == "__main__":
    main()