"""
Emotion Classification using Hybrid Deep Sequential Attention Network
Architecture: CSV Datasets → Text Pre-Processing → Word2Vec Embedding →
              Hybrid Deep Sequential Attention Network → 6-class Emotion Classification

Dataset: https://www.kaggle.com/datasets/parulpandey/emotion-dataset
Emotions: Sadness, Joy, Love, Anger, Fear, Surprise
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
from io import StringIO

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ─────────────────────────────────────────────────────────────────────────────
# 1. INFORMATION GATHERING  (load CSVs – Emotion Dataset-1 & Dataset-2)
# ─────────────────────────────────────────────────────────────────────────────
# ── Self-contained NLP utilities (no network calls needed) ───────────────────
ENGLISH_STOPWORDS = set("""i me my myself we our ours ourselves you your yours
yourself yourselves he him his himself she her hers herself it its itself they
them their theirs themselves what which who whom this that these those am is are
was were be been being have has had having do does did doing a an the and but if
or because as until while of at by for with about against between into through
during before after above below to from up down in out on off over under again
further then once here there when where why how all both each few more most
other some such no nor not only own same so than too very s t can will just
should now d ll m o re ve y ain aren couldn didn doesn hadn hasn haven isn ma
mightn mustn needn shan shouldn wasn weren wouldn""".split())


def word_tokenize(text: str):
    """Simple regex word tokenizer (replaces nltk.word_tokenize)."""
    return re.findall(r"[a-z]+", text.lower())


def pos_tag(tokens):
    """
    Lightweight POS tagger using suffix heuristics.
    Returns list of (word, tag) tuples (NN / VB / JJ / RB / OTHER).
    """
    tagged = []
    for t in tokens:
        if t.endswith(("ing", "ed", "ize", "ise", "ate")):
            tag = "VB"
        elif t.endswith(("ly",)):
            tag = "RB"
        elif t.endswith(("ful", "ous", "ive", "al", "ible", "able", "ic")):
            tag = "JJ"
        elif t.endswith(("tion", "ness", "ment", "ity", "ism", "age")):
            tag = "NN"
        else:
            tag = "NN"
        tagged.append((t, tag))
    return tagged


class PorterStemmer:
    """Minimal Porter-like stemmer (step 1 suffixes only)."""
    _rules = [
        ("ational", "ate"), ("tional", "tion"), ("enci", "ence"),
        ("anci", "ance"), ("izer", "ize"), ("ising", "ise"),
        ("izing", "ize"), ("ational", "ate"), ("alism", "al"),
        ("iness", "i"), ("fulness", "ful"), ("ousness", "ous"),
        ("iveness", "ive"), ("ization", "ize"), ("isation", "ise"),
        ("ness", ""), ("ment", ""), ("tion", "t"), ("sses", "ss"),
        ("ies", "i"), ("ing", ""), ("ed", ""), ("er", ""),
        ("ly", ""), ("es", ""), ("s", ""),
    ]
    def stem(self, word: str) -> str:
        if len(word) <= 3:
            return word
        for suffix, replacement in self._rules:
            if word.endswith(suffix) and len(word) - len(suffix) >= 2:
                return word[: -len(suffix)] + replacement
        return word

# ── Synthetic datasets (mirrors Kaggle emotion dataset format) ────────────────
SAMPLE_DATA_1 = """text,label
i feel so happy today everything is going great,joy
i am really sad about what happened yesterday,sadness
i feel so much love for my family and friends,love
i am furious about the unfair treatment i received,anger
i am scared of what might happen next,fear
wow i never expected that to happen at all,surprise
i am feeling joyful and excited about the trip,joy
this news makes me deeply sorrowful and upset,sadness
i adore you with all of my heart always,love
i am enraged by the injustice in the world,anger
the dark alley filled me with dread and terror,fear
i was completely stunned by the unexpected gift,surprise
feeling blissful and content with life right now,joy
i feel so heartbroken and devastated by the loss,sadness
i cherish every moment spent with loved ones,love
the corrupt system makes me incredibly angry,anger
i am terrified about the upcoming medical procedure,fear
the plot twist left everyone in complete shock,surprise
today i feel wonderful and full of gratitude,joy
i am overwhelmed with grief after the tragedy,sadness
my heart overflows with warmth and affection,love
i cannot stand the lies and betrayal anymore,anger
spiders and dark rooms make me extremely anxious,fear
nobody expected the surprise birthday party at all,surprise
life is beautiful and i am genuinely thrilled,joy
losing my job made me feel utterly miserable,sadness
i feel deep compassion and tenderness for others,love
the reckless driver made me furiously upset,anger
the horror movie left me trembling with fear,fear
the announcement completely caught everyone off guard,surprise
"""

SAMPLE_DATA_2 = """text,label
i am so delighted by the wonderful weather today,joy
the funeral was a heartbreaking and painful experience,sadness
holding hands feels warm gentle and full of love,love
i am boiling with rage at the dishonest politician,anger
the loud thunderstorm at night frightened me badly,fear
i gasped when i opened the mysterious package,surprise
winning the tournament filled me with sheer joy,joy
i feel hollow and broken after the breakup,sadness
you mean the entire world to me always,love
the bully made me incredibly angry and defensive,anger
walking alone at night always makes me nervous,fear
the magician's trick left the crowd in disbelief,surprise
i am exhilarated and overjoyed by the promotion,joy
i cried all night feeling lonely and depressed,sadness
sharing a meal with family fills my heart,love
i explode with anger every time i see injustice,anger
uncertainty about the future gives me constant anxiety,fear
the sudden loud noise startled and surprised everyone,surprise
i feel pure euphoria running through my veins,joy
the abandonment left me feeling empty inside forever,sadness
genuine kindness and care are the truest forms of love,love
the traffic jam made me lose my temper completely,anger
the strange shadow in the hallway frightened me,fear
nobody knew the party was a surprise at all,surprise
i am over the moon about the new baby,joy
this chronic loneliness is tearing me apart slowly,sadness
i feel grateful and loved by those around me,love
i am seething with frustration at the broken system,anger
the suspense in the movie made my heart race,fear
we were all shocked by the unexpected resignation,surprise
"""

EMOTION_LABELS = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}
LABEL_TO_IDX = {v: i for i, v in EMOTION_LABELS.items()}

def load_datasets():
    """Load Emotion Dataset-1 and Dataset-2 (Information Gathering stage)."""
    print("=" * 65)
    print("STAGE 1: INFORMATION GATHERING")
    print("=" * 65)
    df1 = pd.read_csv(StringIO(SAMPLE_DATA_1))
    df2 = pd.read_csv(StringIO(SAMPLE_DATA_2))
    print(f"  Emotion Dataset-1: {len(df1)} samples")
    print(f"  Emotion Dataset-2: {len(df2)} samples")
    combined = pd.concat([df1, df2], ignore_index=True)
    combined["label_idx"] = combined["label"].map(LABEL_TO_IDX)
    print(f"  Combined dataset : {len(combined)} samples")
    print(f"  Label distribution:\n{combined['label'].value_counts().to_string()}")
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# 2. TEXT PRE-PROCESSING
#    Tokenization → POS Tagging → Stop Word Removal → Stemming
# ─────────────────────────────────────────────────────────────────────────────
class TextPreprocessor:
    """Implements the 4-step Text Pre-Processing block from the diagram."""

    def __init__(self):
        self.stop_words = ENGLISH_STOPWORDS
        self.stemmer = PorterStemmer()

    # Step 1 – Tokenization
    def tokenize(self, text: str):
        text = text.lower()
        text = re.sub(r"[^a-z\s]", "", text)
        tokens = word_tokenize(text)
        return tokens

    # Step 2 – POS Tagging
    def pos_tag(self, tokens):
        return pos_tag(tokens)

    # Step 3 – Stop Word Removal
    def remove_stopwords(self, tokens):
        return [t for t in tokens if t not in self.stop_words and len(t) > 1]

    # Step 4 – Stemming
    def stem(self, tokens):
        return [self.stemmer.stem(t) for t in tokens]

    def preprocess(self, text: str):
        tokens = self.tokenize(text)
        tagged = self.pos_tag(tokens)          # POS tagging (used for feature enrichment)
        clean_tokens = self.remove_stopwords(tokens)
        stemmed = self.stem(clean_tokens)
        return stemmed, tagged

    def preprocess_corpus(self, texts):
        print("\n" + "=" * 65)
        print("STAGE 2: TEXT PRE-PROCESSING")
        print("=" * 65)
        processed_texts = []
        all_tagged = []
        for text in texts:
            stemmed, tagged = self.preprocess(text)
            processed_texts.append(stemmed)
            all_tagged.append(tagged)
        print(f"  Tokenization     : ✓")
        print(f"  POS Tagging      : ✓  (sample: {all_tagged[0][:4]})")
        print(f"  Stop Word Removal: ✓")
        print(f"  Stemming         : ✓  (sample: {processed_texts[0][:6]})")
        return processed_texts


# ─────────────────────────────────────────────────────────────────────────────
# 3. DATA SPLIT  (Train / Test)
# ─────────────────────────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split

def split_data(df, processed_texts, test_size=0.2, random_state=42):
    print("\n" + "=" * 65)
    print("STAGE 3: DATA SPLIT")
    print("=" * 65)
    X = processed_texts
    y = df["label_idx"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"  Train size: {len(X_train)} samples")
    print(f"  Test  size: {len(X_test)} samples")
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────────────────────
# 4. WORD EMBEDDING PROCESS  (Word2Vec)
# ─────────────────────────────────────────────────────────────────────────────
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.sequence import pad_sequences

EMBEDDING_DIM = 100
MAX_SEQ_LEN = 30
VOCAB_SIZE_LIMIT = 5000

class Word2VecEmbedder:
    """Train Word2Vec on the corpus, then build embedding matrix for Keras."""

    def __init__(self, embedding_dim=EMBEDDING_DIM):
        self.embedding_dim = embedding_dim
        self.model = None
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.embedding_matrix = None

    def train(self, corpus):
        """corpus: list of token lists."""
        print("\n" + "=" * 65)
        print("STAGE 4: WORD EMBEDDING PROCESS (Word2Vec)")
        print("=" * 65)
        self.model = Word2Vec(
            sentences=corpus,
            vector_size=self.embedding_dim,
            window=5,
            min_count=1,
            workers=4,
            epochs=50,
            sg=1,   # Skip-gram
        )
        # Build vocabulary index
        for word in self.model.wv.index_to_key:
            if len(self.word2idx) < VOCAB_SIZE_LIMIT:
                self.word2idx[word] = len(self.word2idx)

        vocab_size = len(self.word2idx)
        self.embedding_matrix = np.zeros((vocab_size, self.embedding_dim))
        hits = 0
        for word, idx in self.word2idx.items():
            if word in self.model.wv:
                self.embedding_matrix[idx] = self.model.wv[word]
                hits += 1
        print(f"  Word2Vec trained : vector_size={self.embedding_dim}, sg=1 (skip-gram)")
        print(f"  Vocabulary size  : {vocab_size}")
        print(f"  Embedding matrix : {self.embedding_matrix.shape}  ({hits} hits)")
        return self

    def encode(self, corpus):
        """Convert token lists → padded integer sequences."""
        sequences = [
            [self.word2idx.get(t, 1) for t in tokens]
            for tokens in corpus
        ]
        return pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding="post", truncating="post")


# ─────────────────────────────────────────────────────────────────────────────
# 5. HYBRID DEEP SEQUENTIAL ATTENTION NETWORK
#    BiLSTM + CNN + Multi-Head Self-Attention → Emotion Classification
# ─────────────────────────────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, MaxPooling1D,
    Bidirectional, LSTM, GRU, Dense, Dropout,
    LayerNormalization, GlobalAveragePooling1D,
    Concatenate, MultiHeadAttention, Add, Flatten
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

NUM_CLASSES = 6

def build_hdsan(vocab_size, embedding_matrix):
    """
    Hybrid Deep Sequential Attention Network (HDSAN):

    Input → Embedding (Word2Vec weights)
          ├─ CNN branch  : Conv1D(128) → MaxPool → Conv1D(64) → MaxPool
          └─ BiLSTM branch: BiLSTM(128) → GRU(64)
          → Concatenate
          → Multi-Head Self-Attention (4 heads)
          → LayerNorm + Residual
          → GlobalAveragePooling
          → Dense(128, relu) → Dropout(0.4)
          → Dense(64, relu)  → Dropout(0.3)
          → Dense(6, softmax)  [Sadness | Joy | Love | Anger | Fear | Surprise]
    """
    inp = Input(shape=(MAX_SEQ_LEN,), name="token_input")

    # Embedding layer initialised with pre-trained Word2Vec weights
    emb = Embedding(
        input_dim=vocab_size,
        output_dim=EMBEDDING_DIM,
        weights=[embedding_matrix],
        input_length=MAX_SEQ_LEN,
        trainable=True,          # allow fine-tuning
        name="word2vec_embedding",
    )(inp)

    # ── CNN Branch ───────────────────────────────────────────────────────────
    cnn = Conv1D(128, kernel_size=3, activation="relu", padding="same", name="cnn_1")(emb)
    cnn = MaxPooling1D(pool_size=2, name="pool_1")(cnn)
    cnn = Conv1D(64, kernel_size=3, activation="relu", padding="same", name="cnn_2")(cnn)
    cnn = MaxPooling1D(pool_size=2, name="pool_2")(cnn)

    # ── BiLSTM → GRU Branch ──────────────────────────────────────────────────
    rnn = Bidirectional(LSTM(128, return_sequences=True), name="bilstm")(emb)
    rnn = GRU(64, return_sequences=True, name="gru")(rnn)

    # ── Align temporal dimensions before concat ───────────────────────────────
    # CNN output: (batch, 7, 64)  RNN output: (batch, 30, 64)
    # Use GlobalAveragePooling on CNN so shapes agree for attention
    cnn_gap = GlobalAveragePooling1D(name="cnn_gap")(cnn)                # (batch, 64)
    from tensorflow.keras.layers import Reshape, Lambda
    cnn_exp = Reshape((1, 64), name="cnn_reshape")(cnn_gap)
    cnn_tiled = Lambda(lambda x: tf.tile(x, [1, MAX_SEQ_LEN, 1]), name="cnn_tile")(cnn_exp)

    # ── Concatenate CNN + RNN features ───────────────────────────────────────
    merged = Concatenate(axis=-1, name="merge")([rnn, cnn_tiled])        # (batch, 30, 128)

    # ── Multi-Head Self-Attention ─────────────────────────────────────────────
    attn_out = MultiHeadAttention(num_heads=4, key_dim=32, name="mhsa")(
        query=merged, value=merged, key=merged
    )
    attn_out = Add(name="residual")([merged, attn_out])
    attn_out = LayerNormalization(name="layernorm")(attn_out)

    # ── Classification Head ───────────────────────────────────────────────────
    x = GlobalAveragePooling1D(name="gap")(attn_out)
    x = Dense(128, activation="relu", name="fc1")(x)
    x = Dropout(0.4, name="drop1")(x)
    x = Dense(64, activation="relu", name="fc2")(x)
    x = Dropout(0.3, name="drop2")(x)
    out = Dense(NUM_CLASSES, activation="softmax", name="emotion_output")(x)

    model = Model(inputs=inp, outputs=out, name="HDSAN")
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 6. TRAINING MODULE
# ─────────────────────────────────────────────────────────────────────────────
def train_module(model, X_train_enc, y_train, X_test_enc, y_test, epochs=30, batch_size=16):
    print("\n" + "=" * 65)
    print("STAGE 5: TRAINING MODULE")
    print("=" * 65)
    model.summary(print_fn=lambda x: print("  " + x))

    y_train_cat = to_categorical(y_train, NUM_CLASSES)
    y_test_cat = to_categorical(y_test, NUM_CLASSES)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=0),
    ]

    history = model.fit(
        X_train_enc, y_train_cat,
        validation_data=(X_test_enc, y_test_cat),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    return history


# ─────────────────────────────────────────────────────────────────────────────
# 7. TESTING MODULE + CLASSIFICATION RESULTS
# ─────────────────────────────────────────────────────────────────────────────
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_and_report(model, X_enc, y_true, split_name="TEST"):
    print(f"\n{'=' * 65}")
    print(f"STAGE 6: {split_name} MODULE – CLASSIFICATION RESULTS")
    print("=" * 65)
    probs = model.predict(X_enc, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    labels = [EMOTION_LABELS[i] for i in range(NUM_CLASSES)]
    report = classification_report(y_true, y_pred, target_names=labels, digits=4)
    print(report)

    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print("Confusion Matrix:")
    print(cm_df.to_string())
    return y_pred, probs


def predict_emotion(model, embedder, preprocessor, text: str):
    """Run inference on a raw text string."""
    tokens, _ = preprocessor.preprocess(text)
    enc = embedder.encode([tokens])
    probs = model.predict(enc, verbose=0)[0]
    top_idx = int(np.argmax(probs))
    print(f"\n  Input   : \"{text}\"")
    print(f"  Emotion : {EMOTION_LABELS[top_idx].upper()}  (confidence: {probs[top_idx]:.2%})")
    for i, label in EMOTION_LABELS.items():
        bar = "█" * int(probs[i] * 30)
        print(f"    {label:<10}: {bar:<30} {probs[i]:.2%}")
    return EMOTION_LABELS[top_idx], probs


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║   EMOTION CLASSIFICATION – HYBRID DEEP SEQUENTIAL ATTENTION ║")
    print("║   NETWORK (HDSAN)  |  6 Classes                             ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    # ── Stage 1: Load data ────────────────────────────────────────────────────
    df = load_datasets()

    # ── Stage 2: Pre-process ──────────────────────────────────────────────────
    preprocessor = TextPreprocessor()
    processed_texts = preprocessor.preprocess_corpus(df["text"].tolist())

    # ── Stage 3: Split ────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = split_data(df, processed_texts)

    # ── Stage 4: Word2Vec Embeddings ──────────────────────────────────────────
    embedder = Word2VecEmbedder(embedding_dim=EMBEDDING_DIM)
    embedder.train(X_train)                        # train only on train split
    X_train_enc = embedder.encode(X_train)
    X_test_enc  = embedder.encode(X_test)

    # ── Stage 5: Build + Train HDSAN ──────────────────────────────────────────
    vocab_size = len(embedder.word2idx)
    model = build_hdsan(vocab_size, embedder.embedding_matrix)
    history = train_module(model, X_train_enc, y_train, X_test_enc, y_test,
                           epochs=40, batch_size=16)

    # ── Stage 6: Evaluate ─────────────────────────────────────────────────────
    print("\n── TRAINING SET RESULTS ──")
    evaluate_and_report(model, X_train_enc, y_train, split_name="TRAINING")

    print("\n── TEST SET RESULTS ──")
    evaluate_and_report(model, X_test_enc, y_test, split_name="TESTING")

    # ── Demo Predictions ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("DEMO: REAL-TIME EMOTION PREDICTIONS")
    print("=" * 65)
    demo_sentences = [
        "I am so happy and excited about my new job offer!",
        "I feel completely devastated and broken inside.",
        "I love you with all my heart forever.",
        "This injustice makes me absolutely furious!",
        "The dark hallway filled me with terror.",
        "I never expected this amazing surprise!",
    ]
    for sentence in demo_sentences:
        predict_emotion(model, embedder, preprocessor, sentence)

    # ── Save model ────────────────────────────────────────────────────────────
    model.save("/home/claude/hdsan_emotion_model.keras")
    embedder.model.save("/home/claude/word2vec_emotion.model")
    print("\n✓ Model saved to /home/claude/hdsan_emotion_model.keras")
    print("✓ Word2Vec saved to /home/claude/word2vec_emotion.model")


if __name__ == "__main__":
    main()