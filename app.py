import tensorflow as tf
import wikipedia
import re
import numpy as np
from tensorflow.keras.layers import Layer, Embedding, Dense, LayerNormalization, Dropout
from tensorflow.keras import Model
import random

# Constants with better values for optimization
VOCAB_SIZE = 3000
MAX_LENGTH = 1024
EMBED_DIM = 768
NUM_HEADS = 12
DFF = 3072
NUM_LAYERS = 12
DROPOUT_RATE = 0.1

class MultiHeadSelfAttention(Layer):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = Dense(3 * embed_dim)
        self.output_proj = Dense(embed_dim)

    def call(self, x, mask=None, training=False):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        qkv = self.qkv_proj(x)
        qkv = tf.reshape(qkv, [batch_size, seq_len, 3, self.num_heads, self.head_dim])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])

        q, k, v = qkv[0], qkv[1], qkv[2]

        scaled_attention = tf.matmul(q, k, transpose_b=True)
        scaled_attention = scaled_attention / tf.math.sqrt(tf.cast(self.head_dim, tf.float32))

        if mask is not None:
            scaled_attention = tf.where(mask == 0, -1e9, scaled_attention)

        attention_weights = tf.nn.softmax(scaled_attention, axis=-1)
        output = tf.matmul(attention_weights, v)

        output = tf.transpose(output, [0, 2, 1, 3])
        output = tf.reshape(output, [batch_size, seq_len, self.embed_dim])

        return self.output_proj(output)


class FeedForwardNN(Layer):
    def __init__(self, embed_dim, dff, dropout_rate=0.1):
        super().__init__()
        self.dense1 = Dense(dff)
        self.dense2 = Dense(embed_dim)
        self.dropout = Dropout(dropout_rate)

    def call(self, x, training=False):
        x = self.dense1(x)
        x = tf.nn.gelu(x)
        x = self.dropout(x, training=training)
        return self.dense2(x)


class Transformer(Layer):
    def __init__(self, embed_dim, num_heads, dff, dropout_rate=0.1):
        super().__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = FeedForwardNN(embed_dim, dff, dropout_rate)
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, x, mask=None, training=False):
        normalized_x = self.norm1(x)
        att_output = self.att(normalized_x, mask, training=training)
        att_output = self.dropout1(att_output, training=training)
        out1 = x + att_output

        normalized_out1 = self.norm2(out1)
        ffn_output = self.ffn(normalized_out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)

        return out1 + ffn_output


class GPT2(Model):
    def __init__(self, vocab_size, max_length, embed_dim=768, num_heads=12, dff=3072, num_layers=12, dropout_rate=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length

        self.token_emb = Embedding(vocab_size, embed_dim)
        self.pos_emb = self.add_weight(
            name="positional_embeddings",
            shape=[max_length, embed_dim],
            initializer="zeros",
            trainable=True
        )

        self.dropout = Dropout(dropout_rate)
        self.transformer_blocks = [
            Transformer(embed_dim, num_heads, dff, dropout_rate)
            for _ in range(num_layers)
        ]
        self.norm = LayerNormalization(epsilon=1e-6)
        self.out = Dense(vocab_size)

        mask = 1 - tf.linalg.band_part(tf.ones((max_length, max_length)), -1, 0)
        self.causal_mask = tf.cast(mask, tf.float32)

    def call(self, x, training=False):
        seq_len = tf.shape(x)[1]
        mask = self.causal_mask[:seq_len, :seq_len]

        token_emb = self.token_emb(x)
        pos_emb = self.pos_emb[:seq_len]
        x = token_emb + pos_emb
        x = self.dropout(x, training=training)

        for transformer in self.transformer_blocks:
            x = transformer(x, mask, training=training)

        x = self.norm(x)
        return self.out(x)

    def generate(self, input_ids, max_new_tokens=50, temperature=0.8, top_k=40):
        input_ids = tf.convert_to_tensor([input_ids], dtype=tf.int32)

        for _ in range(max_new_tokens):
            context = input_ids[:, -self.max_length:]
            logits = self(context, training=False)
            next_token_logits = logits[:, -1, :]
            next_token_logits = next_token_logits / temperature

            if top_k > 0:
                values, _ = tf.math.top_k(next_token_logits, k=top_k)
                min_value = values[:, -1]
                next_token_logits = tf.where(
                    next_token_logits < min_value[:, tf.newaxis],
                    tf.ones_like(next_token_logits) * -1e10,
                    next_token_logits
                )

            probs = tf.nn.softmax(next_token_logits, axis=-1)
            next_token = tf.random.categorical(tf.math.log(probs), num_samples=1, dtype=tf.int32)
            input_ids = tf.concat([input_ids, next_token], axis=1)

        return input_ids[0].numpy()


class Tokenizer:
    def __init__(self, vocab_size=VOCAB_SIZE):
        self.vocab_size = vocab_size
        self.token2idx = {'<PAD>': 0, '<UNK>': 1, '<EOS>': 2}
        self.idx2token = {0: '<PAD>', 1: '<UNK>', 2: '<EOS>'}
        self.word_pattern = re.compile(r"\b\w+\b")

    def fit_on_text(self, text):
        text = text.lower()
        tokens = self.word_pattern.findall(text)

        freq = {}
        for token in tokens:
            freq[token] = freq.get(token, 0) + 1

        sorted_tokens = sorted(freq.items(), key=lambda x: -x[1])
        vocab_tokens = [t[0] for t in sorted_tokens[:self.vocab_size - 3]]

        for i, token in enumerate(vocab_tokens):
            idx = i + 3
            self.token2idx[token] = idx
            self.idx2token[idx] = token

        return self

    def encode(self, text, max_length=MAX_LENGTH):
        text = text.lower()
        tokens = self.word_pattern.findall(text)
        ids = [self.token2idx.get(t, 1) for t in tokens[:max_length-1]]
        ids.append(2)

        if len(ids) < max_length:
            ids += [0] * (max_length - len(ids))

        return ids

    def decode(self, ids):
        tokens = []
        for id_ in ids:
            if id_ == 0 or id_ == 2:
                break
            tokens.append(self.idx2token.get(id_, '<UNK>'))

        return " ".join(tokens)


def fetch_wiki_text(topic, max_retries=3):
    for _ in range(max_retries):
        try:
            page = wikipedia.page(topic, auto_suggest=False)
            return page.content
        except wikipedia.DisambiguationError as e:
            if e.options:
                try:
                    page = wikipedia.page(e.options[0], auto_suggest=False)
                    return page.content
                except:
                    continue
        except Exception as e:
            print(f"Error fetching {topic}: {e}")
            continue

    return f"Unable to fetch content for '{topic}' after {max_retries} attempts."


def main():
    print("Initializing GPT-2 model...")
    gpt2 = GPT2(
        vocab_size=VOCAB_SIZE,
        max_length=MAX_LENGTH,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        dff=DFF,
        num_layers=NUM_LAYERS,
        dropout_rate=DROPOUT_RATE
    )

    # Build the model with dummy input to enable model.summary()
    dummy_input = tf.constant(np.zeros((1, MAX_LENGTH), dtype=np.int32))
    gpt2(dummy_input)  # Trigger build
    gpt2.summary()

    topic = "Biology"
    print(f"Fetching Wikipedia content for '{topic}'...")

    text = fetch_wiki_text(topic)
    tokenizer = Tokenizer(vocab_size=VOCAB_SIZE).fit_on_text(text)

    input_ids = tokenizer.encode(text[:500])

    print("Generating text with the new model...")
    output_ids = gpt2.generate(input_ids, max_new_tokens=50, temperature=0.8, top_k=40)
    generated_text = tokenizer.decode(output_ids)

    print("=== Generated Output ===")
    print(generated_text[:800])


if __name__ == "__main__":
    main()
