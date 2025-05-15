import streamlit as st
import tensorflow as tf
import numpy as np
from app import GPT2, Tokenizer, VOCAB_SIZE, MAX_LENGTH, EMBED_DIM, NUM_HEADS, DFF, NUM_LAYERS, DROPOUT_RATE

# Initialize model only once using Streamlit caching
@st.cache_resource
def load_model_and_tokenizer():
    model = GPT2(
        vocab_size=VOCAB_SIZE,
        max_length=MAX_LENGTH,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        dff=DFF,
        num_layers=NUM_LAYERS,
        dropout_rate=DROPOUT_RATE
    )
    dummy_input = tf.constant(np.zeros((1, MAX_LENGTH), dtype=np.int32))
    model(dummy_input)  # Build model

    # You can optionally train or load a pre-trained tokenizer
    tokenizer = Tokenizer(vocab_size=VOCAB_SIZE)
    tokenizer.fit_on_text("basic training text like Wikipedia content")  # Replace with real corpus

    return model, tokenizer

gpt2, tokenizer = load_model_and_tokenizer()

st.title("🧠 GPT-2 Chat with Streamlit")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("You:", key="user_input")

if user_input:
    # Encode the user input
    input_ids = tokenizer.encode(user_input)
    # Generate response
    output_ids = gpt2.generate(input_ids, max_new_tokens=50, temperature=0.8, top_k=40)
    response = tokenizer.decode(output_ids[len(input_ids):])  # Exclude prompt

    # Store chat history
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", response))

# Display chat history
for speaker, message in st.session_state.chat_history:
    st.markdown(f"**{speaker}:** {message}")
