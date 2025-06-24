import streamlit as st
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

# --- Fungsi Cache ---
@st.cache_resource
def load_hoax_model():
    return load_model("hoax_lstm_model.h5")  # Pastikan nama dan path benar

@st.cache_resource
def load_tokenizer():
    with open("tokenizer_hoax.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer

# --- Fungsi Sampling Kata ---
def sample_word(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-10) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# --- Fungsi Generate Kalimat ---
def generate_hoax_from_word(seed_word, tokenizer, model, max_seq_len, max_words=10, temperature=0.8):
    seed_text = seed_word.lower()
    output_text = seed_text
    for _ in range(max_words):
        token_list = tokenizer.texts_to_sequences([output_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_seq_len - 1, padding='pre')
        preds = model.predict(token_list, verbose=0)[0]
        predicted_index = sample_word(preds, temperature)
        predicted_word = tokenizer.index_word.get(predicted_index, '')
        if predicted_word == '' or predicted_word == '<OOV>':
            break
        output_text += ' ' + predicted_word
    return output_text

# --- UI Streamlit ---
st.set_page_config(page_title="Hoax Generator", layout="centered")
st.title("🧠 Hoax Text Generator")
st.markdown("Masukkan satu kata, dan model akan mengembangkan menjadi kalimat bernuansa hoax (untuk keperluan edukasi).")

try:
    model = load_hoax_model()
    tokenizer = load_tokenizer()
    max_seq_len = model.input_shape[1] + 1
except Exception as e:
    st.error(f"Gagal memuat model/tokenizer. Pastikan file `.h5` dan `tokenizer_hoax.pkl` tersedia.\n\nError: {e}")
    st.stop()

seed_word = st.text_input("📝 Masukkan kata awal:", "")
temperature = st.slider("🎛️ Suhu kreativitas (temperature)", 0.2, 1.5, 0.8, 0.1)

if st.button("🔮 Generate"):
    if not seed_word.strip():
        st.warning("Masukkan setidaknya satu kata.")
    else:
        with st.spinner("Menghasilkan teks hoax..."):
            result = generate_hoax_from_word(seed_word, tokenizer, model, max_seq_len, temperature=temperature)
            st.success("✅ Teks Berhasil Dihasilkan:")
            st.markdown(f"**{result}**")
