import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import stanza

# Load NLP pipeline dari Stanza (pastikan sudah di-download di app.py)
nlp = stanza.Pipeline(lang='id')

def sample_word(preds, temperature=1.0):
    """
    Sampling kata berdasarkan probabilitas dan suhu (temperature).
    """
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-10) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_hoax_from_word(seed_word, tokenizer, model, max_seq_len, max_words=10, temperature=0.8):
    """
    Menghasilkan kalimat hoax dari kata awal (seed) menggunakan model LSTM.
    """
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

def extract_spok(text):
    """
    Analisis SPOK dari kalimat menggunakan Stanza (Bahasa Indonesia).
    """
    doc = nlp(text)
    spok = []

    for sent in doc.sentences:
        subj, pred, obj, keterangan = None, None, None, []

        for word in sent.words:
            if word.deprel in ['nsubj', 'nsubj:pass']:
                subj = word.text
            elif word.upos == 'VERB':
                pred = word.text
            elif word.deprel in ['obj', 'iobj']:
                obj = word.text
            elif word.deprel == 'obl':
                keterangan.append(word.text)

        spok.append({
            'kalimat': sent.text,
            'subjek': subj,
            'predikat': pred,
            'objek': obj,
            'keterangan': " ".join(keterangan)
        })

    return spok
