from os import getenv

import streamlit as st
import numpy as np
from transformers import DistilBertForSequenceClassification, AutoTokenizer

@st.cache(allow_output_mutation=True)
def load_model(path = getenv('MODEL_PATH')):
    return DistilBertForSequenceClassification.from_pretrained(path, local_files_only=True)

@st.cache(allow_output_mutation=True)
def load_tokenizer():
    return AutoTokenizer.from_pretrained('distilbert-base-cased')


tokenizer = load_tokenizer()
model = load_model()
categories = np.array([
    '🧬 Biology', 
    '🖥️ Computer Science',
    '💱 Economics',
    '🔋 Electrical Engineering',
    '🪙 Finance',
    '➕ Mathematics',
    '🔬 Physics',
    '🎲 Statistics',
    '🤷🏻 Unknown'
])

def run_model(input_text: str, border: float = 0.95):
    tokens = tokenizer.encode(input_text, truncation=True, padding=True, return_tensors='pt')
    preds = model(tokens)
    [probs] = preds.logits.softmax(dim=-1).tolist()

    idx = np.flip(np.argsort(probs))

    total = 0.
    result = []

    for cat, prob in zip(categories[idx], sorted(probs)):
        result.append((cat, prob))
        total += prob
        if total >= border:
            return result

def pretty_print(results):
    return '\n'.join([f'- {name.ljust(25)}: {prob * 100 :. 2f}%' for name, prob in results])

st.write('''
# Hello, StreamLit!

This is an article topic classifier.
''')

input_text = st.text_area('Input abstract of an article you want to classify:')

if input_text != '':
    result = run_model(input_text)

    st.write(f'''
    ### The article is:
    {result}
    ''')
