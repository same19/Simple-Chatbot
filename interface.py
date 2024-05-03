import heapq
import tkinter as tk
from tkinter import simpledialog, messagebox
import torch
import torch.nn
from transformer import TransformerModel
import numpy as np

# Tkinter implementation by ChatGPT

vocab = torch.load("saves/vocab_may1_WT2_transformer_min25f.pt")
transformer = torch.load("saves/model_transformer_may2_0600pm.pt")

def lookup_id(word, vocab=vocab):
    word = word.lower()
    if word not in vocab:
        return vocab["<unk>"]
    return vocab[word]

def lookup_token(word_id, vocab=vocab):
    for word in vocab:
        if vocab[word] == word_id:
            return word
    return "<unk>"

def embed(word):
    return transformer.input_emb(lookup_id(word))

# Get a list of the probability of each word in the dictionary
def get_probabilities(sequence, n=5, include_unk = True):
    with torch.no_grad():
        text = torch.tensor([lookup_id(word) for word in sequence])
        out = torch.softmax(transformer(text)[0][-1],0)
        top = np.argsort(-np.array(out))[:(n+1 if n != None else None)]
        if (not include_unk) and lookup_id("<unk>") in top:
            top = list(top)
            top.remove(lookup_id("<unk>"))
        elif n != None:
            top = top[:-1]
        return [lookup_token(i) for i in top], [out[i] for i in top]

get_probabilities(["hello"], None, True)
# Get the indexes of the top five probabilities in a list of probabilities
def get_top_5_probability_indexes(probabilities):
    largest = heapq.nlargest(5, enumerate(probabilities), key=lambda x: x[1])
    indices = [index for index, value in largest]
    return indices

# # Return the top five words based on the probabilities list
# def pick_top_5_words(probabilities):
#     words = ["hello", "the", "dog", "happy", "is", "sad"]
#     indexes = get_top_5_probability_indexes(probabilities)
#     return [words[index] for index in indexes]

# Create the main application window
class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("LLM Interface")
        self.geometry("1000x300")

        self.sequence = [""]
        self.create_widgets()

    def create_widgets(self):
        self.label_seq = tk.Label(self, text=" ".join(self.sequence), font=("Arial", 14))
        self.label_seq.pack(pady=20)
        self.label = tk.Label(self, text="Select a word:", font=("Arial", 14))
        self.label.pack(pady=20)

        self.word_buttons_frame = tk.Frame(self)
        self.word_buttons_frame.pack(pady=20)

        self.update_words()

    def update_words(self):
        words, prob = get_probabilities(self.sequence, 5, False)

        # Clear previous buttons
        for widget in self.word_buttons_frame.winfo_children():
            widget.destroy()

        # Create a button for each word
        for word in words:
            btn = tk.Button(self.word_buttons_frame, text=word, command=lambda w=word: self.add_word(w))
            btn.pack(side=tk.LEFT, padx=10)

        # Add Exit button
        exit_btn = tk.Button(self.word_buttons_frame, text="EXIT", command=self.close_app)
        exit_btn.pack(side=tk.LEFT, padx=10)

    def add_word(self, word):
        self.sequence.append(word)
        self.label_seq.config(text=" ".join(self.sequence))
        self.update_words()

    def close_app(self):
        self.quit()

if __name__ == "__main__":
    app = Application()
    app.mainloop()
