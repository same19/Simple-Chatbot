import tkinter as tk
import tkinter.ttk as ttk
from tkinter import simpledialog, messagebox
import sv_ttk
import torch
import torch.nn
from transformer import TransformerModel
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import spacy
import plotly.io as pio
import kaleido
from PIL import Image, ImageTk
import time

# Load English tokenizer, tagger, parser and NER
tokenizer = spacy.load("en_core_web_sm").tokenizer

# Tkinter implementation by ChatGPT

vocab = torch.load("saves/vocab_may1_WT2_transformer_min25f.pt")
transformer = torch.load("saves/model_transformer_may5_0100am.pt")

# get first layer of the model
embeddings = list(transformer.input_emb.parameters())[0]
embeddings = embeddings.cpu().detach().numpy()

# normalize the embeddings layer
norms = (embeddings ** 2).sum(axis=1) ** (0.5)
norms = np.reshape(norms, (len(norms), 1))
embeddings_norm = embeddings / norms
embeddings_norm.shape
embeddings_df = pd.DataFrame(embeddings)

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

def normalize(emb):
    emb_norm = (emb ** 2).sum() ** (1 / 2)
    return emb / emb_norm

def embed(word):
    return torch.tensor(embeddings[lookup_id(word)])

def analogy(worda, wordA, wordb, n=5, include_inputs = False):
    vocab[worda], vocab[wordA], vocab[wordb]
    emba = embed(worda)
    embA = embed(wordA)
    embb = embed(wordb)

    embB = embA - emba + embb
    embB = normalize(embB)

    embB = np.reshape(embB, (len(embB), 1))
    dists = np.matmul(embeddings_norm, embB).flatten()

    topn = np.argsort(-dists)[:n+3]
    index = 0
    count = 0
    out = []
    while count < n:
        word_id = topn[index]
        if include_inputs or (lookup_token(word_id) not in [worda, wordA, wordb]):
            out.append((lookup_token(word_id), dists[word_id]))
            print("{}: {:.3f}".format(lookup_token(word_id), dists[word_id]))
            count += 1
        index += 1

def closest_word(embedding, n = 1):
    emb = normalize(embedding)

    emb = np.reshape(emb, (len(emb), 1))
    dists = np.matmul(embeddings_norm, emb).flatten()

    topn = np.argsort(-dists)[:n]
    return [lookup_token(top) for top in topn], [dists[top] for top in topn]

def mathify(word):
    return (word, embed(word))
def multiply(word: tuple, factor):
    a = word[1]*factor
    return (a, closest_word(a,3)[0])
def add(worda: tuple, wordb: tuple):
    a = worda[1]+wordb[1]
    return (a, closest_word(a,3)[0])

# # t-SNE transform
# tsne = TSNE(n_components=2)
# embeddings_df_tsne = tsne.fit_transform(embeddings_df)
# embeddings_df_tsne = pd.DataFrame(embeddings_df_tsne)

# embeddings_df_tsne.index = vocab.keys()

# torch.save(embeddings_df_tsne, "saves/plot_emb_transformer_may5_1130pm.pt")
embeddings_df_tsne = torch.load("saves/plot_emb_transformer_may5_1130pm.pt")

def generate_word_plot(highlighted_words):
    highlight = np.array(pd.Series(embeddings_df_tsne.index).isin(highlighted_words))
    
    embeddings_df_temp = embeddings_df_tsne
    embeddings_df_temp['highlight'] = highlight
    embeddings_df_temp = embeddings_df_temp.sort_values(by='highlight', ascending=True)
    highlight = embeddings_df_temp['highlight']

    color = np.where(highlight, "red", "black")
    size = np.where(highlight, 30, 10)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=embeddings_df_temp[0],
            y=embeddings_df_temp[1],
            mode="text",
            text=embeddings_df_temp.index,
            textposition="middle center",
            textfont=dict(color=color, size=size),
        )
    )
    fig.update_traces(marker=dict(size=size))
    f = "plot.png"
    pio.write_image(fig, 'plot.png', engine='kaleido', scale=5.0)
    return f
generate_word_plot(["first", "game", "film", '"', "second"])

# analogy("mother", "woman", "father", include_inputs=False) #man
# print()
# analogy("kingdom", "king", "empire", include_inputs=False) #emperor
# print()
# analogy("2001", "1", "2002", include_inputs=False) #2
# print()
# analogy("2001", "2002", "2005", include_inputs=False) #2006
# print()
# analogy("1", "3", "4", include_inputs=False) #6
# print()
# analogy("bright", "yellow", "dark", include_inputs=False) #brown
# print()
# analogy("bright", "dark", "cold", include_inputs=False) #hot
# print(multiply(mathify("cold"),-1)[1]) #bright

# Get a list of the probability of each word in the dictionary
def get_probabilities(sequence, n=5, include_unk = True):
    with torch.no_grad():
        text = torch.tensor([lookup_id(word) for word in sequence])
        max_seq_length = 32
        if len(text) > max_seq_length:
            text = text[len(text)-max_seq_length:]
        out = torch.softmax(transformer(text, False)[0][-1],0)
        top = np.argsort(-np.array(out))[:(n+1 if n != None else None)]
        if (not include_unk) and lookup_id("<unk>") in top:
            top = list(top)
            top.remove(lookup_id("<unk>"))
        elif n != None:
            top = top[:-1]
        probs = np.array([out[i] for i in top])
        probs /= sum(probs)
        return [lookup_token(i) for i in top], list(probs)

# Get the indexes of the top five probabilities in a list of probabilities
# def get_top_5_probability_indexes(probabilities):
#     largest = heapq.nlargest(5, enumerate(probabilities), key=lambda x: x[1])
#     indices = [index for index, value in largest]
#     return indices

def display_word(word):
    if "@" in word:
        return str(word[1])
    return word

def format_sequence(l):
    s = ""
    count = 0
    quote_count = []
    bind_next = False
    for word in l:
        if bind_next:
            # print('binding', word, 'to', s)
            s += word
            bind_next = False
        elif len(word)>1 and word[0] == "'": #contraction
            s += word
        elif '@' in word:
            s += word[1]
            bind_next = True
        elif word in ['.', ',', '!', ';', '?']: #punctuation
            s += word
        elif word in ["'", '"']: #symmetric quote characters
            if word in quote_count: #already an open quote
                quote_count.remove(word)
                s += word
            else:
                quote_count.append(word)
                s += " " + word
                bind_next = True
        elif word in ["(", "[", "{"]: #open brackets
            s += " " + word
            bind_next = True
        elif word in [")", "]", "}"]: #open brackets
            s += word
        elif count == 0:
            s += word
        elif bind_next:
            s += word
        else:
            s += " " + word
        count = 1
    return s


class ZoomableImageFrame(tk.Frame):
    def __init__(self, master, image_path, *args, **kwargs):
        tk.Frame.__init__(self, master, *args, **kwargs)
        self.canvas = tk.Canvas(self)
        self.canvas.pack(fill=tk.BOTH, expand=tk.YES)

        self.init_size = (500,300)
        self.image = Image.open(image_path).resize(self.init_size)
        self.image_tk = ImageTk.PhotoImage(self.image)
        self.image_id = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image_tk)

        self.canvas.bind("<MouseWheel>", self.zoom)
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move)

        self.start_x = None
        self.start_y = None
        self.zoom_level = 1.0

    def zoom(self, event):
        factor = 1.1 if event.delta > 0 else 0.9
        self.zoom_level *= factor
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        bbox = self.canvas.bbox(tk.ALL)
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        dx = x - bbox[0]
        dy = y - bbox[1]
        new_width = int(self.image.width * self.zoom_level)
        new_height = int(self.image.height * self.zoom_level)
        resized_image = self.image.resize((new_width, new_height))
        self.image_tk = ImageTk.PhotoImage(resized_image)
        self.canvas.itemconfig(self.image_id, image=self.image_tk)
        self.canvas.move(tk.ALL, -int(dx*(factor-1)), -int(dy*(factor-1)))
        # self.canvas.scan_dragto(int(center_x - dx * factor), int(center_y - dy * factor))

    def on_button_press(self, event):
        self.start_x = event.x
        self.start_y = event.y

    def on_move(self, event):
        if self.start_x is not None and self.start_y is not None:
            delta_x = event.x - self.start_x
            delta_y = event.y - self.start_y
            self.canvas.move(tk.ALL, delta_x, delta_y)
            self.start_x = event.x
            self.start_y = event.y


# Create the main application window
class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("LLM Interface")
        self.geometry("1100x650")

        self.sequence = ["The"]
        self.create_widgets()

    def create_widgets(self):
        self.label = ttk.Label(self, text="Select a word:", font=("Comic Sans", 20))  # Setting text color to white
        self.label.pack(pady=(30,10))

        self.word_buttons_frame = ttk.Frame(self)  # Setting background color to dark gray for frame #, bg="#121212"
        self.word_buttons_frame.pack(pady=(20,20))
        

        self.label_seq = tk.Text(self, font=("Arial", 14), width=100, height=3, wrap=tk.WORD, highlightcolor=self.cget("bg"))  # Using Entry widget instead of Label
        self.label_seq.insert(tk.END, format_sequence(self.sequence))
        # self.label_seq.bind("<FocusOut>", self.update_sequence)  # Binding Return key to update sequence
        self.label_seq.pack(pady=20)
        self.label_seq.bind("<FocusIn>", self.on_text_focus_in)
        self.label_seq.bind("<FocusOut>", self.on_text_focus_out)

        self.plot_button = ttk.Button(self, text="Regenerate Plot", command=self.generate_new_plot)
        self.plot_button.pack(pady=(0,20))

        self.plot_frame = ttk.Frame(self, height=400)  # Setting background color to dark gray for frame #, bg="#121212"
        self.plot_frame.pack(pady=(0,0))
        # image_frame = ZoomableImageFrame(self, "plot.png")
        # image_frame.pack()#fill=tk.BOTH, expand=tk.YES)
        self.update_words()

        self.generate_new_plot(first_time=True)

        

    def on_text_focus_in(self, event):
        self.label_seq.config(highlightbackground="white")  # Change highlight color to white when focused

    def on_text_focus_out(self, event):
        self.label_seq.config(highlightbackground=self.cget("bg"))  # Restore default highlight color when focus is lost

    def generate_new_plot(self, first_time = False):
        if first_time:
            self.tk_image = ImageTk.PhotoImage(Image.open("plot.png").resize((485, 350)))
            print("saved to tk_image")
            self.img_label = tk.Label(self.plot_frame, image=self.tk_image)
            print("labeled")
            self.img_label.pack(pady=20)
            print("packed")
        else:
            generate_word_plot(self.words)
            # print("done generating plot")
            tk_image = ImageTk.PhotoImage(Image.open("plot.png").resize((485, 350)))  # Resize image if needed
            # print("saved to tk_image")
            self.img_label.config(image=tk_image)  # Update existing Label widget's image
            self.img_label.image = tk_image  # Keep a reference to avoid garbage collection
            # print("updated image")

    def update_sequence(self, event=None):
        new_sequence = self.label_seq.get("1.0", tk.END).strip()  # Get the text from the entry widget
        self.sequence = [str(w) for w in tokenizer(new_sequence)]  # Split the text into a list of words
        self.update_words()

    def update_words(self):
        words, probs = get_probabilities(self.sequence, 5, False)
        self.words = words
        # Clear previous buttons
        for widget in self.word_buttons_frame.winfo_children():
            widget.destroy()

        # Create a button for each word
        for word,prob in zip(words,probs):
            btn = ttk.Button(self.word_buttons_frame, text=display_word(word), command=lambda w=word: self.add_word(w), width=8+int(16*prob))  # Setting button color to a darker shade of gray #width=8
            btn.pack(side=tk.LEFT, padx=10)

        update_button = ttk.Button(self.word_buttons_frame, text="Reload Choices", command=self.update_sequence)
        update_button.pack(side=tk.LEFT, padx=10)
        # Add Exit button
        exit_btn = ttk.Button(self.word_buttons_frame, text="Exit", command=self.close_app)  # Setting button color to orange #bg="#FF5722",
        exit_btn.pack(side=tk.LEFT, padx=10)

    def set_sequence_text(self, text):
        self.label_seq.delete("1.0", tk.END)  # Delete existing text from Text widget
        self.label_seq.insert(tk.END, text)
        self.label_seq.see(tk.END)
    
    def add_word(self, word):
        self.update_sequence()
        self.sequence.append(word)
        self.set_sequence_text(format_sequence(self.sequence))
        self.update_words()

    def close_app(self):
        self.quit()

if __name__ == "__main__":
    app = Application()
    sv_ttk.set_theme("light")
    app.mainloop()