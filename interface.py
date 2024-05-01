import heapq
import tkinter as tk
from tkinter import simpledialog, messagebox

# Tkinter implementation by ChatGPT

# Get a list of the probability of each word in the dictionary
def get_probabilities(sequence):
    return [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

# Get the indexes of the top five probabilities in a list of probabilities
def get_top_5_probability_indexes(probabilities):
    largest = heapq.nlargest(5, enumerate(probabilities), key=lambda x: x[1])
    indices = [index for index, value in largest]
    return indices

# Return the top five words based on the probabilities list
def pick_top_5_words(probabilities):
    words = ["hello", "the", "dog", "happy", "is", "sad"]
    indexes = get_top_5_probability_indexes(probabilities)
    return [words[index] for index in indexes]

# Create the main application window
class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("LLM Interface")
        self.geometry("400x300")

        self.sequence = "Start"
        self.create_widgets()

    def create_widgets(self):
        self.label = tk.Label(self, text="Select a word:", font=("Arial", 14))
        self.label.pack(pady=20)

        self.word_buttons_frame = tk.Frame(self)
        self.word_buttons_frame.pack(pady=20)

        self.update_words()

    def update_words(self):
        probabilities = get_probabilities(self.sequence)
        words = pick_top_5_words(probabilities)

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
        self.sequence += " " + word
        self.update_words()

    def close_app(self):
        self.quit()

if __name__ == "__main__":
    app = Application()
    app.mainloop()
