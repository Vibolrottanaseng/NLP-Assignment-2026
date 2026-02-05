import json
import re
from typing import List

from flask import Flask, request, render_template, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Tokenizer (same as notebook)
# ----------------------------
TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]")

def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())

# ----------------------------
# Model (same as notebook)
# ----------------------------
class LSTMLM(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.hid_dim = hid_dim

        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(
            emb_dim, hid_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        self.fc = nn.Linear(hid_dim, vocab_size)

    def init_hidden(self, batch_size, device):
        h = torch.zeros(self.num_layers, batch_size, self.hid_dim, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hid_dim, device=device)
        return (h, c)

    def forward(self, x, hidden):
        x = self.emb(x)
        out, hidden = self.lstm(x, hidden)
        logits = self.fc(out)
        return logits, hidden

# ----------------------------
# Load vocab + model
# ----------------------------
app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("../model/vocab_itos.json", "r", encoding="utf-8") as f:
    itos = json.load(f)

stoi = {tok: i for i, tok in enumerate(itos)}

UNK = stoi["<unk>"]
EOS = stoi["<eos>"]

# MUST match training
EMB_DIM = 1024
HID_DIM = 1024
NUM_LAYERS = 2

model = LSTMLM(
    vocab_size=len(itos),
    emb_dim=EMB_DIM,
    hid_dim=HID_DIM,
    num_layers=NUM_LAYERS,
    dropout=0.0
).to(device)

model.load_state_dict(torch.load("../model/language_model.pt", map_location=device))
model.eval()

# ----------------------------
# Text generation
# ----------------------------
@torch.no_grad()
def generate_text(prompt, max_new_tokens, temperature):
    tokens = tokenize(prompt)
    if not tokens:
        tokens = ["the"]

    ids = [stoi.get(t, UNK) for t in tokens]
    hidden = model.init_hidden(1, device)

    x_prompt = torch.tensor([ids], dtype=torch.long, device=device)
    _, hidden = model(x_prompt, hidden)

    current_id = ids[-1]

    for _ in range(max_new_tokens):
        x = torch.tensor([[current_id]], dtype=torch.long, device=device)
        logits, hidden = model(x, hidden)

        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)

        # forbid <unk>
        probs[0, UNK] = 0
        probs = probs / probs.sum()

        next_id = torch.multinomial(probs, 1).item()
        if next_id == EOS:
            break

        ids.append(next_id)
        current_id = next_id

    return " ".join(itos[i] for i in ids)

# ----------------------------
# Routes
# ----------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    prompt = data.get("prompt", "emma was")
    temperature = float(data.get("temperature", 0.8))
    max_new_tokens = int(data.get("max_new_tokens", 50))

    output = generate_text(prompt, max_new_tokens, temperature)
    return jsonify({"result": output})

if __name__ == "__main__":
    app.run(debug=True)
