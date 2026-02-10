from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from flask import Flask, jsonify, render_template, request

# -----------------------------
# Flask app
# -----------------------------
app = Flask(__name__)

# -----------------------------
# Device + special token indices (from your notebook)
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3

# -----------------------------
# Minimal vocab helper (loads itos list from json)
# -----------------------------
class SimpleVocab:
    def __init__(self, itos: List[str], default_index: int = 0):
        self.itos = itos
        self.stoi = {tok: i for i, tok in enumerate(itos)}
        self.default_index = default_index

    def __len__(self) -> int:
        return len(self.itos)

    def __getitem__(self, token: str) -> int:
        return self.stoi.get(token, self.default_index)

    def lookup_tokens(self, ids: List[int]) -> List[str]:
        return [self.itos[i] for i in ids]

def load_vocab_json(path: str) -> SimpleVocab:
    with open(path, "r", encoding="utf-8") as f:
        itos = json.load(f)
    return SimpleVocab(itos=itos, default_index=UNK_IDX)

# -------------------------------------------
# Model (Transformer with Additive Attention)
# --------------------------------------------
class MultiHeadAdditiveAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        assert hid_dim % n_heads == 0
        self.hid_dim  = hid_dim
        self.n_heads  = n_heads
        self.head_dim = hid_dim // n_heads
        self.device   = device

        self.W1 = nn.Linear(self.head_dim, self.head_dim, bias=False)  # for K (h_i)
        self.W2 = nn.Linear(self.head_dim, self.head_dim, bias=False)  # for Q (s_t)
        self.v  = nn.Linear(self.head_dim, 1, bias=False)

        self.fc_o = nn.Linear(hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None, q_chunk_size=32):
        # query = [batch size, query len, hid dim]
        # key   = [batch size, key len, hid dim]
        # value = [batch size, value len, hid dim]
        B, q_len, _ = query.shape
        k_len = key.shape[1]

        Q = query.view(B, q_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = key  .view(B, k_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = value.view(B, k_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        K_proj = self.W1(K)

        scores_chunks = []
        for start in range(0, q_len, q_chunk_size):
            end = min(start + q_chunk_size, q_len)
            Q_chunk = Q[:, :, start:end, :]
            Q_proj = self.W2(Q_chunk)

            tmp = torch.tanh(K_proj.unsqueeze(2) + Q_proj.unsqueeze(3))
            scores = self.v(tmp).squeeze(-1)  # [B, heads, q_chunk, k_len]
            scores_chunks.append(scores)

        scores = torch.cat(scores_chunks, dim=2)  # [B, heads, q_len, k_len]

        if mask is not None:
            # mask expected: [B, 1, 1, k_len] for src mask OR [B, 1, q_len, q_len] for trg mask
            scores = scores.masked_fill(mask == 0, -1e10)

        attn = torch.softmax(scores, dim=-1)  # [B, heads, q_len, k_len]
        x = torch.matmul(self.dropout(attn), V)  # [B, heads, q_len, head_dim]
        x = x.permute(0, 2, 1, 3).contiguous().view(B, q_len, self.hid_dim)  # [B, q_len, hid_dim]
        x = self.fc_o(x)
        return x, attn

class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(hid_dim, pf_dim)
        self.fc2 = nn.Linear(pf_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm        = nn.LayerNorm(hid_dim)
        self.self_attention       = MultiHeadAdditiveAttentionLayer(hid_dim, n_heads, dropout, device)
        self.feedforward          = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout              = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        _src, _ = self.self_attention(src, src, src, src_mask)
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        _src = self.feedforward(src)
        src = self.ff_layer_norm(src + self.dropout(_src))
        return src

class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device, max_length=512):
        super().__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        self.layers = nn.ModuleList(
            [EncoderLayer(hid_dim, n_heads, pf_dim, dropout, device) for _ in range(n_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(self.device)

    def forward(self, src, src_mask):
        batch_size, src_len = src.shape
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
        for layer in self.layers:
            src = layer(src, src_mask)
        return src

class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm  = nn.LayerNorm(hid_dim)
        self.ff_layer_norm        = nn.LayerNorm(hid_dim)

        self.self_attention    = MultiHeadAdditiveAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAdditiveAttentionLayer(hid_dim, n_heads, dropout, device)
        self.feedforward       = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout           = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))

        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))

        _trg = self.feedforward(trg)
        trg = self.ff_layer_norm(trg + self.dropout(_trg))

        return trg, attention

class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device, max_length=512):
        super().__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        self.layers = nn.ModuleList(
            [DecoderLayer(hid_dim, n_heads, pf_dim, dropout, device) for _ in range(n_layers)]
        )
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        batch_size, trg_len = trg.shape
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
        attention = None
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        output = self.fc_out(trg)
        return output, attention

class Seq2SeqTransformer(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        # [B, 1, 1, S]
        return (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

    def make_trg_mask(self, trg):
        # trg_pad_mask: [B, 1, 1, T]
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()
        # result: [B, 1, T, T]
        return trg_pad_mask & trg_sub_mask

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output, attention

# -----------------------------
# Build model exactly like notebook hyperparams
# -----------------------------
def build_model(input_dim: int, output_dim: int, device: str):
    hid_dim = 256
    enc_layers = 3
    dec_layers = 3
    enc_heads = 8
    dec_heads = 8
    enc_pf_dim = 512
    dec_pf_dim = 512
    enc_dropout = 0.1
    dec_dropout = 0.1

    enc = Encoder(input_dim, hid_dim, enc_layers, enc_heads, enc_pf_dim, enc_dropout, device)
    dec = Decoder(output_dim, hid_dim, dec_layers, dec_heads, dec_pf_dim, dec_dropout, device)

    model = Seq2SeqTransformer(enc, dec, PAD_IDX, PAD_IDX, device).to(device)
    return model

# -----------------------------
# Simple tokenization + numericalization
# (Replace tokenize() with your notebook tokenizers if you want.)
# -----------------------------
def tokenize(text: str) -> List[str]:
    return text.strip().split()

def numericalize(tokens: List[str], vocab: SimpleVocab, max_len: int = 128) -> List[int]:
    ids = [vocab[t] for t in tokens][: max_len - 2]
    return [SOS_IDX] + ids + [EOS_IDX]

def ids_to_text(ids: List[int], vocab: SimpleVocab) -> str:
    cleaned = [i for i in ids if i not in (SOS_IDX, EOS_IDX, PAD_IDX)]
    toks = vocab.lookup_tokens(cleaned)
    return " ".join(toks)

@torch.no_grad()
def greedy_decode(model: Seq2SeqTransformer, src_ids: List[int], max_new_tokens: int = 80):
    model.eval()
    src = torch.tensor(src_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)  # [1, S]
    trg = torch.tensor([[SOS_IDX]], dtype=torch.long, device=DEVICE)           # [1, 1]

    attn_steps: List[List[float]] = []

    for _ in range(max_new_tokens):
        out, attn = model(src, trg)  # out: [1, T, V], attn: [1, heads, T, S]
        next_id = int(out[:, -1, :].argmax(dim=-1).item())

        # average over heads for last step attention => [S]
        if attn is not None:
            attn_last = attn[0, :, -1, :].mean(dim=0).detach().cpu().tolist()
            attn_steps.append(attn_last)

        trg = torch.cat([trg, torch.tensor([[next_id]], device=DEVICE)], dim=1)
        if next_id == EOS_IDX:
            break

    out_ids = trg.squeeze(0).tolist()
    return out_ids, attn_steps

# -----------------------------
# Load assets: vocab + weights
# - src_vocab.json
# - trg_vocab.json
# - additive_attention_model.pt
# -----------------------------
ASSET_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_VOCAB_PATH = os.path.join(ASSET_DIR, "src_vocab.json")
TRG_VOCAB_PATH = os.path.join(ASSET_DIR, "trg_vocab.json")
WEIGHTS_PATH   = os.path.join(ASSET_DIR, "additive_attention_model.pt")

_src_vocab: Optional[SimpleVocab] = None
_trg_vocab: Optional[SimpleVocab] = None
_model: Optional[Seq2SeqTransformer] = None
_model_error: Optional[str] = None

def load_assets():
    global _src_vocab, _trg_vocab, _model, _model_error
    try:
        if not os.path.exists(SRC_VOCAB_PATH):
            raise FileNotFoundError("Missing src_vocab.json")
        if not os.path.exists(TRG_VOCAB_PATH):
            raise FileNotFoundError("Missing trg_vocab.json")
        if not os.path.exists(WEIGHTS_PATH):
            raise FileNotFoundError("Missing additive_attention_model.pt")

        _src_vocab = load_vocab_json(SRC_VOCAB_PATH)
        _trg_vocab = load_vocab_json(TRG_VOCAB_PATH)

        _model = build_model(len(_src_vocab), len(_trg_vocab), DEVICE)
        state = torch.load(WEIGHTS_PATH, map_location=DEVICE)
        _model.load_state_dict(state)
        _model.eval()
        _model_error = None
        print("✅ Model assets loaded.")
    except Exception as e:
        _model = None
        _src_vocab = None
        _trg_vocab = None
        _model_error = str(e)
        print(f"⚠️ Model not loaded: {_model_error}")

load_assets()

def translate(text: str) -> Tuple[str, List[str], List[List[float]]]:
    if _model is None or _src_vocab is None or _trg_vocab is None:
        raise RuntimeError(_model_error or "Model is not loaded")

    tokens = tokenize(text)
    src_ids = numericalize(tokens, _src_vocab, max_len=128)
    out_ids, attn_steps = greedy_decode(_model, src_ids, max_new_tokens=80)
    out_text = ids_to_text(out_ids, _trg_vocab)
    return out_text, tokens, attn_steps

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def home():
    return render_template("index.html")

@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"ok": False, "error": "Please provide non-empty text."}), 400

    try:
        output, src_tokens, attention = translate(text)
        return jsonify({
            "ok": True,
            "output": output,
            "src_tokens": src_tokens,
            "attention": attention,  # steps x src_len
        })
    except Exception as e:
        return jsonify({
            "ok": False,
            "error": str(e),
        }), 500

if __name__ == "__main__":
    # Tip: set FLASK_ENV=development for auto-reload, or just run python app.py
    app.run(host="127.0.0.1", port=5000, debug=True)
