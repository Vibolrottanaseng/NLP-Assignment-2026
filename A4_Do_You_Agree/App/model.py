import torch
import torch.nn as nn
import torch.nn.functional as F

############################################
# YOUR CUSTOM COMPONENTS (MATCH CHECKPOINT)
############################################

class Embedding(nn.Module):
    def __init__(self, vocab_size, max_len, n_segments, d_model):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.seg_embed = nn.Embedding(n_segments, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, seg):
        pos = torch.arange(x.size(1)).unsqueeze(0).expand_as(x).to(x.device)
        out = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.norm(out)


class DummyEncoderLayer(nn.Module):
    # Placeholder — weights will load from checkpoint
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        return x, None


class BERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = Embedding(43992,1000,2,768)
        self.layers = nn.ModuleList(
            [DummyEncoderLayer(768) for _ in range(12)]
        )

    def encode(self, input_ids, segment_ids):
        x = self.embedding(input_ids, segment_ids)
        return x


class SBERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BERT()
        self.classifier = nn.Linear(768*3,3)

    def sentence_embedding(self, sent_ids):
        seg = torch.zeros_like(sent_ids)
        H = self.bert.encode(sent_ids, seg)
        return H[:,0,:]

    def forward(self, prem_ids, hypo_ids):
        u = self.sentence_embedding(prem_ids)
        v = self.sentence_embedding(hypo_ids)
        feats = torch.cat([u,v,torch.abs(u-v)],dim=1)
        logits = self.classifier(feats)
        return logits,u,v

############################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SBERT()

state = torch.load("../sbert_model.pth", map_location=device)

model.load_state_dict(state, strict=False)

model.to(device)
model.eval()

labels = ["entailment","neutral","contradiction"]

############################################

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def predict(premise,hypothesis):

    prem = tokenizer(premise,return_tensors="pt",padding=True,truncation=True,max_length=128)
    hypo = tokenizer(hypothesis,return_tensors="pt",padding=True,truncation=True,max_length=128)

    prem_ids = prem["input_ids"].to(device)
    hypo_ids = hypo["input_ids"].to(device)

    with torch.no_grad():
        logits,_,_ = model(prem_ids,hypo_ids)

    probs = torch.softmax(logits,dim=1)
    pred = torch.argmax(probs,dim=1).item()

    return labels[pred], probs[0][pred].item()