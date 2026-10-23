"""Pure-Python WordPiece tokenizer + onnxruntime runner for sentence-transformers/all-MiniLM-L6-v2.

Equivalent to SentenceTransformer('all-MiniLM-L6-v2').encode(texts, normalize_embeddings=True):
BERT uncased tokenisation -> transformer -> attention-masked mean pooling -> L2 normalisation.
"""
import json, unicodedata
import numpy as np
import onnxruntime as ort


class WordPieceTokenizer:
    def __init__(self, tokenizer_json):
        cfg = json.load(open(tokenizer_json))
        self.vocab = cfg["model"]["vocab"]
        self.unk = self.vocab[cfg["model"]["unk_token"]]
        self.cls, self.sep = self.vocab["[CLS]"], self.vocab["[SEP]"]
        self.max_chars = cfg["model"].get("max_input_chars_per_word", 100)

    @staticmethod
    def _is_punct(ch):
        cp = ord(ch)
        if 33 <= cp <= 47 or 58 <= cp <= 64 or 91 <= cp <= 96 or 123 <= cp <= 126:
            return True
        return unicodedata.category(ch).startswith("P")

    @staticmethod
    def _is_cjk(cp):
        return (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or 0x20000 <= cp <= 0x2A6DF
                or 0x2A700 <= cp <= 0x2B73F or 0x2B740 <= cp <= 0x2B81F or 0x2B820 <= cp <= 0x2CEAF
                or 0xF900 <= cp <= 0xFAFF or 0x2F800 <= cp <= 0x2FA1F)

    def _basic(self, text):
        out = []
        for ch in text:  # clean text + CJK spacing
            cp = ord(ch)
            if cp == 0 or cp == 0xFFFD or (unicodedata.category(ch).startswith("C") and ch not in "\t\n\r"):
                continue
            if ch in " \t\n\r" or unicodedata.category(ch) == "Zs":
                out.append(" ")
            elif self._is_cjk(cp):
                out.append(f" {ch} ")
            else:
                out.append(ch)
        text = "".join(out).lower()
        text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
        tokens = []
        for word in text.split():
            cur = ""
            for ch in word:
                if self._is_punct(ch):
                    if cur:
                        tokens.append(cur); cur = ""
                    tokens.append(ch)
                else:
                    cur += ch
            if cur:
                tokens.append(cur)
        return tokens

    def _wordpiece(self, word):
        if len(word) > self.max_chars:
            return [self.unk]
        ids, start = [], 0
        while start < len(word):
            end, found = len(word), None
            while start < end:
                piece = word[start:end] if start == 0 else "##" + word[start:end]
                if piece in self.vocab:
                    found = self.vocab[piece]; break
                end -= 1
            if found is None:
                return [self.unk]
            ids.append(found); start = end
        return ids

    def encode(self, text, max_len=256):
        ids = [i for w in self._basic(text) for i in self._wordpiece(w)]
        return [self.cls] + ids[: max_len - 2] + [self.sep]


class MiniLMEmbedder:
    def __init__(self, model_path, tokenizer_path, max_len=256, threads=None):
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
        self.tok = WordPieceTokenizer(tokenizer_path)
        self.max_len = max_len

    def encode(self, texts, batch_size=64):
        enc = [self.tok.encode(t, self.max_len) for t in texts]
        order = np.argsort([len(e) for e in enc])  # length-bucketed batches for speed
        out = np.zeros((len(texts), 384), dtype=np.float32)
        for b in range(0, len(order), batch_size):
            idx = order[b:b + batch_size]
            L = max(len(enc[i]) for i in idx)
            ids = np.zeros((len(idx), L), dtype=np.int64); mask = np.zeros_like(ids)
            for r, i in enumerate(idx):
                ids[r, :len(enc[i])] = enc[i]; mask[r, :len(enc[i])] = 1
            h = self.sess.run(None, {"input_ids": ids, "attention_mask": mask,
                                     "token_type_ids": np.zeros_like(ids)})[0]
            m = mask[..., None].astype(np.float32)
            v = (h * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
            out[idx] = v / np.linalg.norm(v, axis=1, keepdims=True)
        return out
