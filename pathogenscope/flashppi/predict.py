"""
FlashPPI inference pipeline.

Wraps the FlashPPI model (tattabio/flashppi) for proteome-scale PPI prediction.
Follows the three-stage pipeline:
  1. Embed all proteins into shared latent space
  2. FAISS nearest-neighbor retrieval (top-k candidates)
  3. Residue-level contact map scoring on retrieved pairs
"""

import os
from dataclasses import dataclass, field

import faiss
import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


@dataclass
class FlashPPIConfig:
    model_name: str = "tattabio/flashppi"
    stage1_top_k: int = 100
    threshold: float = 0.5
    batch_size: int = 64
    max_len: int = 1024
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def load_fasta(fasta_path: str) -> dict[str, str]:
    """Load FASTA file and return {id: sequence} dict."""
    sequences = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        seq = str(record.seq)[:1024]  # truncate to max_len
        sequences[record.id] = seq
    return sequences


class FlashPPIPredictor:
    """Three-stage FlashPPI inference pipeline."""

    def __init__(self, config: FlashPPIConfig | None = None):
        self.config = config or FlashPPIConfig()
        print(f"Loading FlashPPI model on {self.config.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            self.config.model_name, trust_remote_code=True
        ).to(self.config.device)
        self.model.eval()

    @torch.no_grad()
    def _embed_batch(self, sequences: list[str]) -> tuple[np.ndarray, list[torch.Tensor]]:
        """Embed a batch of sequences. Returns (pooled_embeddings, residue_embeddings)."""
        tokens = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_len,
        ).to(self.config.device)

        outputs = self.model(**tokens, output_hidden_states=True)

        # Get pooled embeddings for retrieval
        if hasattr(outputs, "query_embeddings"):
            pooled = outputs.query_embeddings.cpu().numpy()
        else:
            # Fallback: mean pool last hidden state
            hidden = outputs.last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1)
            pooled = pooled.cpu().numpy()

        # Get residue-level embeddings for contact prediction
        residue_embs = []
        if hasattr(outputs, "residue_embeddings"):
            for i in range(len(sequences)):
                seq_len = len(sequences[i])
                residue_embs.append(outputs.residue_embeddings[i, :seq_len].cpu())
        else:
            hidden = outputs.last_hidden_state
            for i in range(len(sequences)):
                seq_len = len(sequences[i])
                residue_embs.append(hidden[i, :seq_len].cpu())

        return pooled, residue_embs

    def embed_proteome(
        self, sequences: dict[str, str]
    ) -> tuple[list[str], np.ndarray, list[torch.Tensor]]:
        """Stage 1: Embed all proteins."""
        ids = list(sequences.keys())
        seqs = list(sequences.values())
        all_pooled = []
        all_residue = []

        for i in tqdm(range(0, len(seqs), self.config.batch_size), desc="Embedding"):
            batch = seqs[i : i + self.config.batch_size]
            pooled, residue = self._embed_batch(batch)
            all_pooled.append(pooled)
            all_residue.extend(residue)

        pooled_matrix = np.vstack(all_pooled).astype(np.float32)
        # L2 normalize for cosine similarity
        norms = np.linalg.norm(pooled_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        pooled_matrix = pooled_matrix / norms

        return ids, pooled_matrix, all_residue

    def retrieve_candidates(
        self, pooled: np.ndarray
    ) -> list[list[tuple[int, float]]]:
        """Stage 2: FAISS nearest-neighbor retrieval."""
        k = min(self.config.stage1_top_k, len(pooled))
        index = faiss.IndexFlatIP(pooled.shape[1])
        index.add(pooled)

        scores, indices = index.search(pooled, k)
        candidates = []
        for i in range(len(pooled)):
            pairs = []
            for j_idx in range(k):
                j = int(indices[i, j_idx])
                if j > i:  # avoid self and duplicates
                    pairs.append((j, float(scores[i, j_idx])))
            candidates.append(pairs)
        return candidates

    @torch.no_grad()
    def score_contact(
        self, res_emb_a: torch.Tensor, res_emb_b: torch.Tensor
    ) -> float:
        """Stage 3: Predict contact map and return max contact score."""
        if hasattr(self.model, "predict_contacts"):
            contact_map = self.model.predict_contacts(
                res_emb_a.unsqueeze(0).to(self.config.device),
                res_emb_b.unsqueeze(0).to(self.config.device),
            )
            contact_map = torch.sigmoid(contact_map).cpu()
        else:
            # Fallback: dot product attention as contact proxy
            a = res_emb_a.float()
            b = res_emb_b.float()
            contact_map = torch.sigmoid(a @ b.T)

        return float(contact_map.max())

    def predict_proteome(
        self,
        fasta_path: str,
        output_path: str | None = None,
    ) -> pd.DataFrame:
        """Run full three-stage pipeline on a proteome FASTA."""
        # Load
        sequences = load_fasta(fasta_path)
        print(f"Loaded {len(sequences)} proteins from {fasta_path}")

        # Stage 1: Embed
        ids, pooled, residue_embs = self.embed_proteome(sequences)

        # Stage 2: Retrieve
        print("Retrieving candidates...")
        candidates = self.retrieve_candidates(pooled)

        # Stage 3: Score contacts
        results = []
        total_pairs = sum(len(c) for c in candidates)
        print(f"Scoring {total_pairs} candidate pairs...")

        with tqdm(total=total_pairs, desc="Contact scoring") as pbar:
            for i, pairs in enumerate(candidates):
                for j, retrieval_score in pairs:
                    contact_score = self.score_contact(
                        residue_embs[i], residue_embs[j]
                    )
                    if contact_score >= self.config.threshold:
                        results.append(
                            {
                                "query_id": ids[i],
                                "match_id": ids[j],
                                "contact_score": round(contact_score, 4),
                            }
                        )
                    pbar.update(1)

        df = pd.DataFrame(results)
        if len(df) > 0:
            df = df.sort_values("contact_score", ascending=False).reset_index(drop=True)

        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            df.to_csv(output_path, index=False)
            print(f"Saved {len(df)} interactions to {output_path}")

        return df


def run(fasta_path: str, output_path: str, **kwargs) -> pd.DataFrame:
    """Convenience function to run FlashPPI prediction."""
    config = FlashPPIConfig(**kwargs)
    predictor = FlashPPIPredictor(config)
    return predictor.predict_proteome(fasta_path, output_path)
