"""
FlashPPI inference pipeline.

Wraps the FlashPPI model (tattabio/flashppi) for proteome-scale PPI prediction.
Follows the three-stage pipeline:
  1. Embed all proteins (encode_protein → CLIP embeddings via forward)
  2. FAISS nearest-neighbor retrieval (top-k candidates)
  3. Residue-level contact map scoring on retrieved pairs
"""

import os
from dataclasses import dataclass

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
    batch_size: int = 16
    max_len: int = 1024
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def load_fasta(fasta_path: str, max_len: int = 1024) -> dict[str, str]:
    """Load FASTA file and return {id: sequence} dict."""
    sequences = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        seq = str(record.seq)[:max_len]
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

    def _tokenize(self, sequence: str) -> dict[str, torch.Tensor]:
        """Tokenize a single protein sequence."""
        tokens = self.tokenizer(
            sequence,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.config.max_len,
        )
        return {k: v.to(self.config.device) for k, v in tokens.items()}

    def _tokenize_batch(self, sequences: list[str]) -> dict[str, torch.Tensor]:
        """Tokenize a batch of protein sequences."""
        tokens = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_len,
        )
        return {k: v.to(self.config.device) for k, v in tokens.items()}

    @torch.no_grad()
    def embed_proteins(
        self, sequences: dict[str, str]
    ) -> tuple[list[str], np.ndarray, list[torch.Tensor], list[torch.Tensor]]:
        """
        Stage 1: Embed all proteins.

        Returns:
            ids: protein identifiers
            clip_embeddings: (N, D) normalized CLIP embeddings for FAISS retrieval
            all_residue_embs: per-protein residue embeddings for contact prediction
            all_masks: per-protein attention masks
        """
        ids = list(sequences.keys())
        seqs = list(sequences.values())
        all_clip = []
        all_residue = []
        all_masks = []

        for i in tqdm(range(0, len(seqs), self.config.batch_size), desc="Embedding"):
            batch_seqs = seqs[i : i + self.config.batch_size]

            for seq in batch_seqs:
                tokens = self._tokenize(seq)
                mask = tokens.get("attention_mask")
                # Single pass: encode_protein gives residue embeddings,
                # then head_q gives CLIP embedding for retrieval
                residue_emb = self.model.encode_protein(tokens["input_ids"], mask)
                clip_emb = self.model.head_q(residue_emb, mask)

                all_residue.append(residue_emb.squeeze(0).cpu())
                all_masks.append(mask.squeeze(0).cpu())
                all_clip.append(clip_emb.squeeze(0).cpu().numpy())

        clip_matrix = np.vstack(all_clip).astype(np.float32)
        # L2 normalize for cosine similarity
        norms = np.linalg.norm(clip_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        clip_matrix = clip_matrix / norms

        return ids, clip_matrix, all_residue, all_masks

    def retrieve_candidates(
        self, clip_embeddings: np.ndarray
    ) -> list[list[tuple[int, float]]]:
        """Stage 2: FAISS nearest-neighbor retrieval."""
        n = len(clip_embeddings)
        k = min(self.config.stage1_top_k, n)
        index = faiss.IndexFlatIP(clip_embeddings.shape[1])
        index.add(clip_embeddings)

        scores, indices = index.search(clip_embeddings, k)
        candidates = []
        for i in range(n):
            pairs = []
            for j_idx in range(k):
                j = int(indices[i, j_idx])
                if j > i:  # avoid self-pairs and duplicates
                    pairs.append((j, float(scores[i, j_idx])))
            candidates.append(pairs)
        return candidates

    @torch.no_grad()
    def score_pair(
        self,
        residue_emb1: torch.Tensor,
        residue_emb2: torch.Tensor,
        mask1: torch.Tensor,
        mask2: torch.Tensor,
    ) -> tuple[float, torch.Tensor]:
        """
        Stage 3: Predict contact map for a protein pair.

        Returns (contact_score, contact_map).
        """
        contact_map, contact_mask = self.model.predict_contacts(
            residue_emb1.unsqueeze(0).to(self.config.device),
            residue_emb2.unsqueeze(0).to(self.config.device),
            mask1.unsqueeze(0).to(self.config.device),
            mask2.unsqueeze(0).to(self.config.device),
        )
        # Apply sigmoid to get probabilities, mask out padding
        cmap = torch.sigmoid(contact_map.squeeze(0)).cpu()
        if contact_mask is not None:
            valid_mask = contact_mask.squeeze(0).cpu().bool()
            masked_cmap = cmap * valid_mask.float()
        else:
            masked_cmap = cmap
        # Contact score = max predicted contact probability
        score = float(masked_cmap.max())
        return score, cmap

    def predict_proteome(
        self,
        fasta_path: str,
        output_path: str | None = None,
        save_contact_maps: bool = False,
    ) -> pd.DataFrame:
        """Run full three-stage pipeline on a proteome FASTA."""
        sequences = load_fasta(fasta_path, self.config.max_len)
        print(f"Loaded {len(sequences)} proteins from {fasta_path}")

        # Stage 1: Embed
        ids, clip_embs, residue_embs, masks = self.embed_proteins(sequences)

        # Stage 2: Retrieve
        print("Retrieving candidates...")
        candidates = self.retrieve_candidates(clip_embs)

        # Stage 3: Score contacts
        results = []
        contact_maps = {}
        total_pairs = sum(len(c) for c in candidates)
        print(f"Scoring {total_pairs} candidate pairs...")

        with tqdm(total=total_pairs, desc="Contact scoring") as pbar:
            for i, pairs in enumerate(candidates):
                for j, retrieval_score in pairs:
                    score, cmap = self.score_pair(
                        residue_embs[i], residue_embs[j],
                        masks[i], masks[j],
                    )
                    if score >= self.config.threshold:
                        results.append({
                            "query_id": ids[i],
                            "match_id": ids[j],
                            "contact_score": round(score, 4),
                        })
                        if save_contact_maps:
                            contact_maps[(ids[i], ids[j])] = cmap
                    pbar.update(1)

        df = pd.DataFrame(results)
        if len(df) > 0:
            # Deduplicate: keep max score per pair
            df = df.sort_values("contact_score", ascending=False)
            df = df.drop_duplicates(subset=["query_id", "match_id"]).reset_index(drop=True)

        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            df.to_csv(output_path, index=False)
            print(f"Saved {len(df)} interactions to {output_path}")

        if save_contact_maps:
            return df, contact_maps
        return df


def run(fasta_path: str, output_path: str, **kwargs) -> pd.DataFrame:
    """Convenience function to run FlashPPI prediction."""
    config = FlashPPIConfig(**kwargs)
    predictor = FlashPPIPredictor(config)
    return predictor.predict_proteome(fasta_path, output_path)
