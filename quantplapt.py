import torch
import re
import onnxruntime
import numpy as np
from typing import List, Dict, Union
from diskcache import Cache
from tqdm import tqdm
from contextlib import contextmanager, nullcontext
from transformers import BertTokenizer, RobertaTokenizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import matplotlib.pyplot as plt
import torch.nn.functional as F


class ONNXEncoder:
    def __init__(self, model_path: str):
        self.session = onnxruntime.InferenceSession(model_path)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_name = self.session.get_outputs()[0].name

    def encode(self, input_ids: np.ndarray, attention_mask: np.ndarray, return_unpooled: bool = False) -> np.ndarray:
        inputs = {
            self.input_names[0]: input_ids,
            self.input_names[1]: attention_mask
        }
        # Run the ONNX model
        output = self.session.run([self.output_name], inputs)[0]

        if return_unpooled:
            return output.astype(np.float32)
        
        # Check output shape and apply pooling if necessary
        if len(output.shape) == 3:  # [batch_size, seq_length, hidden_size]
            # Mean pooling over sequence dimension (axis=1)
            pooled = output.mean(axis=1)  # [batch_size, hidden_size]
        elif len(output.shape) == 2:  # [batch_size, hidden_size]
            pooled = output
        else:
            raise ValueError(f"Unexpected ONNX model output shape: {output.shape}")
        
        return pooled.astype(np.float32)  # Ensure float32 for PyTorch compatibility


class PredictionModule:
    def __init__(self, model_path: str = "models/affinity_predictor.onnx"):
        self.session = onnxruntime.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.mean = 6.51286529169358
        self.scale = 1.5614094578916633

    def convert_to_affinity(self, normalized: float) -> Dict[str, float]:
        neg_log10_affinity_M = float((normalized * self.scale) + self.mean)
        affinity_uM = float((10**6) * (10**(-neg_log10_affinity_M)))
        return {
            "neg_log10_affinity_M": neg_log10_affinity_M,
            "affinity_uM": affinity_uM
        }

    def predict(self, batch_data: np.ndarray) -> List[Dict[str, float]]:
        affinities = []
        for feature in batch_data:
            affinity_normalized = self.session.run(
                None, {self.input_name: [feature], 'TrainingMode': np.array(False)}
            )[0][0][0]
            affinities.append(self.convert_to_affinity(affinity_normalized))
        return affinities


class Plapt:
    def __init__(
        self,
        prediction_module_path: str = "models/affinity_predictor.onnx",
        prot_onnx_path: str = "models/protbert-quant.onnx",
        chem_onnx_path: str = "models/chemberta-dynamic.onnx",
        device: str = 'cuda',
        cache_dir: str = './embedding_cache',
        use_tqdm: bool = False
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.use_tqdm = use_tqdm

        self.prot_tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        self.prot_encoder = ONNXEncoder(prot_onnx_path)

        self.mol_tokenizer = RobertaTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
        self.mol_encoder = ONNXEncoder(chem_onnx_path)

        self.prediction_module = PredictionModule(prediction_module_path)
        self.cache = Cache(cache_dir)

    @contextmanager
    def progress_bar(self, total: int, desc: str):
        if self.use_tqdm:
            with tqdm(total=total, desc=desc) as pbar:
                yield pbar
        else:
            yield nullcontext()

    @staticmethod
    def preprocess_sequence(seq: str) -> str:
        return " ".join(re.sub(r"[UZOB]", "X", seq))

    def tokenize_molecule(self, mol_smiles: Union[str, List[str]]) -> Dict[str, np.ndarray]:
        tokens = self.mol_tokenizer(
            mol_smiles,
            padding=True,
            max_length=278,
            truncation=True,
            return_tensors='np'
        )
        return {'input_ids': tokens['input_ids'], 'attention_mask': tokens['attention_mask']}

    def tokenize_protein(self, prot_seq: Union[str, List[str]]) -> Dict[str, np.ndarray]:
        preprocessed = [self.preprocess_sequence(seq) if isinstance(seq, str) else self.preprocess_sequence(seq[0]) for seq in prot_seq]
        tokens = self.prot_tokenizer(
            preprocessed,
            padding=True,
            max_length=3200,
            truncation=True,
            return_tensors='np'
        )
        return {'input_ids': tokens['input_ids'], 'attention_mask': tokens['attention_mask']}

    def encode_molecules(self, mol_smiles: List[str], batch_size: int) -> torch.Tensor:
        embeddings = []
        with self.progress_bar(len(mol_smiles), "Encoding molecules") as pbar:
            for batch in self.make_batches(mol_smiles, batch_size):
                cached_embeddings = [self.cache.get(smiles) for smiles in batch]
                uncached_indices = [i for i, emb in enumerate(cached_embeddings) if emb is None]

                if uncached_indices:
                    uncached_smiles = [batch[i] for i in uncached_indices]
                    tokens = self.tokenize_molecule(uncached_smiles)
                    new_embeddings = self.mol_encoder.encode(tokens['input_ids'], tokens['attention_mask'])
                    for i, emb in zip(uncached_indices, new_embeddings):
                        cached_embeddings[i] = torch.tensor(emb)
                        self.cache[batch[i]] = torch.tensor(emb)

                embeddings.extend(cached_embeddings)
                if self.use_tqdm:
                    pbar.update(len(batch))

        return torch.stack(embeddings).to(self.device)

    def encode_proteins(self, prot_seqs: List[str], batch_size: int) -> torch.Tensor:
        embeddings = []
        with self.progress_bar(len(prot_seqs), "Encoding proteins") as pbar:
            for batch in self.make_batches(prot_seqs, batch_size):
                cached_embeddings = [self.cache.get(seq) for seq in batch]
                uncached_indices = [i for i, emb in enumerate(cached_embeddings) if emb is None]

                if uncached_indices:
                    uncached_seqs = [batch[i] for i in uncached_indices]
                    tokens = self.tokenize_protein(uncached_seqs)
                    new_embeddings = self.prot_encoder.encode(tokens['input_ids'], tokens['attention_mask'])
                    for i, emb in zip(uncached_indices, new_embeddings):
                        cached_embeddings[i] = torch.tensor(emb)
                        self.cache[batch[i]] = torch.tensor(emb)

                embeddings.extend(cached_embeddings)
                if self.use_tqdm:
                    pbar.update(len(batch))

        return torch.stack(embeddings).to(self.device)

    @staticmethod
    def make_batches(iterable: List, n: int = 1):
        length = len(iterable)
        for ndx in range(0, length, n):
            yield iterable[ndx:min(ndx + n, length)]

    def predict_affinity(self, prot_seqs: List[str], mol_smiles: List[str], prot_batch_size: int = 2, mol_batch_size: int = 16, affinity_batch_size: int = 128) -> List[Dict[str, float]]:
        if len(prot_seqs) != len(mol_smiles):
            raise ValueError("The number of proteins and molecules must be the same.")

        prot_encodings = self.encode_proteins(prot_seqs, prot_batch_size)
        mol_encodings = self.encode_molecules(mol_smiles, mol_batch_size)

        affinities = []
        with self.progress_bar(len(prot_seqs), "Predicting affinities") as pbar:
            for batch in self.make_batches(range(len(prot_seqs)), affinity_batch_size):
                prot_batch = prot_encodings[batch]
                mol_batch = mol_encodings[batch]
                features = torch.cat((prot_batch, mol_batch), dim=1).cpu().numpy()
                batch_affinities = self.prediction_module.predict(features)
                affinities.extend(batch_affinities)
                if self.use_tqdm:
                    pbar.update(len(batch))

        return affinities

    def score_candidates(self, target_protein: str, mol_smiles: List[str], mol_batch_size: int = 16, affinity_batch_size: int = 128) -> List[Dict[str, float]]:
        target_encoding = self.encode_proteins([target_protein], batch_size=1).mean(axis=1)
        mol_encodings = self.encode_molecules(mol_smiles, mol_batch_size).mean(axis=1)
        print("target_encoding shape:", target_encoding.shape)
        print("mol_encodings shape:", mol_encodings.shape)
        affinities = []
        with self.progress_bar(len(mol_smiles), "Scoring candidates") as pbar:
            for batch in self.make_batches(range(len(mol_smiles)), affinity_batch_size):
                mol_batch = mol_encodings[batch]
                repeated_target = target_encoding.repeat(len(batch), 1)
                features = torch.cat((repeated_target, mol_batch), dim=1).cpu().numpy()
                batch_affinities = self.prediction_module.predict(features)
                affinities.extend(batch_affinities)
                if self.use_tqdm:
                    pbar.update(len(batch))

        return affinities


    def compute_protein_pseudo_contact_map(self, prot_seq: str, similarity_threshold: float = 0.3, visualize: bool = False) -> np.ndarray:
        """
        Compute a pseudo-contact map for a protein using residue embeddings from ProtBERT.
        Returns a binary matrix [seq_length, seq_length] where 1 indicates high similarity.
        """
        # Tokenize protein sequence
        preprocessed_seq = self.preprocess_sequence(prot_seq)
        tokens = self.prot_tokenizer(
            [preprocessed_seq], padding=True, max_length=3200, truncation=True, return_tensors='np'
        )

        input_ids = tokens['input_ids']
        attention_mask = tokens['attention_mask']

        # Get residue-level embeddings (unpooled)
        residue_embeddings = self.prot_encoder.encode(
            input_ids, attention_mask, return_unpooled=True
        )[0]  # [seq_length, hidden_size]

        # Remove padding tokens using attention mask
        mask = attention_mask[0].astype(bool)
        residue_embeddings = residue_embeddings[mask]

        # Normalize residue embeddings
        residue_embeddings = F.normalize(torch.tensor(residue_embeddings), p=2, dim=-1).numpy()

        # Compute pairwise cosine similarity
        # similarity_matrix = cosine_similarity(residue_embeddings)  # [seq_len, seq_len]

        # # Threshold to create binary contact map
        # contact_map = (similarity_matrix >= similarity_threshold).astype(np.float32)
        dist_matrix = euclidean_distances(residue_embeddings)
        contact_map = np.exp(-dist_matrix ** 2 / (2 * 0.5 ** 2)) 
        np.fill_diagonal(contact_map, 0)  # Remove self-contacts

        return contact_map


    def compute_protein_ligand_pseudo_contact_map(
        self,
        prot_seq: str,
        smiles: str,
        sigma: float = 0.5
    ) -> np.ndarray:
        """
        Compute a pseudo-contact map for protein-ligand interactions.
        Returns a [seq_length, 1] matrix indicating interaction likelihoods
        between each residue and the ligand using Gaussian similarity.

        Args:
            prot_seq (str): Protein amino acid sequence.
            smiles (str): Ligand SMILES string.
            sigma (float): Gaussian kernel width.

        Returns:
            np.ndarray: Interaction map of shape [seq_length, 1].
        """
        # --- Protein processing ---
        preprocessed_seq = self.preprocess_sequence(prot_seq)
        prot_tokens = self.prot_tokenizer(
            [preprocessed_seq], padding=True, max_length=3200, truncation=True, return_tensors='np'
        )
        residue_embeddings = self.prot_encoder.encode(
            prot_tokens['input_ids'], prot_tokens['attention_mask'], return_unpooled=True
        )[0]  # [seq_length, hidden_size]
        
        # Remove padding residues
        mask = prot_tokens['attention_mask'][0].astype(bool)
        residue_embeddings = residue_embeddings[mask]
        seq_len = residue_embeddings.shape[0]

        # if seq_len != len(prot_seq):
        #     raise ValueError(f"Residue embeddings ({seq_len}) ≠ input sequence length ({len(prot_seq)})")

        # --- Ligand processing ---
        mol_tokens = self.mol_tokenizer(
            [smiles], padding=True, max_length=278, truncation=True, return_tensors='np'
        )
        ligand_embedding = self.mol_encoder.encode(
            mol_tokens['input_ids'], mol_tokens['attention_mask']
        )[0]  # [hidden_size]

        # --- Compute residue-ligand distances ---
        ligand_embedding = ligand_embedding.reshape(1, -1)  # [1, hidden_size]
        distances = euclidean_distances(residue_embeddings, ligand_embedding)  # [seq_len, 1]

        # --- Convert distances to similarities using Gaussian kernel ---
        contact_map = np.exp(-distances ** 2 / (2 * sigma ** 2))  # [seq_len, 1]

        return contact_map.astype(np.float32)


    def plot_contact_map(self, contact_map: np.ndarray, title: str = "Pseudo-Contact Map", output_file: str = None):
        """
        Visualize a contact map using Matplotlib.
        """
        plt.figure(figsize=(8, 8))
        plt.imshow(contact_map, cmap='binary', interpolation='nearest')
        plt.title(title)
        plt.xlabel("Residue/Atom Index")
        plt.ylabel("Residue Index")
        plt.colorbar(label='Contact (1 = high similarity)')
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

