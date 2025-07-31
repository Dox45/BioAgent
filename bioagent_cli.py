import os
import sqlite3
import numpy as np
import onnxruntime as ort
from transformers import BertTokenizerFast
import typer
from typing import List, Optional
from pathlib import Path
import json

app = typer.Typer(help="BioAgent CLI - Protein embedding search and binding affinity prediction tool")

# Import Plapt functionality
try:
    from quantplapt import Plapt
    PLAPT_AVAILABLE = True
except ImportError:
    PLAPT_AVAILABLE = False
    typer.echo("⚠️ Plapt module not found. Binding affinity and contact map features will be unavailable.", err=True)

# Default configuration
DEFAULT_SEQUENCE = "MTNVSGVDFQLRSVPLLSRVGADRADRLRTDMEAAAAGWPGAALLRVDSRNRVLVANGRVLLGAAIELADKPPPEAVFLGRVEGGRHVWAVRAALQPIADPDIPAEAVDLRGLGRIMDDTSSQLVSSASALLNWHDNARFSALDGAPTKPARAGWSRVNPITGHEEFPRIDPAVICLVHDGADRAVLARQAAWPERMFSLLAGFVEAGESFEVCVAREIREEIGLTVRDVRYLGSQQWPFPRSLMVGFHALGDPDEEFSFSDGEIAEAAWFTRDEVRAALAAGDWSSASESKLLLPGSISIARVIIESWAACE"

DEFAULT_DB_PATHS = [
    "hiv_embeddings.sqlite",
    "tb_embeddings.sqlite", 
    "malaria_embeddings.sqlite"
]

# ONNX model configuration
TOKENIZER_NAME = "Rostlab/prot_bert"
MODEL_PATH = "models/protbert-quant.onnx"

# Global variables for model loading
tokenizer = None
ort_session = None

def load_model():
    """Load tokenizer and ONNX model."""
    global tokenizer, ort_session
    if tokenizer is None or ort_session is None:
        try:
            tokenizer = BertTokenizerFast.from_pretrained(TOKENIZER_NAME)
            ort_session = ort.InferenceSession(MODEL_PATH)
            typer.echo("✅ Model and tokenizer loaded successfully")
        except Exception as e:
            typer.echo(f"❌ Failed to load model or tokenizer: {e}", err=True)
            raise typer.Exit(1)

def get_protein_embedding(sequence: str) -> np.ndarray:
    """Generate protein embedding using ProtBERT ONNX model."""
    try:
        sequence = ' '.join(list(sequence.upper()))
        tokens = tokenizer(sequence, return_tensors="np", padding=True, truncation=True, max_length=512)
        inputs = {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"]
        }
        outputs = ort_session.run(None, inputs)[0]
        cls_embedding = outputs[:, 0, :]  # [CLS] token
        return cls_embedding.squeeze().astype(np.float32)
    except Exception as e:
        typer.echo(f"❌ Error generating embedding: {e}", err=True)
        return None

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

def load_embeddings_from_blob_sqlite(db_path: str, table: str = "proteins"):
    """Load protein embeddings from an SQLite database."""
    if not os.path.exists(db_path):
        typer.echo(f"❌ Database file {db_path} does not exist")
        return []
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(f"SELECT id, embedding FROM {table}")
        data = cursor.fetchall()
    except sqlite3.OperationalError as e:
        typer.echo(f"❌ Database error: {e}")
        conn.close()
        return []
    
    entries = []
    for protein_id, emb_blob in data:
        try:
            vec = np.frombuffer(emb_blob, dtype=np.float32)
            if vec.ndim == 1:
                entries.append((protein_id, vec))
        except Exception as e:
            typer.echo(f"⚠️ Skipping invalid embedding for ID {protein_id}: {e}")
    
    conn.close()
    return entries

def find_nearest_protein(query_embedding, entries, threshold: float = 0.5):
    """Find the protein with the highest cosine similarity."""
    best_id, best_score = None, -1
    for protein_id, db_emb in entries:
        if len(db_emb) != len(query_embedding):
            typer.echo(f"⚠️ Dimension mismatch: query={len(query_embedding)}, db={len(db_emb)}")
            continue
        score = cosine_similarity(query_embedding, db_emb)
        if score > best_score:
            best_id, best_score = protein_id, score
    
    if best_score >= threshold:
        return best_id, best_score
    return None, best_score

def search_all_databases(query_embedding, db_paths: List[str], table: str = "proteins", threshold: float = 0.5):
    """Search for similar proteins across multiple SQLite databases."""
    for db_path in db_paths:
        if not os.path.exists(db_path):
            typer.echo(f"❌ Database file {db_path} does not exist")
            continue
        
        typer.echo(f"\n🔎 Searching in {db_path}...")
        entries = load_embeddings_from_blob_sqlite(db_path, table)
        
        if not entries:
            typer.echo(f"No valid embeddings found in {db_path}")
            continue
        
        typer.echo(f"Query embedding dimension: {len(query_embedding)}")
        typer.echo(f"Database embedding dimension: {len(entries[0][1]) if entries else 'N/A'}")
        
        match_id, score = find_nearest_protein(query_embedding, entries, threshold)
        if match_id:
            typer.echo(f"✅ Match found in {db_path} → {match_id} (cosine: {score:.4f})")
            return match_id, db_path, score
    
    typer.echo("❌ No similar protein found in any database.")
    return None, None, None

@app.command()
def search(
    sequence: Optional[str] = typer.Option(None, "--sequence", "-s", help="Protein sequence to search for"),
    sequence_file: Optional[Path] = typer.Option(None, "--file", "-f", help="File containing protein sequence"),
    databases: Optional[List[str]] = typer.Option(None, "--db", "-d", help="Database paths to search"),
    table: str = typer.Option("proteins", "--table", "-t", help="Database table name"),
    threshold: float = typer.Option(0.5, "--threshold", help="Similarity threshold (0.0-1.0)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """Search for similar proteins using embedding similarity."""
    
    # Load model first
    load_model()
    
    # Determine sequence to use
    if sequence_file:
        if not sequence_file.exists():
            typer.echo(f"❌ Sequence file {sequence_file} does not exist", err=True)
            raise typer.Exit(1)
        try:
            sequence = sequence_file.read_text().strip().replace('\n', '').replace(' ', '')
            typer.echo(f"📄 Loaded sequence from {sequence_file} ({len(sequence)} amino acids)")
        except Exception as e:
            typer.echo(f"❌ Error reading sequence file: {e}", err=True)
            raise typer.Exit(1)
    elif sequence:
        sequence = sequence.strip().replace('\n', '').replace(' ', '')
        typer.echo(f"🧬 Using provided sequence ({len(sequence)} amino acids)")
    else:
        sequence = DEFAULT_SEQUENCE
        typer.echo(f"🧬 Using default test sequence ({len(sequence)} amino acids)")
    
    if verbose:
        typer.echo(f"Sequence: {sequence[:50]}{'...' if len(sequence) > 50 else ''}")
    
    # Determine databases to search
    db_paths = databases if databases else DEFAULT_DB_PATHS
    typer.echo(f"🗄️ Searching {len(db_paths)} database(s)")
    
    # Validate threshold
    if not 0.0 <= threshold <= 1.0:
        typer.echo("❌ Threshold must be between 0.0 and 1.0", err=True)
        raise typer.Exit(1)
    
    # Generate embedding
    typer.echo("📥 Generating protein embedding...")
    query_embedding = get_protein_embedding(sequence)
    if query_embedding is None:
        typer.echo("❌ Failed to generate embedding", err=True)
        raise typer.Exit(1)
    
    if verbose:
        typer.echo(f"Generated embedding with dimension: {len(query_embedding)}")
    
    # Search databases
    typer.echo("🚀 Starting search...")
    match_id, db_path, score = search_all_databases(query_embedding, db_paths, table, threshold)
    
    if match_id:
        typer.echo(f"\n🎯 Final Result:")
        typer.echo(f"  Protein ID: {match_id}")
        typer.echo(f"  Database: {db_path}")
        typer.echo(f"  Similarity: {score:.4f}")
    else:
        typer.echo(f"\n❌ No matches found above threshold {threshold}")

@app.command()
def test():
    """Run the original test functionality with default parameters."""
    typer.echo("🧪 Running core functionality test...")
    
    # Load model
    load_model()
    
    # Use default sequence
    typer.echo(f"🧬 Using default test sequence ({len(DEFAULT_SEQUENCE)} amino acids)")
    
    # Generate embedding
    typer.echo("📥 Getting embedding...")
    query_embedding = get_protein_embedding(DEFAULT_SEQUENCE)
    if query_embedding is None:
        typer.echo("❌ Failed to generate embedding", err=True)
        raise typer.Exit(1)
    
    # Search with defaults
    typer.echo("🚀 Searching...")
    search_all_databases(query_embedding, DEFAULT_DB_PATHS, table="proteins", threshold=0.5)

@app.command()
def predict_affinity(
    protein_seq: str = typer.Option(..., "--protein", "-p", help="Protein sequence"),
    molecules: List[str] = typer.Option(..., "--molecule", "-m", help="SMILES strings (can specify multiple)"),
    output_file: Optional[Path] = typer.Option(None, "--output", "-o", help="Output JSON file"),
    prot_batch_size: int = typer.Option(2, "--prot-batch", help="Protein batch size"),
    mol_batch_size: int = typer.Option(16, "--mol-batch", help="Molecule batch size"),
    affinity_batch_size: int = typer.Option(128, "--affinity-batch", help="Affinity batch size"),
    use_progress: bool = typer.Option(True, "--progress/--no-progress", help="Show progress bars"),
    device: str = typer.Option("cuda", "--device", help="Device to use (cuda/cpu)")
):
    """Predict binding affinity between protein and molecules."""
    if not PLAPT_AVAILABLE:
        typer.echo("❌ Plapt module not available. Cannot predict binding affinity.", err=True)
        raise typer.Exit(1)
    
    try:
        typer.echo("🧪 Initializing Plapt model...")
        plapt = Plapt(device=device, use_tqdm=use_progress)
        
        typer.echo(f"🧬 Protein sequence length: {len(protein_seq)}")
        typer.echo(f"💊 Number of molecules: {len(molecules)}")
        
        # Prepare data for batch prediction
        protein_seqs = [protein_seq] * len(molecules)
        
        typer.echo("🔬 Predicting binding affinities...")
        results = plapt.predict_affinity(
            protein_seqs, molecules, 
            prot_batch_size=prot_batch_size,
            mol_batch_size=mol_batch_size,
            affinity_batch_size=affinity_batch_size
        )
        
        # Format results
        formatted_results = []
        for i, (mol, result) in enumerate(zip(molecules, results)):
            formatted_results.append({
                "molecule_index": i,
                "smiles": mol,
                "neg_log10_affinity_M": result["neg_log10_affinity_M"],
                "affinity_uM": result["affinity_uM"]
            })
        
        # Display results
        typer.echo("\n📊 Results:")
        for result in formatted_results:
            typer.echo(f"  Molecule {result['molecule_index']}: {result['smiles'][:50]}{'...' if len(result['smiles']) > 50 else ''}")
            typer.echo(f"    Affinity: {result['affinity_uM']:.2f} μM")
            typer.echo(f"    -log10(Kd): {result['neg_log10_affinity_M']:.3f}\n")
        
        # Save to file if requested
        if output_file:
            output_data = {
                "protein_sequence": protein_seq,
                "results": formatted_results
            }
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            typer.echo(f"💾 Results saved to {output_file}")
            
    except Exception as e:
        typer.echo(f"❌ Error predicting affinity: {e}", err=True)
        raise typer.Exit(1)

@app.command()  
def score_candidates(
    protein_seq: str = typer.Option(..., "--protein", "-p", help="Target protein sequence"),
    molecules: List[str] = typer.Option(..., "--molecule", "-m", help="SMILES strings to score"),
    output_file: Optional[Path] = typer.Option(None, "--output", "-o", help="Output JSON file"),
    mol_batch_size: int = typer.Option(16, "--mol-batch", help="Molecule batch size"),
    affinity_batch_size: int = typer.Option(128, "--affinity-batch", help="Affinity batch size"),
    use_progress: bool = typer.Option(True, "--progress/--no-progress", help="Show progress bars"),
    device: str = typer.Option("cuda", "--device", help="Device to use (cuda/cpu)"),
    sort_results: bool = typer.Option(True, "--sort/--no-sort", help="Sort by affinity")
):
    """Score molecular candidates against a target protein."""
    if not PLAPT_AVAILABLE:
        typer.echo("❌ Plapt module not available. Cannot score candidates.", err=True)
        raise typer.Exit(1)
    
    try:
        typer.echo("🧪 Initializing Plapt model...")
        plapt = Plapt(device=device, use_tqdm=use_progress)
        
        typer.echo(f"🎯 Target protein length: {len(protein_seq)}")
        typer.echo(f"💊 Number of candidates: {len(molecules)}")
        
        typer.echo("🔬 Scoring candidates...")
        results = plapt.score_candidates(
            protein_seq, molecules,
            mol_batch_size=mol_batch_size,
            affinity_batch_size=affinity_batch_size
        )
        
        # Format results
        formatted_results = []
        for i, (mol, result) in enumerate(zip(molecules, results)):
            formatted_results.append({
                "molecule_index": i,
                "smiles": mol,
                "neg_log10_affinity_M": result["neg_log10_affinity_M"],
                "affinity_uM": result["affinity_uM"]
            })
        
        # Sort by affinity if requested
        if sort_results:
            formatted_results.sort(key=lambda x: x["affinity_uM"])
        
        # Display results
        typer.echo("\n📊 Candidate Scores (sorted by affinity):" if sort_results else "\n📊 Candidate Scores:")
        for i, result in enumerate(formatted_results[:10]):  # Show top 10
            typer.echo(f"  #{i+1} - Molecule {result['molecule_index']}: {result['smiles'][:50]}{'...' if len(result['smiles']) > 50 else ''}")
            typer.echo(f"      Affinity: {result['affinity_uM']:.2f} μM")
            typer.echo(f"      -log10(Kd): {result['neg_log10_affinity_M']:.3f}\n")
        
        if len(formatted_results) > 10:
            typer.echo(f"... and {len(formatted_results) - 10} more candidates")
        
        # Save to file if requested
        if output_file:
            output_data = {
                "target_protein": protein_seq,
                "results": formatted_results
            }
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            typer.echo(f"💾 Results saved to {output_file}")
            
    except Exception as e:
        typer.echo(f"❌ Error scoring candidates: {e}", err=True)
        raise typer.Exit(1)

@app.command()
def protein_contact_map(
    protein_seq: str = typer.Option(..., "--protein", "-p", help="Protein sequence"),
    output_file: Optional[Path] = typer.Option(None, "--output", "-o", help="Output image file"),
    similarity_threshold: float = typer.Option(0.3, "--threshold", help="Similarity threshold"),
    visualize: bool = typer.Option(True, "--visualize/--no-visualize", help="Show visualization"),
    device: str = typer.Option("cuda", "--device", help="Device to use (cuda/cpu)")
):
    """Generate protein pseudo-contact map using residue embeddings."""
    if not PLAPT_AVAILABLE:
        typer.echo("❌ Plapt module not available. Cannot generate contact map.", err=True)
        raise typer.Exit(1)
    
    try:
        typer.echo("🧪 Initializing Plapt model...")
        plapt = Plapt(device=device, use_tqdm=False)
        
        typer.echo(f"🧬 Computing contact map for protein ({len(protein_seq)} residues)...")
        contact_map = plapt.compute_protein_pseudo_contact_map(
            protein_seq, 
            similarity_threshold=similarity_threshold,
            visualize=False
        )
        
        typer.echo(f"📊 Contact map shape: {contact_map.shape}")
        typer.echo(f"📈 Contact density: {np.mean(contact_map):.3f}")
        
        if visualize:
            title = f"Protein Contact Map (threshold={similarity_threshold})"
            plapt.plot_contact_map(contact_map, title=title, output_file=str(output_file) if output_file else None)
            
        if output_file and not visualize:
            # Save as numpy array if not visualizing
            np.save(output_file.with_suffix('.npy'), contact_map)
            typer.echo(f"💾 Contact map saved to {output_file.with_suffix('.npy')}")
            
    except Exception as e:
        typer.echo(f"❌ Error generating contact map: {e}", err=True)
        raise typer.Exit(1)

@app.command()
def protein_ligand_contact_map(
    protein_seq: str = typer.Option(..., "--protein", "-p", help="Protein sequence"),
    ligand_smiles: str = typer.Option(..., "--ligand", "-l", help="Ligand SMILES string"),
    output_file: Optional[Path] = typer.Option(None, "--output", "-o", help="Output image file"),
    sigma: float = typer.Option(0.5, "--sigma", help="Gaussian kernel width"),
    visualize: bool = typer.Option(True, "--visualize/--no-visualize", help="Show visualization"),
    device: str = typer.Option("cuda", "--device", help="Device to use (cuda/cpu)")
):
    """Generate protein-ligand pseudo-contact map."""
    if not PLAPT_AVAILABLE:
        typer.echo("❌ Plapt module not available. Cannot generate contact map.", err=True)
        raise typer.Exit(1)
    
    try:
        typer.echo("🧪 Initializing Plapt model...")
        plapt = Plapt(device=device, use_tqdm=False)
        
        typer.echo(f"🧬 Protein length: {len(protein_seq)} residues")
        typer.echo(f"💊 Ligand SMILES: {ligand_smiles}")
        
        typer.echo("🔬 Computing protein-ligand contact map...")
        contact_map = plapt.compute_protein_ligand_pseudo_contact_map(
            protein_seq, ligand_smiles, sigma=sigma
        )
        
        typer.echo(f"📊 Contact map shape: {contact_map.shape}")
        typer.echo(f"📈 Max interaction score: {np.max(contact_map):.3f}")
        typer.echo(f"📈 Mean interaction score: {np.mean(contact_map):.3f}")
        
        # Find top interacting residues
        top_indices = np.argsort(contact_map.flatten())[-10:][::-1]
        typer.echo("\n🔝 Top 10 interacting residues:")
        for i, idx in enumerate(top_indices):
            score = contact_map[idx, 0]
            typer.echo(f"  #{i+1}: Residue {idx+1} - Score: {score:.3f}")
        
        if visualize:
            title = f"Protein-Ligand Contact Map (σ={sigma})"
            plapt.plot_contact_map(contact_map, title=title, output_file=str(output_file) if output_file else None)
            
        if output_file and not visualize:
            # Save as numpy array if not visualizing
            np.save(output_file.with_suffix('.npy'), contact_map)
            typer.echo(f"💾 Contact map saved to {output_file.with_suffix('.npy')}")
            
    except Exception as e:
        typer.echo(f"❌ Error generating protein-ligand contact map: {e}", err=True)
        raise typer.Exit(1)

if __name__ == "__main__":
    app()