# bioagent_cli.py – CLI version of BioAgent for testing core functionality
"""
CLI tool for protein embedding search functionality and binding affinity prediction.
Supports custom sequences, database paths, search parameters, and Plapt functionality.
"""

import os
import sqlite3
import numpy as np
import onnxruntime as ort
from transformers import BertTokenizerFast
import typer
from typing import List, Optional
from pathlib import Path
import json
import re
from dataclasses import dataclass

# Import chat functionality
try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False

app = typer.Typer(help="BioAgent CLI - Protein embedding search and binding affinity prediction tool")

# Import Plapt functionality
try:
    from quantplapt import Plapt
    PLAPT_AVAILABLE = True
except ImportError:
    PLAPT_AVAILABLE = False
    typer.echo("⚠️ Plapt module not found. Binding affinity and contact map features will be unavailable.", err=True)

@dataclass
class FunctionCall:
    """Represents a function call with parameters"""
    function_name: str
    parameters: dict
    confidence: float = 0.0

class BioAgentChat:
    """Chat agent for natural language interaction with BioAgent functions"""
    
    def __init__(
        self, 
        model_path: str = "models/tinyllama.gguf",
        device: str = "auto",
        n_ctx: int = 2048,
        verbose: bool = False
    ):
        """Initialize the chat agent with TinyLlama model"""
        self.model_path = model_path
        self.device = device
        self.verbose = verbose
        
        if not LLAMA_AVAILABLE:
            raise RuntimeError("llama-cpp-python not available. Install with: pip install llama-cpp-python")
        
        # Initialize LLM
        try:
            typer.echo("🤖 Loading TinyLlama model...")
            self.llm = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                verbose=verbose,
                n_gpu_layers=-1 if device != "cpu" else 0
            )
            typer.echo("✅ TinyLlama loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
        
        # Initialize Plapt if available
        self.plapt = None
        if PLAPT_AVAILABLE:
            try:
                typer.echo("🧪 Initializing Plapt...")
                self.plapt = Plapt(device="cuda" if device != "cpu" else "cpu", use_tqdm=True)
                typer.echo("✅ Plapt initialized")
            except Exception as e:
                typer.echo(f"⚠️ Plapt initialization warning: {e}")
        
        # Load embedding search model
        try:
            typer.echo("🔬 Loading embedding search model...")
            load_model()
            typer.echo("✅ Embedding search ready")
        except Exception as e:
            typer.echo(f"⚠️ Embedding search warning: {e}")
        
        # Define available functions
        self.functions = self._define_functions()
        
    def _define_functions(self) -> dict:
        """Define available functions for the agent"""
        functions = {
            "search_protein": {
                "description": "Search for similar proteins in databases using sequence similarity",
                "parameters": {
                    "sequence": {"type": "string", "description": "Protein amino acid sequence"},
                    "threshold": {"type": "number", "description": "Similarity threshold (0.0-1.0)", "default": 0.5},
                    "databases": {"type": "array", "description": "Database paths to search", "default": DEFAULT_DB_PATHS}
                },
                "available": True  # Always available since it's core functionality
            },
            "predict_binding_affinity": {
                "description": "Predict binding affinity between protein and molecules",
                "parameters": {
                    "protein_sequence": {"type": "string", "description": "Protein amino acid sequence"},
                    "molecules": {"type": "array", "description": "List of SMILES strings"},
                },
                "available": PLAPT_AVAILABLE
            },
            "score_drug_candidates": {
                "description": "Score molecular candidates against a target protein",
                "parameters": {
                    "protein_sequence": {"type": "string", "description": "Target protein sequence"},
                    "molecules": {"type": "array", "description": "List of SMILES strings to score"},
                },
                "available": PLAPT_AVAILABLE
            },
            "generate_protein_contact_map": {
                "description": "Generate protein contact map visualization",
                "parameters": {
                    "protein_sequence": {"type": "string", "description": "Protein amino acid sequence"},
                    "threshold": {"type": "number", "description": "Similarity threshold", "default": 0.3}
                },
                "available": PLAPT_AVAILABLE
            },
            "generate_protein_ligand_contact_map": {
                "description": "Generate protein-ligand interaction contact map",
                "parameters": {
                    "protein_sequence": {"type": "string", "description": "Protein amino acid sequence"},
                    "ligand_smiles": {"type": "string", "description": "Ligand SMILES string"},
                    "sigma": {"type": "number", "description": "Gaussian kernel width", "default": 0.5}
                },
                "available": PLAPT_AVAILABLE
            }
        }
        
        # Filter out unavailable functions
        return {k: v for k, v in functions.items() if v["available"]}
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for intent classification"""
        available_functions = list(self.functions.keys())
        
        return f"""You are BioAgent, an AI assistant specialized in protein analysis and drug discovery.

Available functions: {', '.join(available_functions)}

Your task is to:
1. Understand the user's request
2. Classify the intent and determine which function to call
3. Extract required parameters from the user's message
4. Respond with a JSON object containing the function call

Function descriptions:
{json.dumps(self.functions, indent=2)}

Response format:
{{
    "intent": "function_name",
    "parameters": {{
        "param1": "value1",
        "param2": "value2"
    }},
    "explanation": "Brief explanation of what you'll do"
}}

If the request is unclear or missing required parameters, ask for clarification.
If no function matches the request, respond with "intent": "chat" and provide a helpful response.

Examples:
User: "Search for HIV protein sequence MKVL..."
Response: {{"intent": "search_protein", "parameters": {{"sequence": "MKVL..."}}, "explanation": "I'll search for similar proteins in the databases"}}

User: "Generate contact map for protein MKVL..."
Response: {{"intent": "generate_protein_contact_map", "parameters": {{"protein_sequence": "MKVL..."}}, "explanation": "I'll generate a contact map visualization"}}
"""
    
    def _classify_intent(self, user_message: str) -> dict:
        """Classify user intent and extract parameters using LLM"""
        system_prompt = self._create_system_prompt()
        
        prompt = f"""<|system|>
{system_prompt}
<|user|>
{user_message}
<|assistant|>
"""
        
        try:
            response = self.llm(
                prompt,
                max_tokens=512,
                temperature=0.1,
                top_p=0.9,
                stop=["<|user|>", "<|system|>"],
                echo=False
            )
            
            response_text = response["choices"][0]["text"].strip()
            
            # Try to parse JSON response
            try:
                # Find JSON in response
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                else:
                    return {
                        "intent": "chat",
                        "explanation": response_text,
                        "parameters": {}
                    }
            except json.JSONDecodeError:
                return {
                    "intent": "chat",
                    "explanation": f"I couldn't parse the request properly. Could you rephrase that?",
                    "parameters": {}
                }
                
        except Exception as e:
            return {
                "intent": "error",
                "explanation": f"Error processing request: {e}",
                "parameters": {}
            }
    
    def _execute_function(self, intent: str, parameters: dict) -> dict:
        """Execute the classified function with parameters"""
        try:
            if intent == "search_protein":
                return self._search_protein(**parameters)
            elif intent == "predict_binding_affinity":
                return self._predict_binding_affinity(**parameters)
            elif intent == "score_drug_candidates":
                return self._score_drug_candidates(**parameters)
            elif intent == "generate_protein_contact_map":
                return self._generate_protein_contact_map(**parameters)
            elif intent == "generate_protein_ligand_contact_map":
                return self._generate_protein_ligand_contact_map(**parameters)
            else:
                return {"error": f"Unknown function: {intent}"}
        except Exception as e:
            return {"error": f"Function execution failed: {e}"}
    
    def _search_protein(self, sequence: str, threshold: float = 0.5, databases: list = None) -> dict:
        """Execute protein similarity search"""
        if databases is None:
            databases = DEFAULT_DB_PATHS
        
        typer.echo(f"🔍 Searching for similar proteins...")
        query_embedding = get_protein_embedding(sequence)
        if query_embedding is None:
            return {"error": "Failed to generate protein embedding"}
        
        match_id, db_path, score = search_all_databases(query_embedding, databases, threshold=threshold)
        
        return {
            "match_id": match_id,
            "database": db_path,
            "similarity_score": score,
            "success": match_id is not None
        }
    
    def _predict_binding_affinity(self, protein_sequence: str, molecules: list) -> dict:
        """Execute binding affinity prediction"""
        if not PLAPT_AVAILABLE or self.plapt is None:
            return {"error": "Binding affinity prediction not available"}
        
        typer.echo(f"🔬 Predicting binding affinities...")
        protein_seqs = [protein_sequence] * len(molecules)
        results = self.plapt.predict_affinity(protein_seqs, molecules)
        
        formatted_results = []
        for mol, result in zip(molecules, results):
            formatted_results.append({
                "smiles": mol,
                "affinity_uM": result["affinity_uM"],
                "neg_log10_affinity_M": result["neg_log10_affinity_M"]
            })
        
        return {"results": formatted_results, "success": True}
    
    def _score_drug_candidates(self, protein_sequence: str, molecules: list) -> dict:
        """Execute drug candidate scoring"""
        if not PLAPT_AVAILABLE or self.plapt is None:
            return {"error": "Drug candidate scoring not available"}
        
        typer.echo(f"🎯 Scoring drug candidates...")
        results = self.plapt.score_candidates(protein_sequence, molecules)
        
        formatted_results = []
        for mol, result in zip(molecules, results):
            formatted_results.append({
                "smiles": mol,
                "affinity_uM": result["affinity_uM"],
                "neg_log10_affinity_M": result["neg_log10_affinity_M"]
            })
        
        # Sort by affinity
        formatted_results.sort(key=lambda x: x["affinity_uM"])
        
        return {"results": formatted_results, "success": True}
    
    def _generate_protein_contact_map(self, protein_sequence: str, threshold: float = 0.3) -> dict:
        """Generate protein contact map"""
        if not PLAPT_AVAILABLE or self.plapt is None:
            return {"error": "Contact map generation not available"}
        
        typer.echo(f"🧬 Generating protein contact map...")
        contact_map = self.plapt.compute_protein_pseudo_contact_map(
            protein_sequence, similarity_threshold=threshold
        )
        
        # Generate visualization
        output_file = f"protein_contact_map_{len(protein_sequence)}aa.png"
        self.plapt.plot_contact_map(
            contact_map, 
            title=f"Protein Contact Map ({len(protein_sequence)} residues)",
            output_file=output_file
        )
        
        return {
            "contact_map_shape": contact_map.shape,
            "contact_density": float(contact_map.mean()),
            "output_file": output_file,
            "success": True
        }
    
    def _generate_protein_ligand_contact_map(self, protein_sequence: str, ligand_smiles: str, sigma: float = 0.5) -> dict:
        """Generate protein-ligand contact map"""
        if not PLAPT_AVAILABLE or self.plapt is None:
            return {"error": "Protein-ligand contact map not available"}
        
        typer.echo(f"🔗 Generating protein-ligand contact map...")
        contact_map = self.plapt.compute_protein_ligand_pseudo_contact_map(
            protein_sequence, ligand_smiles, sigma=sigma
        )
        
        # Generate visualization
        output_file = f"protein_ligand_contact_map_{len(protein_sequence)}aa.png"
        self.plapt.plot_contact_map(
            contact_map,
            title=f"Protein-Ligand Contact Map",
            output_file=output_file
        )
        
        # Find top interactions
        top_indices = contact_map.flatten().argsort()[-5:][::-1]
        top_interactions = [
            {"residue": int(idx), "score": float(contact_map[idx, 0])}
            for idx in top_indices
        ]
        
        return {
            "contact_map_shape": contact_map.shape,
            "max_interaction": float(contact_map.max()),
            "mean_interaction": float(contact_map.mean()),
            "top_interactions": top_interactions,
            "output_file": output_file,
            "success": True
        }
    
    def chat(self, user_message: str) -> str:
        """Main chat interface"""
        typer.echo(f"👤 User: {user_message}")
        
        # Classify intent and extract parameters
        classification = self._classify_intent(user_message)
        intent = classification.get("intent", "chat")
        parameters = classification.get("parameters", {})
        explanation = classification.get("explanation", "")
        
        typer.echo(f"🤖 Intent: {intent}")
        if parameters:
            typer.echo(f"📋 Parameters: {json.dumps(parameters, indent=2)}")
        
        if intent == "chat":
            return f"🤖 BioAgent: {explanation}"
        elif intent == "error":
            return f"❌ Error: {explanation}"
        else:
            # Execute function
            typer.echo(f"⚙️ Executing {intent}...")
            result = self._execute_function(intent, parameters)
            
            if "error" in result:
                return f"❌ Error: {result['error']}"
            else:
                return self._format_result(intent, result)
    
    def _format_result(self, intent: str, result: dict) -> str:
        """Format function execution results for display"""
        if intent == "search_protein":
            if result["success"]:
                return f"✅ Found match: {result['match_id']} in {result['database']} (similarity: {result['similarity_score']:.4f})"
            else:
                return "❌ No similar proteins found"
        
        elif intent == "predict_binding_affinity":
            output = "🔬 Binding Affinity Results:\n"
            for i, res in enumerate(result["results"]):
                output += f"  Molecule {i+1}: {res['smiles'][:50]}{'...' if len(res['smiles']) > 50 else ''}\n"
                output += f"    Affinity: {res['affinity_uM']:.2f} μM\n"
                output += f"    -log10(Kd): {res['neg_log10_affinity_M']:.3f}\n\n"
            return output
        
        elif intent == "score_drug_candidates":
            output = "🎯 Drug Candidate Scores (sorted by affinity):\n"
            for i, res in enumerate(result["results"][:5]):  # Top 5
                output += f"  #{i+1}: {res['smiles'][:50]}{'...' if len(res['smiles']) > 50 else ''}\n"
                output += f"    Affinity: {res['affinity_uM']:.2f} μM\n\n"
            return output
        
        elif intent == "generate_protein_contact_map":
            return f"🧬 Contact map generated! Shape: {result['contact_map_shape']}, Density: {result['contact_density']:.3f}\n📁 Saved to: {result['output_file']}"
        
        elif intent == "generate_protein_ligand_contact_map":
            output = f"🔗 Protein-ligand contact map generated!\n"
            output += f"📊 Max interaction: {result['max_interaction']:.3f}\n"
            output += f"📈 Mean interaction: {result['mean_interaction']:.3f}\n"
            output += f"🔝 Top interactions:\n"
            for interaction in result['top_interactions'][:3]:
                output += f"  Residue {interaction['residue']}: {interaction['score']:.3f}\n"
            output += f"📁 Saved to: {result['output_file']}"
            return output
        
        else:
            return f"✅ Function executed: {json.dumps(result, indent=2)}"

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


@app.command()
def chat(
    model_path: str = typer.Option("models/tinyllama.gguf", "--model", "-m", help="Path to TinyLlama GGUF model"),
    device: str = typer.Option("auto", "--device", help="Device to use (auto/cpu/cuda)"),
    context_size: int = typer.Option(2048, "--context", help="Context window size"),
    verbose: bool = typer.Option(False, "--verbose", help="Enable verbose model output"),
    test_mode: bool = typer.Option(False, "--test", help="Run test queries first")
):
    """Start interactive chat mode with LLM-powered function calling."""
    
    if not LLAMA_AVAILABLE:
        typer.echo("❌ llama-cpp-python not available. Install with: pip install llama-cpp-python", err=True)
        raise typer.Exit(1)
    
    if not Path(model_path).exists():
        typer.echo(f"❌ Model not found at {model_path}", err=True)
        typer.echo("Please download TinyLlama model or specify correct path with --model", err=True)
        raise typer.Exit(1)
    
    try:
        # Initialize chat agent
        agent = BioAgentChat(
            model_path=model_path,
            device=device,
            n_ctx=context_size,
            verbose=verbose
        )
        
        typer.echo(f"\n🧬 BioAgent Chat Ready!")
        typer.echo(f"📋 Available functions: {', '.join(agent.functions.keys())}")
        typer.echo(f"🤖 Model: {Path(model_path).name}")
        typer.echo(f"💾 Device: {device}")
        typer.echo(f"📝 Context: {context_size} tokens")
        
        # Run test queries if requested
        if test_mode:
            typer.echo(f"\n🧪 Running test queries:")
            test_queries = [
                "Search for HIV protein with sequence MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGQRWELALGRFWDYLRWVQTLSEQVQEELLSSQVTQELRALMDETAQ",
                "Generate contact map for protein MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGQRWELALGRFWDYLRWVQTLSEQVQEELLSSQVTQELRALMDETAQ",
            ]
            
            if PLAPT_AVAILABLE:
                test_queries.extend([
                    "Predict binding affinity between protein MKVL and molecules CCO, CCN",
                    "Score drug candidates CCO and CCN against protein MKVL"
                ])
            
            for i, query in enumerate(test_queries, 1):
                typer.echo(f"\n{'='*60}")
                typer.echo(f"Test {i}: {query[:80]}{'...' if len(query) > 80 else ''}")
                typer.echo(f"{'='*60}")
                try:
                    response = agent.chat(query)
                    typer.echo(response)
                except Exception as e:
                    typer.echo(f"❌ Test {i} failed: {e}")
        
        # Interactive mode
        typer.echo(f"\n{'='*60}")
        typer.echo("💬 Interactive Chat Mode")
        typer.echo("Type your questions in natural language!")
        typer.echo("Examples:")
        typer.echo("  - 'Search for protein sequence MKVL...'")
        typer.echo("  - 'Generate contact map for protein ATCG...'")
        typer.echo("  - 'Predict binding affinity for molecule CCO'")
        typer.echo("Type 'quit', 'exit', or 'q' to exit")
        typer.echo(f"{'='*60}")
        
        while True:
            try:
                user_input = typer.prompt("\n👤 You", default="").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                    
                if not user_input:
                    continue
                    
                # Process chat
                response = agent.chat(user_input)
                typer.echo(f"\n{response}")
                
            except KeyboardInterrupt:
                typer.echo(f"\n\n👋 Chat interrupted. Goodbye!")
                break
            except EOFError:
                typer.echo(f"\n\n👋 Chat ended. Goodbye!")
                break
        
        typer.echo("👋 Thanks for using BioAgent Chat!")
        
    except Exception as e:
        typer.echo(f"❌ Error initializing chat: {e}", err=True)
        if verbose:
            import traceback
            typer.echo(traceback.format_exc(), err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()