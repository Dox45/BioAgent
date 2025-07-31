"""
BioAgent Chat - Offline LLM-powered agent for protein analysis
Uses TinyLlama for intent classification and function calling
"""

import json
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import typer

try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False
    print("❌ llama-cpp-python not available. Install with: pip install llama-cpp-python")


app = typer.Typer(help="BioAgent CLI - Protein embedding search and binding affinity prediction tool")
# Import our existing functionality
try:
    from bioagent_cli import (
        get_protein_embedding, search_all_databases, load_model,
        DEFAULT_DB_PATHS, DEFAULT_SEQUENCE
    )
    EMBEDDING_SEARCH_AVAILABLE = True
except ImportError:
    EMBEDDING_SEARCH_AVAILABLE = False
    print("❌ Embedding search functions not available")

try:
    from quantplapt import Plapt
    PLAPT_AVAILABLE = True
except ImportError:
    PLAPT_AVAILABLE = False
    print("❌ Plapt module not available")


@dataclass
class FunctionCall:
    """Represents a function call with parameters"""
    function_name: str
    parameters: Dict[str, Any]
    confidence: float = 0.0


class BioAgentChat:
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
            raise RuntimeError("llama-cpp-python not available")
        
        # Initialize LLM
        try:
            print("🤖 Loading TinyLlama model...")
            self.llm = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                verbose=verbose,
                n_gpu_layers=-1 if device != "cpu" else 0
            )
            print("✅ TinyLlama loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
        
        # Initialize Plapt if available
        self.plapt = None
        if PLAPT_AVAILABLE:
            try:
                print("🧪 Initializing Plapt...")
                self.plapt = Plapt(device="cuda" if device != "cpu" else "cpu", use_tqdm=True)
                print("✅ Plapt initialized")
            except Exception as e:
                print(f"⚠️ Plapt initialization failed: {e}")
        
        # Load embedding search model if available
        if EMBEDDING_SEARCH_AVAILABLE:
            try:
                print("🔬 Loading embedding search model...")
                load_model()
                print("✅ Embedding search ready")
            except Exception as e:
                print(f"⚠️ Embedding search initialization failed: {e}")
        
        # Define available functions
        self.functions = self._define_functions()
        
    def _define_functions(self) -> Dict[str, Dict]:
        """Define available functions for the agent"""
        functions = {
            "search_protein": {
                "description": "Search for similar proteins in databases using sequence similarity",
                "parameters": {
                    "sequence": {"type": "string", "description": "Protein amino acid sequence"},
                    "threshold": {"type": "number", "description": "Similarity threshold (0.0-1.0)", "default": 0.5},
                    "databases": {"type": "array", "description": "Database paths to search", "default": DEFAULT_DB_PATHS}
                },
                "available": EMBEDDING_SEARCH_AVAILABLE
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
    
    def _classify_intent(self, user_message: str) -> Dict[str, Any]:
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
                    "explanation": f"I couldn't parse the request properly. Response was: {response_text}",
                    "parameters": {}
                }
                
        except Exception as e:
            return {
                "intent": "error",
                "explanation": f"Error processing request: {e}",
                "parameters": {}
            }
    
    def _execute_function(self, intent: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
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
    
    def _search_protein(self, sequence: str, threshold: float = 0.5, databases: List[str] = None) -> Dict[str, Any]:
        """Execute protein similarity search"""
        if not EMBEDDING_SEARCH_AVAILABLE:
            return {"error": "Protein search not available"}
        
        if databases is None:
            databases = DEFAULT_DB_PATHS
        
        print(f"🔍 Searching for similar proteins...")
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
    
    def _predict_binding_affinity(self, protein_sequence: str, molecules: List[str]) -> Dict[str, Any]:
        """Execute binding affinity prediction"""
        if not PLAPT_AVAILABLE or self.plapt is None:
            return {"error": "Binding affinity prediction not available"}
        
        print(f"🔬 Predicting binding affinities...")
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
    
    def _score_drug_candidates(self, protein_sequence: str, molecules: List[str]) -> Dict[str, Any]:
        """Execute drug candidate scoring"""
        if not PLAPT_AVAILABLE or self.plapt is None:
            return {"error": "Drug candidate scoring not available"}
        
        print(f"🎯 Scoring drug candidates...")
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
    
    def _generate_protein_contact_map(self, protein_sequence: str, threshold: float = 0.3) -> Dict[str, Any]:
        """Generate protein contact map"""
        if not PLAPT_AVAILABLE or self.plapt is None:
            return {"error": "Contact map generation not available"}
        
        print(f"🧬 Generating protein contact map...")
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
    
    def _generate_protein_ligand_contact_map(self, protein_sequence: str, ligand_smiles: str, sigma: float = 0.5) -> Dict[str, Any]:
        """Generate protein-ligand contact map"""
        if not PLAPT_AVAILABLE or self.plapt is None:
            return {"error": "Protein-ligand contact map not available"}
        
        print(f"🔗 Generating protein-ligand contact map...")
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
        print(f"👤 User: {user_message}")
        
        # Classify intent and extract parameters
        classification = self._classify_intent(user_message)
        intent = classification.get("intent", "chat")
        parameters = classification.get("parameters", {})
        explanation = classification.get("explanation", "")
        
        print(f"🤖 Intent: {intent}")
        if parameters:
            print(f"📋 Parameters: {parameters}")
        
        if intent == "chat":
            return f"🤖 BioAgent: {explanation}"
        elif intent == "error":
            return f"❌ Error: {explanation}"
        else:
            # Execute function
            print(f"⚙️ Executing {intent}...")
            result = self._execute_function(intent, parameters)
            
            if "error" in result:
                return f"❌ Error: {result['error']}"
            else:
                return self._format_result(intent, result)
    
    def _format_result(self, intent: str, result: Dict[str, Any]) -> str:
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


def main():
    """Test the chat agent"""
    if not LLAMA_AVAILABLE:
        print("❌ Please install llama-cpp-python: pip install llama-cpp-python")
        return
    
    model_path = "models/tinyllama.gguf"
    if not Path(model_path).exists():
        print(f"❌ Model not found at {model_path}")
        print("Please download TinyLlama model or update the path")
        return
    
    try:
        agent = BioAgentChat(model_path=model_path, verbose=False)
        
        print("\n🧬 BioAgent Chat Ready!")
        print("Available functions:", list(agent.functions.keys()))
        print("Type 'quit' to exit\n")
        
        # Test examples
        test_queries = [
            "Search for HIV protein with sequence MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGQRWELALGRFWDYLRWVQTLSEQVQEELLSSQVTQELRALMDETAQ",
            "Generate contact map for protein MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGQRWELALGRFWDYLRWVQTLSEQVQEELLSSQVTQELRALMDETAQ",
            
        ]
        
        print("🧪 Running test queries:")
        for query in test_queries:
            print(f"\n{'='*60}")
            response = agent.chat(query)
            print(response)
        
        print(f"\n{'='*60}")
        print("💬 Interactive mode (type 'quit' to exit):")
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                if user_input:
                    response = agent.chat(user_input)
                    print(response)
            except KeyboardInterrupt:
                break
        
        print("\n👋 Goodbye!")
        
    except Exception as e:
        print(f"❌ Error initializing chat agent: {e}")

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
    main()