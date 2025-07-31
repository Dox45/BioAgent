from quantplapt import Plapt


plapt = Plapt()

# Predict affinity for a single protein and multiple ligands
protein = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
molecules = ["CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F", 
             "COC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F"]

results = plapt.score_candidates(protein, molecules)
print(results)

# Predict affinities for multiple protein-ligand pairs
proteins = ["SEQUENCE1", "SEQUENCE2"]
molecules = ["SMILES1", "SMILES2"]

results = plapt.predict_affinity(proteins, molecules)
print(results)



