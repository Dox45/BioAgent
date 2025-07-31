# from quantplapt import Plapt


# plapt = Plapt()
# protein = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
# molecule = "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F"
# # protein_contact_map = plapt.compute_protein_pseudo_contact_map(protein)
# # plapt.plot_contact_map(protein_contact_map, output_file="protein_pseudo_contact_map.png")
# pl_contact_map = plapt.compute_protein_ligand_pseudo_contact_map(protein, molecule)
# plapt.plot_contact_map(pl_contact_map, output_file="protein_ligand_pseudo_contact_map.png")

# bioagent_cli.py – CLI version of BioAgent for testing core functionality
"""
CLI tool for protein embedding search functionality and binding affinity prediction.
Supports custom sequences, database paths, search parameters, and Plapt functionality.
"""

