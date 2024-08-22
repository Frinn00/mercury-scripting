import numpy as np
import trimesh
from mercury_interface import MercuryInterface

helper = MercuryInterface()
entry = helper.current_entry
mol = entry.molecule
SCALE_FACTOR = 10.0  # scale factor in mm/Angstrom
html_file_name = helper.output_html_file
html_file = open(html_file_name, "w")

# Dictionary to store meshes for each atom type
stl_dict = {}

# Set to keep track of processed hydrogen bonds
processed_hydrogen_bonds = set()

for atom in mol.atoms:
    # Generate a space-fill representation for each atom
    atom_mesh = trimesh.primitives.Sphere(radius=atom.vdw_radius * SCALE_FACTOR,
                                          center=np.multiply(atom.coordinates, SCALE_FACTOR))

    # Add the atom mesh to the dictionary
    if atom.atomic_symbol not in stl_dict:
        stl_dict[atom.atomic_symbol] = trimesh.Trimesh()
    stl_dict[atom.atomic_symbol] += atom_mesh

# Export meshes for each atom type
for key, value in stl_dict.items():
    html_file.write(f'Writing mesh file for {key} <br />')
    stl_file_name = f"{key}_{entry.identifier}_space-filling.stl"
    value.export(stl_file_name)

html_file.write('Space-fill generation complete<br />')
html_file.close()
