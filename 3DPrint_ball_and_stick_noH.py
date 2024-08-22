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
# Set to keep track of processed bonds involving hydrogen
processed_hydrogen_bonds = set()

for atom in mol.atoms:
    # Skip hydrogen atoms
    if atom.atomic_symbol.lower() == 'h':
        continue
    
    atom_mesh = trimesh.primitives.Sphere(radius=atom.vdw_radius * 0.25 * SCALE_FACTOR,
                                           center=np.multiply(atom.coordinates, SCALE_FACTOR))

    bond_mesh = trimesh.Trimesh()  # Initialize an empty mesh for bonds

    for bond in atom.bonds:
        atom1, atom2 = bond.atoms
        # Skip bonds involving hydrogen
        if atom1.atomic_symbol.lower() == 'h' or atom2.atomic_symbol.lower() == 'h':
            continue
        if atom1.label == atom.label:
            p1 = np.multiply(np.array(atom1.coordinates), SCALE_FACTOR)
            p2 = np.multiply(np.array(atom2.coordinates), SCALE_FACTOR)
        elif atom2.label == atom.label:
            p1 = np.multiply(np.array(atom2.coordinates), SCALE_FACTOR)
            p2 = np.multiply(np.array(atom1.coordinates), SCALE_FACTOR)

        mp = (p1 + p2) / 2.0
        d1 = np.subtract(p1, mp)
        length = np.linalg.norm(d1)
        z = np.array([0., 0., 1.])
        t = np.cross(z, d1)
        angle = np.arccos(np.dot(z, d1) / length)
        rot = trimesh.transformations.rotation_matrix(angle, t)

        if bond.bond_type == 2:  # Check if the bond is a double bond
            offset = 0.15 * SCALE_FACTOR  # Adjust this value for appropriate spacing

            # Compute a perpendicular vector to d1
            perp_vector = np.cross(d1, np.array([1, 0, 0]))
            if np.linalg.norm(perp_vector) < 1e-6:
                perp_vector = np.cross(d1, np.array([0, 1, 0]))
            perp_vector = perp_vector / np.linalg.norm(perp_vector) * offset

            for sign in [-1, 1]:
                trans = trimesh.transformations.translation_matrix(-d1 / 2.0 + p1 + sign * perp_vector)
                tf = trans.dot(rot)
                bond_mesh += trimesh.primitives.Cylinder(radius=0.15 * SCALE_FACTOR, height=length,
                                                         sections=20, transform=tf)
        else:
            trans1 = trimesh.transformations.translation_matrix(-d1 / 2.0 + p1)
            tf1 = trans1.dot(rot)
            bond_mesh += trimesh.primitives.Cylinder(radius=0.20 * SCALE_FACTOR, height=length,
                                                     sections=20, transform=tf1)
        # Add processed hydrogen bond to the set
        processed_hydrogen_bonds.add(bond)
    # Add the atom mesh and bond mesh to the dictionary
    if atom.atomic_symbol not in stl_dict:
        stl_dict[atom.atomic_symbol] = trimesh.Trimesh()
    stl_dict[atom.atomic_symbol] += trimesh.util.concatenate([atom_mesh, bond_mesh])

# Export meshes for each atom type
for key, value in stl_dict.items():
    html_file.write(f'Writing mesh file for {key} <br />')
    stl_file_name = f"{key}_{entry.identifier}_ball_and_stick.stl"
    value.export(stl_file_name)

html_file.write('Mesh generation complete<br />')
html_file.close()
