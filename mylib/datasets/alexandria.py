# Datalaoder for loading Alexandria database
import json
from os.path import abspath, dirname
import numpy as np
import torch as tc
from ase.data import atomic_numbers  # this is a dict
from pymatgen.core.lattice import Lattice
from torch.utils.data import DataLoader as DL
from torch.utils.data import Dataset as DS
import os

def load_dataset_to_tmp():
    """
    This function will load the Alexandria dataset to ./tmp/Alexandria
    """ 
    # check if ./tmp/Alexandria exsits
    if os.path.exists("./tmp/Alexandria"):
        print("Alexandria dataset already exsits in ./tmp/Alexandria")
        return
    os.makedirs("./tmp/Alexandria", exist_ok=True)
    os.system("cp -r ~/data_coses1/gengyao/Alexandria/ ./tmp/Alexandria")
    return 

def read_json(path: str) -> dict:
    with open(path, "r") as f:
        ans = json.load(f)
    return ans

def atoms_to_graph(atoms: list[dict], matrix: list[list], cutoff: float|int):
    """
    args:
        - atom: the 4th return from dataset.
            - This is a list of dictionaries where the diction have 'element' and 'abc' keys.
        - matrix: is a 2d list with shape 3X3.
        - cutoff: the threshold for bond generation.
        - curoff_mode: could be "min_length" or "edges_num"
            - for "min_length": the bond length will not larger than ${cutoff}.
            - for "edges_num": there will have ${cutoff} bonds for each atoms.
    return:
        - node_features
        - edges_index
        - edges_weight
        
    """

    def pbc_distance(a, b, matrix):
        """
        args:
            - a, b are the tensors with shape(3),
            - matrix is the tensor with shape(3,3),
        return:
            - TODO
        """
        lat = Lattice(matrix)
        dist, _ = lat.get_distance_and_image(a, b)
        return dist

    num_atoms = len(atoms)
    atoms_list = []
    bonds_list = []
    """
    for atom in atoms:
        atomic_number = atomic_numbers[atom['element']]
        coord = atom['abc'] # This is a 1d list with shape (3, )
        pass
    """
    # Get atoms_list
    for atom in atoms:
        # TODO: add the atom't xyz to atoms_list
        atoms_list.append([
            atomic_numbers[atom["element"]],
            atom["xyz"],
        ])
    # Get bonds_list
    for y in range(num_atoms):
        for x in range(num_atoms):
            dist = pbc_distance(atoms[y]["xyz"], atoms[x]["xyz"], matrix)
            if dist >= 0:
                bonds_list.append([y, x, dist])
    # Here we have atoms_list and bonds_list.
    # Process the cutoff
    # Set all distance < cutoff to 0
    nodes_features = []
    edges_index = [[], []]
    edges_weight = []
    for atom in atoms_list:
        nodes_features.append([atom[0], atom[1][0], atom[1][1], atom[1][2]])
    for b in bonds_list:
        if b[2]<cutoff:
            edges_index[0].append(b[0])
            edges_index[1].append(b[1])
            edges_weight.append(b[2])
    return [nodes_features, edges_index, edges_weight] 
    # return atoms_list, bonds_list


class Dataset(DS):
    def __init__(self, root:str, mode:str="graph", cutoff=1.5, partial=(0.0, 1.0), device='cpu', tensorize_work=False):
        """
        args:
            - mode : should be 'coor' or 'graph', 'coor' is not for training purpose.
            - cutoff: if mode='graph', you should specify the cutoff for bond generation.
            - partial: load part of data into memory.
                - this parameters load the data in range (from, to), where from, to are fraction.
        returns:
        """
        self.energy = []
        self.e_above_hull = []
        self.matrix = []
        self.atoms = []
        self.mode = mode
        self.cutoff = cutoff
        self.device = device
        self.tensorize_work = tensorize_work
        for i in [
            "000"
        ]:  # in each json file you can replace the list here by ['000', '001', '002'], but seems we have no more memory to do this.
            jdict = read_json(
                f"{root}/tmp/Alexandria/alexandria_{i}.json"
            )
            for each_structure in jdict["entries"][int(len(jdict["entries"])*partial[0]):int(len(jdict["entries"])*partial[1])]:  # in each structure
                energy = each_structure["energy"]
                e_above_hull = each_structure["data"]["e_above_hull"]
                # breakpoint()
                matrix = each_structure["structure"]["lattice"]["matrix"]
                atoms = []
                for each_atom in each_structure["structure"]["sites"]:  # in each atom
                    # breakpoint()
                    atoms.append(
                        {
                            "element": each_atom["species"][0]["element"],
                            "xyz": each_atom["xyz"],
                            "abc": each_atom["abc"],
                        }
                    )
                self.energy += [energy]
                self.e_above_hull += [e_above_hull]
                self.matrix += [matrix]
                self.atoms += [atoms]
        return

    def __len__(self) -> int:
        return self.energy.__len__()
    
    def tensorize(self, energy, e_above_hull, matrix, graph):
        '''
        This method will convert alexandria.Dataset graph output into tensor type
        args:
            - TODO
        return:
            - TODO
        '''
        if not self.tensorize_work:
            return energy, e_above_hull, matrix, graph
        # Create the adjacency matrix.
        return (tc.tensor(energy).to(self.device), 
                tc.tensor(e_above_hull).to(self.device), 
                tc.tensor(matrix).to(self.device), 
                tc.tensor(graph[0]).to(self.device).float(), # This is the nodes_features
                tc.tensor(graph[1]).to(self.device), # This is the edge_index
                tc.tensor(graph[2]).to(self.device).float(), # This is the edge_weight
        )
    
    def __getitem__(self, x):
        """
        return: energy, e_above_hull, matrix, atoms
        """
        if self.mode == "coor":
            return self.energy[x], self.e_above_hull[x], self.matrix[x], self.atoms[x]
        if self.mode == "graph":
            # Handel the slice x input.
            if type(x)==slice:
                ans_graph = []
                for each_pair in zip(self.atoms[x], self.matrix[x]):
                    ans_graph.append(
                        atoms_to_graph(*each_pair, cutoff=self.cutoff)
                    )
            else:
                ans_graph = atoms_to_graph(self.atoms[x], self.matrix[x], cutoff=self.cutoff)
            return self.tensorize(
                self.energy[x],
                self.e_above_hull[x],
                self.matrix[x],
                ans_graph,
            )
        # unreachable
        raise Exception('In this dataset, mode should in ["coor", "graph"]')
    pass
    
# %% [markdown]
# # Comming code is just for test purpose.

# %%