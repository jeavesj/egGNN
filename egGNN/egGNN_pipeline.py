#! /usr/bin/env -S python3 -Bu
# coding: utf-8
# @Auth: Jor<qhjiao@mail.sdu.edu.cn>
# @Date: Mon 06 Jun 2022 04:40:33 PM HKT
# @Desc: run egGNN

import os
import torch

from rdkit import Chem
from dgllife.utils import load_molecule

from egGNN.model import EGG
from egGNN.feat import ConstructGraph


CWD = os.path.dirname(os.path.abspath(__file__))


def preprocess(ligand: str, protein: str):
    """construct the graph"""
    constructor = ConstructGraph()
    #  ligand, _ = load_molecule(ligand)
    with open(ligand) as f:
        smiles = f.read().split()[0]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        sdf_path = ligand.replace('.smi', '.sdf')
        mol = Chem.MolFromMolFile(sdf_path, sanitize=False)
        if mol is not None:
            Chem.SanitizeMol(mol, catchErrors=True)
    ligand = mol
    protein, protein_coords = load_molecule(protein)
    ligand_graph = constructor.construct_ligand_graph(ligand, add_self_loop=False, num_virtual_nodes=0)
    protein_graph = constructor.construct_pocket_graph(protein, protein_coords, cutoff=11)

    return ligand_graph, protein_graph


def run_egGNN(args):
    # preprocess
    ligand, protein = preprocess(args.ligand, args.protein)

    # run model
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    model = torch.load(os.path.join(CWD, 'checkpoint/16-8596.sav'), map_location=device)
    for m in model.modules():
        if isinstance(m, torch.nn.GELU) and not hasattr(m, 'approximate'):
            m.approximate = 'none'
        if isinstance(m, torch.nn.ParameterList) and not hasattr(m, '_size'):
            m._size = len(m._parameters)
    model.eval()

    affinity = model((ligand, protein), device)[0].item()
    print(f"the affinity of {args.ligand_name} and {args.protein_name} is {affinity:.2f}")
    return affinity
