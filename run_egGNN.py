#!/usr/bin/env -S python3 -Bu
# coding: utf-8
# @Auth: Jor<qhjiao@mail.sdu.edu.cn>
# @Date: Tue 07 Jun 2022 09:10:48 AM HKT
# @Desc: ligand protein binding affinity prediction

import os
import argparse

from egGNN.model import *
from egGNN.egGNN_pipeline import run_egGNN
import time

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ligand', type=str, help='the ligand path (absolute)')
    parser.add_argument('--protein', type=str, help='the protein path (absolute)')
    parser.add_argument('--gpu', type=int, default=0, help='the gpu num you use')
    parser.add_argument('--out_csv', type=str, required=True, help='Path to write predictions and times')

    args = parser.parse_args()
    return args


def already_done(args):
    if not os.path.exists(args.out_csv):
        return False
    pdbid = args.ligand_name.split('_')[0]
    with open(args.out_csv) as f:
        for line in f:
            if line.startswith(pdbid + ','):
                return True
    return False


def run(args):
    if already_done(args):
        print(f"Skipping {args.ligand_name}, already in {args.out_csv}")
        return
    t0 = time.time()
    affinity = run_egGNN(args)
    t1 = time.time() - t0
    
    if not os.path.exists(args.out_csv):
        with open(args.out_csv, 'w') as f:
            f.write(f'pdbid,pK_predicted,t_tot_s\n')
        
    with open(args.out_csv, 'a') as f:
        f.write(f"{args.ligand_name.split('_')[0]},{affinity},{t1}\n")


if __name__ == '__main__':
    args = get_parser()
    args.ligand_name = os.path.basename(args.ligand).split('.')[0]
    args.protein_name = os.path.basename(args.protein).split('.')[0]
    run(args)
