# Edge-Gated Graph Neural Network for Predicting Protein-Ligand Binding Affinities

```
Authors: Qihong Jiao, Zongzhao Qiu, Yuxiao Wang, Cheng Chen, Zhenghe Yang, Xuefeng Cui*
    - *: To whom correspondence should be addressed.
Contact: xfcui@email.sdu.edu.cn
Publish: https://ieeexplore.ieee.org/document/9669846
```


## Usage

Clone this repository by:
```bash
git clone https://github.com/xfcui/egGNN.git
```

Install packages by:
```bash
pip install -r requirements.txt
```

NOTE:  
To generate smiles from SDFs, [DEAttentionDTA](https://github.com/whatamazing1/DEAttentionDTA) provides a great script in `pre-code/sdf_to_smi.py`!

Run:
```bash
./run_ligpro.py --ligand SMILES --protein **.pdb --gpu GPU_NUM
```
