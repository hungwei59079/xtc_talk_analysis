from lgdo import lh5
from xtc_utils import XTCMatrix
import argparse

xtc_file = '/global/u2/h/hungwei/legend-dataflow-new/inputs/dataprod/overrides/evt/xtc/p08/r015/l200-p08-r015-xtc-T%-par_evt_xtc.lh5'
parser=argparse.ArgumentParser()
parser.add_argument('--xtc_file', type=str, default=xtc_file, help='Path to the XTC file')
args = parser.parse_args()

loaded_matrix = lh5.read("/xtc/xtalk_matrix_negative", args.xtc_file).nda * 100
print(loaded_matrix)
print(f"Loaded matrix shape: {loaded_matrix.shape}")

if loaded_matrix.shape[0] != loaded_matrix.shape[1]:
    raise ValueError("Loaded matrix is not square. Please check the XTC file.")

xtc_matrix = XTCMatrix(loaded_matrix.shape[0], 'neg', matrix=loaded_matrix)
xtc_matrix.plot("results/", filename="loaded_xtalk_matrix.png")
