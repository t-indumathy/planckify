import sys
sys.path.insert(0, '/mnt/d/code/planckify/experiments/gemma4_e2b_onnx')
from run_inference_cpu import DEFAULT_MODEL_DIR
resolved = DEFAULT_MODEL_DIR.resolve()
print('DEFAULT_MODEL_DIR:', resolved)
print('Exists:', resolved.exists())
