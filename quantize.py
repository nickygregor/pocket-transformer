import argparse
from pathlib import Path
from model import load,save
p=argparse.ArgumentParser(description='Convert float32 model weights to per-row symmetric int8.')
p.add_argument('source');p.add_argument('destination');a=p.parse_args()
if Path(a.source).resolve()==Path(a.destination).resolve():p.error('Choose a different output file')
save(a.destination,load(a.source),quantized=True)
print(f'{Path(a.source).stat().st_size:,} → {Path(a.destination).stat().st_size:,} bytes')
