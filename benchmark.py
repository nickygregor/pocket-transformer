"""Measure local median decode times. Results depend on hardware and workload."""
import argparse,json,statistics,subprocess
p=argparse.ArgumentParser();p.add_argument('models',nargs='+');p.add_argument('--binary',default='./build/pocket');p.add_argument('--runs',type=int,default=5);a=p.parse_args()
if a.runs<1:p.error('runs must be positive')
results=[]
for model in a.models:
    texts=[]
    for cached in (True,False):
        command=[a.binary,'--model',model,'--prompt','the ','--steps','64']+([] if cached else ['--no-cache'])
        subprocess.run(command,check=True,capture_output=True) # warm-up
        runs=[json.loads(subprocess.check_output(command)) for _ in range(a.runs)]
        texts.append(runs[0]['text'])
        results.append({'model':model,'cached':cached,'median_decode_ms':statistics.median(r['decode_ms'] for r in runs),'weight_bytes':runs[0]['weight_bytes'],'runs':a.runs})
    if texts[0]!=texts[1]:raise RuntimeError('Cached and uncached greedy outputs differ')
print(json.dumps(results,indent=2))
