#!/usr/bin/env python3
"""CPU-only structure-SSIM proxy audit; NOT the authors' exact mSSIM."""
import json
import numpy as np
from scipy.ndimage import gaussian_filter, shift

def score(a,b):
    a=a.ravel()-a.mean(); b=b.ravel()-b.mean()
    return float((np.mean(a*b)+1e-5)/(np.sqrt(np.mean(a*a)*np.mean(b*b))+1e-5))

def patch_conf(a,b):
    return np.array([[score(a[y:y+16,x:x+16],b[y:y+16,x:x+16])
        for x in range(0,128,16)] for y in range(0,128,16)])

results={}
for dy in (0,1,2,3):
    rates=[]
    for seed in range(12):
        rng=np.random.default_rng(seed)
        a=gaussian_filter(rng.normal(size=(128,128)),1.5)
        a=(a-a.min())/(a.max()-a.min())
        b=shift(a,(dy,0),order=1,mode="nearest")
        rates.append(float(np.mean(patch_conf(a,b)>=0.8)))
    results[f"static_pose_{dy}px"]=float(np.mean(rates))
rates=[]
for seed in range(12):
    rng=np.random.default_rng(seed)
    a=gaussian_filter(rng.normal(size=(128,128)),1.5)
    a=(a-a.min())/(a.max()-a.min())
    b=a.copy(); b[48:80,48:80]=b[48:80,48:80]*0.6+0.3
    rates.append(float(np.mean(patch_conf(a,b)[3:5,3:5]>=0.8)))
results["appearance_only_changed_region_false_static"]=float(np.mean(rates))
print(json.dumps({"method":"structure-only proxy; not exact Cross-Temporal mSSIM","seeds":12,"threshold":0.8,"results":results},indent=2))
