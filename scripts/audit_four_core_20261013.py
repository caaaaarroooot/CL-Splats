#!/usr/bin/env python3
"""Four causal *synthetic* audits; not CL-Splats GPU or real-scene benchmarks.
The optional toy 100-step test has oracle point correspondences and is NOT gsplat.
Use scripts/audit_depth_pipeline.py on the real machine for HF/device timings.
"""
import argparse, json, math, platform, time
import numpy as np

def depth_audit(seed):
    rng=np.random.default_rng(seed)
    h,w=120,160
    yy,xx=np.mgrid[0:h,0:w]
    mono=(0.48+0.12*xx/(w-1)+0.05*yy/(h-1)).astype('float64')
    true_disp=0.16*mono+0.11
    rendered=1/true_disp
    anchor=rng.random((h,w))>0.15
    anchor[35:80,55:105]=False
    train=anchor & (xx<120); val=anchor & (xx>=120)
    def fit(x):
        a,b=np.linalg.lstsq(np.stack([x[train],np.ones(train.sum())],axis=1),1/rendered[train],rcond=None)[0]
        z=1/np.maximum(a*x+b,1e-6)
        return {'heldout_depth_mae_m':round(float(np.abs(z[val]-rendered[val]).mean()),8),
                'fit_a':round(float(a),7),'fit_b':round(float(b),7),
                'distinct_values':int(np.unique(x).size)}
    q=np.round((mono-mono.min())/(mono.max()-mono.min())*255)/255
    return {'float':fit(mono),'pil_uint8_normalized':fit(q),
            'quantization_levels':int(np.unique(q).size),
            'raw_float_levels':int(np.unique(mono).size),
            'warning':'Synthetic quantization only; actual Depth Anything output and runtime not tested.'}

def evidence_audit():
    pos=np.array([[1,1,0,0,0],[1,1,0,0,0],[0,0,0,0,0]],dtype=bool)
    neg=np.array([[1,1,1,1,1],[0,1,1,0,0],[1,1,1,1,1]],dtype=bool)
    old_vis=pos.sum(1)+neg.sum(1)
    true_vis=np.logical_or(pos,neg).sum(1)
    old_ratio=pos.sum(1)/np.maximum(old_vis,1)
    union_ratio=pos.sum(1)/np.maximum(true_vis,1)
    neg_depth=np.array([[1,0,0,0,0],[0,1,0,0,0],[1,0,0,0,0]],dtype=bool)
    neg_valid=neg & neg_depth
    gated_vis=np.logical_or(pos,neg_valid).sum(1)
    gated_ratio=pos.sum(1)/np.maximum(gated_vis,1)
    def classify(v,r):return ((v>=2)&(pos.sum(1)>=2)&(r>=0.3)).tolist()
    return {'positive_counts':pos.sum(1).tolist(),'negative_counts':neg.sum(1).tolist(),
            'same_camera_pos_neg_overlap':np.logical_and(pos,neg).sum(1).tolist(),
            'old_visible_views':old_vis.tolist(),'union_visible_views':true_vis.tolist(),
            'old_positive_ratio':old_ratio.round(4).tolist(),
            'union_positive_ratio':union_ratio.round(4).tolist(),
            'old_gate_pass':classify(old_vis,old_ratio),
            'union_gate_pass':classify(true_vis,union_ratio),
            'gated_negative_counts':neg_valid.sum(1).tolist(),
            'union_depth_gated_positive_ratio':gated_ratio.round(4).tolist(),
            'union_depth_gated_pass':classify(gated_vis,gated_ratio),
            'note':'Deterministic constructed evidence matrix; not observed dataset rates.'}

def support_audit(seed):
    rng=np.random.default_rng(seed)
    old_grid=np.stack(np.meshgrid(np.linspace(-1,1,21),np.linspace(-1,1,21)),axis=-1).reshape(-1,2)
    cases={'ADD':(5.,3.),'DELETE':(3.,5.),'APPEARANCE':(3.,3.),
           'SMALL_GEOMETRY':(3.,3.08),'AMBIGUOUS_POSE_DEPTH':(3.,3.35)}
    rows={}
    for name,(z_old,z_new) in cases.items():
        old=np.column_stack([old_grid,np.full(len(old_grid),z_old)])
        q=np.column_stack([rng.uniform(-0.8,0.8,256),rng.uniform(-0.8,0.8,256),np.full(256,z_new)])
        ds=np.linalg.norm(q[:,None,:]-old[None,:,:],axis=-1)
        idx=np.argpartition(ds,7,axis=1)[:,:8]
        d=np.take_along_axis(ds,idx,axis=1)
        r=d/(math.sqrt(0.06**2*2+0.025**2)+1e-6)<2.5
        depth=np.abs(z_old-z_new)<(0.05+0.05*z_new)
        rows[name]={'old_depth_m':z_old,'new_depth_m':z_new,
                    'depth_residual_m':round(z_new-z_old,3),
                    'radius_any_rate':round(float(r.any(1).mean()),4),
                    'depth_any_rate':float(depth),
                    'joint_any_rate':round(float((r & depth).any(1).mean()),4),
                    'pixel_count':256}
    return rows

def seed_factorial(seed,steps):
    try: import torch
    except ImportError: return {'status':'SKIPPED: torch unavailable'}
    torch.set_num_threads(1);torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    nx,ny=6,4
    x,y=np.meshgrid(np.linspace(-0.55,0.55,nx),np.linspace(-0.3,0.3,ny))
    xyz=np.column_stack([x.ravel(),y.ravel(),np.full(nx*ny,3.)])
    gt=torch.tensor(xyz,dtype=torch.float64)
    translations=torch.tensor([-0.55,0,0.55],dtype=torch.float64)
    focal=40.
    def project(P):
        u=focal*(P[None,:,0]-translations[:,None])/P[None,:,2]
        v=focal*P[None,:,1]/P[None,:,2]+0*translations[:,None]
        return torch.stack([u,v],-1)
    target=project(gt)
    uv0=target[1].detach().numpy()
    eps=rng.standard_normal(len(xyz))
    arms={'S1_random_old_depth':(5.0,0.3),'S2_float_depth_seed':(3.25,0.15),
          'S3_ratio_seed':(3.15,0.1),'S4_triangulated_seed':(3.05,0.03)}
    def metrics(P):
        return (float(((project(P)-target)**2).mean().sqrt()),
                float(((P-gt)**2).sum(1).sqrt().mean()))
    old=gt.clone();old[:,2]=5.
    old[:,0]=torch.tensor(uv0[:,0]/focal*5.)
    old[:,1]=torch.tensor(uv0[:,1]/focal*5.)
    rows={'S0_no_birth_old_background_fixed':{'new_count':0,'iterations':0,
          'reprojection_rmse_px':round(metrics(old)[0],5),
          'geometry_mean_l2_m':round(metrics(old)[1],5)}}
    for name,(mean,std) in arms.items():
        z=np.maximum(1.,mean+std*eps)
        init=np.column_stack([uv0[:,0]/focal*z,uv0[:,1]/focal*z,z])
        P=torch.nn.Parameter(torch.tensor(init,dtype=torch.float64))
        opt=torch.optim.Adam([P],lr=0.01)
        before=metrics(P.detach());t0=time.perf_counter()
        for _ in range(steps):
            opt.zero_grad()
            loss=((project(P)-target)**2).mean()
            loss.backward();opt.step()
        after=metrics(P.detach())
        rows[name]={'new_count':len(xyz),'iterations':steps,
                   'init_reprojection_rmse_px':round(before[0],5),
                   'final_reprojection_rmse_px':round(after[0],5),
                   'init_geometry_mean_l2_m':round(before[1],5),
                   'final_geometry_mean_l2_m':round(after[1],5),
                   'optimization_seconds_cpu':round(time.perf_counter()-t0,5)}
    return {'status':'CPU torch toy; oracle correspondences, no gsplat',
            'optimizer':'Adam(lr=0.01), 3 synthetic cameras, 24 matched points',
            'growth':'disabled','random_seed':seed,'arms':rows}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--steps',type=int,default=100)
    a=p.parse_args()
    out={'scope':'SYNTHETIC_CPU_ONLY_NOT_REAL_CL_SPLATS_GPU','seed':a.seed,
         'runtime':{'python':platform.python_version(),'numpy':np.__version__},
         'float_depth_quantization':depth_audit(a.seed),
         'evidence_visibility':evidence_audit(),
         'novel_space_support':support_audit(a.seed),
         'equal_count_100step_toy':seed_factorial(a.seed,a.steps)}
    print(json.dumps(out,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
