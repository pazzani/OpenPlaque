from __future__ import annotations

import gc
import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

BASELINE = '0593b453959f5a353d644267fbeef24b514ef4d7'
ALGORITHM = 'lcx-local-av-junction-ribbon-v1.0-lowmem'
OUTPUT_DIRNAME = 'LCX_Local_AV_Junction_Ribbon_v1'
SOURCE_CACHE = Path('Cache/Secondary_3D_Vesselness_Topology_v1')
MASTER = Path('Cache/Master_Coronary_Anatomy_Baseline_v2/master_anatomy_summary.json')
LAD_PATH = Path('Cache/LAD_Frozen_Proximal_Reacquisition_v1/combined_lad_centerline.csv')
RCA_PATH = Path('PCAT_RCA_10_50/rca_centerline_smoothed_zyx.csv')
TS = Path('TotalSegmentator_Cardiovascular_Cache_v1')
HEART = TS / 'heartchambers_highres'
COR_CURRENT = TS / 'coronary_arteries/coronary_arteries.nii.gz'
COR_LEGACY = TS / 'coronary_arteries_LEGACY/coronary_arteries.nii.gz'
LA = HEART / 'heart_atrium_left.nii.gz'
LV = HEART / 'heart_ventricle_left.nii.gz'
RA = HEART / 'heart_atrium_right.nii.gz'
RV = HEART / 'heart_ventricle_right.nii.gz'
MYO = HEART / 'heart_myocardium.nii.gz'
PRIOR = Path('Joint_Three_Vessel_Template_Classifier_v1')
PRIOR_RANKING = PRIOR / 'LCX_joint_candidate_ranking.csv'
LEAF_FILES = [PRIOR / f'candidate_{i:02d}_source_path.csv' for i in range(1, 6)]

C6_ID, C7_ID, C9_ID = 6, 7, 9
STATUS_TOPOLOGY_FAIL = 'LOCAL_AV_JUNCTION_TOPOLOGY_PREREQUISITE_FAILED'
STATUS_CONTROL_FAIL = 'LOCAL_AV_JUNCTION_CONTROL_FAILED'
STATUS_AMBIG = 'LCX_VS_OM_LOCAL_JUNCTION_AMBIGUOUS'
STATUS_C6 = 'C6_LCX_C7_OM_LOCAL_JUNCTION_CANDIDATES_REQUIRE_VISUAL_QC'
STATUS_C7 = 'C7_LCX_C6_OM_LOCAL_JUNCTION_CANDIDATES_REQUIRE_VISUAL_QC'


def _req(p):
    p = Path(p)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return p


def _write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=True, default=str), encoding='utf-8')


@dataclass(frozen=True)
class Geometry:
    origin: np.ndarray
    spacing: np.ndarray
    direction: np.ndarray

    def xyz_to_zyx(self, pts):
        pts = np.atleast_2d(np.asarray(pts, float))
        idx = ((pts - self.origin) @ np.linalg.inv(self.direction).T) / self.spacing
        return idx[:, ::-1]

    def zyx_to_xyz(self, pts):
        idx = np.atleast_2d(np.asarray(pts, float))[:, ::-1]
        return self.origin + (idx * self.spacing) @ self.direction.T


def _source(cache):
    arr = np.load(_req(cache / 'series7_int16.npy'), mmap_mode='r')
    m = json.loads(_req(cache / 'series7_int16.json').read_text())
    sp = np.asarray(m['spacing_zyx'], float)[::-1]
    iop = np.asarray(m['image_orientation_patient'], float)
    row, col = iop[:3], iop[3:]
    slc = np.cross(row, col)
    d = np.array([[row[0], col[0], slc[0]], [row[1], col[1], slc[1]], [row[2], col[2], slc[2]]], float)
    return Geometry(np.asarray(m['positions_lps_mm'][0], float), sp, d), arr


def _sample(arr, g, pts, order=1, cval=-1024.0):
    return map_coordinates(arr, g.xyz_to_zyx(pts).T, order=order, mode='constant', cval=cval)


def _load_path(path, g):
    d = pd.read_csv(_req(path))
    for cols in [('lps_x_mm','lps_y_mm','lps_z_mm'), ('x_mm','y_mm','z_mm')]:
        if all(c in d.columns for c in cols):
            return d[list(cols)].to_numpy(float)
    for cols in [('zyx_z','zyx_y','zyx_x'), ('source_z','source_y','source_x'), ('z','y','x')]:
        if all(c in d.columns for c in cols):
            return g.zyx_to_xyz(d[list(cols)].to_numpy(float))
    raise ValueError(f'Unrecognized path columns in {path}: {list(d.columns)}')


def _arc(p):
    p = np.asarray(p, float)
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))] if len(p) > 1 else np.zeros(len(p))


def _interp(p, q):
    p = np.asarray(p, float); a = _arc(p); q = np.asarray(q, float)
    return np.column_stack([np.interp(q, a, p[:, k]) for k in range(3)])


def _resample(p, step=.25):
    a = _arc(p)
    if len(a) < 2 or a[-1] <= 0:
        return np.asarray(p, float), a
    q = np.arange(0, a[-1] + 1e-9, step)
    if q[-1] < a[-1] - 1e-6:
        q = np.r_[q, a[-1]]
    return _interp(p, q), q


def _orient(paths):
    ref = np.asarray(paths[0], float)[0]
    out = []
    for p in paths:
        p = np.asarray(p, float)
        if np.linalg.norm(p[-1] - ref) < np.linalg.norm(p[0] - ref):
            p = p[::-1].copy()
        out.append(p)
    return out


def _consensus_prefix(paths, step=.25, tol=.60, sustain=4):
    paths = _orient(paths)
    minlen = min(_arc(p)[-1] for p in paths)
    q = np.arange(0, minlen + 1e-9, step)
    s = np.stack([_interp(p, q) for p in paths], axis=0)
    centroid = np.median(s, axis=0)
    dev = np.max(np.linalg.norm(s - centroid[None, :, :], axis=2), axis=0)
    first_bad = None
    start = max(1, int(round(5.0 / step)))
    for i in range(start, max(start, len(q) - sustain + 1)):
        if np.all(dev[i:i+sustain] > tol):
            first_bad = i; break
    end_i = first_bad - 1 if first_bad is not None else len(q) - 1
    return centroid[:end_i+1], q[:end_i+1], dev[:end_i+1], None if first_bad is None else float(q[first_bad])


def _truncation(long_path, short_path, step=.25):
    long_path, short_path = _orient([long_path, short_path])
    L, S = _arc(long_path)[-1], _arc(short_path)[-1]
    q = np.arange(0, S + 1e-9, step)
    if q[-1] < S - 1e-6: q = np.r_[q, S]
    d = np.linalg.norm(_interp(long_path, q) - _interp(short_path, q), axis=1)
    return {
        'C6_length_mm': float(L), 'C9_length_mm': float(S), 'C6_extension_beyond_C9_mm': float(L-S),
        'p95_prefix_separation_mm': float(np.percentile(d,95)), 'max_prefix_separation_mm': float(np.max(d)),
        'truncation_gate_pass': bool((L-S) >= 2.0 and np.percentile(d,95) <= .35 and np.max(d) <= .60)
    }


def _img_zyx_to_xyz(im, pts):
    idx = np.atleast_2d(np.asarray(pts, float))[:, ::-1]
    o = np.asarray(im.GetOrigin()); sp = np.asarray(im.GetSpacing()); d = np.asarray(im.GetDirection()).reshape(3,3)
    return o + (idx * sp) @ d.T


def _surface_tree(path, max_points=300000):
    im = sitk.ReadImage(str(_req(path)))
    surf = sitk.LabelContour(sitk.Cast(im > 0, sitk.sitkUInt8), False)
    a = sitk.GetArrayViewFromImage(surf)
    z = np.argwhere(a > 0)
    if not len(z): raise ValueError(f'No foreground in {path}')
    if len(z) > max_points: z = z[::int(math.ceil(len(z)/max_points))]
    xyz = _img_zyx_to_xyz(surf, z).astype(np.float32, copy=False)
    del z, a, surf, im; gc.collect()
    return cKDTree(xyz)


def _mask_tree(path, max_points=350000):
    im = sitk.ReadImage(str(_req(path)))
    a = sitk.GetArrayViewFromImage(im)
    z = np.argwhere(a > 0)
    if not len(z): raise ValueError(f'No foreground in {path}')
    if len(z) > max_points: z = z[::int(math.ceil(len(z)/max_points))]
    xyz = _img_zyx_to_xyz(im, z).astype(np.float32, copy=False)
    del z, a, im; gc.collect()
    return cKDTree(xyz)


def _local_ribbon(anchor, atrium_tree, ventricle_tree, myocardium_tree, radius=20.0):
    ids = myocardium_tree.query_ball_point(np.asarray(anchor,float), radius)
    pts = np.asarray(myocardium_tree.data[np.asarray(ids,dtype=int)], float)
    if len(pts) < 100: raise RuntimeError(f'Only {len(pts)} local myocardial surface points near anchor')
    da = atrium_tree.query(pts)[0]; dv = ventricle_tree.query(pts)[0]
    chosen = None; used = None
    for balance, prox in [(2.5,7.0),(3.5,8.5),(5.0,10.0),(6.5,12.0)]:
        m = (np.abs(da-dv) <= balance) & (np.maximum(da,dv) <= prox)
        if m.sum() >= 80:
            chosen = pts[m]; used = (balance, prox); break
    if chosen is None:
        m = np.argsort(np.abs(da-dv) + .35*np.maximum(da,dv))[:min(200,len(pts))]
        chosen = pts[m]; used = (float(np.max(np.abs(da[m]-dv[m]))), float(np.max(np.maximum(da[m],dv[m]))))
    # Voxel-like deduplication stabilizes local PCA and limits tree size.
    key = np.round(chosen / .7).astype(int)
    _, ix = np.unique(key, axis=0, return_index=True)
    chosen = chosen[np.sort(ix)]
    tree = cKDTree(chosen.astype(np.float32, copy=False))
    return tree, {
        'anchor_lps_mm': [float(x) for x in anchor], 'radius_mm': float(radius),
        'n_local_myocardium_points': int(len(pts)), 'n_ribbon_points': int(len(chosen)),
        'balance_cut_mm': float(used[0]), 'proximity_cut_mm': float(used[1])
    }


def _tangents(p):
    p = np.asarray(p,float)
    if len(p) < 2: return np.zeros_like(p)
    d = np.gradient(p, axis=0); n = np.linalg.norm(d,axis=1); n[n<1e-9]=1
    return d/n[:,None]


def _local_ribbon_tangent(ribbon_tree, query_point, k=30):
    kk = min(k, len(ribbon_tree.data))
    _, idx = ribbon_tree.query(np.asarray(query_point,float), k=kk)
    hood = np.atleast_2d(np.asarray(ribbon_tree.data[idx],float))
    hood = hood - hood.mean(axis=0)
    if len(hood) < 3: return np.zeros(3)
    _, _, vh = np.linalg.svd(hood, full_matrices=False)
    v = vh[0]; n = np.linalg.norm(v)
    return v/n if n > 1e-9 else np.zeros(3)


def _slope(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float); m=np.isfinite(x)&np.isfinite(y)
    return 0.0 if m.sum()<3 or np.ptp(x[m])<1e-6 else float(np.polyfit(x[m],y[m],1)[0])


def _score_against_ribbon(path, ribbon_tree, atrium_tree, ventricle_tree, max_mm=8.0):
    p,q = _resample(path,.25)
    keep = q <= min(q[-1],max_mm)+1e-9; p,q = p[keep],q[keep]
    d, _ = ribbon_tree.query(p)
    t = _tangents(p)
    aligns=[]
    for x,v in zip(p,t):
        r = _local_ribbon_tangent(ribbon_tree,x)
        aligns.append(abs(float(np.dot(v,r))) if np.linalg.norm(r)>0 else 0.0)
    aligns=np.asarray(aligns,float)
    da=atrium_tree.query(p)[0]; dv=ventricle_tree.query(p)[0]
    med_d=float(np.median(d)); end_d=float(d[-1]); sl=_slope(q,d); align=float(np.median(aligns)); bal=float(np.median(np.abs(da-dv)))
    closeness=float(np.exp(-med_d/6.0)); retention=float(np.exp(-max(0.0,sl)/.45)); balance=float(np.exp(-bal/5.0))
    score=float(.40*closeness + .30*align + .15*retention + .15*balance)
    return {
        'junction_identity_score':score,'median_ribbon_distance_mm':med_d,'p90_ribbon_distance_mm':float(np.percentile(d,90)),
        'endpoint_ribbon_distance_mm':end_d,'ribbon_distance_slope_mm_per_mm':sl,'junction_retention_score':retention,
        'median_ribbon_tangent_alignment':align,'median_abs_atrium_minus_ventricle_distance_mm':bal,
        'profile_arc_mm':q,'profile_ribbon_distance_mm':d,'profile_alignment':aligns,'points':p
    }


def _slice_from_arc(path,start_arc,max_mm=8.0):
    total=_arc(path)[-1]; q=np.arange(start_arc,min(total,start_arc+max_mm)+1e-9,.25)
    if len(q)<2: raise ValueError('Insufficient path after anchor')
    return _interp(path,q)


def _support(tree,path):
    p,_=_resample(path,.25); return float(np.mean(tree.query(p)[0] <= 1.0))


def _hu(src,g,path):
    p,_=_resample(path,.25); h=_sample(src,g,p)
    return float(np.mean((h>=120)&(h<=1200))), float(np.median(h))


def _plane(g,src,center,tangent,half=7.,step=.2):
    t=tangent/max(np.linalg.norm(tangent),1e-9); axes=np.eye(3); seed=axes[np.argmin(np.abs(axes@t))]
    n=np.cross(t,seed); n/=max(np.linalg.norm(n),1e-9); b=np.cross(t,n); b/=max(np.linalg.norm(b),1e-9)
    q=np.arange(-half,half+1e-9,step); yy,xx=np.meshgrid(q,q,indexing='ij'); pts=center[None,None,:]+xx[...,None]*n+yy[...,None]*b
    return _sample(src,g,pts.reshape(-1,3)).reshape(len(q),len(q)),q


def synthetic_local_junction_self_test():
    x=np.linspace(0,10,101); ribbon=np.column_stack([x,np.zeros_like(x),np.zeros_like(x)]); rt=cKDTree(ribbon)
    good=np.column_stack([x,np.full_like(x,.5),np.zeros_like(x)]); bad=np.column_stack([x,2+.5*x,np.zeros_like(x)])
    # fake chamber surfaces symmetric about ribbon
    at=cKDTree(np.vstack([ribbon+[0,2,0],ribbon+[0,-2,0]])); vt=cKDTree(np.vstack([ribbon+[0,2,1],ribbon+[0,-2,1]]))
    g=_score_against_ribbon(good,rt,at,vt,8); b=_score_against_ribbon(bad,rt,at,vt,8)
    assert g['median_ribbon_distance_mm'] < b['median_ribbon_distance_mm']
    return {'ok':True,'good_distance':g['median_ribbon_distance_mm'],'bad_distance':b['median_ribbon_distance_mm']}


def run(drive_root='/content/drive/MyDrive/OpenPlaque', output_dir=None):
    root=Path(drive_root); out=Path(output_dir) if output_dir else root/OUTPUT_DIRNAME; out.mkdir(parents=True,exist_ok=True)
    _write_json(out/'run_state.json',{'status':'STARTED','algorithm':ALGORITHM,'baseline_commit':BASELINE})
    required=[root/SOURCE_CACHE/'series7_int16.npy',root/SOURCE_CACHE/'series7_int16.json',root/MASTER,root/LAD_PATH,root/RCA_PATH,root/COR_CURRENT,root/COR_LEGACY,root/LA,root/LV,root/RA,root/RV,root/MYO,root/PRIOR_RANKING]+[root/p for p in LEAF_FILES]
    for p in required:_req(p)

    master=json.loads((root/MASTER).read_text()); g,src=_source(root/SOURCE_CACHE); lad=_load_path(root/LAD_PATH,g); rca=_load_path(root/RCA_PATH,g)
    leaves=_orient([_load_path(root/p,g) for p in LEAF_FILES]); ranking=pd.read_csv(root/PRIOR_RANKING)
    mapping={}
    for i in range(5): mapping[int(ranking.iloc[i]['candidate_id'])]=leaves[i]
    for cid in (C6_ID,C7_ID,C9_ID):
        if cid not in mapping: raise RuntimeError(f'Missing source candidate C{cid}')
    c6,c7,c9=mapping[6],mapping[7],mapping[9]
    f2con,f2q,f2dev,firstsplit=_consensus_prefix([c6,c7,c9],.25,.60,4); split_arc=float(f2q[-1]); trunc=_truncation(c6,c9)
    topo_pass=bool(trunc['truncation_gate_pass'] and 19.0 <= split_arc <= 21.0)

    print('Building sparse cardiac and coronary trees...')
    la=_surface_tree(root/LA); lv=_surface_tree(root/LV); ra=_surface_tree(root/RA); rv=_surface_tree(root/RV); myo=_surface_tree(root/MYO)
    cur=_mask_tree(root/COR_CURRENT); leg=_mask_tree(root/COR_LEGACY)

    left_anchor=_interp(c6,[split_arc])[0]
    left_ribbon,left_build=_local_ribbon(left_anchor,la,lv,myo,20.0)
    # RCA control anchor at 30 mm, well inside the accepted 10-50 mm AV-groove segment.
    rca_total=_arc(rca)[-1]; rca_anchor_arc=min(30.0,max(8.0,0.58*rca_total)); right_anchor=_interp(rca,[rca_anchor_arc])[0]
    right_ribbon,right_build=_local_ribbon(right_anchor,ra,rv,myo,20.0)
    _write_json(out/'local_junction_ribbon_build_summary.json',{'left':left_build,'right':right_build,'split_arc_mm':split_arc,'rca_anchor_arc_mm':rca_anchor_arc})

    rca_seg=_slice_from_arc(rca,max(0.0,rca_anchor_arc-3.0),10.0)
    rca_m=_score_against_ribbon(rca_seg,right_ribbon,ra,rv,10.0)
    # LAD negative: use the LAD segment spatially closest to the left split anchor.
    lad_rs,lad_q=_resample(lad,.25); j=int(np.argmin(np.linalg.norm(lad_rs-left_anchor[None,:],axis=1))); lad_start=max(0.0,float(lad_q[j])-3.0); lad_seg=_slice_from_arc(lad,lad_start,6.0)
    lad_m=_score_against_ribbon(lad_seg,left_ribbon,la,lv,6.0)
    control_margin=float(rca_m['junction_identity_score']-lad_m['junction_identity_score'])
    control_pass=bool(rca_m['median_ribbon_distance_mm'] <= 8.0 and rca_m['median_ribbon_tangent_alignment'] >= .45 and control_margin >= .08)
    controls={
        'RCA_right_local_junction_score':rca_m['junction_identity_score'],'RCA_median_ribbon_distance_mm':rca_m['median_ribbon_distance_mm'],'RCA_median_tangent_alignment':rca_m['median_ribbon_tangent_alignment'],
        'LAD_left_local_negative_score':lad_m['junction_identity_score'],'LAD_median_ribbon_distance_mm':lad_m['median_ribbon_distance_mm'],'LAD_median_tangent_alignment':lad_m['median_ribbon_tangent_alignment'],
        'control_margin':control_margin,'control_pass':control_pass
    }
    _write_json(out/'local_junction_controls.json',controls)

    c6_post=_slice_from_arc(c6,split_arc,8.0); c7_post=_slice_from_arc(c7,split_arc,8.0)
    m6=_score_against_ribbon(c6_post,left_ribbon,la,lv,8.0); m7=_score_against_ribbon(c7_post,left_ribbon,la,lv,8.0)
    for cid,p,m in [(6,c6_post,m6),(7,c7_post,m7)]:
        m['current_support_fraction']=_support(cur,p); m['legacy_support_fraction']=_support(leg,p); m['robust_hu_fraction'],m['median_hu']=_hu(src,g,p)
    rows=[]
    for cid,m in [(6,m6),(7,m7)]:
        rows.append({k:v for k,v in {'source_candidate_id':cid,**m}.items() if not isinstance(v,np.ndarray)})
    pd.DataFrame(rows).to_csv(out/'C6_C7_local_junction_identity_scores.csv',index=False)

    win_id=6 if m6['junction_identity_score']>=m7['junction_identity_score'] else 7; lose_id=7 if win_id==6 else 6; win=m6 if win_id==6 else m7; lose=m7 if win_id==6 else m6
    score_margin=float(win['junction_identity_score']-lose['junction_identity_score']); dist_adv=float(lose['median_ribbon_distance_mm']-win['median_ribbon_distance_mm']); align_adv=float(win['median_ribbon_tangent_alignment']-lose['median_ribbon_tangent_alignment']); end_adv=float(lose['endpoint_ribbon_distance_mm']-win['endpoint_ribbon_distance_mm']); slope_adv=float(lose['ribbon_distance_slope_mm_per_mm']-win['ribbon_distance_slope_mm_per_mm'])
    support_pass=bool(min(win['current_support_fraction'],win['legacy_support_fraction'],win['robust_hu_fraction'],lose['current_support_fraction'],lose['legacy_support_fraction'],lose['robust_hu_fraction'])>=.90)
    identity_gate=bool(topo_pass and control_pass and support_pass and score_margin>=.08 and win['junction_retention_score']>=.55 and (dist_adv>=1.5 or align_adv>=.15) and (slope_adv>=.20 or end_adv>=2.0))
    decision={'winner_source_candidate_id':win_id,'loser_source_candidate_id':lose_id,'score_margin':score_margin,'median_ribbon_distance_advantage_mm':dist_adv,'tangent_alignment_advantage':align_adv,'departure_slope_advantage_mm_per_mm':slope_adv,'endpoint_ribbon_distance_advantage_mm':end_adv,'dual_mask_and_HU_support_pass':support_pass,'identity_gate_pass':identity_gate}
    _write_json(out/'LCX_vs_OM_local_junction_decision.json',decision)

    # QC plots
    plt.figure(figsize=(6,4)); plt.bar(['C6','C7'],[m6['junction_identity_score'],m7['junction_identity_score']]); plt.ylabel('local junction identity score'); plt.title(f'Local AV-junction ribbon | control pass={control_pass}'); plt.tight_layout(); plt.savefig(out/'01_local_junction_scores.png',dpi=180); plt.close()
    plt.figure(figsize=(7,4)); plt.plot(m6['profile_arc_mm'],m6['profile_ribbon_distance_mm'],label='C6'); plt.plot(m7['profile_arc_mm'],m7['profile_ribbon_distance_mm'],label='C7'); plt.xlabel('downstream arc (mm)'); plt.ylabel('distance to local AV-junction ribbon (mm)'); plt.legend(); plt.tight_layout(); plt.savefig(out/'02_local_junction_distance_profiles.png',dpi=180); plt.close()
    plt.figure(figsize=(7,4)); plt.plot(m6['profile_arc_mm'],m6['profile_alignment'],label='C6'); plt.plot(m7['profile_arc_mm'],m7['profile_alignment'],label='C7'); plt.xlabel('downstream arc (mm)'); plt.ylabel('|tangent · local ribbon PCA tangent|'); plt.ylim(0,1.05); plt.legend(); plt.tight_layout(); plt.savefig(out/'03_local_junction_alignment_profiles.png',dpi=180); plt.close()
    fig=plt.figure(figsize=(10,8)); ax=fig.add_subplot(111,projection='3d'); lp=np.asarray(left_ribbon.data); ax.scatter(lp[:,0],lp[:,1],lp[:,2],s=4,alpha=.20,label='local LA-LV junction ribbon'); ax.plot(c6_post[:,0],c6_post[:,1],c6_post[:,2],lw=3,label='C6'); ax.plot(c7_post[:,0],c7_post[:,1],c7_post[:,2],lw=3,label='C7'); ax.scatter([left_anchor[0]],[left_anchor[1]],[left_anchor[2]],s=55,label='F2 split anchor'); ax.legend(); ax.set_title('Local daughter-blind LA-LV junction ribbon'); plt.tight_layout(); plt.savefig(out/'04_local_junction_geometry.png',dpi=180); plt.close()
    fig,axes=plt.subplots(2,3,figsize=(12,8))
    for r,(cid,p) in enumerate([(6,c6_post),(7,c7_post)]):
        pp,qq=_resample(p,.25); picks=np.linspace(0,len(pp)-1,3).astype(int); tt=_tangents(pp)
        for j,ix in enumerate(picks):
            im,qv=_plane(g,src,pp[ix],tt[ix]); ax=axes[r,j]; ax.imshow(im,cmap='gray',vmin=-100,vmax=900,extent=[qv[0],qv[-1],qv[-1],qv[0]]); ax.scatter([0],[0],s=25); ax.set_title(f'C{cid} +{qq[ix]:.1f} mm'); ax.set_xlabel('mm'); ax.set_ylabel('mm')
    plt.tight_layout(); plt.savefig(out/'05_C6_C7_orthogonal_source_qc.png',dpi=180); plt.close()

    if not topo_pass: status=STATUS_TOPOLOGY_FAIL
    elif not control_pass: status=STATUS_CONTROL_FAIL
    elif identity_gate: status=STATUS_C6 if win_id==6 else STATUS_C7
    else: status=STATUS_AMBIG
    summary={
        'status':status,'algorithm':ALGORITHM,'baseline_commit':BASELINE,'master_status':master.get('status'),'LCX_master_status':'UNRESOLVED',
        'topology_prerequisite_pass':topo_pass,'split_arc_mm':split_arc,'C6_C9_truncation':trunc,'local_junction_controls':controls,
        'C6_metrics':{k:v for k,v in m6.items() if not isinstance(v,np.ndarray)},'C7_metrics':{k:v for k,v in m7.items() if not isinstance(v,np.ndarray)},'decision':decision,
        'template_similarity_used_in_decision':False,'prior_AV_groove_score_used_in_decision':False,
        'decision_rule':'Daughter-blind local myocardial AV-junction ribbon anchored at the established F2 split. RCA/right-junction positive and LAD/left-junction negative controls must pass. Winner requires >=0.08 score margin, >=0.55 retention, >=0.90 dual-mask/HU support, plus >=1.5 mm distance or >=0.15 tangent-alignment advantage and explicit loser departure (>=0.20 mm/mm slope or >=2 mm endpoint-distance advantage).',
        'scientific_boundary':'A positive result nominates an LCX-like junction-following daughter and an OM-like junction-departing daughter. It does not establish clinical vessel identity; LM remains unresolved.'
    }
    _write_json(out/'summary.json',summary)
    report=out/'OPENPLAQUE_LCX_LOCAL_AV_JUNCTION_RIBBON_REPORT.html'; report.write_text(f"<html><body><h1>OpenPlaque LCX Local AV-Junction Ribbon</h1><p><b>Status:</b> {status}</p><p>Topology prerequisite: {topo_pass}. Control pass: {control_pass} (margin {control_margin:.3f}).</p><p>C6 score {m6['junction_identity_score']:.3f}; C7 score {m7['junction_identity_score']:.3f}; winner C{win_id}; identity gate {identity_gate}.</p><p>Template and prior AV-groove scores have zero decision weight.</p></body></html>",encoding='utf-8')
    _write_json(out/'run_state.json',{'status':'COMPLETE','scientific_status':status,'algorithm':ALGORITHM,'baseline_commit':BASELINE})
    zpath=out/'OPENPLAQUE_LCX_LOCAL_AV_JUNCTION_RIBBON_REPORT_BACK.zip'
    with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            if p!=zpath and p.is_file(): z.write(p,p.name)
    return {'summary':summary,'report':str(report),'zip':str(zpath)}
