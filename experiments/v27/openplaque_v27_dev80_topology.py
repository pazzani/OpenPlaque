"""OpenPlaque V27: frozen V26 completion on remaining DEV52, then descriptive 3D topology QA.
Research only; not a coronary-tree validation or clinical inference.
"""
import argparse, gc, hashlib, html, importlib.util, json, os, sys, traceback
from pathlib import Path

VERSION="openplaque-v27-completion-topology-20261017-2"
ROOT=Path("/content/drive/MyDrive/OpenPlaque")
BASE=ROOT/"OpenPlaque_Evaluation_Dataset_v0_2"
OUT=BASE/"evaluation_v27_topology_full_dev80"
V26=BASE/"evaluation_v26_lm_holdout"
V23=BASE/"evaluation_v23_heart_aorta_roi"
CT_ROOT=ROOT/"ExternalData/ImageCAS_OriginalCCTA_MatchedTo_ImageCASX_v1/volumes"
REF_ROOT=ROOT/"Cache/ImageCAS_X_Pretrained_v1/ImageCAS-X_dataset.zip_extracted/ImageCAS-X_dataset/segmentations"
V26_CODE=Path("/content/openplaque_v26_pinned.py")
EXPECTED_V26_SHA="ca2b576c40dacce8ce6447419c2e14ba4154825f14c6e183558e3b53df2c4d41"
DEV_SHA="ef2bd3bdbe7dc9cbf4639e62d80dceb53108acff64617e0f17b4f29be4f"
MODEL_SHA="f789eede621816433240232455d7a7862c7eafc69a979a50938e73349a6796b9"
HEADS={"LM":(1,0),"LAD":(2,1),"LCX":(3,2),"RCA":(9,3)}

def digest(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
    return h.hexdigest()

def load(p):return json.loads(Path(p).read_text())

def save(p,thing):
    p=Path(p);p.parent.mkdir(exist_ok=True,parents=True)
    tmp=p.with_name(p.name+".partial")
    tmp.write_text(json.dumps(thing,indent=2,sort_keys=True,allow_nan=False))
    os.replace(tmp,p)

def frozen():
    if digest(V26_CODE)!=EXPECTED_V26_SHA:raise RuntimeError("STOP: V26 source fingerprint mismatch")
    spec=importlib.util.spec_from_file_location("openplaque_v26_pinned",str(V26_CODE))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    if m.VERSION!="openplaque-v26-lm-ridge-soft15-holdout-20261016-1" or m.PRIMARY!="lm_ridge_only":
        raise RuntimeError("STOP: wrong V26 method")
    m.OUT=OUT/"new52_primary"
    return m

def cohort():
    ids=[x.strip() for x in (BASE/"internal_dev_80_frozen.txt").read_text().splitlines() if x.strip()]
    if len(ids)!=80 or len(set(ids))!=80:raise RuntimeError("Wrong DEV80 size")
    if hashlib.sha256("".join(x+"\n" for x in sorted(ids,key=int)).encode()).hexdigest()!=DEV_SHA:
        raise RuntimeError("STOP: frozen DEV80 checksum")
    manifest=load(V26/"manifest_v26.json");status=load(V26/"run_status_v26.json")
    if status["status"]!="COMPLETE_16_NEW_DEV_CASES" or manifest["model_sha256"]!=MODEL_SHA or manifest["primary_variant"]!="lm_ridge_only":
        raise RuntimeError("STOP: V26 source provenance")
    previous=list(manifest["pilot_case_ids"])+list(manifest["independent_development_case_ids"])
    if len(previous)!=len(set(previous))!=28:raise RuntimeError("Malformed prior case IDs")
    if len(previous)!=28 or not set(previous)<=set(ids):raise RuntimeError("Invalid previous DEV28")
    remaining=[x for x in sorted(ids,key=int) if x not in set(previous)]
    if len(remaining)!=52:raise RuntimeError("Incorrect remaining DEV52")
    for c in previous:
        if not (V26/"per_case"/(c+".json")).is_file():raise FileNotFoundError("V26 case "+c)
        if not (V26/"candidate_masks"/(c+".v26_primary_4head_native.nii.gz")).is_file():
            raise FileNotFoundError("V26 mask "+c)
    plan=dict(version=VERSION,old_28=previous,new_52=remaining,frozen_dev_sha256=DEV_SHA,
        frozen_model_sha256=MODEL_SHA,frozen_v26_code_sha256=EXPECTED_V26_SHA,
        candidate_method="V26 lm_ridge_only unchanged",aorta_proxy_mm=3.,
        major_component_volume_mm3=10.,gate=dict(min_removed=.85,min_each_head_retained=.90,min_eligible=10),
        no_training=True,no_test_reads=True,no_reference_guided_proposals=True,
        no_model_or_artery_promotion=True,unknown_background_not_negative=True)
    p=OUT/"manifest_v27.json"
    if p.exists() and load(p)!=plan:raise RuntimeError("STOP: mismatching V27 experiment manifest")
    if not p.exists():save(p,plan)
    return previous,remaining

def files_for(c,previous):
    parent=V26 if c in set(previous) else OUT/"new52_primary"
    return parent/"per_case"/(c+".json"),parent/"candidate_masks"/(c+".v26_primary_4head_native.nii.gz")

def check_mask(c,previous):
    rp,mp=files_for(c,previous)
    d=load(rp)
    if d.get("version")!="openplaque-v26-lm-ridge-soft15-holdout-20261016-1" or d.get("primary_mask_sha256")!=digest(mp):
        raise RuntimeError("Candidate provenance mismatch "+c)
    return d,mp

def generate(previous,remaining,force=False):
    m=frozen()
    for i,c in enumerate(remaining,1):
        r,state=m.per_case(c,"V27_REMAINING_DEV52",force=force)
        check_mask(c,previous)
        print("Frozen V26 generation",i,"/52",c,state,flush=True)
        if i%5==0:save(OUT/"progress_v27.json",dict(stage="candidate_generation",done=i,total=52,last_case=c))

def components(mask,rootband,other,positive,spacing):
    import numpy as np
    from scipy import ndimage as ndi
    count=int(mask.sum());voxel=float(np.prod(spacing))
    if count==0:
        return dict(voxels=0,components=0,major_components=0,root_proxy_components=0,
                    fraction_in_root_proxy_components=0.,fraction_cross_head=0.,
                    cross_head_voxels=0,largest_component_fraction=0.,
                    bulky_components=0,positive_fraction_in_root_proxy_components=None)
    cc,n=ndi.label(mask,structure=ndi.generate_binary_structure(3,1))
    counts=np.bincount(cc.ravel())[1:]
    touching=np.unique(cc[rootband&mask]);touching=touching[touching>0]
    keep=np.zeros(n+1,dtype=bool);keep[touching]=True
    poshits=int((mask&positive).sum()) if positive is not None else 0
    pfrac=float(np.count_nonzero(mask&positive&keep[cc])/poshits) if poshits else None
    result=dict(voxels=count,components=int(n),
        major_components=int(np.count_nonzero(counts*voxel>=10.)),
        root_proxy_components=int(len(touching)),
        fraction_in_root_proxy_components=float(counts[keep[1:]].sum()/count),
        fraction_cross_head=float(np.count_nonzero(mask&other)/count),
        cross_head_voxels=int(np.count_nonzero(mask&other)),
        largest_component_fraction=float(counts.max()/count),
        bulky_components=int(np.count_nonzero(counts*voxel>10000.)),
        positive_fraction_in_root_proxy_components=pfrac)
    del cc,counts
    return result

def topology(c,previous,force=False):
    import numpy as np,nibabel as nib
    from scipy.ndimage import distance_transform_edt
    rowpath=OUT/"topology_per_case"/(c+".json")
    old,maskpath=check_mask(c,previous)
    aortapath=V23/"totalseg_masks"/c/"aorta.nii.gz"
    asha=digest(aortapath)
    if rowpath.exists() and not force:
        row=load(rowpath)
        if row.get("version")==VERSION and row.get("mask_sha256")==old["primary_mask_sha256"] and row.get("aorta_sha256")==asha:
            return row,"REUSED"
    ct=nib.load(str(CT_ROOT/(c+".img.nii.gz")))
    pred=nib.load(str(maskpath));ref=nib.load(str(REF_ROOT/(c+".coronary.nii.gz")))
    if not(ct.shape==pred.shape==ref.shape and np.allclose(ct.affine,pred.affine,atol=1e-3) and np.allclose(ct.affine,ref.affine,atol=1e-3)):
        raise RuntimeError("STOP geometry "+c)
    spacing=tuple(float(x) for x in ct.header.get_zooms()[:3])
    m=frozen()
    heart=m.native_mask(V23/"totalseg_masks"/c/"heart.nii.gz",ct)
    crop=m.crop_bbox(heart,spacing);del heart
    aorta=m.native_mask(aortapath,ct)[crop]
    near=distance_transform_edt(~aorta,sampling=spacing)<=3.
    del aorta
    bits=np.asarray(pred.dataobj[crop],dtype=np.uint8)
    labels=np.asarray(ref.dataobj[crop],dtype=np.uint8) # post hoc ONLY; frozen candidate already exists
    if np.any(bits>15):raise RuntimeError("Invalid 4head bits "+c)
    arteries={}
    for name,(label,j) in HEADS.items():
        mask=(bits&(1<<j))!=0
        other=(bits&(15^(1<<j)))!=0
        sr=old["variants"]["lm_ridge_only"][name]
        known=(labels==label) if sr["eligible_reference"] else None
        x=components(mask,near,other,known,spacing)
        if x["voxels"]!=sr["prediction_voxels"]:raise RuntimeError("V26 candidate count changed "+c+" "+name)
        x.update(raw_predictions=sr["raw_prediction_voxels"],
                 V22_hits=sr["v22_positive_hits"],retained_hits=sr["retained_reference_hits"],
                 eligible_reference=sr["eligible_reference"],
                 known_other_reference_hits=sr["hits_other_known_major_labels"])
        arteries[name]=x
    row=dict(case_id=c,version=VERSION,mask_sha256=old["primary_mask_sha256"],
        aorta_sha256=asha,source="V26_28" if c in set(previous) else "NEW52",
        arteries=arteries,not_validated_coronary=True,root_proxy_not_ostium=True,no_test_reads=True)
    save(rowpath,row)
    del bits,labels,near
    gc.collect()
    return row,"NEW"

def summary(previous,remaining,stage):
    import numpy as np
    def collect(ids):
        rows=[load(OUT/"topology_per_case"/(c+".json")) for c in ids if (OUT/"topology_per_case"/(c+".json")).exists()]
        result=dict(expected=len(ids),complete=len(rows),missing=[c for c in ids if not (OUT/"topology_per_case"/(c+".json")).exists()],arteries={})
        totalraw=totalpred=0
        for name in HEADS:
            a=[r["arteries"][name] for r in rows]
            raw=sum(x["raw_predictions"] for x in a);pred=sum(x["voxels"] for x in a)
            totalraw+=raw;totalpred+=pred
            elig=[x for x in a if x["eligible_reference"]]
            hits=sum(x["retained_hits"] or 0 for x in elig)
            V22hits=sum(x["V22_hits"] or 0 for x in elig)
            result["arteries"][name]=dict(eligible_cases=len(elig),
                positive_hit_retention=hits/max(1,V22hits),
                removed=1-pred/max(1,raw),
                median_major_components=float(np.median([x["major_components"] for x in a])) if a else None,
                median_root_proxy_fraction=float(np.median([x["fraction_in_root_proxy_components"] for x in a])) if a else None,
                median_crosshead_fraction=float(np.median([x["fraction_cross_head"] for x in a])) if a else None,
                median_bulky_components=float(np.median([x["bulky_components"] for x in a])) if a else None)
        result["weighted_removed"]=1-totalpred/max(1,totalraw)
        return result
    prior=collect(previous);new=collect(remaining);combined=collect(previous+remaining)
    if new["complete"]==52:
        x=new["arteries"]
        passed=bool(new["weighted_removed"]>=.85 and all(x[n]["positive_hit_retention"]>=.90 and x[n]["eligible_cases"]>=10 for n in HEADS))
        gate=dict(complete=True,passed=passed,weighted_removed=new["weighted_removed"],
            per_artery_retention={n:x[n]["positive_hit_retention"] for n in HEADS},
            meaning="Candidate-screen criterion ONLY, not validated anatomy")
    else:gate=dict(complete=False,passed=False,reason="Remaining52 not complete")
    d=dict(version=VERSION,stage=stage,previous_28=prior,remaining_52=new,all80=combined,
           new52_candidate_gate=gate,no_test_reads=True,coronary_tree_validated=False,
           unannotated_voxels_are_unknown=True)
    save(OUT/"summary_v27.json",d)
    h=['<html><meta charset="utf-8"><title>OpenPlaque V27 topology</title>',
       '<style>body{font:16px system-ui;max-width:960px;margin:30px auto}td,th{border:1px solid #aaa;padding:8px}table{border-collapse:collapse}</style>',
       '<h1>OpenPlaque V27: frozen V26 + 3D topology diagnostic</h1>',
       '<p>Research only: component counts, proximity to modeled aorta, and overlaps DO NOT prove artery identity, proper ostial connection, or specificity. Unannotated voxels are unknown.</p>',
       '<p>Stage: '+html.escape(stage)+' | Completed '+str(new["complete"])+'/52 new and '+str(combined["complete"])+'/80 combined. Candidate gate: '+("PASS" if gate["passed"] else "FAIL OR INCOMPLETE")+'.</p>',
       '<table><tr><th>Artery</th><th>New52 positive retention</th><th>New52 median 3D major components</th><th>All80 median major components</th><th>All80 median fraction in aorta-proxy components</th></tr>']
    for n in HEADS:
        a=new["arteries"][n];b=combined["arteries"][n]
        h.append(f'<tr><td>{n}</td><td>{a["positive_hit_retention"]:.1%}</td><td>{a["median_major_components"]}</td><td>{b["median_major_components"]}</td><td>{b["median_root_proxy_fraction"]}</td></tr>')
    h+=['</table><p>Root proximity ≤3mm is a CT-derived aorta-mask proxy, not a confirmed coronary origin.</p>',
        '<h2>Machine-readable data</h2><pre>'+html.escape(json.dumps(d,indent=2))+'</pre></html>']
    (OUT/"OPENPLAQUE_V27_TOPOLOGY_REPORT.html").write_text("\n".join(h))
    return d

def qa_images(previous,remaining,limit=6):
    import numpy as np,nibabel as nib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows=[load(OUT/"topology_per_case"/(c+".json")) for c in previous+remaining]
    rows.sort(key=lambda r:(-sum(z["cross_head_voxels"] for z in r["arteries"].values()),int(r["case_id"])))
    selected=[r["case_id"] for r in rows[:limit]]
    folder=OUT/"qa_png";folder.mkdir(exist_ok=True,parents=True)
    for c in selected:
        _,p=files_for(c,previous)
        arr=np.asarray(nib.load(str(p)).dataobj,dtype=np.uint8)
        ct=np.asarray(nib.load(str(CT_ROOT/(c+".img.nii.gz"))).dataobj,dtype=np.float32)
        n=np.zeros(arr.shape,np.uint8)
        for _,j in HEADS.values():n+=((arr&(1<<j))!=0).astype(np.uint8)
        z=int(np.sum(n>=2,axis=(0,1)).argmax()) if (n>=2).any() else int(np.count_nonzero(arr,axis=(0,1)).argmax())
        fig,ax=plt.subplots(figsize=(7,7))
        ax.imshow(ct[:,:,z].T,origin="lower",cmap="gray",vmin=-100,vmax=650)
        palette=["yellow","cyan","magenta","lime"]
        for name,(_,bit) in HEADS.items():
            color=palette[bit]
            mask=np.ma.masked_where((arr[:,:,z]&(1<<bit))==0,np.ones(arr[:,:,z].shape)).T
            from matplotlib.colors import ListedColormap
            ax.imshow(mask,origin="lower",cmap=ListedColormap([color]),alpha=.35)
        ax.set_title("V27 case "+c+" native voxel axial k="+str(z)+" | proxy QA, not clinical orientation");ax.axis("off")
        fig.savefig(folder/(c+".png"),dpi=115,bbox_inches="tight");plt.close(fig)
        del arr,ct,n;gc.collect()
    save(OUT/"qa_image_index_v27.json",dict(selected=selected,method="top 6 by native frozen-candidate cross-head overlap, no labels"))
    return selected

def run(force=False):
    previous,remaining=cohort()
    generate(previous,remaining,force=force)
    todo=previous+remaining
    for i,c in enumerate(todo,1):
        try:
            _,state=topology(c,previous,force=force)
            print("3D QA",i,"/80",c,state,flush=True)
            if i%8==0:summary(previous,remaining,"TOPOLOGY_IN_PROGRESS")
        except Exception as e:
            save(OUT/"run_status_v27.json",dict(stage="ERROR",case=c,error=str(e),traceback=traceback.format_exc()[-1400:]))
            raise
    d=summary(previous,remaining,"FULL_DEV80_COMPLETE")
    selection=qa_images(previous,remaining)
    save(OUT/"run_status_v27.json",dict(stage="COMPLETE_80_RESEARCH_QA",
        new52_candidate_gate=d["new52_candidate_gate"],attention_cases=selection,
        sealed_test_reads=0,anatomy_not_promoted=True))
    print("V27 completed, gate:",json.dumps(d["new52_candidate_gate"],indent=2))

def test():
    import numpy as np
    a=np.zeros((16,16,12),bool);a[1:4,2:5,3:6]=1;a[10:13,10:13,7:9]=1
    near=np.zeros_like(a);near[1:3,2:5,3:6]=1
    other=np.zeros_like(a);other[11,11,8]=1
    pos=np.zeros_like(a);pos[2,3,4]=1;pos[11,11,8]=1
    x=components(a,near,other,pos,(.5,.5,1))
    assert x["components"]==2 and x["root_proxy_components"]==1
    assert x["cross_head_voxels"]==1 and x["positive_fraction_in_root_proxy_components"]==.5
    print("PASS synthetic disconnected components, aorta-proxy flag, collisions, posthoc-positive QA")

if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("stage",choices=["test","preflight","run","report"]);a.add_argument("--force",action="store_true")
    args=a.parse_args()
    if args.stage=="test":test()
    elif args.stage=="preflight":
        p,r=cohort();frozen();print("PASS: 28 existing, 52 new; frozen source checksum")
    elif args.stage=="run":run(args.force)
    else:
        p,r=cohort();print(summary(p,r,"REPORT_ONLY")["new52_candidate_gate"])
