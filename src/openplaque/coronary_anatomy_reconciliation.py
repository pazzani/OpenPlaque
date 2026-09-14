from __future__ import annotations

import json, math, zipfile
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

ALGORITHM_VERSION = "coronary-anatomy-reconciliation-v1.0"

def _json_write(obj, path): Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")
def _first_existing(paths):
    for p in paths:
        p=Path(p)
        if p.exists(): return p
    return None

def _read_path(path):
    df=pd.read_csv(path)
    if not {"z","y","x"}.issubset(df.columns): raise ValueError(f"{path} lacks z,y,x")
    return df[["z","y","x"]].to_numpy(float), df

def arc_length(points_mm):
    p=np.asarray(points_mm,float)
    return 0.0 if len(p)<2 else float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum())

def source_zyx_to_lps(meta, zyx):
    p=np.atleast_2d(np.asarray(zyx,float)); positions=np.asarray(meta["positions_lps_mm"],float)
    orient=np.asarray(meta["image_orientation_patient"],float); row_cos,col_cos=orient[:3],orient[3:]
    sp=np.asarray(meta["spacing_zyx"],float)
    slice_vec=(positions[-1]-positions[0])/max(len(positions)-1,1) if len(positions)>1 else np.cross(row_cos,col_cos)*sp[0]
    out=positions[0][None,:]+p[:,0,None]*slice_vec[None,:]
    out+=p[:,2,None]*sp[2]*row_cos[None,:]; out+=p[:,1,None]*sp[1]*col_cos[None,:]
    return out

def _unit(v):
    v=np.asarray(v,float); return v/max(float(np.linalg.norm(v)),1e-12)

def _orth_basis(t):
    t=_unit(t); ref=np.array([1.,0.,0.]) if abs(t[0])<0.82 else np.array([0.,1.,0.])
    u=_unit(np.cross(t,ref)); v=_unit(np.cross(t,u)); return u,v

def _plane_memmap(ct, point_zyx, tangent_mm_zyx, spacing_zyx, half_mm=5.0, pix_mm=0.18):
    t=_unit(tangent_mm_zyx); u,v=_orth_basis(t); g=np.arange(-half_mm,half_mm+1e-9,pix_mm,dtype=np.float32)
    vv,uu=np.meshgrid(g,g,indexing="ij"); sp=np.asarray(spacing_zyx,np.float32); center=np.asarray(point_zyx,np.float32)*sp
    pos=center[None,None,:]+uu[...,None]*u+vv[...,None]*v; vox=pos/sp
    out=ndi.map_coordinates(ct,[vox[...,0],vox[...,1],vox[...,2]],output=np.float32,order=1,mode="nearest",prefilter=False)
    return out,g

def _lumen_metrics(im, grid, threshold_hu=220.):
    bright=(im>=threshold_hu)&(im<=1200.); lab,_=ndi.label(bright,structure=np.ones((3,3),np.uint8)); cy=cx=len(grid)//2
    chosen=int(lab[cy,cx])
    if chosen==0:
        yy,xx=np.nonzero(bright)
        if len(yy):
            d=np.hypot(grid[yy],grid[xx]); j=int(np.argmin(d)); chosen=int(lab[yy[j],xx[j]]) if float(d[j])<=0.9 else 0
    if chosen<=0:
        return {"radius_mm":np.nan,"centroid_shift_mm":np.nan,"circularity":np.nan,"center_hu":float(im[cy,cx]),"component_median_hu":np.nan,"plane_score":0.0}
    comp=lab==chosen; pix=float(abs(grid[1]-grid[0])); yy,xx=np.nonzero(comp); area=float(comp.sum())*pix*pix
    radius=math.sqrt(area/math.pi); cu,cv=float(np.mean(grid[xx])),float(np.mean(grid[yy])); shift=float(math.hypot(cu,cv))
    er=ndi.binary_erosion(comp); per=max(float(np.logical_and(comp,~er).sum())*pix,pix); circ=float(np.clip(4*math.pi*area/(per*per),0,1))
    med=float(np.median(im[comp])); center=float(im[cy,cx]); rscore=math.exp(-0.5*((radius-1.7)/1.0)**2); sscore=math.exp(-0.5*(shift/0.75)**2)
    score=float(0.35*rscore+0.30*sscore+0.20*min(1,circ/0.45)+0.15*min(1,max(0,(med-180)/350)))
    return {"radius_mm":radius,"centroid_shift_mm":shift,"circularity":circ,"center_hu":center,"component_median_hu":med,"plane_score":score}

def _tangent_zyx(path, idx, spacing):
    a=max(0,int(idx)-4); b=min(len(path)-1,int(idx)+4)
    return np.array([1.,0.,0.]) if a==b else _unit((path[b]-path[a])*np.asarray(spacing,float))

def _pair_metrics(name_a,a_lps,name_b,b_lps):
    tb=cKDTree(np.asarray(b_lps,float)); d,ib=tb.query(a_lps); j=int(np.argmin(d)); k=int(ib[j]); eps=[]
    for ea in [0,len(a_lps)-1]:
        for eb in [0,len(b_lps)-1]:
            eps.append({"vessel_a":name_a,"endpoint_a":"start" if ea==0 else "end","vessel_b":name_b,"endpoint_b":"start" if eb==0 else "end","distance_mm":float(np.linalg.norm(a_lps[ea]-b_lps[eb]))})
    return {"vessel_a":name_a,"vessel_b":name_b,"minimum_distance_mm":float(d[j]),"median_a_to_b_mm":float(np.median(d)),"p90_a_to_b_mm":float(np.percentile(d,90)),"fraction_a_within_2mm":float(np.mean(d<=2.0)),"nearest_a_index":j,"nearest_b_index":k},eps

def synthetic_reconciliation_self_test():
    meta={"spacing_zyx":[1,.5,.5],"positions_lps_mm":[[10,20,30],[10,20,31],[10,20,32]],"image_orientation_patient":[1,0,0,0,1,0]}
    lps=source_zyx_to_lps(meta,[[1,2,4]])[0]; pm,_=_pair_metrics("a",np.array([[0.,0.,0.],[1,0,0]]),"b",np.array([[3.,0.,0.],[4,0,0]]))
    return {"passed":bool(np.allclose(lps,[12,21,31]) and abs(pm["minimum_distance_mm"]-2)<1e-8),"lps":lps.tolist(),"pair_min_mm":pm["minimum_distance_mm"]}

class CoronaryAnatomyReconciliationWorkflow:
    def __init__(self,root="/content/drive/MyDrive/OpenPlaque",reuse=None):
        self.root=Path(root); self.cache=self.root/"Cache"/"Coronary_Anatomy_Reconciliation_v1"; self.out=self.root/"Coronary_Anatomy_Reconciliation_Report"
        self.cache.mkdir(parents=True,exist_ok=True); self.out.mkdir(parents=True,exist_ok=True); self.reuse={"inputs":True,"metrics":True,"figures":True,"report":True}
        if reuse:self.reuse.update({k:bool(v) for k,v in reuse.items()})
        self.ct=self.meta=self.spacing=None; self.paths_zyx={}; self.paths_lps={}; self.source_files={}; self.pairwise=self.endpoints=self.junction_qc=None; self.summary=None
    def cache_status(self):
        names={"inputs":"input_snapshot.json","metrics":"reconciliation_summary.json","figures":"figures.done","report":"report.done"}
        return pd.DataFrame([{"component":k,"reuse":self.reuse[k],"cache_exists":(self.cache/v).exists()} for k,v in names.items()])
    def _resolve_inputs(self):
        r=self.root; ct=meta=None
        for d in [r/"Cache"/"LAD_Confirmed_Backtrack_v1",r/"Cache"/"LAD_Origin_Backtrack_v1",r/"Cache"/"Left_Coronary_Ostium_Neck_v1"]:
            a,b=d/"series7_int16.npy",d/"series7_int16.json"
            if a.exists() and b.exists():
                try:m=json.loads(b.read_text())
                except Exception:continue
                if "positions_lps_mm" in m and "image_orientation_patient" in m:ct,meta=a,b;break
        files={
            "RCA_VALIDATED":_first_existing([r/"PCAT_RCA_10_50"/"rca_centerline_smoothed_zyx.csv",r/"PCAT_RCA_10_50_Reproducibility_Lock"/"rca_centerline_smoothed_zyx.csv"]),
            "LAD_FROZEN":_first_existing([r/"Cache"/"LAD_Confirmed_Backtrack_v1"/"frozen_lad_centerline.csv",r/"LAD_Confirmed_Backtrack_Report"/"frozen_lad_centerline.csv"]),
            "TRUNK_CANDIDATE":_first_existing([r/"LAD_Takeoff_Confirmation_Report"/"trunk_centerline.csv",r/"Cache"/"LAD_Takeoff_Confirmation_v1"/"trunk_centerline.csv"]),
            "SECONDARY_CANDIDATE":_first_existing([r/"Secondary_Branch_Lateral_Divergence_Report"/"branch_centerline.csv",r/"Cache"/"Secondary_Branch_Lateral_Divergence_v1"/"branch_centerline.csv"])}
        missing=[k for k,p in files.items() if p is None]
        if ct is None or meta is None:missing.append("SOURCE_SERIES7_WITH_DICOM_GEOMETRY")
        if missing:raise FileNotFoundError("Missing reconciliation input(s): "+", ".join(missing))
        return files,ct,meta
    def load_inputs(self):
        files,ct_path,meta_path=self._resolve_inputs(); self.meta=json.loads(meta_path.read_text()); self.spacing=np.asarray(self.meta["spacing_zyx"],float); self.ct=np.load(ct_path,mmap_mode="r")
        self.source_files={k:str(v) for k,v in files.items()}; self.source_files.update({"SOURCE_CT":str(ct_path),"SOURCE_META":str(meta_path)}); rows=[]
        for name,path in files.items():
            zyx,_=_read_path(path); lps=source_zyx_to_lps(self.meta,zyx); self.paths_zyx[name]=zyx; self.paths_lps[name]=lps
            for i,(q,w) in enumerate(zip(zyx,lps)):rows.append({"object":name,"point_index":i,"z":q[0],"y":q[1],"x":q[2],"lps_x_mm":w[0],"lps_y_mm":w[1],"lps_z_mm":w[2]})
        pd.DataFrame(rows).to_csv(self.cache/"centerlines_common_lps.csv",index=False)
        snap={"algorithm":ALGORITHM_VERSION,"source_ct":str(ct_path),"source_meta":str(meta_path),"source_shape":list(self.ct.shape),"spacing_zyx_mm":self.spacing.tolist(),"inputs":self.source_files,"note":"Every centerline is converted through the same series-7 DICOM geometry before comparisons."}; _json_write(snap,self.cache/"input_snapshot.json"); return snap
    def _aorta_distance_map(self):
        fp=self.root/"RCA_Ostium_TotalSegmentator"/"aorta_series7_totalseg.nii.gz"
        if not fp.exists():return None,None
        try:
            import SimpleITK as sitk; arr=sitk.GetArrayFromImage(sitk.ReadImage(str(fp))).astype(bool)
            if tuple(arr.shape)!=tuple(self.ct.shape):return None,None
            return arr,ndi.distance_transform_edt(~arr,sampling=self.spacing).astype(np.float32)
        except Exception:return None,None
    def compute_metrics(self):
        if not self.paths_lps:self.load_inputs()
        names=list(self.paths_lps); pair_rows=[]; endpoint_rows=[]
        for i in range(len(names)):
            for j in range(i+1,len(names)):
                row,eps=_pair_metrics(names[i],self.paths_lps[names[i]],names[j],self.paths_lps[names[j]]); pair_rows.append({k:v for k,v in row.items() if not k.startswith("nearest_")}); endpoint_rows.extend(eps)
        self.pairwise=pd.DataFrame(pair_rows); self.endpoints=pd.DataFrame(endpoint_rows); self.pairwise.to_csv(self.cache/"pairwise_distances.csv",index=False); self.endpoints.to_csv(self.cache/"endpoint_relationships.csv",index=False)
        aorta_mask,aorta_dist=self._aorta_distance_map(); aorta_rows=[]
        if aorta_dist is not None:
            for name,p in self.paths_zyx.items():
                for idx,label in [(0,"start"),(len(p)-1,"end")]:
                    q=np.asarray(p[idx],float); d=float(map_coordinates(aorta_dist,q[:,None],order=1,mode="nearest",prefilter=False)[0]); qq=np.clip(np.rint(q).astype(int),0,np.asarray(aorta_mask.shape)-1)
                    aorta_rows.append({"object":name,"endpoint":label,"aorta_distance_mm":d,"inside_aorta":bool(aorta_mask[tuple(qq)])})
        aorta_df=pd.DataFrame(aorta_rows); aorta_df.to_csv(self.cache/"aorta_endpoint_distances.csv",index=False)
        def minpair(a,b):
            q=self.pairwise[((self.pairwise.vessel_a==a)&(self.pairwise.vessel_b==b))|((self.pairwise.vessel_a==b)&(self.pairwise.vessel_b==a))]; return float(q.minimum_distance_mm.iloc[0])
        def minep(a,b):
            q=self.endpoints[((self.endpoints.vessel_a==a)&(self.endpoints.vessel_b==b))|((self.endpoints.vessel_a==b)&(self.endpoints.vessel_b==a))]; return float(q.distance_mm.min())
        tl_min,tl_ep=minpair("TRUNK_CANDIDATE","LAD_FROZEN"),minep("TRUNK_CANDIDATE","LAD_FROZEN"); tr_min,tr_ep=minpair("TRUNK_CANDIDATE","RCA_VALIDATED"),minep("TRUNK_CANDIDATE","RCA_VALIDATED")
        trunk_aorta=float("nan")
        if len(aorta_df):
            q=aorta_df[aorta_df.object=="TRUNK_CANDIDATE"]
            if len(q):trunk_aorta=float(q.aorta_distance_mm.min())
        lad_conn=bool(tl_ep<=2.0 and tl_min<=1.5); rca_conn=bool(tr_ep<=2.0 and tr_min<=1.5); aorta_ok=bool(np.isfinite(trunk_aorta) and trunk_aorta<=2.0)
        if lad_conn and aorta_ok and not rca_conn:status="TRUNK_TO_FROZEN_LAD_CONNECTIVITY_SUPPORTED"
        elif rca_conn and not lad_conn:status="TRUNK_CANDIDATE_ALIGNS_WITH_RCA_NOT_FROZEN_LAD"
        elif not lad_conn:status="TRUNK_TO_FROZEN_LAD_CONNECTIVITY_NOT_SUPPORTED"
        else:status="TRUNK_CONNECTIVITY_UNRESOLVED"
        self.summary={"algorithm":ALGORITHM_VERSION,"status":status,"automatic_relabeling_performed":False,"common_coordinate_frame":"DICOM LPS millimetres","lengths_mm":{n:arc_length(self.paths_lps[n]) for n in names},"trunk_to_frozen_lad":{"minimum_curve_distance_mm":tl_min,"minimum_endpoint_distance_mm":tl_ep,"connectivity_gate":lad_conn},"trunk_to_validated_rca":{"minimum_curve_distance_mm":tr_min,"minimum_endpoint_distance_mm":tr_ep,"connectivity_gate":rca_conn},"trunk_to_aorta":{"minimum_endpoint_distance_mm":trunk_aorta,"aorta_proximity_gate":aorta_ok},"interpretation_rule":"LM label remains provisional unless trunk meets frozen LAD and aorta in common LPS coordinates without instead matching RCA."}; _json_write(self.summary,self.cache/"reconciliation_summary.json"); return self.summary
    def build_junction_qc(self):
        if self.summary is None:self.compute_metrics()
        pairs=[("TRUNK_CANDIDATE","LAD_FROZEN"),("TRUNK_CANDIDATE","RCA_VALIDATED"),("LAD_FROZEN","RCA_VALIDATED"),("SECONDARY_CANDIDATE","LAD_FROZEN")]; rows=[]; figs=[]
        for a,b in pairs:
            A,B=self.paths_lps[a],self.paths_lps[b]; tb=cKDTree(B); d,j=tb.query(A); ia=int(np.argmin(d)); ib=int(j[ia])
            for name,idx in [(a,ia),(b,ib)]:
                p=self.paths_zyx[name]; t=_tangent_zyx(p,idx,self.spacing); im,g=_plane_memmap(self.ct,p[idx],t,self.spacing); m=_lumen_metrics(im,g); rows.append({"pair":f"{a}__{b}","object":name,"point_index":idx,"nearest_pair_distance_mm":float(d[ia]),**m}); figs.append((f"{a} ↔ {b}\n{name}",im,m))
        self.junction_qc=pd.DataFrame(rows); self.junction_qc.to_csv(self.cache/"junction_qc.csv",index=False); return self.junction_qc,figs
    def make_figures(self):
        if self.summary is None:self.compute_metrics()
        _,qfigs=self.build_junction_qc(); names=[]
        fig=plt.figure(figsize=(10,8)); ax=fig.add_subplot(111,projection="3d")
        for name,p in self.paths_lps.items():ax.plot(p[:,0],p[:,1],p[:,2],lw=2.4,label=name); ax.scatter(p[[0,-1],0],p[[0,-1],1],p[[0,-1],2],s=18)
        ax.set_xlabel("L (x) mm"); ax.set_ylabel("P (y) mm"); ax.set_zlabel("S (z) mm"); ax.set_title("Common DICOM LPS frame"); ax.legend(fontsize=8); fig.tight_layout(); f=self.out/"01_common_lps_coronary_geometry.png"; fig.savefig(f,dpi=180); plt.close(fig); names.append(f.name)
        fig,axs=plt.subplots(1,3,figsize=(16,5)); cfg=[("XY",0,(2,1)),("XZ",1,(2,0)),("YZ",2,(1,0))]
        for ax,(title,axis,dims) in zip(axs,cfg):
            ax.imshow(np.max(self.ct,axis=axis),cmap="gray",vmin=-100,vmax=800,origin="upper")
            for name,p in self.paths_zyx.items():ax.plot(p[:,dims[0]],p[:,dims[1]],lw=1.8,label=name)
            ax.set_title(title)
        axs[0].legend(fontsize=7); fig.suptitle("Source-series-7 CCTA MIPs — all inputs on one voxel grid"); fig.tight_layout(); f=self.out/"02_source_ccta_mips_all_centerlines.png"; fig.savefig(f,dpi=180); plt.close(fig); names.append(f.name)
        lad=self.paths_lps["LAD_FROZEN"]; c=lad[0]; fig=plt.figure(figsize=(10,8)); ax=fig.add_subplot(111,projection="3d")
        for name,p in self.paths_lps.items():
            mask=np.linalg.norm(p-c[None,:],axis=1)<=35
            if mask.any():q=p[mask]; ax.plot(q[:,0],q[:,1],q[:,2],lw=2.4,label=name)
        ax.scatter([c[0]],[c[1]],[c[2]],s=45,marker="x",label="frozen LAD proximal endpoint"); ax.set_title("Left-coronary reconciliation neighborhood"); ax.legend(fontsize=8); fig.tight_layout(); f=self.out/"03_left_coronary_reconciliation_zoom.png"; fig.savefig(f,dpi=180); plt.close(fig); names.append(f.name)
        fig,ax=plt.subplots(figsize=(11,6)); d=self.pairwise.sort_values("minimum_distance_mm"); labels=d.vessel_a+" ↔ "+d.vessel_b; ax.barh(np.arange(len(d)),d.minimum_distance_mm); ax.set_yticks(np.arange(len(d)),labels); ax.axvline(1.5,ls="--",lw=1); ax.set_xlabel("minimum curve-to-curve distance (mm)"); ax.set_title("Common-frame pairwise distances"); fig.tight_layout(); f=self.out/"04_pairwise_distances.png"; fig.savefig(f,dpi=180); plt.close(fig); names.append(f.name)
        n=len(qfigs); cols=4; rowsn=int(math.ceil(n/cols)); fig,axs=plt.subplots(rowsn,cols,figsize=(14,3.4*rowsn)); axs=np.atleast_1d(axs).ravel()
        for ax,(title,im,m) in zip(axs,qfigs):ax.imshow(im,cmap="gray",vmin=-100,vmax=900,origin="lower"); ax.set_title(f"{title}\nr={m['radius_mm']:.2f} shift={m['centroid_shift_mm']:.2f} score={m['plane_score']:.2f}",fontsize=8); ax.axis("off")
        for ax in axs[n:]:ax.axis("off")
        fig.suptitle("Source-CCTA planes at nearest-approach locations"); fig.tight_layout(); f=self.out/"05_nearest_approach_source_planes.png"; fig.savefig(f,dpi=180); plt.close(fig); names.append(f.name)
        fig,ax=plt.subplots(figsize=(13,5.5)); ax.axis("off"); s=self.summary; lines=["CORONARY ANATOMY RECONCILIATION",f"Status: {s['status']}","",f"Trunk↔frozen LAD curve {s['trunk_to_frozen_lad']['minimum_curve_distance_mm']:.2f} mm; endpoint {s['trunk_to_frozen_lad']['minimum_endpoint_distance_mm']:.2f} mm",f"Trunk↔validated RCA curve {s['trunk_to_validated_rca']['minimum_curve_distance_mm']:.2f} mm; endpoint {s['trunk_to_validated_rca']['minimum_endpoint_distance_mm']:.2f} mm",f"Trunk↔aorta endpoint {s['trunk_to_aorta']['minimum_endpoint_distance_mm']:.2f} mm","","No automatic vessel relabeling."]; ax.text(.02,.98,"\n".join(lines),va="top",family="monospace",fontsize=11); fig.tight_layout(); f=self.out/"06_reconciliation_evidence_summary.png"; fig.savefig(f,dpi=180); plt.close(fig); names.append(f.name)
        _json_write(names,self.cache/"figure_manifest.json"); (self.cache/"figures.done").write_text("ok"); return names
    def build_report(self):
        if self.summary is None:self.compute_metrics()
        if not (self.cache/"figures.done").exists():self.make_figures()
        if self.junction_qc is None:self.build_junction_qc()
        s=self.summary; html=f"""<html><head><meta charset='utf-8'><title>OpenPlaque Coronary Anatomy Reconciliation</title><style>body{{font-family:Arial;max-width:1400px;margin:24px auto;padding:0 18px}}img{{max-width:100%;margin:8px 0 24px}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:6px;border-bottom:1px solid #ddd}}.warn{{background:#fff3cd;padding:12px;border-left:4px solid #c99600}}</style></head><body><h1>OpenPlaque — Coronary Anatomy Reconciliation</h1><div class='warn'><b>Status: {s['status']}</b><br>All comparisons are in DICOM LPS millimetres. No automatic vessel relabeling was performed.</div><h2>Key connectivity metrics</h2><ul><li>Trunk ↔ frozen LAD: curve {s['trunk_to_frozen_lad']['minimum_curve_distance_mm']:.3f} mm; endpoint {s['trunk_to_frozen_lad']['minimum_endpoint_distance_mm']:.3f} mm.</li><li>Trunk ↔ validated RCA: curve {s['trunk_to_validated_rca']['minimum_curve_distance_mm']:.3f} mm; endpoint {s['trunk_to_validated_rca']['minimum_endpoint_distance_mm']:.3f} mm.</li><li>Trunk ↔ aorta endpoint: {s['trunk_to_aorta']['minimum_endpoint_distance_mm']:.3f} mm.</li></ul><h2>Pairwise distances</h2>{self.pairwise.to_html(index=False)}<h2>Endpoint relationships</h2>{self.endpoints.sort_values('distance_mm').head(16).to_html(index=False)}<h2>Nearest-approach source-CCTA QC</h2>{self.junction_qc.to_html(index=False)}<h2>Figures</h2>"""
        for name in json.loads((self.cache/"figure_manifest.json").read_text()):html+=f"<h3>{name}</h3><img src='{name}'>"
        html+="</body></html>"; out=self.out/"OPENPLAQUE_CORONARY_ANATOMY_RECONCILIATION_REPORT.html"; out.write_text(html,encoding="utf-8"); (self.cache/"report.done").write_text("ok"); return out
    def package(self):
        self.build_report()
        for fn in ["input_snapshot.json","centerlines_common_lps.csv","pairwise_distances.csv","endpoint_relationships.csv","aorta_endpoint_distances.csv","junction_qc.csv","reconciliation_summary.json","figure_manifest.json"]:
            src=self.cache/fn
            if src.exists():(self.out/fn).write_bytes(src.read_bytes())
        z=self.out/"OPENPLAQUE_CORONARY_ANATOMY_RECONCILIATION_REPORT_BACK.zip"
        with zipfile.ZipFile(z,"w",zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(self.out.iterdir()):
                if p!=z and p.is_file():zf.write(p,arcname=p.name)
        return z
