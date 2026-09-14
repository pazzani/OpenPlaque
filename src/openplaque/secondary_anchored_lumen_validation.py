from __future__ import annotations

"""Mandatory accepted-endpoint anchor + serial source-CCTA lumen validation.

Reuses the validated v2 3-D vesselness representation, but does not reuse its verdict.
A continuation is supported only if the graph passes through the accepted secondary-branch
endpoint and the distal path remains a compact centered lumen on serial source-CCTA planes.
Research use only; vessel identity is never assigned automatically.
"""

import json, math, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from . import secondary_3d_vesselness_topology as base
from .secondary_3d_vesselness_topology_v2 import Secondary3DVesselnessTopologyWorkflow, finite_dist_coords

ALGORITHM_VERSION = "secondary-anchored-lumen-validation-v1.0"


def _component_metrics(im, grid, comp):
    yy, xx = np.nonzero(comp)
    pix = float(abs(grid[1] - grid[0]))
    area = int(comp.sum()) * pix * pix
    radius = math.sqrt(area / math.pi)
    cu, cv = float(np.mean(grid[xx])), float(np.mean(grid[yy]))
    shift = float(math.hypot(cu, cv))
    er = ndi.binary_erosion(comp)
    perimeter = max(float(np.logical_and(comp, ~er).sum()) * pix, pix)
    circ = float(np.clip(4.0 * math.pi * area / (perimeter * perimeter), 0, 1))
    return {
        "radius_mm": float(radius),
        "centroid_shift_mm": shift,
        "circularity": circ,
        "component_median_hu": float(np.median(im[comp])),
    }


def _center_component(im, grid, threshold_hu, max_shift_mm=0.95):
    bright = (im >= float(threshold_hu)) & (im <= 1200.0)
    lab, nlab = ndi.label(bright, structure=np.ones((3, 3), np.uint8))
    out = []
    for k in range(1, int(nlab) + 1):
        comp = lab == k
        if int(comp.sum()) < 7:
            continue
        m = _component_metrics(im, grid, comp)
        yy, xx = np.nonzero(comp)
        dmin = float(np.min(np.hypot(grid[xx], grid[yy])))
        if m["centroid_shift_mm"] <= max_shift_mm or dmin <= 0.45:
            out.append((m["centroid_shift_mm"] + 0.35*dmin, m))
    return None if not out else min(out, key=lambda x: x[0])[1]


def _max_consecutive_false(v):
    best = cur = 0
    for x in v:
        if bool(x): cur = 0
        else:
            cur += 1; best = max(best, cur)
    return int(best)


def synthetic_lumen_validator_self_test():
    grid = np.arange(-4.4, 4.4 + 1e-9, 0.16)
    yy, xx = np.meshgrid(grid, grid, indexing="ij")
    tube = 100.0 + 650.0 * ((xx*xx + yy*yy) <= 1.8**2)
    broad = 100.0 + 650.0 * (xx >= -0.2)
    mt, mb = _center_component(tube, grid, 220), _center_component(broad, grid, 220)
    passed = bool(mt and mb and 1.5 <= mt["radius_mm"] <= 2.1 and mb["radius_mm"] > 2.65)
    return {"passed": passed, "tube_radius_mm": None if not mt else mt["radius_mm"], "broad_radius_mm": None if not mb else mb["radius_mm"]}


class SecondaryAnchoredLumenValidationWorkflow(Secondary3DVesselnessTopologyWorkflow):
    def __init__(self, root="/content/drive/MyDrive/OpenPlaque", reuse=None):
        super().__init__(root=root, reuse={})
        self.cache = self.root / "Cache" / "Secondary_Anchored_Lumen_Validation_v1"
        self.out = self.root / "Secondary_Anchored_Lumen_Validation_Report"
        self.cache.mkdir(parents=True, exist_ok=True); self.out.mkdir(parents=True, exist_ok=True)
        self.reuse = {"inputs": True, "anchored_search": True, "figures": True, "report": True}
        if reuse: self.reuse.update({k: bool(v) for k, v in reuse.items()})
        self.prov = []
        self.anchor_point = None; self.anchor_path = None; self.best_qc = None; self.calibration = None
        self.prior_v2_extension = None

    def _record(self, component, action, path="", note=""):
        self.prov.append({"component": component, "reuse_requested": self.reuse.get(component), "action": action, "path": str(path), "note": note})
        pd.DataFrame(self.prov).to_csv(self.out / "cache_provenance.csv", index=False)

    def cache_status(self):
        names = {"inputs":"input_snapshot.json", "anchored_search":"summary.json", "figures":"figures.done", "report":"report.done"}
        return pd.DataFrame([{"component":k,"reuse":self.reuse[k],"cache_exists":(self.cache/v).exists()} for k,v in names.items()])

    def load_inputs(self, control_arc_mm=12.7):
        src = self.root / "Cache" / "Secondary_3D_Vesselness_Topology_v1"
        need = [src/"series7_int16.npy", src/"series7_int16.json", src/"vesselness.npy", src/"best_scale.npy", src/"vesselness_meta.json", src/"frozen_geometry.json"]
        seedf = self.root/"Secondary_Branch_Lateral_Divergence_Report"/"branch_centerline.csv"
        reff = self.root/"LAD_Takeoff_Root_Alternatives_Report"/"alternative_01_centerline.csv"
        trunkf = self.root/"LAD_Takeoff_Confirmation_Report"/"trunk_centerline.csv"
        need += [seedf, reff, trunkf]
        if not all(p.exists() for p in need):
            raise FileNotFoundError("Missing prior v2/accepted-anatomy outputs: " + "; ".join(str(p) for p in need if not p.exists()))
        meta=json.loads((src/"series7_int16.json").read_text()); vm=json.loads((src/"vesselness_meta.json").read_text()); fg=json.loads((src/"frozen_geometry.json").read_text())
        self.spacing=np.asarray(meta["spacing_zyx"],float); self.ct=np.load(src/"series7_int16.npy",mmap_mode="r")
        self.vesselness=np.load(src/"vesselness.npy",mmap_mode="r"); self.best_scale=np.load(src/"best_scale.npy",mmap_mode="r"); self.vessel_threshold=float(vm["adaptive_threshold"])
        self.roi_lo=np.asarray(fg["roi_lo_zyx"],int); self.roi_hi=np.asarray(fg["roi_hi_zyx"],int)
        self.roi=np.asarray(self.ct[self.roi_lo[0]:self.roi_hi[0], self.roi_lo[1]:self.roi_hi[1], self.roi_lo[2]:self.roi_hi[2]],dtype=np.float32)
        self.seed=pd.read_csv(seedf)[["z","y","x"]].to_numpy(float); self.reference=pd.read_csv(reff)[["z","y","x"]].to_numpy(float); self.trunk=pd.read_csv(trunkf)[["z","y","x"]].to_numpy(float)
        v2ext=src/"extension_path.csv"
        if v2ext.exists(): self.prior_v2_extension=pd.read_csv(v2ext)[["z","y","x"]].to_numpy(float)
        sp=base.resample_path(self.seed,self.spacing,0.12); s=base.arc_mm(sp,self.spacing); self.control_arc=float(control_arc_mm)
        i=int(np.argmin(np.abs(s-self.control_arc))); self.control_point=sp[i].copy(); self.anchor_point=sp[-1].copy(); self.seed_endpoint=self.anchor_point.copy()
        self.start_point=self.control_point.copy(); self.start_arc=self.control_arc
        j=max(0,len(sp)-9); self.terminal_tangent=base._unit((sp[-1]-sp[j])*self.spacing)
        snap={"algorithm":ALGORITHM_VERSION,"control_arc_mm":self.control_arc,"accepted_seed_end_arc_mm":float(s[-1]),"control_zyx":self.control_point.tolist(),"mandatory_anchor_accepted_endpoint_zyx":self.anchor_point.tolist(),"mandatory_anchor_radius_mm":0.65,"vessel_threshold":self.vessel_threshold,"search_target":None,"note":"Accepted endpoint is mandatory; distal support also requires serial compact-lumen source-CCTA validation."}
        base._write_json(snap,self.cache/"input_snapshot.json"); self._record("inputs","loaded_v2_vesselness_and_accepted_anatomy",self.cache/"input_snapshot.json")
        self._calibrate_lumen(); return snap

    def _build_mask_and_cost(self):
        hu=np.asarray(self.roi,float); v=np.asarray(self.vesselness,float); bs=np.asarray(self.best_scale,float)
        vm=json.loads((self.root/"Cache"/"Secondary_3D_Vesselness_Topology_v1"/"vesselness_meta.json").read_text()); med=float(vm["control_vesselness_median"])
        vn=np.clip(v/max(1.5*med,1e-4),0,1); mask=(hu>=170)&(hu<=1200)&(v>=self.vessel_threshold)
        cost=0.30+4.5*(1-vn)**2; cost+=0.30*np.clip((bs-1.15)/0.50,0,1); cost+=0.20*np.clip((hu-900)/300,0,1)
        return mask,cost.astype(np.float32)

    def _plane_component(self, point, tangent, threshold, max_shift=0.95):
        im,grid=base.orthogonal_plane(self.ct,point,tangent,self.spacing,half_mm=4.4,pix_mm=0.16)
        return im,_center_component(im,grid,threshold,max_shift)

    def _calibrate_lumen(self):
        sp=base.resample_path(self.seed,self.spacing,0.30); s=base.arc_mm(sp,self.spacing); p=sp[(s>=10.0)&(s<=self.control_arc)]
        center=ndi.map_coordinates(np.asarray(self.ct),[p[:,0],p[:,1],p[:,2]],order=1,mode="nearest"); refhu=float(np.median(center)); thr=float(max(170,min(300,0.38*refhu)))
        rows=[]
        for i,pt in enumerate(p):
            i0,i1=max(0,i-2),min(len(p)-1,i+2); _,m=self._plane_component(pt,(p[i1]-p[i0])*self.spacing,thr,0.80)
            if m is not None: rows.append(m)
        if not rows: raise RuntimeError("Could not calibrate lumen on accepted branch")
        df=pd.DataFrame(rows); good=df[(df.radius_mm>=0.65)&(df.radius_mm<=2.65)&(df.centroid_shift_mm<=0.80)]
        if len(good)<3: good=df
        self.calibration={"reference_center_hu":refhu,"bright_threshold_hu":thr,"median_radius_mm":float(good.radius_mm.median()),"p90_radius_mm":float(good.radius_mm.quantile(.9)),"median_circularity":float(good.circularity.median()),"n_planes":int(len(good))}
        base._write_json(self.calibration,self.cache/"lumen_calibration.json")

    def _validate_lumen(self,path):
        p=base.resample_path(path,self.spacing,0.35); s=base.arc_mm(p,self.spacing); rows=[]; thr=self.calibration["bright_threshold_hu"]; rref=self.calibration["median_radius_mm"]
        for i,pt in enumerate(p):
            i0,i1=max(0,i-2),min(len(p)-1,i+2); _,m=self._plane_component(pt,(p[i1]-p[i0])*self.spacing,thr)
            row={"arc_mm":float(s[i]),"z":pt[0],"y":pt[1],"x":pt[2]}
            if m is None:
                row.update({"component_found":False,"radius_mm":np.nan,"centroid_shift_mm":np.nan,"circularity":np.nan,"component_median_hu":np.nan,"plane_pass":False,"plane_score":0.0})
            else:
                passed=bool(0.65<=m["radius_mm"]<=2.65 and m["centroid_shift_mm"]<=0.75 and m["circularity"]>=0.30 and m["component_median_hu"]>=max(220,0.45*self.calibration["reference_center_hu"]))
                rscore=math.exp(-.5*((m["radius_mm"]-rref)/max(.55*rref,.70))**2); sscore=math.exp(-.5*(m["centroid_shift_mm"]/.55)**2); cscore=float(np.clip(m["circularity"]/max(self.calibration["median_circularity"],.35),0,1)); hscore=float(np.clip(m["component_median_hu"]/max(self.calibration["reference_center_hu"],1),0,1.2)/1.2)
                row.update({"component_found":True,**m,"plane_pass":passed,"plane_score":float(.35*rscore+.35*sscore+.18*cscore+.12*hscore)})
            rows.append(row)
        q=pd.DataFrame(rows); found=q[q.component_found]; first=q[q.arc_mm<=2.0]
        m={"lumen_plane_pass_fraction":float(q.plane_pass.mean()) if len(q) else 0.0,"lumen_first2mm_pass_fraction":float(first.plane_pass.mean()) if len(first) else 0.0,"lumen_median_plane_score":float(q.plane_score.median()) if len(q) else 0.0,"lumen_max_consecutive_failures":_max_consecutive_false(q.plane_pass.tolist()) if len(q) else 999,"lumen_median_radius_mm":float(found.radius_mm.median()) if len(found) else np.nan,"lumen_p90_radius_mm":float(found.radius_mm.quantile(.9)) if len(found) else np.nan,"lumen_median_shift_mm":float(found.centroid_shift_mm.median()) if len(found) else np.nan}
        m["lumen_gate"]=bool(m["lumen_plane_pass_fraction"]>=.85 and m["lumen_first2mm_pass_fraction"]>=.80 and m["lumen_median_plane_score"]>=.62 and m["lumen_max_consecutive_failures"]<=1 and np.isfinite(m["lumen_p90_radius_mm"]) and m["lumen_p90_radius_mm"]<=2.65 and np.isfinite(m["lumen_median_shift_mm"]) and m["lumen_median_shift_mm"]<=.60)
        return q,m

    def _candidate_metrics_anchor(self, path):
        p=base.resample_path(path,self.spacing,.20); s=base.arc_mm(p,self.spacing); length=float(s[-1]); disp=float(np.linalg.norm((p[-1]-self.anchor_point)*self.spacing)); proj=float(np.dot((p[-1]-self.anchor_point)*self.spacing,self.terminal_tangent)); turns=base._path_turns_deg(p,self.spacing); v=base._sample_trilinear(self.vesselness,self._global_to_local(p)); bs=base._sample_trilinear(self.best_scale,self._global_to_local(p)); hu=base._sample_trilinear(self.roi,self._global_to_local(p)); old=base.resample_path(self.seed,self.spacing,.15); old_sep=float(np.min(np.linalg.norm((old-p[-1])*self.spacing,axis=1))); ref=np.vstack([self.reference,self.trunk]); ref_sep=float(np.min(np.linalg.norm((ref-p[-1])*self.spacing,axis=1)))
        return {"new_length_mm":length,"endpoint_displacement_mm":disp,"forward_projection_mm":proj,"tortuosity":length/max(disp,1e-6),"max_turn_deg":float(np.max(turns)) if len(turns) else 0.0,"median_vesselness":float(np.median(v)),"p10_vesselness":float(np.percentile(v,10)),"median_best_scale_mm":float(np.median(bs)),"p90_best_scale_mm":float(np.percentile(bs,90)),"median_center_hu":float(np.median(hu)),"endpoint_old_seed_separation_mm":old_sep,"endpoint_reference_separation_mm":ref_sep}

    def run(self,max_cost=150.0,min_projection_mm=1.5):
        sf=self.cache/"summary.json"; cf=self.cache/"candidate_validation.csv"; qf=self.cache/"best_candidate_qc.csv"; af=self.cache/"anchor_path.csv"; ef=self.cache/"extension_path.csv"; ff=self.cache/"full_path.csv"
        if self.ct is None: self.load_inputs()
        if self.reuse["anchored_search"] and all(p.exists() for p in (sf,cf,qf,af,ef,ff)):
            self.summary=json.loads(sf.read_text()); self.candidates=pd.read_csv(cf); self.best_qc=pd.read_csv(qf); self.anchor_path=pd.read_csv(af)[["z","y","x"]].to_numpy(float); self.extension_path=pd.read_csv(ef)[["z","y","x"]].to_numpy(float); self.full_path=pd.read_csv(ff)[["z","y","x"]].to_numpy(float); return self.summary
        mask,cost=self._build_mask_and_cost(); am=mask.copy(); am[self._sphere_mask(self.control_point,.55)]=True; goal=self._sphere_mask(self.anchor_point,.65); am[goal]=True
        _,prev,reached=self._dijkstra(am,cost,self._nearest_voxel(self.control_point),goal_mask=goal,max_cost=80)
        if reached<0:
            self.anchor_path=np.asarray([self.control_point]); self.extension_path=np.asarray([self.anchor_point]); self.full_path=self.anchor_path; self.candidates=pd.DataFrame(); self.best_qc=pd.DataFrame(); self.summary={"algorithm":ALGORITHM_VERSION,"status":"ANCHOR_CONTROL_FAILED","accepted_continuation":False,"anchor_pass":False,"interpretation":"3-D graph could not reach the accepted branch endpoint; distal inference is invalid."}
        else:
            loc=self._reconstruct(prev,reached,self.roi.shape); self.anchor_path=base.resample_path(self._local_to_global(loc),self.spacing,.20)
            known=base.resample_path(self.seed,self.spacing,.12); ks=base.arc_mm(known,self.spacing); seg=known[ks>=self.control_arc-.1]; dd,_=cKDTree(seg*self.spacing).query(self.anchor_path*self.spacing)
            targetd=float(np.linalg.norm((self.anchor_path[-1]-self.anchor_point)*self.spacing)); knownlen=float(ks[-1]-self.control_arc); alen=float(base.arc_mm(self.anchor_path,self.spacing)[-1])
            anchor_pass=bool(targetd<=.65 and np.median(dd)<=.55 and np.percentile(dd,90)<=.85 and alen<=max(2.0,1.8*knownlen)); anchor={"anchor_pass":anchor_pass,"anchor_target_distance_mm":targetd,"anchor_path_length_mm":alen,"accepted_control_to_endpoint_length_mm":knownlen,"anchor_median_distance_to_seed_mm":float(np.median(dd)),"anchor_p90_distance_to_seed_mm":float(np.percentile(dd,90))}
            if not anchor_pass:
                self.extension_path=np.asarray([self.anchor_point]); self.full_path=self.anchor_path; self.candidates=pd.DataFrame(); self.best_qc=pd.DataFrame(); self.summary={"algorithm":ALGORITHM_VERSION,"status":"ANCHOR_CONTROL_FAILED","accepted_continuation":False,**anchor,"interpretation":"The endpoint sphere was reached, but the route did not faithfully track the accepted 12.7-to-end segment."}
            else:
                dm=mask.copy(); dm[self._sphere_mask(self.anchor_point,.60)]=True; dist,p2,_=self._dijkstra(dm,cost,self._nearest_voxel(self.anchor_point),goal_mask=None,max_cost=max_cost); coords=finite_dist_coords(dist,self.roi.shape)
                if len(coords):
                    phys=(coords+self.roi_lo)*self.spacing; ap=self.anchor_point*self.spacing; proj=(phys-ap)@self.terminal_tangent; disp=np.linalg.norm(phys-ap,axis=1); vals=self.vesselness[tuple(coords.T)]; ok=(proj>=min_projection_mm)&(disp>=1.8)&(vals>=self.vessel_threshold); coords,proj,vals=coords[ok],proj[ok],vals[ok]
                rows=[]; paths=[]; qcs=[]
                if len(coords):
                    order=np.argsort(proj+1.5*vals)[::-1]; chosen=[]
                    for ii in order:
                        g=coords[ii]+self.roi_lo
                        if any(np.linalg.norm((g-q)*self.spacing)<1.0 for q in chosen): continue
                        chosen.append(g.copy()); flat=int(np.ravel_multi_index(tuple(coords[ii]),self.roi.shape)); loc=self._reconstruct(p2,flat,self.roi.shape)
                        if loc is None or len(loc)<2: continue
                        path=base.resample_path(self._local_to_global(loc),self.spacing,.20); m=self._candidate_metrics_anchor(path); qc,lm=self._validate_lumen(path)
                        geom=bool(m["new_length_mm"]>=2 and m["endpoint_displacement_mm"]>=1.8 and m["forward_projection_mm"]>=1.5 and m["tortuosity"]<=2 and m["max_turn_deg"]<=75 and m["p10_vesselness"]>=.65*self.vessel_threshold and m["endpoint_reference_separation_mm"]>=1.7)
                        supported=bool(geom and lm["lumen_gate"] and m["new_length_mm"]>=4 and m["endpoint_displacement_mm"]>=3 and m["forward_projection_mm"]>=2.5 and m["endpoint_old_seed_separation_mm"]>=1 and m["p90_best_scale_mm"]<=1.50)
                        rank=.24*min(m["new_length_mm"]/6,1)+.18*min(m["forward_projection_mm"]/5,1)+.22*lm["lumen_plane_pass_fraction"]+.18*lm["lumen_median_plane_score"]+.10*(1/max(m["tortuosity"],1))+.08*min(m["endpoint_old_seed_separation_mm"]/2,1)
                        rows.append({"candidate":len(rows)+1,**m,"geometry_gate":geom,**lm,"supported_continuation":supported,"rank_score":float(rank)}); paths.append(path); qcs.append(qc)
                        if len(rows)>=24: break
                self.candidates=pd.DataFrame(rows)
                if len(self.candidates):
                    self.candidates=self.candidates.sort_values(["supported_continuation","lumen_gate","geometry_gate","rank_score"],ascending=[False,False,False,False]); best_id=int(self.candidates.iloc[0].candidate); idx=best_id-1; best=rows[idx]; self.extension_path=paths[idx]; self.best_qc=qcs[idx]; self.full_path=base.resample_path(np.vstack([self.anchor_path,self.extension_path[1:]]),self.spacing,.20)
                    if best["supported_continuation"]: status="SUPPORTED_ANCHORED_LUMEN_EXTENSION"; interp="Mandatory endpoint anchor passed and a >=4 mm distal path also passed serial compact-lumen source-CCTA validation; vessel identity remains unassigned."
                    elif best["geometry_gate"]: status="3D_TUBULAR_CORRIDOR_FOUND_LUMEN_REJECTED"; interp="The anchored graph found a distal tubular corridor, but serial source-CCTA planes did not support a compact coronary lumen strongly enough."
                    else: status="NO_SUPPORTED_ANCHORED_3D_EXTENSION"; interp="The mandatory anchor passed, but distal paths failed geometry and/or lumen validation."
                    self.summary={"algorithm":ALGORITHM_VERSION,"status":status,"accepted_continuation":bool(best["supported_continuation"]),**anchor,**best,"interpretation":interp}
                else:
                    self.extension_path=np.asarray([self.anchor_point]); self.full_path=self.anchor_path; self.best_qc=pd.DataFrame(); self.summary={"algorithm":ALGORITHM_VERSION,"status":"NO_ANCHORED_3D_EXTENSION","accepted_continuation":False,**anchor,"interpretation":"Mandatory anchor passed, but no distal endpoint >=1.5 mm forward was reachable."}
        self.candidates.to_csv(cf,index=False); self.best_qc.to_csv(qf,index=False)
        for f,a in [(af,self.anchor_path),(ef,self.extension_path),(ff,self.full_path)]:
            a=np.asarray(a,float); pd.DataFrame({"arc_mm":base.arc_mm(a,self.spacing),"z":a[:,0],"y":a[:,1],"x":a[:,2]}).to_csv(f,index=False)
        base._write_json(self.summary,sf); self._record("anchored_search","computed",sf); return self.summary

    def make_figures(self):
        if self.summary is None: self.run()
        names=["01_anchored_geometry.png","02_anchor_control.png","03_lumen_cross_sections.png","04_lumen_profiles.png"]; sp=base.resample_path(self.seed,self.spacing,.15)
        fig=plt.figure(figsize=(13,4))
        for k,(a,b,t) in enumerate([(2,1,"X-Y"),(2,0,"X-Z"),(1,0,"Y-Z")],1):
            ax=fig.add_subplot(1,3,k); ax.plot(sp[:,a]*self.spacing[a],sp[:,b]*self.spacing[b],label="accepted branch")
            if self.prior_v2_extension is not None: ax.plot(self.prior_v2_extension[:,a]*self.spacing[a],self.prior_v2_extension[:,b]*self.spacing[b],alpha=.3,label="prior v2 corridor")
            if self.anchor_path is not None: ax.plot(self.anchor_path[:,a]*self.spacing[a],self.anchor_path[:,b]*self.spacing[b],lw=3,label="anchor path")
            if self.extension_path is not None and len(self.extension_path)>1: ax.plot(self.extension_path[:,a]*self.spacing[a],self.extension_path[:,b]*self.spacing[b],lw=3,label="v3 distal")
            ax.set_title(t); ax.set_aspect("equal",adjustable="box");
            if k==1: ax.legend(fontsize=7)
        fig.suptitle(f"Anchored lumen validation — {self.summary['status']}"); fig.tight_layout(); fig.savefig(self.out/names[0],dpi=160); plt.close(fig)
        fig,ax=plt.subplots(figsize=(8,4.5)); known=base.resample_path(self.seed,self.spacing,.12); ks=base.arc_mm(known,self.spacing); seg=known[ks>=self.control_arc-.1]; ax.plot(seg[:,2]*self.spacing[2],seg[:,1]*self.spacing[1],label="accepted 12.7→endpoint");
        if self.anchor_path is not None: ax.plot(self.anchor_path[:,2]*self.spacing[2],self.anchor_path[:,1]*self.spacing[1],lw=3,label="recovered anchor")
        ax.set_aspect("equal",adjustable="box"); ax.set_title(f"Mandatory endpoint anchor: pass={self.summary.get('anchor_pass')}"); ax.legend(); fig.tight_layout(); fig.savefig(self.out/names[1],dpi=160); plt.close(fig)
        fig,axes=plt.subplots(2,5,figsize=(13,5.4)); axes=axes.ravel()
        if self.extension_path is not None and len(self.extension_path)>2:
            p=base.resample_path(self.extension_path,self.spacing,.35); s=base.arc_mm(p,self.spacing); idxs=list(np.argsort(self.best_qc.plane_score.to_numpy())[:5]) if self.best_qc is not None and len(self.best_qc) else []; idxs += list(np.linspace(0,len(p)-1,5).astype(int)); idxs=list(dict.fromkeys(int(i) for i in idxs if i<len(p)))[:10]
            for ax,i in zip(axes,idxs):
                i0,i1=max(0,i-2),min(len(p)-1,i+2); im,_=base.orthogonal_plane(self.ct,p[i],(p[i1]-p[i0])*self.spacing,self.spacing,half_mm=4.4,pix_mm=.16); ax.imshow(im,cmap="gray",vmin=100,vmax=900); ax.set_title(f"{s[i]:.1f} mm"); ax.axis("off")
        for ax in axes:
            if not ax.has_data(): ax.axis("off")
        fig.suptitle("Worst + distributed source-CCTA planes along best anchored distal path"); fig.tight_layout(); fig.savefig(self.out/names[2],dpi=160); plt.close(fig)
        fig,ax=plt.subplots(figsize=(9,4.6))
        if self.best_qc is not None and len(self.best_qc):
            q=self.best_qc; ax.plot(q.arc_mm,q.radius_mm,label="radius (mm)"); ax.axhline(2.65,ls="--",label="radius limit"); ax.plot(q.arc_mm,q.centroid_shift_mm,label="center shift (mm)"); ax2=ax.twinx(); ax2.plot(q.arc_mm,q.plane_score,alpha=.55,label="plane score"); ax.set_xlabel("Distal path arc (mm)"); ax.set_title("Serial compact-lumen validation"); lines=ax.get_lines()+ax2.get_lines(); ax.legend(lines,[l.get_label() for l in lines],fontsize=8,loc="best")
        fig.tight_layout(); fig.savefig(self.out/names[3],dpi=160); plt.close(fig); (self.cache/"figures.done").write_text("done"); self._record("figures","computed",self.cache/"figures.done"); return names

    def make_report(self):
        if self.summary is None: self.run()
        self.make_figures(); html=self.out/"OPENPLAQUE_SECONDARY_ANCHORED_LUMEN_VALIDATION_REPORT.html"; rows="".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k,v in self.summary.items() if k!="interpretation")
        html.write_text(f"<html><body><h1>OpenPlaque Secondary Branch — Anchored Lumen Validation</h1><p><b>Status: {self.summary['status']}</b></p><p>{self.summary.get('interpretation','')}</p><table border='1' cellpadding='4'>{rows}</table><h2>Anchored geometry</h2><img src='01_anchored_geometry.png' width='95%'><h2>Mandatory endpoint anchor</h2><img src='02_anchor_control.png' width='85%'><h2>Source-CCTA lumen planes</h2><img src='03_lumen_cross_sections.png' width='95%'><h2>Lumen profiles</h2><img src='04_lumen_profiles.png' width='90%'><p>Research use only. Vesselness alone is not sufficient; serial compact-lumen validation is required. Vessel identity remains unassigned.</p></body></html>")
        (self.cache/"report.done").write_text("done"); self._record("report","computed",html); return html

    def package(self):
        report=self.make_report(); zpath=self.out/"OPENPLAQUE_SECONDARY_ANCHORED_LUMEN_VALIDATION_REPORT_BACK.zip"; include=[report,self.cache/"input_snapshot.json",self.cache/"lumen_calibration.json",self.cache/"summary.json",self.cache/"candidate_validation.csv",self.cache/"best_candidate_qc.csv",self.cache/"anchor_path.csv",self.cache/"extension_path.csv",self.cache/"full_path.csv",self.out/"cache_provenance.csv"]+[self.out/n for n in ["01_anchored_geometry.png","02_anchor_control.png","03_lumen_cross_sections.png","04_lumen_profiles.png"]]
        with zipfile.ZipFile(zpath,"w",compression=zipfile.ZIP_DEFLATED) as z:
            for p in include:
                if p.exists(): z.write(p,arcname=p.name)
        return zpath
