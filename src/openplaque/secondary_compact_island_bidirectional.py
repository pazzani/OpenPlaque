from __future__ import annotations

"""Bidirectional compact-lumen validation of the distal secondary-branch island.

The experiment seeds only inside the compact-looking 13.3-13.6 mm island, estimates a
local tangent from source-CCTA plane quality, then tracks independently backward and
forward using local compact-lumen evidence only. The tracker has no target to the known
12.2 mm segment or to any distal point. Accepted-branch geometry is used only to locate
the island seed and for post-hoc arc correspondence.

Research use only. No LCX identity is assigned automatically.
"""

import json
import math
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

ALGORITHM_VERSION = "secondary-compact-island-bidirectional-v1.0"


def _unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / max(n, 1e-12)


def _arc_mm(path, spacing):
    p = np.asarray(path, float)
    if len(p) == 0:
        return np.array([], float)
    if len(p) == 1:
        return np.array([0.0], float)
    d = np.linalg.norm(np.diff(p, axis=0) * np.asarray(spacing, float), axis=1)
    return np.r_[0.0, np.cumsum(d)]


def _resample_path(path, spacing, step_mm=0.10):
    p = np.asarray(path, float)
    if len(p) <= 1:
        return p.copy()
    s = _arc_mm(p, spacing)
    x = np.arange(0.0, s[-1] + 1e-9, float(step_mm))
    if len(x) == 0 or x[-1] < s[-1] - 0.03:
        x = np.r_[x, s[-1]]
    return np.column_stack([np.interp(x, s, p[:, j]) for j in range(3)])


def _orth_basis(t):
    t = _unit(t)
    ref = np.array([1.0, 0.0, 0.0]) if abs(t[0]) < 0.82 else np.array([0.0, 1.0, 0.0])
    u = _unit(np.cross(t, ref))
    v = _unit(np.cross(t, u))
    return u, v


def _angle_deg(a, b):
    a, b = _unit(a), _unit(b)
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def _write_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2))


def _plane_memmap(ct, point_zyx, tangent_mm, spacing, half_mm=4.0, pix_mm=0.16):
    t = _unit(tangent_mm)
    u, v = _orth_basis(t)
    grid = np.arange(-half_mm, half_mm + 1e-9, pix_mm)
    vv, uu = np.meshgrid(grid, grid, indexing="ij")
    center_mm = np.asarray(point_zyx, float) * np.asarray(spacing, float)
    xyz_mm = center_mm[None, None, :] + uu[..., None] * u + vv[..., None] * v
    zyx = xyz_mm / np.asarray(spacing, float)
    im = ndi.map_coordinates(
        ct,
        [zyx[..., 0], zyx[..., 1], zyx[..., 2]],
        output=np.float32,
        order=1,
        mode="nearest",
    )
    return im, grid, u, v


def _component_metrics(im, grid, comp):
    yy, xx = np.nonzero(comp)
    pix = float(abs(grid[1] - grid[0]))
    area = float(comp.sum()) * pix * pix
    radius = math.sqrt(area / math.pi)
    cu = float(np.mean(grid[xx])); cv = float(np.mean(grid[yy]))
    shift = float(math.hypot(cu, cv))
    er = ndi.binary_erosion(comp)
    perimeter = max(float(np.logical_and(comp, ~er).sum()) * pix, pix)
    circ = float(np.clip(4.0 * math.pi * area / (perimeter * perimeter), 0, 1))
    return {
        "radius_mm": radius,
        "centroid_shift_mm": shift,
        "circularity": circ,
        "component_median_hu": float(np.median(im[comp])),
        "offset_u_mm": cu,
        "offset_v_mm": cv,
    }


def _compact_component(im, grid, threshold_hu, max_shift_mm=0.90, compact_only=False):
    bright = (im >= float(threshold_hu)) & (im <= 1200.0)
    lab, nlab = ndi.label(bright, structure=np.ones((3, 3), np.uint8))
    candidates = []
    for k in range(1, int(nlab) + 1):
        comp = lab == k
        if int(comp.sum()) < 7:
            continue
        m = _component_metrics(im, grid, comp)
        yy, xx = np.nonzero(comp)
        dmin = float(np.min(np.hypot(grid[xx], grid[yy])))
        near = m["centroid_shift_mm"] <= max_shift_mm or dmin <= 0.40
        if not near:
            continue
        if compact_only and not (0.65 <= m["radius_mm"] <= 2.65):
            continue
        candidates.append((m["centroid_shift_mm"] + 0.30 * dmin, m))
    return None if not candidates else min(candidates, key=lambda x: x[0])[1]


def _direction_fan(base_dir, shell_deg=(0, 8, 16, 24, 32), n_azimuth=8):
    t = _unit(base_dir)
    u, v = _orth_basis(t)
    out = []
    for ang in shell_deg:
        if float(ang) == 0.0:
            out.append(t.copy()); continue
        th = math.radians(float(ang))
        for j in range(int(n_azimuth)):
            ph = 2.0 * math.pi * j / float(n_azimuth)
            d = math.cos(th) * t + math.sin(th) * (math.cos(ph) * u + math.sin(ph) * v)
            out.append(_unit(d))
    return out


def synthetic_island_self_test():
    grid = np.arange(-4.0, 4.0 + 1e-9, 0.16)
    yy, xx = np.meshgrid(grid, grid, indexing="ij")
    tube = 100.0 + 650.0 * (((xx - 0.22) ** 2 + (yy + 0.14) ** 2) <= 1.8**2)
    broad = 100.0 + 650.0 * (xx >= -0.10)
    mt = _compact_component(tube, grid, 220.0)
    mb = _compact_component(broad, grid, 220.0)
    dirs = _direction_fan([0, 1, 0])
    return {
        "passed": bool(mt and mb and 1.5 <= mt["radius_mm"] <= 2.1 and mb["radius_mm"] > 2.65 and len(dirs) == 33),
        "tube_radius_mm": None if mt is None else mt["radius_mm"],
        "broad_radius_mm": None if mb is None else mb["radius_mm"],
        "n_tracking_directions": len(dirs),
    }


class SecondaryCompactIslandBidirectionalWorkflow:
    def __init__(self, root="/content/drive/MyDrive/OpenPlaque", reuse=None):
        self.root = Path(root)
        self.cache = self.root / "Cache" / "Secondary_Compact_Island_Bidirectional_v1"
        self.out = self.root / "Secondary_Compact_Island_Bidirectional_Report"
        self.cache.mkdir(parents=True, exist_ok=True); self.out.mkdir(parents=True, exist_ok=True)
        self.reuse = {"inputs": True, "seed": True, "tracking": True, "figures": True, "report": True}
        if reuse:
            self.reuse.update({k: bool(v) for k, v in reuse.items()})
        self.ct = None; self.spacing = None; self.seed_path = None
        self.calibration = None; self.seed_point = None; self.seed_tangent = None; self.seed_arc = None
        self.seed_candidates = None; self.forward = None; self.backward = None; self.terminal = None; self.summary = None

    def cache_status(self):
        names = {"inputs":"input_snapshot.json","seed":"seed_candidates.csv","tracking":"summary.json","figures":"figures.done","report":"report.done"}
        return pd.DataFrame([{"component":k,"reuse":self.reuse[k],"cache_exists":(self.cache/v).exists()} for k,v in names.items()])

    def load_inputs(self):
        src = self.root / "Cache" / "Secondary_3D_Vesselness_Topology_v1"
        ctf = src / "series7_int16.npy"; metaf = src / "series7_int16.json"
        seedf = self.root / "Secondary_Branch_Lateral_Divergence_Report" / "branch_centerline.csv"
        need = [ctf, metaf, seedf]
        if not all(p.exists() for p in need):
            raise FileNotFoundError("Missing source CT / accepted branch: " + "; ".join(str(p) for p in need if not p.exists()))
        meta = json.loads(metaf.read_text())
        self.spacing = np.asarray(meta["spacing_zyx"], float)
        self.ct = np.load(ctf, mmap_mode="r")
        self.seed_path = pd.read_csv(seedf)[["z","y","x"]].to_numpy(float)
        self._calibrate()
        snap = {
            "algorithm": ALGORITHM_VERSION,
            "source_ct": str(ctf),
            "accepted_branch": str(seedf),
            "island_seed_search_arc_mm": [13.25, 13.60],
            "tracking_step_mm": 0.20,
            "maximum_each_direction_mm": 3.0,
            "tracking_direction_hypotheses_per_step": 33,
            "search_target": None,
            "note": "Accepted geometry is used only to locate the compact island seed and post-hoc arc correspondence. Bidirectional tracking has no proximal or distal target.",
        }
        _write_json(snap, self.cache/"input_snapshot.json")
        return snap

    def _calibrate(self):
        sp = _resample_path(self.seed_path, self.spacing, 0.25); s = _arc_mm(sp, self.spacing)
        pts = sp[(s >= 10.0) & (s <= 12.2)]
        hu = ndi.map_coordinates(self.ct, [pts[:,0], pts[:,1], pts[:,2]], output=np.float32, order=1, mode="nearest")
        refhu = float(np.median(hu)); thr = float(max(170.0, min(300.0, 0.38*refhu)))
        rows=[]
        for i,p in enumerate(pts):
            i0=max(0,i-2); i1=min(len(pts)-1,i+2); t=(pts[i1]-pts[i0])*self.spacing
            im,g,_,_=_plane_memmap(self.ct,p,t,self.spacing,4.0,0.16); m=_compact_component(im,g,thr,0.80)
            if m is not None: rows.append(m)
        df=pd.DataFrame(rows); good=df[(df.radius_mm>=0.65)&(df.radius_mm<=2.65)&(df.centroid_shift_mm<=0.80)]
        if len(good)<3: good=df
        self.calibration={"reference_center_hu":refhu,"bright_threshold_hu":thr,"median_radius_mm":float(good.radius_mm.median()),"median_circularity":float(good.circularity.median()),"n_planes":int(len(good))}
        _write_json(self.calibration,self.cache/"lumen_calibration.json")

    def _score_measure(self, m):
        if m is None: return False, 0.0
        passed=bool(0.65<=m["radius_mm"]<=2.65 and m["centroid_shift_mm"]<=0.70 and m["circularity"]>=0.30 and m["component_median_hu"]>=max(220.0,0.45*self.calibration["reference_center_hu"]))
        rref=self.calibration["median_radius_mm"]
        rscore=math.exp(-0.5*((m["radius_mm"]-rref)/max(0.55*rref,0.70))**2)
        sscore=math.exp(-0.5*(m["centroid_shift_mm"]/0.50)**2)
        cscore=float(np.clip(m["circularity"]/max(self.calibration["median_circularity"],0.35),0,1))
        hscore=float(np.clip(m["component_median_hu"]/max(self.calibration["reference_center_hu"],1.0),0,1.2)/1.2)
        return passed,float(0.38*rscore+0.34*sscore+0.16*cscore+0.12*hscore)

    def _local_prior_tangent(self, sp, s, arc):
        i=int(np.argmin(np.abs(s-arc))); i0=max(0,i-4); i1=min(len(sp)-1,i+4)
        return _unit((sp[i1]-sp[i0])*self.spacing)

    def find_seed(self):
        sf=self.cache/"seed_candidates.csv"
        if self.reuse["seed"] and sf.exists() and (self.cache/"seed.json").exists():
            self.seed_candidates=pd.read_csv(sf); d=json.loads((self.cache/"seed.json").read_text())
            self.seed_point=np.asarray(d["seed_zyx"],float); self.seed_tangent=np.asarray(d["seed_tangent_mm"],float); self.seed_arc=float(d["seed_arc_mm"]); return d
        sp=_resample_path(self.seed_path,self.spacing,0.05); s=_arc_mm(sp,self.spacing)
        rows=[]
        for arc in np.arange(13.25,13.60+1e-9,0.05):
            i=int(np.argmin(np.abs(s-arc))); p=sp[i]; prior=self._local_prior_tangent(sp,s,arc)
            for d in _direction_fan(prior,shell_deg=(0,10,20,30),n_azimuth=8):
                im,g,_,_=_plane_memmap(self.ct,p,d,self.spacing,4.0,0.16); m=_compact_component(im,g,self.calibration["bright_threshold_hu"],0.90)
                passed,score=self._score_measure(m)
                if m is None: continue
                rank=score+0.08*math.exp(-_angle_deg(d,prior)/25.0)
                rows.append({"arc_mm":float(arc),"z":float(p[0]),"y":float(p[1]),"x":float(p[2]),"plane_pass":passed,"plane_score":score,"rank_score":rank,"radius_mm":m["radius_mm"],"centroid_shift_mm":m["centroid_shift_mm"],"circularity":m["circularity"],"component_median_hu":m["component_median_hu"],"direction_z_mm":float(d[0]),"direction_y_mm":float(d[1]),"direction_x_mm":float(d[2]),"angle_to_geometry_deg":_angle_deg(d,prior)})
        df=pd.DataFrame(rows).sort_values(["plane_pass","rank_score"],ascending=[False,False]).reset_index(drop=True)
        self.seed_candidates=df; df.to_csv(sf,index=False)
        if not len(df) or not bool(df.iloc[0].plane_pass):
            raise RuntimeError("No stable compact-lumen seed found in 13.25-13.60 mm island")
        r=df.iloc[0]; self.seed_point=np.array([r.z,r.y,r.x],float); self.seed_tangent=_unit(np.array([r.direction_z_mm,r.direction_y_mm,r.direction_x_mm],float)); self.seed_arc=float(r.arc_mm)
        d={"seed_arc_mm":self.seed_arc,"seed_zyx":self.seed_point.tolist(),"seed_tangent_mm":self.seed_tangent.tolist(),"seed_plane_score":float(r.plane_score),"seed_radius_mm":float(r.radius_mm),"seed_shift_mm":float(r.centroid_shift_mm)}
        _write_json(d,self.cache/"seed.json"); return d

    def _posthoc_arc(self, p):
        sp=_resample_path(self.seed_path,self.spacing,0.05); s=_arc_mm(sp,self.spacing); q=sp*self.spacing; tree=cKDTree(q); dist,i=tree.query(np.asarray(p)*self.spacing,k=1)
        return float(s[int(i)]),float(dist)

    def _track_one(self, sign, max_length_mm=3.0, step_mm=0.20):
        label="forward" if sign>0 else "backward"
        cur=self.seed_point.copy(); cur_dir=self.seed_tangent.copy()*float(sign); pts=[cur.copy()]; rows=[]; terminal_rows=[]; cum=0.0
        arc0,d0=self._posthoc_arc(cur)
        rows.append({"direction":label,"step_index":0,"track_length_mm":0.0,"z":cur[0],"y":cur[1],"x":cur[2],"radius_mm":np.nan,"centroid_shift_mm":np.nan,"plane_score":np.nan,"turn_deg":0.0,"actual_step_mm":0.0,"posthoc_nearest_arc_mm":arc0,"posthoc_distance_to_accepted_mm":d0})
        step_index=0
        while cum+0.5*step_mm<=max_length_mm:
            proposals=[]; reject={}
            for d in _direction_fan(cur_dir):
                nominal_phys=cur*self.spacing+float(step_mm)*d; nominal=nominal_phys/self.spacing
                if np.any(nominal<2.0) or np.any(nominal>=np.asarray(self.ct.shape)-3.0):
                    reject["outside_volume"]=reject.get("outside_volume",0)+1; continue
                im,g,u,v=_plane_memmap(self.ct,nominal,d,self.spacing,3.8,0.16)
                m0=_compact_component(im,g,self.calibration["bright_threshold_hu"],0.95,compact_only=False)
                if m0 is None:
                    reject["no_component"]=reject.get("no_component",0)+1; continue
                if not (0.65<=m0["radius_mm"]<=2.65):
                    reject["radius_too_large_or_small"]=reject.get("radius_too_large_or_small",0)+1; continue
                corrected_phys=nominal_phys+m0["offset_u_mm"]*u+m0["offset_v_mm"]*v; corrected=corrected_phys/self.spacing
                vec=corrected_phys-cur*self.spacing; actual_step=float(np.linalg.norm(vec)); actual_dir=_unit(vec)
                if not (0.10<=actual_step<=0.45):
                    reject["step_length"]=reject.get("step_length",0)+1; continue
                turn=_angle_deg(cur_dir,actual_dir)
                if turn>48.0:
                    reject["turn_gt_48"]=reject.get("turn_gt_48",0)+1; continue
                if float(np.dot(actual_dir,cur_dir))<0.60:
                    reject["insufficient_forward_progress"]=reject.get("insufficient_forward_progress",0)+1; continue
                im2,g2,_,_=_plane_memmap(self.ct,corrected,actual_dir,self.spacing,3.8,0.16)
                m=_compact_component(im2,g2,self.calibration["bright_threshold_hu"],0.70,compact_only=False)
                passed,score=self._score_measure(m)
                if m is None:
                    reject["no_component_after_recenter"]=reject.get("no_component_after_recenter",0)+1; continue
                if not passed:
                    reason="radius_too_large_or_small" if not (0.65<=m["radius_mm"]<=2.65) else "lumen_gate"
                    reject[reason]=reject.get(reason,0)+1; continue
                continuity=math.exp(-turn/35.0); rank=0.82*score+0.18*continuity
                proposals.append((rank,corrected,actual_dir,m,score,turn,actual_step))
            if not proposals:
                terminal_rows.append({"direction":label,"terminal_step_index":step_index+1,"track_length_mm":cum,"rejection_counts_json":json.dumps(reject,sort_keys=True),"n_directions_tested":len(_direction_fan(cur_dir))})
                break
            _,pnew,dnew,m,score,turn,actual_step=max(proposals,key=lambda x:x[0])
            if len(pts)>4:
                old=np.asarray(pts[:-2])*self.spacing
                if np.min(np.linalg.norm(old-pnew*self.spacing,axis=1))<0.30:
                    terminal_rows.append({"direction":label,"terminal_step_index":step_index+1,"track_length_mm":cum,"rejection_counts_json":json.dumps({"loop_guard":1}),"n_directions_tested":len(_direction_fan(cur_dir))}); break
            cur=pnew; cur_dir=dnew; cum+=actual_step; step_index+=1; pts.append(cur.copy()); pa,pd=self._posthoc_arc(cur)
            rows.append({"direction":label,"step_index":step_index,"track_length_mm":cum,"z":cur[0],"y":cur[1],"x":cur[2],"radius_mm":m["radius_mm"],"centroid_shift_mm":m["centroid_shift_mm"],"circularity":m["circularity"],"component_median_hu":m["component_median_hu"],"plane_score":score,"turn_deg":turn,"actual_step_mm":actual_step,"posthoc_nearest_arc_mm":pa,"posthoc_distance_to_accepted_mm":pd})
        return pd.DataFrame(rows),pd.DataFrame(terminal_rows),np.asarray(pts,float)

    def run_tracking(self):
        if self.seed_point is None: self.find_seed()
        if self.reuse["tracking"] and (self.cache/"summary.json").exists():
            self.forward=pd.read_csv(self.cache/"forward_track.csv"); self.backward=pd.read_csv(self.cache/"backward_track.csv"); self.terminal=pd.read_csv(self.cache/"terminal_diagnostics.csv"); self.summary=json.loads((self.cache/"summary.json").read_text()); return self.summary
        f,ft,fp=self._track_one(+1); b,bt,bp=self._track_one(-1)
        self.forward=f; self.backward=b; self.terminal=pd.concat([ft,bt],ignore_index=True)
        f.to_csv(self.cache/"forward_track.csv",index=False); b.to_csv(self.cache/"backward_track.csv",index=False); self.terminal.to_csv(self.cache/"terminal_diagnostics.csv",index=False)
        np.savetxt(self.cache/"forward_path.csv",fp,delimiter=",",header="z,y,x",comments=""); np.savetxt(self.cache/"backward_path.csv",bp,delimiter=",",header="z,y,x",comments="")
        fl=float(f.track_length_mm.max()) if len(f) else 0.0; bl=float(b.track_length_mm.max()) if len(b) else 0.0
        minarc=float(min(b.posthoc_nearest_arc_mm.min() if len(b) else self.seed_arc, f.posthoc_nearest_arc_mm.min() if len(f) else self.seed_arc))
        maxarc=float(max(b.posthoc_nearest_arc_mm.max() if len(b) else self.seed_arc, f.posthoc_nearest_arc_mm.max() if len(f) else self.seed_arc))
        back_reaches_prox=bool(len(b) and b.posthoc_nearest_arc_mm.min()<=12.30 and b.posthoc_distance_to_accepted_mm.min()<=0.45)
        forward_beyond=bool(len(f) and f.posthoc_nearest_arc_mm.max()>=13.80 and fl>=0.40)
        if fl>=0.50 and bl>=0.50: status="ISLAND_BIDIRECTIONAL_COMPACT_SEGMENT"
        elif fl>=0.50: status="ISLAND_FORWARD_COMPACT_ONLY"
        elif bl>=0.50: status="ISLAND_BACKWARD_COMPACT_ONLY"
        else: status="ISLAND_SHORT_LOCAL_APPEARANCE"
        self.summary={"algorithm":ALGORITHM_VERSION,"status":status,"accepted_continuation":False,"island_seed_arc_mm":self.seed_arc,"forward_compact_length_mm":fl,"backward_compact_length_mm":bl,"total_bidirectional_compact_extent_mm":fl+bl,"posthoc_min_nearest_accepted_arc_mm":minarc,"posthoc_max_nearest_accepted_arc_mm":maxarc,"posthoc_backward_reaches_12_3mm_neighborhood":back_reaches_prox,"posthoc_forward_reaches_geometric_endpoint_neighborhood":forward_beyond,"vessel_identity_assigned":False,"interpretation":"The compact island is evaluated as an independent local object. Bidirectional tracking uses only serial compact-lumen source-CCTA evidence and local continuity; post-hoc accepted-arc correspondence does not guide tracking."}
        _write_json(self.summary,self.cache/"summary.json"); return self.summary

    def make_figures(self):
        if self.summary is None: self.run_tracking()
        names=[]
        sp=_resample_path(self.seed_path,self.spacing,0.10); q=sp*self.spacing
        fig=plt.figure(figsize=(8,6)); ax=fig.add_subplot(111,projection="3d"); ax.plot(q[:,2],q[:,1],q[:,0],lw=1.2,label="accepted geometry")
        for df,label in [(self.backward,"backward"),(self.forward,"forward")]:
            if len(df):
                p=df[["z","y","x"]].to_numpy()*self.spacing; ax.plot(p[:,2],p[:,1],p[:,0],lw=2.5,label=label)
        s=self.seed_point*self.spacing; ax.scatter([s[2]],[s[1]],[s[0]],s=55,label="island seed"); ax.set_title("Compact-island bidirectional tracking"); ax.legend(); fig.tight_layout(); fn="01_bidirectional_geometry.png"; fig.savefig(self.out/fn,dpi=160); plt.close(fig); names.append(fn)
        rows=[]
        for df,sgn in [(self.backward,-1),(self.forward,+1)]:
            if len(df):
                x=df.copy(); x["signed_mm"]=sgn*x.track_length_mm; rows.append(x)
        allp=pd.concat(rows,ignore_index=True).sort_values("signed_mm") if rows else pd.DataFrame()
        if len(allp):
            fig,ax=plt.subplots(figsize=(8,5)); ax.plot(allp.signed_mm,allp.radius_mm,"o-"); ax.axhline(2.65,ls="--"); ax.set_xlabel("signed distance from island seed (mm)"); ax.set_ylabel("radius (mm)"); ax.set_title("Serial compact-lumen radius"); fig.tight_layout(); fn="02_bidirectional_radius.png"; fig.savefig(self.out/fn,dpi=160); plt.close(fig); names.append(fn)
        selections=[("seed",self.seed_point,self.seed_tangent)]
        if len(self.backward)>1:
            r=self.backward.iloc[-1]; selections.append(("backward last",np.array([r.z,r.y,r.x]),-self.seed_tangent))
        if len(self.forward)>1:
            r=self.forward.iloc[-1]; selections.append(("forward last",np.array([r.z,r.y,r.x]),self.seed_tangent))
        fig,axs=plt.subplots(1,len(selections),figsize=(4*len(selections),4)); axs=np.atleast_1d(axs)
        for ax,(title,p,d) in zip(axs,selections):
            im,g,_,_=_plane_memmap(self.ct,p,d,self.spacing,4.0,0.16); ax.imshow(im,cmap="gray",vmin=100,vmax=900,extent=[g[0],g[-1],g[-1],g[0]]); ax.axhline(0,lw=.5); ax.axvline(0,lw=.5); ax.set_title(title); ax.set_xlabel("mm")
        fig.tight_layout(); fn="03_bidirectional_cross_sections.png"; fig.savefig(self.out/fn,dpi=160); plt.close(fig); names.append(fn)
        (self.cache/"figures.done").write_text("done"); return names

    def package(self):
        if self.summary is None: self.run_tracking()
        if not (self.cache/"figures.done").exists(): self.make_figures()
        html=self.out/"OPENPLAQUE_SECONDARY_COMPACT_ISLAND_BIDIRECTIONAL_REPORT.html"
        s=self.summary
        html.write_text(f"<html><body><h1>OpenPlaque Compact Island Bidirectional Validation</h1><p><b>Status:</b> {s['status']}</p><p><b>Backward compact length:</b> {s['backward_compact_length_mm']:.3f} mm</p><p><b>Forward compact length:</b> {s['forward_compact_length_mm']:.3f} mm</p><p><b>Total local compact extent:</b> {s['total_bidirectional_compact_extent_mm']:.3f} mm</p><p>Tracking had no proximal or distal target. Accepted anatomy was used only to locate the island seed and for post-hoc correspondence.</p><p><img src='01_bidirectional_geometry.png' style='max-width:100%'></p><p><img src='02_bidirectional_radius.png' style='max-width:100%'></p><p><img src='03_bidirectional_cross_sections.png' style='max-width:100%'></p></body></html>")
        for p in [self.cache/"input_snapshot.json",self.cache/"lumen_calibration.json",self.cache/"seed.json",self.cache/"seed_candidates.csv",self.cache/"forward_track.csv",self.cache/"backward_track.csv",self.cache/"terminal_diagnostics.csv",self.cache/"forward_path.csv",self.cache/"backward_path.csv",self.cache/"summary.json"]:
            if p.exists(): (self.out/p.name).write_bytes(p.read_bytes())
        zipf=self.out/"OPENPLAQUE_SECONDARY_COMPACT_ISLAND_BIDIRECTIONAL_REPORT_BACK.zip"
        with zipfile.ZipFile(zipf,"w",zipfile.ZIP_DEFLATED) as z:
            for p in self.out.iterdir():
                if p.is_file() and p!=zipf: z.write(p,p.name)
        (self.cache/"report.done").write_text("done"); return zipf
