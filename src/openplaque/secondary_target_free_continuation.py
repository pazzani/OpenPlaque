from __future__ import annotations

"""Target-free continuation of the accepted OpenPlaque secondary branch.
The prior 17.9-mm relaunch is comparison-only; it is never a target or score term.
Research use only. No LCX identity is assigned automatically.
"""
import json, math
from pathlib import Path
import numpy as np
import pandas as pd
from .lcx_gap_bridge import LCXGapBridgeWorkflow, _angle, _nearest_dist, _serial_qc, _unit, _write_json, arc_mm, resample_path

ALGORITHM_VERSION = "secondary-target-free-continuation-v1.0"

class SecondaryTargetFreeContinuationWorkflow(LCXGapBridgeWorkflow):
    COMPONENTS = ("source_ct", "frozen_geometry", "continuation_search")

    def __init__(self, root="/content/drive/MyDrive/OpenPlaque", reuse=None):
        self.root = Path(root)
        self.cache = self.root / "Cache" / "Secondary_Target_Free_Continuation_v1"
        self.out = self.root / "Secondary_Target_Free_Continuation_Report"
        self.cache.mkdir(parents=True, exist_ok=True); self.out.mkdir(parents=True, exist_ok=True)
        self.reuse = {k: True for k in self.COMPONENTS}
        if reuse: self.reuse.update({k: bool(v) for k, v in reuse.items()})
        self.prov = []
        self.ct = self.meta = self.spacing = None
        self.seed = self.seed_prefix = self.reference = self.trunk = self.rca = None
        self.prior_course = None
        self.source_point = self.initial_direction = None
        self.best_path = self.best_qc = self.candidates = self.summary = self.diag = None

    def cache_status(self):
        names = {"source_ct":"series7_int16.npy", "frozen_geometry":"frozen_geometry.json",
                 "continuation_search":"continuation_summary.json"}
        return pd.DataFrame([{"component":k,"reuse":self.reuse[k],"cache_exists":(self.cache/v).exists()} for k,v in names.items()])

    def load_frozen_geometry(self):
        if self.ct is None: self.load_source_ct()
        seed = self.root/"Secondary_Branch_Lateral_Divergence_Report"/"branch_centerline.csv"
        ss = self.root/"Secondary_Branch_Lateral_Divergence_Report"/"branch_summary.json"
        ref = self.root/"LAD_Takeoff_Root_Alternatives_Report"/"alternative_01_centerline.csv"
        trunk = self.root/"LAD_Takeoff_Confirmation_Report"/"trunk_centerline.csv"
        rca = self.root/"LAD_Takeoff_Confirmation_Report"/"validated_rca_calibration.json"
        prior = self.root/"LCX_Distal_Relaunch_Report"/"best_combined_centerline.csv"
        if not all(p.exists() for p in (seed,ss,ref,trunk)):
            raise FileNotFoundError("Required accepted secondary-branch / LAD-trunk outputs not found")
        if not json.loads(ss.read_text()).get("accepted", False):
            raise RuntimeError("Accepted lateral-divergence branch missing")
        self.seed = pd.read_csv(seed)[["z","y","x"]].to_numpy(float)
        self.reference = pd.read_csv(ref)[["z","y","x"]].to_numpy(float)
        self.trunk = pd.read_csv(trunk)[["z","y","x"]].to_numpy(float)
        self.prior_course = pd.read_csv(prior)[["z","y","x"]].to_numpy(float) if prior.exists() else None
        self.rca = json.loads(rca.read_text()) if rca.exists() else {"median_radius_mm":1.616906,"median_center_hu":551.061478}

        seed_qc, qsum = _serial_qc(self.seed, self.ct, self.spacing, self.rca, 0.50)
        good = seed_qc[(seed_qc.plane_pass==True)&(seed_qc.plane_score>=0.90)&(seed_qc.radius_mm<=2.25)&
                       (seed_qc.recenter_shift_mm<=0.80)&(seed_qc.arc_mm>=11.4)&(seed_qc.arc_mm<=13.1)]
        source_arc = float(good.arc_mm.max()) if len(good) else 12.2
        sp = resample_path(self.seed, self.spacing, 0.18); s = arc_mm(sp, self.spacing)
        i = int(np.argmin(abs(s-source_arc))); self.source_point = sp[i].copy(); source_arc=float(s[i]); self.seed_prefix=sp[:i+1]
        j = int(np.argmin(abs(s-max(0.0,source_arc-2.0)))); j=min(j,max(0,i-1))
        self.initial_direction = _unit((sp[i]-sp[j])*self.spacing)
        prior_return = None
        if self.prior_course is not None:
            pp=resample_path(self.prior_course,self.spacing,0.18); ps=arc_mm(pp,self.spacing); k=int(np.argmin(abs(ps-17.9)))
            prior_return=float(np.linalg.norm((pp[k]-self.source_point)*self.spacing))
        snap={"algorithm":ALGORITHM_VERSION,"source_arc_mm":source_arc,"source_zyx":self.source_point.tolist(),
              "initial_direction_zyx_mm":self.initial_direction.tolist(),"accepted_seed_qc":qsum,
              "prior_relaunch_17p9_straight_distance_from_source_mm":prior_return,"search_target":None,
              "note":"Prior relaunch is comparison-only; no distal target is used."}
        _write_json(snap,self.cache/"frozen_geometry.json"); seed_qc.to_csv(self.out/"accepted_seed_qc.csv",index=False)
        self._record("frozen_geometry","loaded_and_fixed_target_free_source",self.cache/"frozen_geometry.json")
        return snap

    def _source_direction_hypotheses(self):
        p=resample_path(self.seed,self.spacing,0.18); s=arc_mm(p,self.spacing)
        i=int(np.argmin(np.linalg.norm((p-self.source_point)*self.spacing,axis=1))); out=[]
        for back in (0.8,1.4,2.2,3.2):
            j=int(np.argmin(abs(s-max(0.0,s[i]-back))))
            if j<i:
                d=_unit((p[i]-p[j])*self.spacing)
                if all(_angle(d,q)>=8 for q in out): out.append(d)
        return out or [self.initial_direction.copy()]

    @staticmethod
    def _basis(t):
        t=_unit(t); ref=np.array([1.,0.,0.]) if abs(t[0])<0.82 else np.array([0.,1.,0.])
        u=_unit(np.cross(t,ref)); return u,_unit(np.cross(t,u))

    def _old_sep(self,point):
        s=arc_mm(self.seed_prefix,self.spacing); old=self.seed_prefix[s<=max(0.0,s[-1]-1.25)]
        return _nearest_dist(point,old,self.spacing)

    def _self_sep(self,point,path):
        p=np.asarray(path,float); return float("inf") if len(p)<=5 else _nearest_dist(point,p[:-5],self.spacing)

    def _metrics(self,path):
        p=np.asarray(path,float); L=float(arc_mm(p,self.spacing)[-1]); d=float(np.linalg.norm((p[-1]-self.source_point)*self.spacing))
        proj=float(np.dot((p[-1]-self.source_point)*self.spacing,self.initial_direction))
        return {"length_mm":L,"endpoint_displacement_mm":d,"forward_projection_mm":proj,
                "endpoint_old_branch_separation_mm":self._old_sep(p[-1]),
                "endpoint_reference_separation_mm":_nearest_dist(p[-1],np.vstack([self.reference,self.trunk]),self.spacing),
                "tortuosity":L/max(d,1e-6)}

    def search_continuation(self,max_new_mm=10.0,beam_width=50):
        sf=self.cache/"continuation_summary.json"; pf=self.cache/"continuation_centerline.csv"; qf=self.cache/"continuation_qc.csv"
        cf=self.cache/"continuation_candidates.csv"; df=self.cache/"step_diagnostics.csv"
        if self.source_point is None: self.load_frozen_geometry()
        if self.reuse["continuation_search"] and all(p.exists() for p in (sf,pf,qf,cf,df)):
            self.summary=json.loads(sf.read_text()); self.best_path=pd.read_csv(pf)[["z","y","x"]].to_numpy(float)
            self.best_qc=pd.read_csv(qf); self.candidates=pd.read_csv(cf); self.diag=pd.read_csv(df); return self.summary

        beams=[{"point":self.source_point.copy(),"direction":d,"path":[self.source_point.copy()],"scores":[],"hard":[],"turns":[],"oldsep":[]} for d in self._source_direction_hypotheses()]
        pool=[]; diag=[]
        for step_i in range(int(math.ceil(max_new_mm/0.28))):
            props=[]; step_mm=0.26 if step_i<8 else 0.30
            for bi,st in enumerate(beams):
                base=st["direction"]; dirs=[(base,0.)]; u,v=self._basis(base)
                for ang in (8,16,24,32,40):
                    aa=math.radians(ang)
                    for ph in np.linspace(0,2*math.pi,12,endpoint=False):
                        dirs.append((_unit(math.cos(aa)*base+math.sin(aa)*(math.cos(ph)*u+math.sin(ph)*v)),float(ang)))
                for d,nominal in dirs:
                    r=self._propose(st["point"],d,step_mm)
                    if r is None: continue
                    turn=_angle(st["direction"],r["direction"])
                    if turn>58: continue
                    old_sep=self._old_sep(r["point"]); self_sep=self._self_sep(r["point"],st["path"])
                    path=st["path"]+[r["point"].copy()]; m=self._metrics(path)
                    if m["length_mm"]>=1.2 and old_sep<0.80: continue
                    if m["length_mm"]>=1.8 and self_sep<0.62: continue
                    if m["length_mm"]>=2.5 and m["tortuosity"]>2.15: continue
                    if m["length_mm"]>=3.0 and m["forward_projection_mm"]<0.55: continue
                    smooth=float(np.clip(np.dot(st["direction"],r["direction"]),-1,1))
                    obj=0.48*r["score"]+0.17*float(r["hard"])+0.16*((smooth+1)/2)+0.11*min(old_sep/2.2,1)+0.08*min(max(m["forward_projection_mm"],0)/4,1)
                    ns={"point":r["point"],"direction":r["direction"],"path":path,"scores":st["scores"]+[r["score"]],
                        "hard":st["hard"]+[r["hard"]],"turns":st["turns"]+[turn],"oldsep":st["oldsep"]+[old_sep]}
                    props.append((obj,ns)); diag.append({"step_index":step_i,"beam_index":bi,"nominal_turn_deg":nominal,
                        "actual_turn_deg":turn,"step_mm":r["step"],"plane_score":r["score"],"hard":r["hard"],
                        "radius_mm":r["radius_mm"],"recenter_shift_mm":r["recenter_shift_mm"],"new_length_mm":m["length_mm"],
                        "endpoint_displacement_mm":m["endpoint_displacement_mm"],"forward_projection_mm":m["forward_projection_mm"],
                        "old_branch_separation_mm":old_sep,"self_separation_mm":self_sep,"reference_separation_mm":r["refsep"]})
            if not props: break
            def rank(item):
                _,st=item; m=self._metrics(st["path"])
                return 0.36*np.mean(st["scores"])+0.16*np.mean(st["hard"])+0.20*min(m["length_mm"]/7,1)+0.12*min(m["endpoint_displacement_mm"]/5,1)+0.09*min(st["oldsep"][-1]/2.2,1)+0.07*min(max(m["forward_projection_mm"],0)/4,1)
            props.sort(key=rank,reverse=True); keep=[]
            for _,st in props:
                if all(np.linalg.norm((st["point"]-q["point"])*self.spacing)>=0.28 for q in keep): keep.append(st)
                if len(keep)>=beam_width: break
            pool.extend(keep); beams=keep
            if max(self._metrics(st["path"])["length_mm"] for st in beams)>=max_new_mm: break

        rows=[]; evaluated=[]
        for st in pool:
            path=np.asarray(st["path"],float); m=self._metrics(path)
            if m["length_mm"]<3.5: continue
            qdf,q=_serial_qc(path,self.ct,self.spacing,self.rca,0.36); hard=float(np.mean(st["hard"])); max_turn=float(max(st["turns"]))
            continuous=bool(q["soft_pass_fraction"]>=0.90 and q["plane_pass_fraction"]>=0.82 and q["median_plane_score"]>=0.80 and q["max_consecutive_soft_failures"]<=1 and hard>=0.78 and max_turn<=58)
            novel=bool(m["length_mm"]>=5.5 and m["endpoint_displacement_mm"]>=3.2 and m["forward_projection_mm"]>=1.3 and m["endpoint_old_branch_separation_mm"]>=1.0 and m["tortuosity"]<=1.85)
            supported=continuous and novel
            score=0.22*min(m["length_mm"]/7,1)+0.22*q["plane_pass_fraction"]+0.18*q["median_plane_score"]+0.12*q["soft_pass_fraction"]+0.10*hard+0.08*min(m["endpoint_displacement_mm"]/5,1)+0.08*min(m["endpoint_old_branch_separation_mm"]/2,1)
            row={"candidate":len(rows)+1,**m,"plane_pass_fraction":q["plane_pass_fraction"],"soft_pass_fraction":q["soft_pass_fraction"],
                 "median_plane_score":q["median_plane_score"],"median_radius_mm":q["median_radius_mm"],"max_consecutive_soft_failures":q["max_consecutive_soft_failures"],
                 "step_hard_fraction":hard,"max_turn_deg":max_turn,"continuous_gate":continuous,"genuinely_new_gate":novel,"supported_continuation":supported,"rank_score":score}
            rows.append(row); evaluated.append((score,row,qdf,path))
        self.diag=pd.DataFrame(diag); self.diag.to_csv(df,index=False); self.candidates=pd.DataFrame(rows)
        if len(self.candidates): self.candidates=self.candidates.sort_values(["supported_continuation","continuous_gate","rank_score"],ascending=[False,False,False])
        self.candidates.to_csv(cf,index=False)
        if not evaluated:
            self.best_path=np.asarray([self.source_point]); self.best_qc=pd.DataFrame(); self.summary={"algorithm":ALGORITHM_VERSION,"status":"NO_EVALUABLE_FORWARD_COURSE","accepted_continuation":False,"interpretation":"No target-free candidate >=3.5 mm survived for serial QC."}
        else:
            evaluated.sort(key=lambda x:(x[1]["supported_continuation"],x[1]["continuous_gate"],x[0]),reverse=True); _,row,qdf,path=evaluated[0]
            self.best_path=resample_path(path,self.spacing,0.20); self.best_qc=qdf
            if row["supported_continuation"]: status="SUPPORTED_TARGET_FREE_CONTINUATION"; interp="Continuous >=5.5-mm geometrically new coronary-like continuation passes serial source-CCTA QC; identity remains unassigned."
            elif row["continuous_gate"]: status="CONTINUOUS_BUT_NOT_GEOMETRICALLY_NEW"; interp="A locally continuous trajectory was found but failed anti-loopback/new-course geometry."
            else: status="NO_SUPPORTED_TARGET_FREE_CONTINUATION"; interp="Multi-mm trajectories were found, but none met complete continuity plus new-course criteria."
            self.summary={"algorithm":ALGORITHM_VERSION,**row,"status":status,"accepted_continuation":bool(row["supported_continuation"]),"interpretation":interp}
        pd.DataFrame({"arc_mm":arc_mm(self.best_path,self.spacing),"z":self.best_path[:,0],"y":self.best_path[:,1],"x":self.best_path[:,2]}).to_csv(pf,index=False)
        self.best_qc.to_csv(qf,index=False); _write_json(self.summary,sf); self._record("continuation_search","recomputed_and_cached",sf,f"status={self.summary.get('status')}")
        return self.summary
