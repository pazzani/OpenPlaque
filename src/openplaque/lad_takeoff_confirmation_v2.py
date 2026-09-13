from __future__ import annotations

"""Selection fix for focused LAD takeoff confirmation.

The v1 secondary-branch search required 0.75 mm separation from the known
LAD/trunk after the very first nominal 0.72 mm step. A real bifurcating branch
cannot reliably satisfy that geometry. This subclass keeps the same source-plane
lumen gates but uses a distance-from-origin-dependent independence requirement:
0.40 mm initially, 0.90 mm after 1.2 mm, and 1.40 mm after 2.5 mm. Final branch
acceptance still requires >=2.3 mm endpoint separation.
"""

import math
import numpy as np
import pandas as pd

from .lad_takeoff_confirmation import (
    LADTakeoffConfirmationWorkflow as _Base,
    _angle_deg,
    _cone,
    _json_write,
    _unit,
    arc_mm,
    resample_path,
    serial_qc,
)

ALGORITHM_VERSION = "lad-takeoff-confirmation-v1.1-branch-separation-fix"


class LADTakeoffConfirmationWorkflow(_Base):
    def trace_secondary_branch(self, max_length_mm=10.0, beam_width=12):
        sf = self.cache / "secondary_branch_summary.json"
        pf = self.cache / "secondary_branch_centerline.csv"
        qf = self.cache / "secondary_branch_qc.csv"
        cf = self.cache / "secondary_branch_candidates.csv"
        if self.trunk_summary is None:
            self.evaluate_trunk()
        if self.reuse["secondary_branch"] and all(p.exists() for p in (sf,pf,qf,cf)):
            self.branch_summary = __import__('json').loads(sf.read_text())
            self.branch = pd.read_csv(pf)[["z","y","x"]].to_numpy(float)
            self.branch_qc = pd.read_csv(qf)
            self.branch_candidates = pd.read_csv(cf)
            self._record("secondary_branch", "reused", sf)
            return self.branch_summary

        td = resample_path(self.lad_distal,self.spacing,0.25)
        tr = resample_path(self.trunk,self.spacing,0.25)
        lad_dir = _unit((td[min(len(td)-1,8)]-td[0])*self.spacing)
        trunk_prox_dir = _unit((tr[0]-tr[-1])*self.spacing)
        seeds=[]
        for d in _cone(lad_dir,(35,50,65,80,95,110,125),16):
            if _angle_deg(d,lad_dir) < 32 or _angle_deg(d,trunk_prox_dir) < 28:
                continue
            r=self._propose_branch(self.takeoff,d,self.alt1,rescue=True)
            if r is None or r["reference_separation_mm"] < 0.40:
                continue
            seeds.append({"point":r["point"],"direction":r["direction"],"path":[self.takeoff.copy(),r["point"].copy()],"scores":[r["plane_score"]],"hard":[r["hard"]],"soft":[r["soft"]],"sep":[r["reference_separation_mm"]],"obj":r["plane_score"]})
        if not seeds:
            self.branch=np.asarray([self.takeoff]); self.branch_qc=pd.DataFrame(); self.branch_candidates=pd.DataFrame()
            self.branch_summary={"status":"FAIL","accepted":False,"reason":"no_distinct_coronary_sized_secondary_seed"}
            pd.DataFrame({"z":[self.takeoff[0]],"y":[self.takeoff[1]],"x":[self.takeoff[2]]}).to_csv(pf,index=False)
            self.branch_qc.to_csv(qf,index=False); self.branch_candidates.to_csv(cf,index=False); _json_write(self.branch_summary,sf)
            self._record("secondary_branch","recomputed_and_cached",sf,self.branch_summary["reason"])
            return self.branch_summary
        seeds.sort(key=lambda st:np.mean(st["scores"]),reverse=True)
        beams=seeds[:beam_width]; pool=list(beams)
        nsteps=int(math.ceil(max_length_mm/0.72))
        for _step in range(1,nsteps):
            props=[]
            for st in beams:
                local=[]
                cur_len=float(arc_mm(np.asarray(st["path"]),self.spacing)[-1])
                minsep = 1.40 if cur_len >= 2.5 else (0.90 if cur_len >= 1.2 else 0.40)
                for d in _cone(st["direction"],(0,18,32,46),10):
                    r=self._propose_branch(st["point"],d,self.alt1,rescue=False)
                    if r is not None and r["reference_separation_mm"] >= minsep:
                        local.append(r)
                if not local:
                    for d in _cone(st["direction"],(55,70),10):
                        r=self._propose_branch(st["point"],d,self.alt1,rescue=True)
                        if r is not None and r["reference_separation_mm"] >= minsep:
                            local.append(r)
                for r in local:
                    smooth=float(np.clip(np.dot(_unit(st["direction"]),_unit(r["direction"])),-1,1))
                    obj=0.62*r["plane_score"]+0.14*float(r["hard"])+0.10*((smooth+1)/2)+0.14*min(r["reference_separation_mm"]/4.0,1.0)
                    props.append({"point":r["point"],"direction":r["direction"],"path":st["path"]+[r["point"].copy()],"scores":st["scores"]+[r["plane_score"]],"hard":st["hard"]+[r["hard"]],"soft":st["soft"]+[r["soft"]],"sep":st["sep"]+[r["reference_separation_mm"]],"obj":st["obj"]+obj})
            if not props:
                break
            def rank(st):
                L=float(arc_mm(np.asarray(st["path"]),self.spacing)[-1])
                return 0.42*np.mean(st["scores"])+0.22*np.mean(st["hard"])+0.20*min(L/8.0,1)+0.16*min(np.median(st["sep"])/3.0,1)
            props.sort(key=rank,reverse=True)
            keep=[]
            for st in props:
                if all(np.linalg.norm((st["point"]-q["point"])*self.spacing)>=0.65 for q in keep):
                    keep.append(st)
                if len(keep)>=beam_width:
                    break
            pool.extend(keep); beams=keep

        rows=[]; evaluated=[]
        for st in pool:
            path=np.asarray(st["path"],float); L=float(arc_mm(path,self.spacing)[-1])
            if L<2.5:
                continue
            qdf,qsum=serial_qc(path,self.ct,self.spacing,self.rca,0.75,allow_larger=False,label="SECONDARY_BRANCH")
            sep=float(np.median(st["sep"])); endsep=float(st["sep"][-1]); initial_ang=_angle_deg((path[1]-path[0])*self.spacing,lad_dir)
            rank=0.34*min(L/8,1)+0.28*qsum["plane_pass_fraction"]+0.24*qsum["median_plane_score"]+0.14*min(endsep/4,1)
            rows.append({"candidate":len(rows)+1,"length_mm":L,"plane_pass_fraction":qsum["plane_pass_fraction"],"median_plane_score":qsum["median_plane_score"],"median_radius_mm":qsum["median_radius_mm"],"median_reference_separation_mm":sep,"endpoint_reference_separation_mm":endsep,"initial_angle_from_lad_deg":initial_ang,"rank_score":rank})
            evaluated.append((rank,L,qsum,qdf,path,st))
        if not evaluated:
            self.branch=np.asarray([self.takeoff]); self.branch_qc=pd.DataFrame(); self.branch_candidates=pd.DataFrame(rows)
            self.branch_summary={"status":"FAIL","accepted":False,"reason":"no_secondary_branch_persisted_beyond_2_5_mm"}
        else:
            evaluated.sort(key=lambda x:x[0],reverse=True)
            rank,L,qsum,qdf,path,st=evaluated[0]
            self.branch=resample_path(path,self.spacing,0.30); self.branch_qc=qdf
            accepted=bool(L>=6.0 and qsum["plane_pass_fraction"]>=0.72 and qsum["soft_pass_fraction"]>=0.85 and qsum["median_plane_score"]>=0.78 and st["sep"][-1]>=2.3)
            self.branch_summary={**qsum,"status":"PASS" if accepted else "REVIEW","accepted":accepted,"rank_score":float(rank),"endpoint_reference_separation_mm":float(st["sep"][-1]),"median_reference_separation_mm":float(np.median(st["sep"])),"initial_angle_from_lad_deg":float(_angle_deg((path[1]-path[0])*self.spacing,lad_dir)),"interpretation":"persistent independent coronary-like secondary branch from candidate; branch is not automatically labeled LCX" if accepted else "secondary direction did not meet persistence/independence gate"}
        self.branch_candidates=pd.DataFrame(rows).sort_values("rank_score",ascending=False) if rows else pd.DataFrame()
        pd.DataFrame({"arc_mm":arc_mm(self.branch,self.spacing),"z":self.branch[:,0],"y":self.branch[:,1],"x":self.branch[:,2]}).to_csv(pf,index=False)
        self.branch_qc.to_csv(qf,index=False); self.branch_candidates.to_csv(cf,index=False); _json_write(self.branch_summary,sf)
        self._record("secondary_branch","recomputed_and_cached",sf,f"status={self.branch_summary.get('status')}, length={self.branch_summary.get('length_mm',0):.1f} mm")
        return self.branch_summary
