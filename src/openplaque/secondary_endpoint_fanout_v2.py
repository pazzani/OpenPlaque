from __future__ import annotations

"""Robust wrapper for endpoint fan-out: local-direction projection + safe empty-cache reload."""

import json
import numpy as np
import pandas as pd

from . import secondary_3d_vesselness_topology as base
from .secondary_endpoint_fanout import SecondaryEndpointFanoutWorkflow as _V1

ALGORITHM_VERSION = "secondary-endpoint-fanout-v1.1-local-direction"


def _safe_csv(path, columns=None):
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=[] if columns is None else columns)


class SecondaryEndpointFanoutWorkflow(_V1):
    def __init__(self, root="/content/drive/MyDrive/OpenPlaque", reuse=None):
        super().__init__(root=root, reuse=reuse)
        self.cache = self.root / "Cache" / "Secondary_Endpoint_Fanout_v1_1"
        self.out = self.root / "Secondary_Endpoint_Fanout_Report"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.out.mkdir(parents=True, exist_ok=True)

    def load_inputs(self, control_arc_mm=12.7):
        snap = super().load_inputs(control_arc_mm)
        snap = dict(snap); snap["algorithm"] = ALGORITHM_VERSION
        snap["long_extension_projection"] = "surviving_local_terminal_direction"
        base._write_json(snap, self.cache / "input_snapshot.json")
        return snap

    def run(self, top_local_survivors=3, max_cost=105):
        sf=self.cache/"summary.json"; ecf=self.cache/"extension_candidates.csv"; bff=self.cache/"best_full_path.csv"; bqf=self.cache/"best_full_qc.csv"
        if self.fanout_directions is None: self.search_fanout()
        if self.reuse["vesselness_extension"] and all(p.exists() for p in (sf,ecf,bff,bqf)):
            self.summary=json.loads(sf.read_text()); self.extension_candidates=_safe_csv(ecf); self.best_full_path=_safe_csv(bff)[["z","y","x"]].to_numpy(float); self.best_full_qc=_safe_csv(bqf); return self.summary

        n=int(len(self.fanout_directions)); ns=int(self.fanout_directions.strict_early_survivor.sum()); nf=int(self.fanout_directions.full_local_survivor.sum()); mx=float(self.fanout_directions.local_length_mm.max()) if n else 0.0
        if ns==0:
            self.extension_candidates=pd.DataFrame(columns=["candidate","source_direction_id","supported_continuation"]); self.best_full_path=np.asarray(self.best_local_path,float); self.best_full_qc=pd.DataFrame(columns=["arc_mm","radius_mm","centroid_shift_mm","plane_score"])
            self.summary={"algorithm":ALGORITHM_VERSION,"status":"NO_COMPACT_LUMEN_FANOUT_SURVIVOR","accepted_continuation":False,"fanout_directions_tested":n,"strict_early_survivors":0,"full_local_survivors":0,"max_local_length_mm":mx,"interpretation":"No direction from the accepted endpoint maintained compact coronary-like lumen through the strict first 2.4 mm; vesselness extension was not attempted."}
        else:
            rows=[]; paths=[]; qcs=[]
            for _,sr in self.local_survivors.head(int(top_local_survivors)).iterrows():
                rr=self.step_diagnostics[(self.step_diagnostics.direction_id==int(sr.direction_id))&self.step_diagnostics.accepted_step].sort_values("step_index")
                lp=np.asarray([self.anchor_point.copy()]+[np.array([r.corrected_z,r.corrected_y,r.corrected_x],float) for _,r in rr.iterrows()],float)
                full=self._extend_one(lp,max_cost)
                if full is None: continue
                local_t=base._unit((lp[-1]-lp[max(0,len(lp)-4)])*self.spacing) if len(lp)>1 else self.terminal_tangent
                qc,lm=self._validate_lumen(full); s=base.arc_mm(full,self.spacing); length=float(s[-1]); disp=float(np.linalg.norm((full[-1]-self.anchor_point)*self.spacing)); turns=base._path_turns_deg(full,self.spacing); vessel=base._sample_trilinear(self.vesselness,self._global_to_local(full)); scale=base._sample_trilinear(self.best_scale,self._global_to_local(full)); ref=np.vstack([self.reference,self.trunk]); refsep=float(np.min(np.linalg.norm((ref-full[-1])*self.spacing,axis=1))); proj=float(np.dot((full[-1]-lp[-1])*self.spacing,local_t)); geom=bool(length>=4 and disp>=3 and proj>=1.5 and length/max(disp,1e-6)<=2 and (float(np.max(turns)) if len(turns) else 0)<=75 and float(np.percentile(vessel,10))>=.65*self.vessel_threshold and refsep>=1.7 and float(np.percentile(scale,90))<=1.50); supported=bool(geom and lm["lumen_gate"]); rank=float(.22*min(length/8,1)+.18*min(max(proj,0)/6,1)+.25*lm["lumen_plane_pass_fraction"]+.20*lm["lumen_median_plane_score"]+.15/max(length/max(disp,1e-6),1)); rows.append({"candidate":len(rows)+1,"source_direction_id":int(sr.direction_id),"local_length_mm":float(sr.local_length_mm),"full_length_mm":length,"endpoint_displacement_mm":disp,"forward_projection_from_local_end_mm":proj,"tortuosity":length/max(disp,1e-6),"max_turn_deg":float(np.max(turns)) if len(turns) else 0.0,"median_vesselness":float(np.median(vessel)),"p10_vesselness":float(np.percentile(vessel,10)),"p90_best_scale_mm":float(np.percentile(scale,90)),"endpoint_reference_separation_mm":refsep,"geometry_gate":geom,**lm,"supported_continuation":supported,"rank_score":rank}); paths.append(full); qcs.append(qc)
            self.extension_candidates=pd.DataFrame(rows) if rows else pd.DataFrame(columns=["candidate","source_direction_id","supported_continuation"])
            if rows:
                self.extension_candidates=self.extension_candidates.sort_values(["supported_continuation","lumen_gate","geometry_gate","rank_score"],ascending=[False,False,False,False]); bid=int(self.extension_candidates.iloc[0].candidate)-1; best=rows[bid]; self.best_full_path=paths[bid]; self.best_full_qc=qcs[bid]
                status="SUPPORTED_ENDPOINT_FANOUT_EXTENSION" if best["supported_continuation"] else "LOCAL_COMPACT_LUMEN_FOUND_LONG_EXTENSION_REJECTED" if best["geometry_gate"] else "LOCAL_COMPACT_LUMEN_FOUND_NO_LONG_EXTENSION"; interp="A compact-lumen direction survived locally and the longer extension also passed serial lumen validation; vessel identity remains unassigned." if best["supported_continuation"] else "Compact-lumen local direction(s) survived, but the longer 3-D extension was not validated as the same coronary lumen."
                self.summary={"algorithm":ALGORITHM_VERSION,"status":status,"accepted_continuation":bool(best["supported_continuation"]),"fanout_directions_tested":n,"strict_early_survivors":ns,"full_local_survivors":nf,"max_local_length_mm":mx,**best,"interpretation":interp}
            else:
                self.best_full_path=np.asarray(self.best_local_path,float); self.best_full_qc=pd.DataFrame(columns=["arc_mm","radius_mm","centroid_shift_mm","plane_score"]); self.summary={"algorithm":ALGORITHM_VERSION,"status":"LOCAL_COMPACT_LUMEN_FOUND_NO_REACHABLE_3D_EXTENSION","accepted_continuation":False,"fanout_directions_tested":n,"strict_early_survivors":ns,"full_local_survivors":nf,"max_local_length_mm":mx,"interpretation":"Compact-lumen local direction(s) survived, but no eligible longer vesselness endpoint was reachable."}

        self.extension_candidates.to_csv(ecf,index=False); p=np.asarray(self.best_full_path,float); pd.DataFrame({"arc_mm":base.arc_mm(p,self.spacing),"z":p[:,0],"y":p[:,1],"x":p[:,2]}).to_csv(bff,index=False); self.best_full_qc.to_csv(bqf,index=False); base._write_json(self.summary,sf); return self.summary