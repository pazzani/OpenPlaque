from __future__ import annotations

"""Local endpoint fan-out with compact-lumen gating before any vesselness extension."""

import json, math, zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from . import secondary_3d_vesselness_topology as base
from .secondary_3d_vesselness_topology_v2 import finite_dist_coords
from .secondary_anchored_lumen_validation import SecondaryAnchoredLumenValidationWorkflow, _component_metrics

ALGORITHM_VERSION = "secondary-endpoint-fanout-v1.0"


def _component_with_offset(im, grid, threshold_hu, max_shift_mm=1.10):
    bright=(im>=float(threshold_hu))&(im<=1200.0)
    lab,nlab=ndi.label(bright,structure=np.ones((3,3),np.uint8)); out=[]
    for k in range(1,int(nlab)+1):
        comp=lab==k
        if int(comp.sum())<7: continue
        m=_component_metrics(im,grid,comp); yy,xx=np.nonzero(comp)
        cu=float(np.mean(grid[xx])); cv=float(np.mean(grid[yy])); dmin=float(np.min(np.hypot(grid[xx],grid[yy])))
        if m["centroid_shift_mm"]<=max_shift_mm or dmin<=0.45:
            out.append((m["centroid_shift_mm"]+0.35*dmin,cu,cv,m))
    if not out: return None
    _,cu,cv,m=min(out,key=lambda x:x[0]); return {**m,"offset_u_mm":cu,"offset_v_mm":cv}


def initial_directions(tangent,shell_deg=(0,10,20,30,40,50,60,70),n_azimuth=16):
    t=base._unit(np.asarray(tangent,float)); u,v=base._orth_basis(t); out=[]; did=0
    for ang in shell_deg:
        if float(ang)==0:
            did+=1; out.append({"direction_id":did,"angle_deg":0.0,"azimuth_deg":0.0,"direction":t.copy()}); continue
        th=math.radians(float(ang))
        for j in range(int(n_azimuth)):
            ph=2*math.pi*j/float(n_azimuth); d=math.cos(th)*t+math.sin(th)*(math.cos(ph)*u+math.sin(ph)*v)
            did+=1; out.append({"direction_id":did,"angle_deg":float(ang),"azimuth_deg":float(math.degrees(ph)),"direction":base._unit(d)})
    return out


def synthetic_fanout_self_test():
    g=np.arange(-4.4,4.4+1e-9,.16); yy,xx=np.meshgrid(g,g,indexing="ij")
    tube=100+650*(((xx-.48)**2+(yy+.32)**2)<=1.75**2); broad=100+650*(xx>=-.15)
    mt=_component_with_offset(tube,g,220); mb=_component_with_offset(broad,g,220); ds=initial_directions([0,1,0])
    ok=bool(mt and mb and 1.5<=mt["radius_mm"]<=2.05 and abs(mt["offset_u_mm"]-.48)<=.2 and abs(mt["offset_v_mm"]+.32)<=.2 and mb["radius_mm"]>2.65 and len(ds)==113)
    return {"passed":ok,"tube_radius_mm":None if not mt else mt["radius_mm"],"broad_radius_mm":None if not mb else mb["radius_mm"],"n_directions":len(ds)}


class SecondaryEndpointFanoutWorkflow(SecondaryAnchoredLumenValidationWorkflow):
    def __init__(self,root="/content/drive/MyDrive/OpenPlaque",reuse=None):
        super().__init__(root=root,reuse={}); self.cache=self.root/"Cache"/"Secondary_Endpoint_Fanout_v1"; self.out=self.root/"Secondary_Endpoint_Fanout_Report"; self.cache.mkdir(parents=True,exist_ok=True); self.out.mkdir(parents=True,exist_ok=True)
        self.reuse={"inputs":True,"fanout_search":True,"vesselness_extension":True,"figures":True,"report":True}
        if reuse: self.reuse.update({k:bool(v) for k,v in reuse.items()})
        self.prov=[]; self.fanout_directions=self.step_diagnostics=self.local_survivors=None; self.best_local_path=self.extension_candidates=self.best_full_path=self.best_full_qc=None

    def cache_status(self):
        names={"inputs":"input_snapshot.json","fanout_search":"fanout_directions.csv","vesselness_extension":"summary.json","figures":"figures.done","report":"report.done"}
        return pd.DataFrame([{"component":k,"reuse":self.reuse[k],"cache_exists":(self.cache/v).exists()} for k,v in names.items()])

    def load_inputs(self,control_arc_mm=12.7):
        snap=super().load_inputs(control_arc_mm); snap=dict(snap); snap.update({"algorithm":ALGORITHM_VERSION,"fanout_origin":"accepted_secondary_branch_endpoint","strict_early_survival_mm":2.4,"local_search_max_mm":4.2,"search_target":None,"note":"Compact coronary-like lumen is required from the first fan-out step before vesselness extension."}); base._write_json(snap,self.cache/"input_snapshot.json"); return snap

    def _strict_plane(self,p,t,max_shift=.85):
        im,g=base.orthogonal_plane(self.ct,p,t,self.spacing,half_mm=4.4,pix_mm=.16); m=_component_with_offset(im,g,self.calibration["bright_threshold_hu"],max_shift)
        if m is None: return None,False,0.0
        rref=float(self.calibration["median_radius_mm"]); passed=bool(.65<=m["radius_mm"]<=2.65 and m["centroid_shift_mm"]<=.85 and m["circularity"]>=.30 and m["component_median_hu"]>=max(220,.45*self.calibration["reference_center_hu"]))
        rs=math.exp(-.5*((m["radius_mm"]-rref)/max(.55*rref,.70))**2); ss=math.exp(-.5*(m["centroid_shift_mm"]/.55)**2); cs=float(np.clip(m["circularity"]/max(self.calibration["median_circularity"],.35),0,1)); hs=float(np.clip(m["component_median_hu"]/max(self.calibration["reference_center_hu"],1),0,1.2)/1.2)
        return m,passed,float(.36*rs+.36*ss+.16*cs+.12*hs)

    def _track(self,spec,step_mm=.30,strict_mm=2.4,max_mm=4.2):
        path=[np.asarray(self.anchor_point,float).copy()]; t=base._unit(spec["direction"]); rows=[]; reason="max_local_reached"
        for k in range(1,int(math.ceil(max_mm/step_mm))+1):
            prop=path[-1]+t*step_mm/self.spacing; im,g=base.orthogonal_plane(self.ct,prop,t,self.spacing,half_mm=4.4,pix_mm=.16); m=_component_with_offset(im,g,self.calibration["bright_threshold_hu"],1.10)
            row={"direction_id":spec["direction_id"],"initial_angle_deg":spec["angle_deg"],"initial_azimuth_deg":spec["azimuth_deg"],"step_index":k,"arc_before_mm":float(base.arc_mm(np.asarray(path),self.spacing)[-1])}
            if m is None: row.update({"accepted_step":False,"rejection_reason":"no_center_component"}); rows.append(row); reason="no_center_component"; break
            u,v=base._orth_basis(t); corr=prop+(m["offset_u_mm"]*u+m["offset_v_mm"]*v)/self.spacing; actual=(corr-path[-1])*self.spacing; alen=float(np.linalg.norm(actual)); nt=base._unit(actual); turn=float(np.degrees(np.arccos(np.clip(np.dot(t,nt),-1,1)))); m2,pass2,score=self._strict_plane(corr,nt,.70)
            fail=None
            if not (.18<=alen<=.72): fail="step_length"
            elif turn>58: fail="turn_gt_58"
            elif m2 is None: fail="no_component_after_recenter"
            elif not pass2: fail="radius_too_large" if m2["radius_mm"]>2.65 else "radius_too_small" if m2["radius_mm"]<.65 else "off_center" if m2["centroid_shift_mm"]>.85 else "low_circularity" if m2["circularity"]<.30 else "lumen_qc_fail"
            row.update({"actual_step_mm":alen,"turn_deg":turn,"corrected_z":corr[0],"corrected_y":corr[1],"corrected_x":corr[2],"pre_radius_mm":m["radius_mm"],"pre_shift_mm":m["centroid_shift_mm"]})
            if fail:
                if m2 is not None: row.update({"radius_mm":m2["radius_mm"],"centroid_shift_mm":m2["centroid_shift_mm"],"circularity":m2["circularity"],"component_median_hu":m2["component_median_hu"],"plane_score":score})
                row.update({"accepted_step":False,"rejection_reason":fail}); rows.append(row); reason=fail; break
            path.append(corr.copy()); t=base._unit(.80*nt+.20*t); arc=float(base.arc_mm(np.asarray(path),self.spacing)[-1]); row.update({"radius_mm":m2["radius_mm"],"centroid_shift_mm":m2["centroid_shift_mm"],"circularity":m2["circularity"],"component_median_hu":m2["component_median_hu"],"plane_score":score,"arc_after_mm":arc,"accepted_step":True,"rejection_reason":""}); rows.append(row)
            if arc>=max_mm-.05: break
        arr=np.asarray(path,float); length=float(base.arc_mm(arr,self.spacing)[-1]); good=[r for r in rows if r.get("accepted_step")]; scores=[r["plane_score"] for r in good]; radii=[r["radius_mm"] for r in good]; shifts=[r["centroid_shift_mm"] for r in good]
        disp=float(np.linalg.norm((arr[-1]-arr[0])*self.spacing)) if len(arr)>1 else 0.0
        sm={"direction_id":spec["direction_id"],"initial_angle_deg":spec["angle_deg"],"initial_azimuth_deg":spec["azimuth_deg"],"local_length_mm":length,"accepted_steps":len(good),"terminal_reason":reason,"strict_early_survivor":bool(length>=strict_mm),"full_local_survivor":bool(length>=max_mm-.20),"median_plane_score":float(np.median(scores)) if scores else 0.0,"min_plane_score":float(np.min(scores)) if scores else 0.0,"median_radius_mm":float(np.median(radii)) if radii else np.nan,"p90_radius_mm":float(np.percentile(radii,90)) if radii else np.nan,"median_shift_mm":float(np.median(shifts)) if shifts else np.nan,"endpoint_displacement_mm":disp,"tortuosity":length/max(disp,1e-6) if disp else np.nan}
        return arr,rows,sm

    def search_fanout(self,step_mm=.30,strict_mm=2.4,max_mm=4.2):
        ff=self.cache/"fanout_directions.csv"; df=self.cache/"fanout_step_diagnostics.csv"; sf=self.cache/"local_survivors.csv"; pf=self.cache/"best_local_path.csv"
        if self.ct is None: self.load_inputs()
        if self.reuse["fanout_search"] and all(p.exists() for p in (ff,df,sf,pf)):
            self.fanout_directions=pd.read_csv(ff); self.step_diagnostics=pd.read_csv(df); self.local_survivors=pd.read_csv(sf); self.best_local_path=pd.read_csv(pf)[["z","y","x"]].to_numpy(float); return self.fanout_directions
        sums=[]; diags=[]; paths={}
        for spec in initial_directions(self.terminal_tangent):
            p,r,s=self._track(spec,step_mm,strict_mm,max_mm); sums.append(s); diags.extend(r); paths[int(spec["direction_id"])]=p
        self.fanout_directions=pd.DataFrame(sums).sort_values(["strict_early_survivor","full_local_survivor","local_length_mm","median_plane_score"],ascending=[False,False,False,False]); self.step_diagnostics=pd.DataFrame(diags); self.local_survivors=self.fanout_directions[self.fanout_directions.strict_early_survivor].copy(); bid=int((self.local_survivors if len(self.local_survivors) else self.fanout_directions).iloc[0].direction_id); self.best_local_path=paths[bid]
        self.fanout_directions.to_csv(ff,index=False); self.step_diagnostics.to_csv(df,index=False); self.local_survivors.to_csv(sf,index=False); p=self.best_local_path; pd.DataFrame({"arc_mm":base.arc_mm(p,self.spacing),"z":p[:,0],"y":p[:,1],"x":p[:,2]}).to_csv(pf,index=False); return self.fanout_directions

    def _extend_one(self,local_path,max_cost=105):
        mask,cost=self._build_mask_and_cost(); ep=local_path[-1]; t=base._unit((local_path[-1]-local_path[max(0,len(local_path)-4)])*self.spacing) if len(local_path)>1 else self.terminal_tangent; mm=mask.copy(); mm[self._sphere_mask(ep,.55)]=True; dist,prev,_=self._dijkstra(mm,cost,self._nearest_voxel(ep),goal_mask=None,max_cost=max_cost); coords=finite_dist_coords(dist,self.roi.shape)
        if not len(coords): return None
        phys=(coords+self.roi_lo)*self.spacing; epp=ep*self.spacing; proj=(phys-epp)@t; disp=np.linalg.norm(phys-epp,axis=1); vals=self.vesselness[tuple(coords.T)]; ok=(proj>=1.5)&(disp>=1.8)&(vals>=self.vessel_threshold); coords,proj,vals=coords[ok],proj[ok],vals[ok]
        if not len(coords): return None
        ii=int(np.argmax(proj+1.5*vals)); flat=int(np.ravel_multi_index(tuple(coords[ii]),self.roi.shape)); loc=self._reconstruct(prev,flat,self.roi.shape)
        if loc is None or len(loc)<2: return None
        tail=base.resample_path(self._local_to_global(loc),self.spacing,.20); return base.resample_path(np.vstack([local_path,tail[1:]]),self.spacing,.20)

    def run(self,top_local_survivors=3,max_cost=105):
        sf=self.cache/"summary.json"; ecf=self.cache/"extension_candidates.csv"; bff=self.cache/"best_full_path.csv"; bqf=self.cache/"best_full_qc.csv"
        if self.fanout_directions is None: self.search_fanout()
        if self.reuse["vesselness_extension"] and all(p.exists() for p in (sf,ecf,bff,bqf)):
            self.summary=json.loads(sf.read_text()); self.extension_candidates=pd.read_csv(ecf); self.best_full_path=pd.read_csv(bff)[["z","y","x"]].to_numpy(float); self.best_full_qc=pd.read_csv(bqf); return self.summary
        n=int(len(self.fanout_directions)); ns=int(self.fanout_directions.strict_early_survivor.sum()); nf=int(self.fanout_directions.full_local_survivor.sum()); mx=float(self.fanout_directions.local_length_mm.max()) if n else 0.0
        if ns==0:
            self.extension_candidates=pd.DataFrame(); self.best_full_path=self.best_local_path; self.best_full_qc=pd.DataFrame(); self.summary={"algorithm":ALGORITHM_VERSION,"status":"NO_COMPACT_LUMEN_FANOUT_SURVIVOR","accepted_continuation":False,"fanout_directions_tested":n,"strict_early_survivors":0,"full_local_survivors":0,"max_local_length_mm":mx,"interpretation":"No direction from the accepted endpoint maintained compact coronary-like lumen through the strict first 2.4 mm; vesselness extension was not attempted."}
        else:
            rows=[]; paths=[]; qcs=[]
            for _,sr in self.local_survivors.head(int(top_local_survivors)).iterrows():
                rr=self.step_diagnostics[(self.step_diagnostics.direction_id==int(sr.direction_id))&self.step_diagnostics.accepted_step].sort_values("step_index"); lp=[self.anchor_point.copy()]+[np.array([r.corrected_z,r.corrected_y,r.corrected_x],float) for _,r in rr.iterrows()]; full=self._extend_one(np.asarray(lp,float),max_cost)
                if full is None: continue
                qc,lm=self._validate_lumen(full); s=base.arc_mm(full,self.spacing); length=float(s[-1]); disp=float(np.linalg.norm((full[-1]-self.anchor_point)*self.spacing)); turns=base._path_turns_deg(full,self.spacing); vessel=base._sample_trilinear(self.vesselness,self._global_to_local(full)); scale=base._sample_trilinear(self.best_scale,self._global_to_local(full)); ref=np.vstack([self.reference,self.trunk]); refsep=float(np.min(np.linalg.norm((ref-full[-1])*self.spacing,axis=1))); proj=float(np.dot((full[-1]-self.anchor_point)*self.spacing,self.terminal_tangent)); geom=bool(length>=4 and disp>=3 and proj>=2.5 and length/max(disp,1e-6)<=2 and (float(np.max(turns)) if len(turns) else 0)<=75 and float(np.percentile(vessel,10))>=.65*self.vessel_threshold and refsep>=1.7 and float(np.percentile(scale,90))<=1.50); supported=bool(geom and lm["lumen_gate"]); rank=float(.22*min(length/8,1)+.18*min(proj/6,1)+.25*lm["lumen_plane_pass_fraction"]+.20*lm["lumen_median_plane_score"]+.15/max(length/max(disp,1e-6),1)); rows.append({"candidate":len(rows)+1,"source_direction_id":int(sr.direction_id),"local_length_mm":float(sr.local_length_mm),"full_length_mm":length,"endpoint_displacement_mm":disp,"forward_projection_mm":proj,"tortuosity":length/max(disp,1e-6),"max_turn_deg":float(np.max(turns)) if len(turns) else 0.0,"median_vesselness":float(np.median(vessel)),"p10_vesselness":float(np.percentile(vessel,10)),"p90_best_scale_mm":float(np.percentile(scale,90)),"endpoint_reference_separation_mm":refsep,"geometry_gate":geom,**lm,"supported_continuation":supported,"rank_score":rank}); paths.append(full); qcs.append(qc)
            self.extension_candidates=pd.DataFrame(rows)
            if len(rows):
                self.extension_candidates=self.extension_candidates.sort_values(["supported_continuation","lumen_gate","geometry_gate","rank_score"],ascending=[False,False,False,False]); bid=int(self.extension_candidates.iloc[0].candidate)-1; best=rows[bid]; self.best_full_path=paths[bid]; self.best_full_qc=qcs[bid]; status="SUPPORTED_ENDPOINT_FANOUT_EXTENSION" if best["supported_continuation"] else "LOCAL_COMPACT_LUMEN_FOUND_LONG_EXTENSION_REJECTED" if best["geometry_gate"] else "LOCAL_COMPACT_LUMEN_FOUND_NO_LONG_EXTENSION"; interp="A compact-lumen direction survived locally and the longer extension also passed serial lumen validation; vessel identity remains unassigned." if best["supported_continuation"] else "Compact-lumen local direction(s) survived, but the longer 3-D extension was not validated as the same coronary lumen."
                self.summary={"algorithm":ALGORITHM_VERSION,"status":status,"accepted_continuation":bool(best["supported_continuation"]),"fanout_directions_tested":n,"strict_early_survivors":ns,"full_local_survivors":nf,"max_local_length_mm":mx,**best,"interpretation":interp}
            else:
                self.best_full_path=self.best_local_path; self.best_full_qc=pd.DataFrame(); self.summary={"algorithm":ALGORITHM_VERSION,"status":"LOCAL_COMPACT_LUMEN_FOUND_NO_REACHABLE_3D_EXTENSION","accepted_continuation":False,"fanout_directions_tested":n,"strict_early_survivors":ns,"full_local_survivors":nf,"max_local_length_mm":mx,"interpretation":"Compact-lumen local direction(s) survived, but no eligible longer vesselness endpoint was reachable."}
        self.extension_candidates.to_csv(ecf,index=False); p=np.asarray(self.best_full_path,float); pd.DataFrame({"arc_mm":base.arc_mm(p,self.spacing),"z":p[:,0],"y":p[:,1],"x":p[:,2]}).to_csv(bff,index=False); self.best_full_qc.to_csv(bqf,index=False); base._write_json(self.summary,sf); return self.summary

    def make_figures(self):
        if self.summary is None: self.run()
        names=["01_fanout_geometry.png","02_local_survivor_planes.png","03_fanout_outcomes.png","04_best_extension_validation.png"]; sp=base.resample_path(self.seed,self.spacing,.15)
        fig=plt.figure(figsize=(13,4))
        for k,(a,b,t) in enumerate([(2,1,"X-Y"),(2,0,"X-Z"),(1,0,"Y-Z")],1):
            ax=fig.add_subplot(1,3,k); ax.plot(sp[:,a]*self.spacing[a],sp[:,b]*self.spacing[b],label="accepted branch")
            if self.prior_v2_extension is not None: ax.plot(self.prior_v2_extension[:,a]*self.spacing[a],self.prior_v2_extension[:,b]*self.spacing[b],alpha=.2,label="rejected v2 corridor")
            if self.best_local_path is not None and len(self.best_local_path)>1: ax.plot(self.best_local_path[:,a]*self.spacing[a],self.best_local_path[:,b]*self.spacing[b],lw=3,label="best local fan-out")
            if self.best_full_path is not None and len(self.best_full_path)>len(self.best_local_path): ax.plot(self.best_full_path[:,a]*self.spacing[a],self.best_full_path[:,b]*self.spacing[b],lw=2,label="best longer path")
            ax.set_title(t); ax.set_aspect("equal",adjustable="box");
            if k==1: ax.legend(fontsize=7)
        fig.suptitle(f"Endpoint fan-out — {self.summary['status']}"); fig.tight_layout(); fig.savefig(self.out/names[0],dpi=160); plt.close(fig)
        fig,axes=plt.subplots(2,5,figsize=(13,5.4)); axes=axes.ravel(); p=np.asarray(self.best_local_path,float)
        if len(p)>=2:
            rp=base.resample_path(p,self.spacing,.30); rs=base.arc_mm(rp,self.spacing); ids=np.linspace(1,len(rp)-1,min(10,max(1,len(rp)-1))).astype(int)
            for ax,i in zip(axes,ids):
                i0,i1=max(0,i-2),min(len(rp)-1,i+2); im,_=base.orthogonal_plane(self.ct,rp[i],(rp[i1]-rp[i0])*self.spacing,self.spacing,half_mm=4.4,pix_mm=.16); ax.imshow(im,cmap="gray",vmin=100,vmax=900); ax.set_title(f"{rs[i]:.1f} mm"); ax.axis("off")
        for ax in axes:
            if not ax.has_data(): ax.axis("off")
        fig.suptitle("Best compact-lumen fan-out direction"); fig.tight_layout(); fig.savefig(self.out/names[1],dpi=160); plt.close(fig)
        fig,ax=plt.subplots(figsize=(9,5)); fd=self.fanout_directions; ax.scatter(fd.initial_angle_deg,fd.local_length_mm,s=18); sv=fd[fd.strict_early_survivor];
        if len(sv): ax.scatter(sv.initial_angle_deg,sv.local_length_mm,s=55,marker="x",label="strict survivors")
        ax.axhline(2.4,ls="--",label="strict threshold"); ax.axhline(4.2,ls=":",label="full local target"); ax.set_xlabel("Initial angle from terminal tangent (deg)"); ax.set_ylabel("Compact-lumen length (mm)"); ax.set_title("Fan-out outcomes"); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(self.out/names[2],dpi=160); plt.close(fig)
        fig,ax=plt.subplots(figsize=(9,4.8)); q=self.best_full_qc
        if q is not None and len(q):
            ax.plot(q.arc_mm,q.radius_mm,label="radius (mm)"); ax.axhline(2.65,ls="--",label="radius limit"); ax.plot(q.arc_mm,q.centroid_shift_mm,label="center shift (mm)"); ax2=ax.twinx(); ax2.plot(q.arc_mm,q.plane_score,alpha=.55,label="plane score"); lines=ax.get_lines()+ax2.get_lines(); ax.legend(lines,[x.get_label() for x in lines],fontsize=8,loc="best"); ax.set_xlabel("Path arc from endpoint (mm)")
        else: ax.text(.5,.5,"No longer vesselness extension available",ha="center",va="center"); ax.set_axis_off()
        fig.tight_layout(); fig.savefig(self.out/names[3],dpi=160); plt.close(fig); (self.cache/"figures.done").write_text("done"); return names

    def make_report(self):
        if self.summary is None: self.run()
        self.make_figures(); html=self.out/"OPENPLAQUE_SECONDARY_ENDPOINT_FANOUT_REPORT.html"; rows="".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k,v in self.summary.items() if k!="interpretation"); html.write_text(f"<html><body><h1>OpenPlaque Secondary Branch — Endpoint Fan-Out</h1><p><b>Status: {self.summary['status']}</b></p><p>{self.summary.get('interpretation','')}</p><table border='1' cellpadding='4'>{rows}</table><h2>Fan-out geometry</h2><img src='01_fanout_geometry.png' width='95%'><h2>Best local planes</h2><img src='02_local_survivor_planes.png' width='95%'><h2>Direction outcomes</h2><img src='03_fanout_outcomes.png' width='85%'><h2>Longer extension validation</h2><img src='04_best_extension_validation.png' width='90%'><p>Research use only. Compact lumen is required before vesselness extension. Vessel identity remains unassigned.</p></body></html>"); (self.cache/"report.done").write_text("done"); return html

    def package(self):
        report=self.make_report(); zp=self.out/"OPENPLAQUE_SECONDARY_ENDPOINT_FANOUT_REPORT_BACK.zip"; inc=[report,self.cache/"input_snapshot.json",self.cache/"lumen_calibration.json",self.cache/"fanout_directions.csv",self.cache/"fanout_step_diagnostics.csv",self.cache/"local_survivors.csv",self.cache/"best_local_path.csv",self.cache/"extension_candidates.csv",self.cache/"best_full_path.csv",self.cache/"best_full_qc.csv",self.cache/"summary.json"]+[self.out/n for n in ["01_fanout_geometry.png","02_local_survivor_planes.png","03_fanout_outcomes.png","04_best_extension_validation.png"]]
        with zipfile.ZipFile(zp,"w",compression=zipfile.ZIP_DEFLATED) as z:
            for p in inc:
                if p.exists(): z.write(p,arcname=p.name)
        return zp