#!/usr/bin/env python3
"""Four-round geometry-aware Bayesian tensor refocus experiments."""

from __future__ import annotations
import argparse,json,math,random,time
from pathlib import Path

import numpy as np
import torch

from geoaware.bayes_models import ExactFeatureBayes
from geoaware.data import load_active_matter
from geoaware.masks import make_observation_split
from geoaware.models import SirenINR
from geoaware.operator_tucker_baselines import NeuralFunctionalCP,NeuralFunctionalTucker
from geoaware.tensor_bayes import OperatorBayesianCP,OperatorBayesianTucker
from geoaware.tensor_data import (operator_cp_tensor,operator_tucker_tensor,
                                  operator_mixed_tensor,operator_nonaligned_tensor,
                                  operator_basis_mismatch_tensor,
                                  diffusion_green_tensor,
                                  explicit_mode_bases,
                                  flat_product_features)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def metrics(truth,mean,std,held):
    y=truth[held]; p=mean[held]; err=p-y
    rmse=err.square().mean().sqrt()
    result={"rmse":float(rmse),"nrmse":float(rmse/y.std().clamp_min(1e-8)),
            "mae":float(err.abs().mean())}
    # Deterministic functional/INR baselines answer the reconstruction
    # question only.  Fabricating a residual variance for them would make NLL
    # and coverage comparisons look method-matched when they are not.
    if std is None:
        return result
    s=std[held].clamp_min(1e-7); z=err.abs()/s
    result.update({
        "nll":float((.5*(err/s).square()+torch.log(s)+.5*math.log(2*math.pi)).mean()),
        "coverage95":float((z<=1.96).float().mean()),"width95":float(3.92*s.mean()),
        "selective_gain50":float(1-torch.sqrt(err[torch.argsort(s)[:len(s)//2]].square().mean())/rmse)})
    return result


def fit_functional_baseline(name,data,obs,y,steps,seed,device,tucker_ranks,rank):
    """Fit a deterministic continuous baseline using observed entries only."""
    seed_all(seed); dev=torch.device(device if torch.cuda.is_available() else "cpu")
    if name=="neural_functional_cp":
        model=NeuralFunctionalCP(data.periodic,rank=rank,hidden=48)
    elif name=="neural_functional_tucker":
        model=NeuralFunctionalTucker(data.periodic,ranks=tucker_ranks,hidden=48)
    elif name=="siren_inr":
        model=SirenINR(len(data.shape),hidden=96,depth=3,omega=20.)
    else:
        raise ValueError(name)
    model=model.to(dev); coordinates=data.flat_coordinates(); x=coordinates[obs].to(dev); yy=y.to(dev)
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-6)
    best=(float("inf"),None)
    for _ in range(steps):
        prediction=model(x)
        regularization=model.regularization() if hasattr(model,"regularization") else prediction.new_zeros(())
        loss=(prediction-yy).square().mean()+1e-5*regularization
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.); optimizer.step()
        score=float(loss.detach())
        if score<best[0]:
            best=(score,{key:value.detach().cpu().clone() for key,value in model.state_dict().items()})
    model.load_state_dict(best[1]); model.eval(); chunks=[]
    with torch.no_grad():
        for start in range(0,len(coordinates),8192):
            chunks.append(model(coordinates[start:start+8192].to(dev)).cpu())
    return (torch.cat(chunks),
            sum(parameter.numel() for parameter in model.parameters()),
            best[0])


def load_task(name,mismatch=0.,basis_cutoff=8,truth_modes=14):
    if name=="cp": return operator_cp_tensor()
    if name=="tucker": return operator_tucker_tensor()
    if name=="mixed": return operator_mixed_tensor(mismatch)
    if name=="basis_mismatch": return operator_basis_mismatch_tensor(mismatch)
    if name=="diffusion_green":
        return diffusion_green_tensor(contrast=mismatch,basis_cutoff=basis_cutoff,
                                      truth_modes=truth_modes)
    if name=="nonaligned": return operator_nonaligned_tensor()
    if name=="active": return load_active_matter(spatial_stride=2)
    raise ValueError(name)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--task",choices=["cp","tucker","mixed","basis_mismatch","diffusion_green","nonaligned","active"],default="cp")
    p.add_argument("--mismatch",type=float,default=0.)
    p.add_argument("--basis-cutoff",type=int,default=8)
    p.add_argument("--truth-modes",type=int,default=14)
    p.add_argument("--models",default="geo_bcp,geo_bcp_noard,wrong_bcp,discrete_bcp,flat_geo_gp")
    p.add_argument("--ratios",default=".005,.01"); p.add_argument("--masks",default="random,periodic_gap")
    p.add_argument("--seeds",default="0,1,2"); p.add_argument("--rank",type=int,default=10)
    p.add_argument("--tucker-ranks",default="4,5,5")
    p.add_argument("--steps",type=int,default=1600); p.add_argument("--power",type=float,default=1.5)
    p.add_argument("--reg",type=float,default=.002); p.add_argument("--ard-cycles",type=int,default=1)
    p.add_argument("--factor-laplace",action="store_true"); p.add_argument("--noise",type=float,default=.1)
    p.add_argument("--init",choices=["random","flat_gp"],default="random")
    p.add_argument("--split-calibration",action="store_true")
    p.add_argument("--device",default="cuda"); args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    data=load_task(args.task,args.mismatch,args.basis_cutoff,args.truth_modes); clean=data.values.clone(); truth=clean.flatten(); rows=[]
    models=args.models.split(",")
    for mask in args.masks.split(","):
      for ratio in map(float,args.ratios.split(",")):
       for seed in map(int,args.seeds.split(",")):
        split=make_observation_split(data,ratio,mask,seed); obs=torch.where(split.observed)[0]
        g=torch.Generator().manual_seed(seed+4401); noisy=truth.clone()
        noisy[obs]+=torch.randn(len(obs),generator=g)*args.noise*truth[obs].std()
        center=float(noisy[obs].mean()); scale=float(noisy[obs].std().clamp_min(1e-6)); y=(noisy[obs]-center)/scale
        initial_cache={}
        for name in models:
            seed_all(seed); started=time.perf_counter()
            if name in ("neural_functional_cp","neural_functional_tucker","siren_inr"):
                ranks=tuple(map(int,args.tucker_ranks.split(",")))
                normalized_mean,parameter_count,best_objective=fit_functional_baseline(
                    name,data,obs,y,args.steps,seed,args.device,ranks,args.rank)
                mean=normalized_mean*scale+center; std=None; effective_rank=None
                meta={"parameters":parameter_count,"inference":"deterministic_regularized_fit",
                      "uncertainty_metrics_supported":False,"optimizer":"AdamW",
                      "learning_rate":2e-3,"gradient_steps":args.steps,
                      "initialization":"random","best_observed_objective":best_objective}
            elif name=="flat_geo_gp":
                phi,eig=flat_product_features(data,512)
                split_scale=1.0
                if args.split_calibration:
                    perm=torch.randperm(len(obs),generator=torch.Generator().manual_seed(seed+7301))
                    ncal=max(8,len(obs)//4); cal=obs[perm[:ncal]]; train=obs[perm[ncal:]]
                    pre_center=float(noisy[train].mean()); pre_scale=float(noisy[train].std().clamp_min(1e-6))
                    pre_y=(noisy[train]-pre_center)/pre_scale
                    preliminary=ExactFeatureBayes(phi,eig,"loo",False,args.device).fit(train,pre_y)
                    pp=preliminary.predict(); z=(noisy[cal]-pre_center)/pre_scale
                    split_scale=float(torch.quantile(
                        (z-pp.mean[cal]).abs()/pp.raw_std[cal].clamp_min(1e-7),.95)/1.96)
                    split_scale=max(.5,min(10.,split_scale))
                model=ExactFeatureBayes(phi,eig,"loo",False,args.device).fit(obs,y)
                pred=model.predict(); mean=pred.mean*scale+center; std=pred.raw_std*scale*split_scale
                effective_rank=None; meta=pred.hyperparameters | {"split_calibration":split_scale}
            else:
                kind=("correct" if name.startswith("geo") else
                      "permuted" if name.startswith("wrong") else "discrete")
                basis,eig=explicit_mode_bases(data,kind)
                def make_fit(fit_obs,fit_y,fit_steps):
                    if name.endswith("btucker"):
                        ranks=tuple(map(int,args.tucker_ranks.split(",")))
                        fitted=OperatorBayesianTucker(basis,eig,ranks,args.power,device=args.device)
                    else:
                        fitted=OperatorBayesianCP(basis,eig,args.rank,args.power,
                            ard=(name!="geo_bcp_noard"),factor_laplace=args.factor_laplace,device=args.device)
                    initial=None
                    if args.init=="flat_gp" and kind!="discrete":
                        key=(kind,tuple(fit_obs.tolist()))
                        if key not in initial_cache:
                            init_phi,init_eig=flat_product_features(data,512,kind)
                            init_fit=ExactFeatureBayes(init_phi,init_eig,"loo",False,args.device).fit(fit_obs,fit_y)
                            initial_cache[key]=init_fit.predict().mean.reshape(data.shape)
                        initial=initial_cache[key]
                    fit_kwargs={"steps":fit_steps,"reg_weight":args.reg,"seed":seed,
                                "initial_tensor":initial}
                    if not name.endswith("btucker"):fit_kwargs["ard_cycles"]=args.ard_cycles
                    fitted.fit(data.flat_indices()[fit_obs],fit_y,**fit_kwargs)
                    return fitted
                split_scale=1.0
                if args.split_calibration:
                    perm=torch.randperm(len(obs),generator=torch.Generator().manual_seed(seed+7301))
                    ncal=max(8,len(obs)//4); cal=obs[perm[:ncal]]; train=obs[perm[ncal:]]
                    # The preliminary calibration fit has its own train-only
                    # normalization. Calibration values do not influence its
                    # center, scale, factors, initializer, or posterior.
                    pre_center=float(noisy[train].mean())
                    pre_scale=float(noisy[train].std().clamp_min(1e-6))
                    train_y=(noisy[train]-pre_center)/pre_scale
                    preliminary=make_fit(train,train_y,max(500,args.steps//2))
                    pp=preliminary.predict(data.flat_indices()[cal])
                    z=(noisy[cal]-pre_center)/pre_scale
                    split_scale=float(torch.quantile((z-pp.mean).abs()/pp.std.clamp_min(1e-7),.95)/1.96)
                    split_scale=max(.5,min(10.,split_scale))
                model=make_fit(obs,y,args.steps)
                pred=model.predict(data.flat_indices()); mean=pred.mean*scale+center; std=pred.std*scale*split_scale
                effective_rank=pred.effective_rank; meta=pred.metadata | {
                    "parameters":sum(parameter.numel() for parameter in model.parameters()),
                    "inference":"regularized_factor_fit_plus_conditional_empirical_bayes",
                    "optimizer":"AdamW","learning_rate":3e-3,
                    "gradient_steps":args.steps,"initialization":args.init,
                    "component_precision":pred.component_precision.tolist(),
                    "component_energy":pred.component_energy.tolist(),
                    "factor_spectral_energy":[x.tolist() for x in pred.factor_spectral_energy],
                    "split_calibration":split_scale}
            row={"task":data.name,"shape":data.shape,"model":name,"mask":mask,"ratio":ratio,
                 "ratio_actual":split.ratio_actual,"n_observed":len(obs),"seed":seed,
                 "metrics":metrics(truth,mean,std,split.held_out),"effective_rank":effective_rank,
                 "observed_fit":{"mse":float((mean[obs]-noisy[obs]).square().mean()),
                    "nrmse":float((mean[obs]-noisy[obs]).square().mean().sqrt()/
                                  noisy[obs].std().clamp_min(1e-8))},
                 "metadata":meta,"elapsed_seconds":time.perf_counter()-started,"arguments":vars(args)}
            rows.append(row); print(f"{data.name} {mask} {ratio:g} s{seed} {name} "
                                    f"NRMSE={row['metrics']['nrmse']:.3f} rank={effective_rank}",flush=True)
    (args.output/"results.json").write_text(json.dumps({"dataset":{"name":data.name,"shape":data.shape,
        "source":data.source,"description":data.description,"metadata":data.metadata},
        "arguments":vars(args),"results":rows},indent=2,default=str))


if __name__=="__main__":main()
