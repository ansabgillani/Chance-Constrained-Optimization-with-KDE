"""Run the fully local evidence matrix used by Chapters 8 and 9.

Every stochastic input is seeded and saved.  The runner never removes an
existing file; ``--force`` only permits replacing files with the same name.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
import numpy as np

from ..baselines import gaussian_parametric
from ..constraints import (boole_risk_allocation, empirical_violation,
                           kde_safe_probability, kde_violation_upper,
                           max_violation, smooth_max)
from ..evaluation.io import write_array, write_run_bundle
from ..evaluation.metrics import evaluate_residuals
from ..kde import silverman_bandwidth
from ..problems import make_energy_dispatch, make_lunar_landing, make_static_problem
from ..solvers import solve_dispatch, solve_static
from ..solvers.collocation import solve_lunar_stages
from ..uncertainty import (sample_bimodal, sample_gaussian, sample_heavy_tailed,
                           sample_renewable_error, sample_skewed)


def _distribution(name, n, seed):
    if name == "gaussian": return sample_gaussian(n, std=1.0, seed=seed)
    if name == "bimodal": return sample_bimodal(n, means=(-1.2, 1.2), stds=(.35, .35), seed=seed)
    if name == "skewed": return sample_skewed(n, scale=1.0, seed=seed)
    if name == "heavy_tailed": return sample_heavy_tailed(n, df=3.5, scale=.8, seed=seed)
    raise ValueError(name)


def _static(seed, n_train, n_test, epsilon):
    problem = make_static_problem()
    rows = []
    for offset, name in enumerate(("gaussian", "bimodal", "skewed", "heavy_tailed")):
        train_a, meta = _distribution(name, n_train, seed + offset)
        train_b, _ = _distribution(name, n_train, seed + offset + 100)
        test_a, _ = _distribution(name, n_test, seed + offset + 10000)
        test_b, _ = _distribution(name, n_test, seed + offset + 10100)
        train, test = np.column_stack((train_a, train_b)), np.column_stack((test_a, test_b))
        for method in ("nominal", "scenario", "unbiased_kde", "local_shifted_epanechnikov"):
            h = silverman_bandwidth(train[:, 0])
            def residual(x, z): return np.asarray(problem.uncertain_residuals(x, z), float).reshape(-1)
            if method == "nominal":
                mean = np.mean(train, axis=0)
                fn = lambda x: float(np.max(residual(x, mean[None, :])))
            elif method == "scenario":
                fn = lambda x: float(np.max(residual(x, train)))
            elif method == "unbiased_kde":
                fn = lambda x: float((1-epsilon) - kde_safe_probability(residual(x, train), h, "gaussian"))
            else:
                fn = lambda x: float(kde_violation_upper(residual(x, train), h, "epanechnikov", True)-epsilon)
            started = time.perf_counter()
            solved = solve_static(problem, initial_guess=np.array([.5, .5]), chance_constraint=fn, maxiter=300)
            x = solved["decision"]
            tr, te = residual(x, train), residual(x, test)
            est = (empirical_violation(tr) if method in ("nominal", "scenario")
                   else 1-kde_safe_probability(tr, h, "gaussian"))
            rows.append(evaluate_residuals(tr, te, epsilon=epsilon, estimated_violation=est,
                objective=solved["objective"], runtime_seconds=time.perf_counter()-started,
                success=solved["success"], message=solved["message"], benchmark="static_nonlinear",
                method=method, seed=seed+offset, metadata={"distribution": name, "bandwidth": h,
                "decision": x.tolist(), "uncertainty_metadata": meta, "solver_iterations": solved["nit"]}))
    return rows


def _dispatch(seed, n_train, n_test, epsilon):
    p = make_energy_dispatch(); train, meta = sample_renewable_error(n_train, seed=seed)
    test, _ = sample_renewable_error(n_test, seed=seed+10000); rows=[]
    for method in ("nominal", "scenario", "unbiased_kde", "local_shifted_epanechnikov"):
        h = silverman_bandwidth(train)
        def residual(x, z): return p.uncertain_residuals(x, z)[:, 0]
        if method == "nominal": fn=lambda x: float(np.max(residual(x, np.array([np.mean(train)]))))
        elif method == "scenario": fn=lambda x: float(np.max(residual(x, train)))
        elif method == "unbiased_kde": fn=lambda x: float((1-epsilon)-kde_safe_probability(residual(x, train),h))
        else: fn=lambda x: float(kde_violation_upper(residual(x, train),h,"epanechnikov",True)-epsilon)
        solved=solve_dispatch(p, initial_guess=np.array([40.,40.]), chance_constraint=fn, maxiter=300)
        tr,te=residual(solved["decision"],train),residual(solved["decision"],test)
        est=1-kde_safe_probability(tr,h,"gaussian")
        rows.append(evaluate_residuals(tr,te,epsilon=epsilon,estimated_violation=est,objective=solved["objective"],
            runtime_seconds=solved["runtime_seconds"],success=solved["success"],message=solved["message"],
            benchmark="energy_dispatch",method=method,seed=seed,metadata={"decision":solved["decision"].tolist(),"bandwidth":h,"uncertainty_metadata":meta}))
    return rows, train, test


def _joint(seed, epsilon):
    rng=np.random.default_rng(seed); r=rng.normal(-.15,.35,size=(4000,2)); alloc=boole_risk_allocation(epsilon,2)
    out=[]
    for method, v in (("boole_equal", np.any(r>0,axis=1)), ("max_exact", max_violation(r,axis=1)>0),
                      ("logsumexp", smooth_max(r,.1,axis=1)>0)):
        out.append({"schema_version":"1.0","benchmark":"joint_static","method":method,"seed":seed,
          "n_train":0,"n_test":len(r),"epsilon":epsilon,"objective":None,"train_violation":None,
          "test_violation":float(np.mean(v)),"estimated_violation":float(np.mean(v)),"gap":float(epsilon-np.mean(v)),
          "runtime_seconds":None,"success":True,"message":"joint statistic evaluation",
          "joint_test_violation":float(np.mean(v)),"individual_violation":np.mean(r>0,axis=0).tolist(),
          "risk_allocation":alloc.tolist(),"tau":.1 if method=="logsumexp" else None})
    return out, r


def run_all(output="results", *, seed=20260906, force=False):
    root=Path(output); root.mkdir(parents=True,exist_ok=True); samples=root/"samples"; samples.mkdir(exist_ok=True)
    rows=_static(seed,250,4000,.1); dispatch,dt,de=_dispatch(seed+1,250,4000,.1); rows.extend(dispatch)
    joint,joint_samples=_joint(seed+2,.1); rows.extend(joint)
    lunar=make_lunar_landing(); terminal_rng=np.random.default_rng(seed+4); terminal=terminal_rng.normal(0,lunar.terminal_sigma,250); path,_=sample_renewable_error(250,seed=seed+4)
    staged=solve_lunar_stages(lunar,terminal_samples=terminal,path_samples=path,maxiter=80)
    for stage in staged["stages"]:
        rows.append({"schema_version":"1.0","benchmark":"lunar_landing","method":stage["stage"],"seed":seed+4,"n_train":250,"n_test":0,"epsilon":lunar.epsilon_a,
                     "objective":stage["objective"],"train_violation":None,"test_violation":None,"estimated_violation":None,"gap":None,
                     "runtime_seconds":stage["runtime_seconds"],"success":stage["success"],"message":stage["message"],"solver_diagnostics":stage["constraint_diagnostics"]})
    for d_i, name in enumerate(("gaussian","bimodal","skewed","heavy_tailed")):
        write_array(np.column_stack((_distribution(name,250,seed+d_i)[0],
                                     _distribution(name,250,seed+d_i+100)[0])),
                    samples/(f"static_{name}_train.npy"),force=force)
        write_array(np.column_stack((_distribution(name,4000,seed+d_i+10000)[0],
                                     _distribution(name,4000,seed+d_i+10100)[0])),
                    samples/(f"static_{name}_test.npy"),force=force)
    for name,arr in (("dispatch_train",dt),("dispatch_test",de),("joint_residuals",joint_samples),
                     ("lunar_terminal",terminal),("lunar_path",path)):
        write_array(arr,samples/(name+".npy"),force=force)
    metadata={"seed":seed,"protocol":"independent seeded train/test; local SciPy solvers","records":len(rows),
      "configuration":{"n_train":250,"n_test":4000,"epsilon":.1,"static_distributions":["gaussian","bimodal","skewed","heavy_tailed"],
      "solver":{"method":"SLSQP","maxiter":300},"lunar_maxiter":80},"lunar_stage_order":staged["stage_order"],
      "sample_files":"samples/*.npy","local_bias_name":"local_shifted_epanechnikov","published_comparison":"Keil values are external report values, not local outputs"}
    write_run_bundle(rows,metadata,root,force=force); return rows


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output",default="results"); ap.add_argument("--seed",type=int,default=20260906); ap.add_argument("--force",action="store_true"); args=ap.parse_args(); run_all(args.output,seed=args.seed,force=args.force)

if __name__ == "__main__": main()
