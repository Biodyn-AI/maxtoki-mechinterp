import json
R="<REPO_ROOT>/projects/maxtoki/runs/exhaustive-mapping-217M/outputs/"
d=json.load(open(R+"experiment2_v2/combinatorial_summary.json"))
print("Hypothesis: the deepest hook overwrites everything upstream, so AB==B, AC==C, BC==C, ABC==C")
for t in d["triplets"]:
    rAB,rAC,rBC=t["pairwise_ratio_AB"],t["pairwise_ratio_AC"],t["pairwise_ratio_BC"]
    a_b=1/rAB-1          # eff_A/eff_B  from AB = B/(A+B)
    b_c=1/rBC-1          # eff_B/eff_C  from BC = C/(B+C)
    a_c=a_b*b_c
    pred_AC=1/(1+a_c)
    pred_3=1/(a_c+b_c+1)
    pred_marg=1-b_c
    print(f"T{t['triplet']}: A:B:C = {a_c:.3f}:{b_c:.3f}:1 | AC obs {rAC:.4f} pred {pred_AC:.4f} | 3way obs {t['threeway_ratio']:.4f} pred {pred_3:.4f} | margC|AB obs {t['marginal_C_given_AB']:.4f} pred {pred_marg:.4f}")
d1=json.load(open(R+"experiment2/combinatorial_summary.json"))
for t in d1["triplets"]:
    print("v1", t["features"], t["pairwise_ratio"], t["threeway_ratio"], t["marginal_C_given_AB"], t["n_targets_with_effect"])
# steering ratios
s=json.load(open(R+"experiment3/steering_summary.json"))
p5=s["per_layer_summary_alpha5"]; p2=s["per_layer_summary_alpha2"]
L0=p5["L0"]["mean_delta_s"]
for k in p5: print(k, "a5 ds", p5[k]["mean_delta_s"], "a2 ds", p2[k]["mean_delta_s"], "L0/this", round(L0/p5[k]["mean_delta_s"],2), "a5/a2", round(p5[k]["mean_delta_s"]/p2[k]["mean_delta_s"],4))
