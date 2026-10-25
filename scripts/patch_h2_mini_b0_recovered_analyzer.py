from pathlib import Path
p=Path("scripts/analyze_h2_mini_b0_expansion.py")
s=p.read_text(encoding="utf-8")
assert "resolve_recovered" not in s
s=s.replace('    for filename in sorted({p.name for p in directory.glob("*.csv")}-set(m["batches"])):',
'''    overrides={};auxiliary=set();recovery=None
    if root.resolve()==ROOT.resolve() and manifest_path.resolve()==EXP_MANIFEST.resolve():
        try:
            from h2_mini_b0_recovered_evidence import resolve_recovered
            overrides,auxiliary,recovery=resolve_recovered(m,root,directory)
        except (ValueError,KeyError,OSError) as exc:
            error("Recovered provenance integrity failure",detail=str(exc))
    for filename in sorted({p.name for p in directory.glob("*.csv")}-set(m["batches"])-auxiliary):''')
s=s.replace('        p=directory/fn','        p=overrides.get(fn,directory/fn)')
s=s.replace('"manifest_sha256":file_sha(manifest_path),"records":by}', '"manifest_sha256":file_sha(manifest_path),"records":by,"recovery_manifest":recovery}')
start=s.index('    eligible=[];outcomes=[]')
end=s.index('    timing=',start)
s=s[:start]+'''    # The audit never merges pools or computes capacity, including when its gate fails.
    outcomes=[];capacity=None
    audit["operative_gate"]="B0_confirmation disagreement <=10% on all180; current human instruction"
    audit["status"]=("INCOMPLETE_EXPANDED_REVIEW" if not complete else
        "EXPANDED_AUDIT_GATE_PASS_POST_REVIEW_AUDIT_REQUIRED" if audit["B0_confirmation_gate_pass"] else
        "STOP_EXPANDED_AUDIT_DISAGREEMENT")
    diagnostics=pair_diagnostics(data)
'''+s[end:]
s=s.replace('"expanded_audit":audit,"old75_metrics"', '"expanded_audit":audit,"diagnostics":diagnostics,"recovery_manifest":data.get("recovery_manifest"),"old75_metrics"')
s=s.replace(' or audit["decision_gate_pass"] is not True','')
s=s.replace('then exact capacity and minimum-needed reserve review preparation if required',
            'then exact capacity; any reserve review requires separate human approval')
s=s.replace('    save(args.output,report)','''    if args.output.exists():
        previous=args.output.read_bytes()
        history=ROOT/"results/h2_mini_b0_review_history"
        history.mkdir(parents=True,exist_ok=True)
        snapshot=history/("expanded_analysis_before_"+sha(previous)[:16]+".json")
        if snapshot.exists():
            if snapshot.read_bytes()!=previous:raise ValueError("Historical snapshot mismatch")
        else:snapshot.write_bytes(previous)
        report["prior_report_snapshot"]=str(snapshot.relative_to(ROOT)).replace("\\\\","/")
    save(args.output,report)''')
extra='''def pair_diagnostics(data):
    baseline=data["base_data"];mapping={r["review_id"]:r for r in data["manifest"]["records"]}
    pairs=[]
    def length_bin(n):
        lower=1
        for upper in (16,24,32,48,64,128,256,1280):
            if n<=upper:return str(lower)+"-"+str(upper)
            lower=upper+1
        return ">1280"
    for rid in data["manifest"]["old_audited_training_ids"]+list(mapping):
        r=data["records"][rid]
        primary=next(a for a in r["reviewer_assignments"] if a["review_role"]=="PRIMARY")
        first=baseline["states"][(primary["reviewer"],rid)]["values"]
        if rid in mapping:
            if data["states"][rid]["status"]!="completed":continue
            second_who=mapping[rid]["second_reviewer"];second=data["states"][rid]["values"];cohort="new105"
        else:
            assignment=next(a for a in r["reviewer_assignments"] if a["review_role"]=="SECOND_REVIEW")
            second_who=assignment["reviewer"];second=baseline["states"][(second_who,rid)]["values"];cohort="original75"
        fields=["reviewer_decision","b0_confirmed","confidence"]
        pairs.append({"review_id":rid,"cohort":cohort,"language":r["language"],"token_length":r["deberta_tokens"],
            "token_length_bin":length_bin(r["deberta_tokens"]),"primary_reviewer":primary["reviewer"],"second_reviewer":second_who,
            "primary":{f:first[f] for f in fields},"second":{f:second[f] for f in fields},
            "decision_disagreement":first["reviewer_decision"]!=second["reviewer_decision"],
            "B0_disagreement":first["b0_confirmed"]!=second["b0_confirmed"],
            "safety_critical":{first["reviewer_decision"],second["reviewer_decision"]}=={"ACCEPT_BENIGN","REJECT_TRUE_INJECTION"}})
    axes={
      "language":lambda r:r["language"],"token_length_bin":lambda r:r["token_length_bin"],
      "primary_reviewer":lambda r:r["primary_reviewer"],"second_reviewer":lambda r:r["second_reviewer"],
      "primary_confidence":lambda r:r["primary"]["confidence"],"second_confidence":lambda r:r["second"]["confidence"],
      "confidence_pair":lambda r:r["primary"]["confidence"]+"/"+r["second"]["confidence"]}
    breakdown={}
    for axis,key in axes.items():
        groups={}
        for r in pairs:groups.setdefault(key(r),[]).append(r)
        breakdown[axis]={k:base.pair_summary([(r["primary"],r["second"]) for r in rows]) for k,rows in sorted(groups.items())}
    return {"pair_count":len(pairs),"all_pair_evidence_metadata_only":pairs,"breakdown":breakdown,
      "disagreement_ids":[r["review_id"] for r in pairs if r["decision_disagreement"] or r["B0_disagreement"]],
      "safety_critical_ids":[r["review_id"] for r in pairs if r["safety_critical"]],
      "interpretation":"Descriptive denominators, not causal language/reviewer effects; primary and second reviewer identities are coupled by opposite-reviewer design."}

'''
s=s.replace('def analyze_expansion(data):',extra+'def analyze_expansion(data):')
compile(s,str(p),"exec")
p.write_text(s,encoding="utf-8")
