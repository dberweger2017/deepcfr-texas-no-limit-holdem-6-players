"""Bind selected-base calibration roots to the published #145 evidence."""

import argparse
import json
from pathlib import Path
import subprocess

from scripts.hu20_search_runtime import atomic_json
from src.arena.schedule import digest
from src.blueprint.hu20_turn_solver import file_hash


def freeze(run, planned, part_a, report, published_commit):
    if part_a["status"]!="complete":raise ValueError("Select the base from complete Part A")
    state=json.loads((run/"result.json").read_text())
    if state["status"]!="completed" or state["jobs_done"]!=state["jobs_total"]:
        raise ValueError("#145 main evidence is incomplete")
    # A public Git object must contain the exact final report bytes being used.
    relative=str(report.resolve().relative_to(Path.cwd().resolve()))
    committed=subprocess.check_output(["git","show",published_commit+":"+relative])
    if committed!=report.read_bytes():raise ValueError("Final report bytes differ from published commit")
    refs=subprocess.check_output(["git","branch","-r","--contains",published_commit],text=True)
    if not refs.strip():raise ValueError("Final-report commit is not present in a fetched remote branch")
    base=part_a["base_decision"]["base"]
    specs={s["name"]:s for s in planned["models"] if s["strategy"]==base}
    items=[];excluded=[];missing=[]
    for directory in sorted((run/"spots").iterdir()):
        job=json.loads((directory/"job.json").read_text());spec=job["policy"]
        if spec["name"] not in specs:continue
        if spec["sha256"]!=specs[spec["name"]]["sha256"]:raise ValueError("#145 artifact differs")
        path=directory/"result.json"
        if not path.exists():missing.append(str(directory));continue
        result=json.loads(path.read_text());root=job["job"]["root"]
        provenance={"result_path":str(path),"result_sha256":file_hash(path),
                    "job_sha256":file_hash(directory/"job.json"),"final_report_commit":published_commit}
        if result["event"]=="spot_excluded":
            excluded.append({"root":root,"policy":specs[spec["name"]],"reason":result.get("reason"),
                             "provenance":provenance});continue
        native=next(g for g in result["gates"] if g["gate"]=="V1")["passed"]
        request=directory/Path(result["attempt_path"]).name/"prepared/request.json"
        if file_hash(request)!=result["request_sha256"]:raise ValueError("Reference request differs")
        prepared=json.loads(request.read_text());provenance["request_sha256"]=file_hash(request)
        items.append({"root":root,"policy":specs[spec["name"]],"reference_native_verified":native,
            "reference_ranges":prepared["ranges"],"e_bp_pct_pot":result["e_bp_pct_pot"],
            "provenance":provenance})
    if missing or len(items)+len(excluded)!=48*3:
        raise ValueError("All 48 roots and three selected-base lineages must be accounted for")
    return {"145_final_report_pushed":True,"final_report_sha256":file_hash(report),
        "final_report_commit":published_commit,"base":base,"items":items,"exclusions":excluded,
        "part_a_sha256":digest(part_a),"reference_law":"#145 original public base likelihood product"}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("run","planned","part-a","report","out"):p.add_argument("--"+n,type=Path,required=True)
    p.add_argument("--published-commit",required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError("Preserve reference freeze")
    atomic_json(a.out,freeze(a.run,json.loads(a.planned.read_text()),json.loads(a.part_a.read_text()),
                            a.report,a.published_commit))


if __name__ == "__main__":main()
