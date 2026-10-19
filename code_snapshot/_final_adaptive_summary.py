import json
import hashlib
from pathlib import Path
for name in [
    "output/research/warmflow_ut/adaptive_source_posterior_fixed_v1/audit.json",
    "output/research/warmflow_mit_seed1final/adaptive_source_posterior_fixed_v1/audit.json",
    "output/research/warmflow_mit/adaptive_source_posterior_fixed_v1/audit.json",
]:
    data = json.loads(Path(name).read_text())
    print(name, data["status"], data["outcome"], data["checks"]["validation_only"], data["checks"]["no_post_test_search"])
for name in [
    "output/research/warmflow_ut/adaptive_source_posterior_fixed_v1/results.json",
    "output/research/warmflow_mit_seed1final/adaptive_source_posterior_fixed_v1/results.json",
    "output/research/warmflow_mit/adaptive_source_posterior_fixed_v1/results.json",
    "research/PAPER_RESULTS.md",
]:
    print("sha", name, hashlib.sha256(Path(name).read_bytes()).hexdigest())
