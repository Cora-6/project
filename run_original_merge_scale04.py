#!/usr/bin/env python3
import json
import re
from pathlib import Path

NOTEBOOK = Path("/home/viplab/liukejia/project/merge_glb_muufl.ipynb")
ROOT = Path("/data/muufl_grid_50m_pipeline")
OUT = ROOT / "merged_glbs_original_scale04"
OUT.mkdir(parents=True, exist_ok=True)

jobs = [
    (ROOT / "vggt_generated_exact_reproject", OUT / "merged_generated_exact_original_scale04.glb"),
    (ROOT / "vggt_3dtile_exact_reproject", OUT / "merged_3dtile_exact_original_scale04.glb"),
]

nb = json.loads(NOTEBOOK.read_text())
source = "".join(nb["cells"][0]["source"])

for input_dir, output_file in jobs:
    code = re.sub(r"^input_dir\s*=.*$", f"input_dir = {str(input_dir)!r}", source, flags=re.MULTILINE)
    code = re.sub(r"^output_file\s*=.*$", f"output_file = {str(output_file)!r}", code, flags=re.MULTILINE)
    code = re.sub(r"^scale\s*=.*$", "scale = 0.4", code, flags=re.MULTILINE)
    namespace = {"__name__": "__main__"}
    exec(compile(code, str(NOTEBOOK) + "#cell0", "exec"), namespace)
    print(f"ORIGINAL_MERGE_DONE {output_file}", flush=True)
