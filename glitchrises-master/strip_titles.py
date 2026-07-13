import json

with open("rapid_crust_coupling.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        new_source = []
        for line in cell["source"]:
            if "ax1.set_title" in line or "ax2.set_title" in line or "ax3.set_title" in line or "ax4.set_title" in line:
                continue
            new_source.append(line)
        cell["source"] = new_source

with open("rapid_crust_coupling.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
