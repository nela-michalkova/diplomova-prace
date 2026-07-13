import json

with open("rapid_crust_coupling.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        new_source = []
        for line in cell["source"]:
            new_line = line.replace('df_res_2["M_coeff"]', 'df_res_2["M_core"]')
            new_source.append(new_line)
        cell["source"] = new_source

with open("rapid_crust_coupling.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
