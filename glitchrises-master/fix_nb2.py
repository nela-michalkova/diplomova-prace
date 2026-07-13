import json

with open("rapid_crust_coupling.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = "".join(cell["source"])
        if "rename_cols = {" in source:
            # We will just replace the whole dict
            new_dict = """rename_cols = {
    "R_cci [km]": "\\\\shortstack{$R_{\\\\rm cci}$ \\\\\\\\ $[\\\\text{km}]$}",
    "M_coeff": "$M_{\\\\rm coeff}$",
    "M_core": "\\\\shortstack{$M_{\\\\rm core}$ \\\\\\\\ $[M_{\\\\odot}]$}",
    "M_NS [M_sun]": "\\\\shortstack{$M_{\\\\rm NS}$ \\\\\\\\ $[M_{\\\\odot}]$}",
    "R_NS [km]": "\\\\shortstack{$R_{\\\\rm NS}$ \\\\\\\\ $[\\\\text{km}]$}",
    "Max Profil A [muHz]": "\\\\shortstack{$\\\\Delta\\\\nu_{\\\\rm max}^A$ \\\\\\\\ $[\\\\mu\\\\text{Hz}]$}",
    "Čas max A [s]": "\\\\shortstack{$t_{\\\\rm max}^A$ \\\\\\\\ $[\\\\text{s}]$}",
    "Max Profil B [muHz]": "\\\\shortstack{$\\\\Delta\\\\nu_{\\\\rm max}^B$ \\\\\\\\ $[\\\\mu\\\\text{Hz}]$}",
    "Čas max B [s]": "\\\\shortstack{$t_{\\\\rm max}^B$ \\\\\\\\ $[\\\\text{s}]$}",
    "Max Profil C [muHz]": "\\\\shortstack{$\\\\Delta\\\\nu_{\\\\rm max}^C$ \\\\\\\\ $[\\\\mu\\\\text{Hz}]$}",
    "Čas max C [s]": "\\\\shortstack{$t_{\\\\rm max}^C$ \\\\\\\\ $[\\\\text{s}]$}"
}"""
            import re
            source = re.sub(r'rename_cols = \{.*?\}', new_dict, source, flags=re.DOTALL)
            cell["source"] = [line + "\n" for line in source.split("\n")]
            # Strip trailing newline from the last element if needed
            if cell["source"]: cell["source"][-1] = cell["source"][-1][:-1]

with open("rapid_crust_coupling.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
