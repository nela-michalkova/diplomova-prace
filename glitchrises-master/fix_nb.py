import json

with open("rapid_crust_coupling.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        new_source = []
        for line in cell["source"]:
            if '"R_cci [km]": "$R_{\\\\rm cci}$ [km]",' in line:
                line = line.replace('"$R_{\\\\rm cci}$ [km]"', '"\\\\shortstack{$R_{\\\\rm cci}$ \\\\\\\\ $[\\\\text{km}]$}"')
            elif '"M_core": "$M_{\\\\rm core}$",' in line:
                line = line.replace('"$M_{\\\\rm core}$"', '"\\\\shortstack{$M_{\\\\rm core}$ \\\\\\\\ $[M_{\\\\odot}]$}"')
            elif '"M_NS [M_sun]": "$M_{\\\\rm NS}$ [$M_{\\\\odot}$]",' in line:
                line = line.replace('"$M_{\\\\rm NS}$ [$M_{\\\\odot}$]"', '"\\\\shortstack{$M_{\\\\rm NS}$ \\\\\\\\ $[M_{\\\\odot}]$}"')
            elif '"R_NS [km]": "$R_{\\\\rm NS}$ [km]",' in line:
                line = line.replace('"$R_{\\\\rm NS}$ [km]"', '"\\\\shortstack{$R_{\\\\rm NS}$ \\\\\\\\ $[\\\\text{km}]$}"')
            elif '"Max Profil A [muHz]": "Max Profil A [$\\\\mu$Hz]",' in line:
                line = line.replace('"Max Profil A [$\\\\mu$Hz]"', '"\\\\shortstack{$\\\\Delta\\\\nu_{\\\\rm max}^A$ \\\\\\\\ $[\\\\mu\\\\text{Hz}]$}"')
            elif '"Max Profil B [muHz]": "Max Profil B [$\\\\mu$Hz]",' in line:
                line = line.replace('"Max Profil B [$\\\\mu$Hz]"', '"\\\\shortstack{$\\\\Delta\\\\nu_{\\\\rm max}^B$ \\\\\\\\ $[\\\\mu\\\\text{Hz}]$}"')
            elif '"Max Profil C [muHz]": "Max Profil C [$\\\\mu$Hz]"' in line:
                line = line.replace('"Max Profil C [$\\\\mu$Hz]"', '"\\\\shortstack{$\\\\Delta\\\\nu_{\\\\rm max}^C$ \\\\\\\\ $[\\\\mu\\\\text{Hz}]$}"')
            # Handle Čas variables that are not explicitly in rename_cols but might need to be
            new_source.append(line)
        
        # Check if we need to add Čas rename
        if "rename_cols = {" in "".join(cell["source"]) and '"Čas max A [s]": ' not in "".join(new_source):
            # We will just inject the missing entries inside rename_cols
            pass # Actually we don't need to overcomplicate it if the notebook is already mostly okay, but let's just make it robust:
        
        cell["source"] = new_source

with open("rapid_crust_coupling.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
