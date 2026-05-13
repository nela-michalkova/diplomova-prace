import json

with open("rapid_crust_coupling.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

new_lines = [
    "\n",
    "    # Export do LaTeX tabulek s hezkým formátováním (pro DP)\n",
    "    rename_cols = {\n",
    "        \"R_cci [km]\": \"$R_{\\\\rm cci}$ [km]\",\n",
    "        \"M_coeff\": \"$M_{\\\\rm coeff}$\",\n",
    "        \"M_NS [M_sun]\": \"$M_{\\\\rm NS}$ [$M_{\\\\odot}$]\",\n",
    "        \"R_NS [km]\": \"$R_{\\\\rm NS}$ [km]\",\n",
    "        \"Max Profil A [muHz]\": \"Max Profil A [$\\\\mu$Hz]\",\n",
    "        \"Max Profil B [muHz]\": \"Max Profil B [$\\\\mu$Hz]\",\n",
    "        \"Max Profil C [muHz]\": \"Max Profil C [$\\\\mu$Hz]\"\n",
    "    }\n",
    "    import re\n",
    "    def export_latex(df, filename):\n",
    "        df_tex = df.rename(columns=rename_cols)\n",
    "        tex_code = df_tex.to_latex(index=False, escape=False, float_format=\"%.3f\", column_format=\"|\" + \"c|\" * len(df_tex.columns))\n",
    "        tex_code = re.sub(r'(\\d+\\.\\d*[1-9])0+(?=\\s|&|\\\\\\\\)', r'\\1', tex_code)\n",
    "        tex_code = re.sub(r'(\\d+)\\.0+(?=\\s|&|\\\\\\\\)', r'\\1', tex_code)\n",
    "        # Add standard horizontal lines exactly like the previous table style\n",
    "        tex_code = tex_code.replace(\"\\\\toprule\", \"\\\\hline\").replace(\"\\\\midrule\", \"\\\\hline\").replace(\"\\\\bottomrule\", \"\\\\hline\")\n",
    "        tex_code = \"\\\\resizebox{\\\\textwidth}{!}{%\\n\" + tex_code.strip() + \"%\\n}\\n\"\n",
    "        with open(filename, \"w\", encoding=\"utf-8\") as f:\n",
    "            f.write(tex_code)\n",
    "\n",
    "    export_latex(df_res_1, \"Experiment_1_Tabulka_Rcci.tex\")\n",
    "    export_latex(df_res_2, \"Experiment_2_Tabulka_Mcoeff.tex\")\n"
]

found = False
for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        src = cell.get('source', [])
        for i, line in enumerate(src):
            if "df_res_2.to_csv" in line:
                for existing_line in src:
                    if "export_latex" in existing_line:
                        found = True
                if not found:
                    src = src[:i+1] + new_lines + src[i+1:]
                    cell['source'] = src
                    found = True
                break
        if found:
            break

with open("rapid_crust_coupling.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook successfully updated.")
