import pandas as pd

df_res_1 = pd.read_csv("Experiment_1_Tabulka_Rcci.csv")
df_res_2 = pd.read_csv("Experiment_2_Tabulka_Mcoeff.csv")

rename_cols = {
    "R_cci [km]": "\\shortstack{$R_{\\rm cci}$ \\\\ $[\\text{km}]$}",
    "M_core": "\\shortstack{$M_{\\rm core}$ \\\\ $[M_{\\odot}]$}",
    "M_NS [M_sun]": "\\shortstack{$M_{\\rm NS}$ \\\\ $[M_{\\odot}]$}",
    "R_NS [km]": "\\shortstack{$R_{\\rm NS}$ \\\\ $[\\text{km}]$}",
    "Max Profil A [muHz]": "\\shortstack{$\\Delta\\nu_{\\rm max}^A$ \\\\ $[\\mu\\text{Hz}]$}",
    "Čas max A [s]": "\\shortstack{$t_{\\rm max}^A$ \\\\ $[\\text{s}]$}",
    "Max Profil B [muHz]": "\\shortstack{$\\Delta\\nu_{\\rm max}^B$ \\\\ $[\\mu\\text{Hz}]$}",
    "Čas max B [s]": "\\shortstack{$t_{\\rm max}^B$ \\\\ $[\\text{s}]$}",
    "Max Profil C [muHz]": "\\shortstack{$\\Delta\\nu_{\\rm max}^C$ \\\\ $[\\mu\\text{Hz}]$}",
    "Čas max C [s]": "\\shortstack{$t_{\\rm max}^C$ \\\\ $[\\text{s}]$}"
}

import re
def export_latex(df, filename):
    df_tex = df.rename(columns=rename_cols)
    tex_code = df_tex.to_latex(index=False, escape=False, float_format="%.3f", column_format="|" + "c|" * len(df_tex.columns))
    tex_code = re.sub(r'(\d+\.\d*[1-9])0+(?=\s|&|\\\\)', r'\1', tex_code)
    tex_code = re.sub(r'(\d+)\.0+(?=\s|&|\\\\)', r'\1', tex_code)
    # Add standard horizontal lines
    tex_code = tex_code.replace("\\toprule", "\\hline").replace("\\midrule", "\\hline").replace("\\bottomrule", "\\hline")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(tex_code)

export_latex(df_res_1, "Experiment_1_Tabulka_Rcci.tex")
export_latex(df_res_2, "Experiment_2_Tabulka_Mcoeff.tex")
print("Tables successfully exported with shortstack.")
