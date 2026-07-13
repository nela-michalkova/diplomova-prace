import pandas as pd
import re

df_res_1 = pd.read_csv("Experiment_1_Tabulka_Rcci.csv")
df_res_2 = pd.read_csv("Experiment_2_Tabulka_Mcoeff.csv")

rename_cols = {
    "R_cci [km]": "\\makecell{$R_{\\rm cci}$ \\\\[1ex] $[\\text{km}]$}",
    "M_coeff": "$M_{\\rm coeff}$",
    "M_core": "\\makecell{$M_{\\rm core}$ \\\\[1ex] $[M_{\\odot}]$}",
    "M_NS [M_sun]": "\\makecell{$M_{\\rm NS}$ \\\\[1ex] $[M_{\\odot}]$}",
    "R_NS [km]": "\\makecell{$R_{\\rm NS}$ \\\\[1ex] $[\\text{km}]$}",
    "Max Profil A [muHz]": "\\makecell{$\\Delta\\nu_{\\rm max, A}$ \\\\[1ex] $[\\mu\\text{Hz}]$}",
    "Čas max A [s]": "\\makecell{$t_{\\rm max, A}$ \\\\[1ex] $[\\text{s}]$}",
    "Max Profil B [muHz]": "\\makecell{$\\Delta\\nu_{\\rm max, B}$ \\\\[1ex] $[\\mu\\text{Hz}]$}",
    "Čas max B [s]": "\\makecell{$t_{\\rm max, B}$ \\\\[1ex] $[\\text{s}]$}",
    "Max Profil C [muHz]": "\\makecell{$\\Delta\\nu_{\\rm max, C}$ \\\\[1ex] $[\\mu\\text{Hz}]$}",
    "Čas max C [s]": "\\makecell{$t_{\\rm max, C}$ \\\\[1ex] $[\\text{s}]$}"
}

def export_latex(df, filename):
    df_tex = df.rename(columns=rename_cols)
    tex_code = df_tex.to_latex(index=False, escape=False, float_format="%.3f", column_format="|" + "c|" * len(df_tex.columns))
    tex_code = re.sub(r'(\d+\.\d*[1-9])0+(?=\s|&|\\\\)', r'\1', tex_code)
    tex_code = re.sub(r'(\d+)\.0+(?=\s|&|\\\\)', r'\1', tex_code)
    # Removing vertical lines since booktabs works best without them
    tex_code = tex_code.replace("{|c|c|c|c|c|c|c|c|c|c|}", "{cccccccccc}")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(tex_code)

export_latex(df_res_1, "Experiment_1_Tabulka_Rcci.tex")
export_latex(df_res_2, "Experiment_2_Tabulka_Mcoeff.tex")
