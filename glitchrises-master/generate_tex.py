import pandas as pd
import re

df_res_1 = pd.read_csv("Experiment_1_Tabulka_Rcci.csv")
df_res_2 = pd.read_csv("Experiment_2_Tabulka_Mcoeff.csv")

# Transform table from Experiment 1: drop 'Kolo' and rename 'M_coeff' to 'M_core'
if "Kolo" in df_res_1.columns:
    df_res_1 = df_res_1.drop(columns=["Kolo"])
if "M_coeff" in df_res_1.columns:
    df_res_1 = df_res_1.rename(columns={"M_coeff": "M_core"})

df_res_1.to_csv("Experiment_1_Tabulka_Rcci.csv", index=False)

# Transform table from Experiment 2: drop 'Kolo' and rename 'M_coeff' to 'M_core'
if "Kolo" in df_res_2.columns:
    df_res_2 = df_res_2.drop(columns=["Kolo"])
if "M_coeff" in df_res_2.columns:
    df_res_2 = df_res_2.rename(columns={"M_coeff": "M_core"})

df_res_2.to_csv("Experiment_2_Tabulka_Mcoeff.csv", index=False)

rename_cols = {
    "R_cci [km]": "$R_{\\rm cci}$ [km]",
    "M_coeff": "$M_{\\rm coeff}$",
    "M_core": "$M_{\\rm core}$",
    "M_NS [M_sun]": "$M_{\\rm NS}$ [$M_{\\odot}$]",
    "R_NS [km]": "$R_{\\rm NS}$ [km]",
    "Max Profil A [muHz]": "Max Profil A [$\\mu$Hz]",
    "Max Profil B [muHz]": "Max Profil B [$\\mu$Hz]",
    "Max Profil C [muHz]": "Max Profil C [$\\mu$Hz]"
}

def export_latex(df, filename):
    df_tex = df.rename(columns=rename_cols)
    tex_code = df_tex.to_latex(index=False, escape=False, float_format="%.3f", column_format="|" + "c|" * len(df_tex.columns))
    tex_code = re.sub(r'(\d+\.\d*[1-9])0+(?=\s|&|\\\\|\\n)', r'\1', tex_code)
    tex_code = re.sub(r'(\d+)\.0+(?=\s|&|\\\\|\\n)', r'\1', tex_code)
    # Restore hline
    tex_code = tex_code.replace("\\toprule", "\\hline").replace("\\midrule", "\\hline").replace("\\bottomrule", "\\hline")
    tex_code = "\\resizebox{\\textwidth}{!}{%\n" + tex_code.strip() + "%\n}\n"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(tex_code)

export_latex(df_res_1, "Experiment_1_Tabulka_Rcci.tex")
export_latex(df_res_2, "Experiment_2_Tabulka_Mcoeff.tex")
print("TeX tables generated.")

