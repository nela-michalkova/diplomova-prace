import pandas as pd
import matplotlib.pyplot as plt

df_res_2 = pd.read_csv("Experiment_2_Tabulka_Mcoeff.csv")

ref_max_A = 245.42
ref_time_A = 3.91

# 3. graf: Maximum Profil A vs M_coeff
fig3, ax3 = plt.subplots(figsize=(8, 6))
ax3.plot(1.4, ref_max_A, marker='+', color='red', markersize=10, markeredgewidth=2, linestyle='none', label=rf'Sada 1 ($10$ km, $1.4\ M_{{ \odot }}$)')
ax3.plot(df_res_2["M_core"], df_res_2["Max Profil A [muHz]"], marker='x', color='purple', markersize=10, markeredgewidth=2, linestyle='none', label='Sada 2')
ax3.set_xlabel(r"$M_{\rm core}$ [$M_{\odot}$]", fontsize=14)
ax3.set_ylabel(r"$\Delta\nu_{\rm max}$ [$\mu$Hz]", fontsize=14)
ax3.grid(True, linestyle='--', alpha=0.6)
ax3.legend(fontsize=12)
plt.savefig("zavislost_maxA_Mcoeff.png", bbox_inches='tight')

# 4. graf: Čas maxima Profil A vs M_coeff
fig4, ax4 = plt.subplots(figsize=(8, 6))
ax4.plot(1.4, ref_time_A, marker='+', color='red', markersize=10, markeredgewidth=2, linestyle='none', label=rf'Sada 1 ($10$ km, $1.4\ M_{{ \odot }}$)')
ax4.plot(df_res_2["M_core"], df_res_2["Čas max A [s]"], marker='x', color='blue', markersize=10, markeredgewidth=2, linestyle='none', label='Sada 2')
ax4.set_xlabel(r"$M_{\rm core}$ [$M_{\odot}$]", fontsize=14)
ax4.set_ylabel(r"$t_{\rm max}$ [s]", fontsize=14)
ax4.grid(True, linestyle='--', alpha=0.6)
ax4.legend(fontsize=12)
plt.savefig("zavislost_casA_Mcoeff.png", bbox_inches='tight')
