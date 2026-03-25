#!/usr/bin/env python
# coding: utf-8

# # Glitch rises as a test for rapid superfluid coupling in neutron stars -- theoretical model
# ---

# In[3]:


#!/usr/bin/env python
"""rapid_crust_coupling.ipynb: a notebook to recreate results of Graber et al. (2018)"""

__author__ = "Vanessa Graber"
__copyright__ = "Copyright 2018"
__credits__ = ["Vanessa Graber"]
__license__ = "MIT"
__maintainer__ = "Vanessa Graber"
__email__ = "vanessa.graber@mcgill.ca"


# This notebook contains code to reproduce part of the results of [Graber et al. (2018)](http://arxiv.org/abs/1804.02706). Focusing on the theoretical predictions, this notebook includes the integration of the TOV equations for a specific crust model to obtain the mutual friction coefficients $\mathcal{B}$ as a function of radius $\tilde{r}$ and relative mass fraction $\Delta M /M$, and the numerical integration of a simple three-component neutron star toy model, providing the time evolution of the angular velocities $\Omega_{\rm sf}, \Omega_{\rm core}$ and $\Omega_{\rm crust}$.

# Comparison to the first pulse-to-pulse observation of a glitch in the Vela pulsar ([Palfreyman et al. 2018](https://www.nature.com/articles/s41586-018-0001-x)) can be found in an accompanying [notebook](https://github.com/vanessagraber/glitchrises/blob/master/data_comparison.ipynb).

# In[6]:


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.integrate import odeint
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.interpolate import InterpolatedUnivariateSpline
import matplotlib.animation as animation
from IPython.display import HTML
import sys
import os


# In[7]:


# %#matplotlib inline

# Set `usetex=False' if you do not have LaTeX installed.
rc("text", usetex=False)
rc("font", family="serif")
mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"


# In[8]:


if os.name == "nt":
    ffmpeg = os.path.join(
        os.path.dirname(sys.executable),
        os.path.join("Library", os.path.join("bin", "ffmpeg.exe")),
    )
else:
    ffmpeg = os.path.join(os.path.dirname(sys.executable), "ffmpeg")


# Physical constants in cgs units and conversion factors:

# In[9]:


from scipy.constants import pi, c, hbar, u, G, m_e, k, eV

hbar = hbar * 1e7
m = u * 1e3
m_e = m_e * 1e3
q = 4.80320425e-10
c = c * 1e2
k_B = k
kappa = hbar * pi / m
G = G * 1e3
Msun = 1.98855e33
mu = 0.5 * m
eV = eV * 1e7
fm = 1e-13
cm = 1e13


# ## Microscopic parameters
# ---

# Import the microscopic parameters from Negele and Vautherin (1973), Epstein and Baym (1992) and Donati and Pizzochero (2006).

# In[10]:


import pandas as pd
import os

os.chdir("/Users/nelachrapkova/Desktop/Relativistická astrofyzika/diplomova-prace/glitchrises-master")
df_EoS = pd.read_csv("microscopic_parameters.txt", sep="&", skiprows=5)

df_EoS.columns = ["n_b", "Z", "N", "x", "n_s", r"$\xi$", "E_p", "R_N", "E_l", "E_s"]
df_EoS.head()


# In[11]:


df_EoS[r"$\rho$ [$10^{12}$]"] = m * df_EoS["n_b"] * fm ** (-3) * 1e-12
df_EoS["A"] = np.round(df_EoS["Z"] * (1 + 1 / df_EoS["x"]))
df_EoS["R_WS"] = (3 * (df_EoS["N"] + df_EoS["Z"]) / (4 * pi * df_EoS["n_b"])) ** (1 / 3)
df_EoS["n_l [$10^{-6}$]"] = 3 / (4 * pi * df_EoS["R_WS"] ** 3) * 1e6
df_EoS["a"] = (2 / (df_EoS["n_l [$10^{-6}$]"] * 1e-6)) ** (1 / 3)
df_EoS["$\Delta$"] = (
    hbar**2
    * (3 * pi**2 * df_EoS["n_s"] * fm ** (-3)) ** (1 / 3)
    / (pi * m * df_EoS[r"$\xi$"] * fm * eV * 1e6)
)


# In[12]:


df_EoS[r"$\rho$ [$10^{12}$]"] = np.round(df_EoS[r"$\rho$ [$10^{12}$]"], 1)
df_EoS["R_WS"] = np.round(df_EoS["R_WS"], 1)
df_EoS["n_l [$10^{-6}$]"] = np.round(df_EoS["n_l [$10^{-6}$]"], 1)
df_EoS["a"] = np.round(df_EoS["a"], 1)
df_EoS["$\Delta$"] = np.round(df_EoS["$\Delta$"], 2)


# In[13]:


df_EoS


# ## TOV equation
# ---

# ### Equation of state

# For the inner crust above neutron drip we use the fitted EoS of Negele and Vautherin (1973) (see their equations on page 317). The baryon density has to be given in $\text{cm}^{-3}$ and the pressure is returned in ${\rm dyn}/{\rm cm}^2$.

# In[14]:


fit_params_EoS = np.array(
    [
        -4.0,
        2.8822899e-1,
        5.9150523e-1,
        9.0185940e-2,
        -1.1025614e-1,
        2.9377479e-2,
        -3.2618465e-3,
        1.3543555e-4,
    ]
)


# In[15]:


def sum_energy(n_b, c_i):
    """sum given by Negele and Vautherin to fit the energy, answer in MeV"""
    return np.sum(
        [c * np.log(n_b * 1e-35) ** (i - 1) for i, c in enumerate(c_i) if i > 0]
    )


def sum_pressure(n_b, c_i):
    """sum given by Negele and Vautherin to fit the pressure"""
    return np.sum(
        [
            c * (i - 1) * np.log(n_b * 1e-35) ** (i - 2)
            for i, c in enumerate(c_i)
            if i > 0
        ]
    )


def P_inner_crust(rho, c_i):
    """pressure in the inner crust given in dyne/cm^2"""
    return (
        (rho / m)
        * sum_pressure(rho / m, c_i)
        * np.exp(sum_energy(rho / m, c_i))
        * 1e6
        * eV
    )


# Below neutron drip the Negele and Vautherin EoS no longer applies and we take the pressure to be dominated by relativistic electrons with $Y_{\rm e} \approx 0.4$.

# In[16]:


rho_drip = 4e11


# In[17]:


def P_outer_crust(rho):
    """pressure in the outer crust given in dyne/cm^2"""
    Y_e = 0.4
    P_e = (
        hbar
        * c
        * (3 * pi**2 / m) ** (4 / 3)
        / (12 * pi**2)
        * (Y_e * rho) ** (4 / 3)
    )
    return P_e


# EoS and inverse EoS for the full crust:

# In[18]:


def pressure(rho, c_i, rho_drip):
    """pressure calculated as a function of density given in g/cm^3, answer in dyne/cm^2"""
    if rho < 0:
        return 0
    if rho >= rho_drip:
        return P_inner_crust(rho, c_i)
    else:
        return P_outer_crust(rho)


# In[19]:


def minimize(rho, P, c_i, rho_drip):
    """function to minimize in the inverse EoS"""
    return pressure(rho, c_i, rho_drip) - P


def density(P, c_i, rho_drip):
    """density calculated as a function of pressure given in dyne/cm^2, answer in g/cm^3"""
    if P < 0:
        return 0
    else:
        return brentq(minimize, 0, 2e14, args=(P, c_i, rho_drip), xtol=1e-6)


# ### Integration

# To obtain the stellar structure for a given EoS, the TOV equations have to be solved together with the continuity equation. We integrate the system for the crust only and consider the crust-core interface to be located at $n_{\rm b} \approx 0.08 \, \text{fm}^{-3}$ and the core to have $R_{\rm cci} \approx 10 \, \text{km}$ and $M_{\rm core} \approx 1.4 \, M_{\odot}$.

# In[20]:


R_cci = round(10 * 1e5)
M_coeff = 2.2
M_core = M_coeff * Msun
rho_cci = 0.08 * cm**3 * m
dr = 1e1
r = np.arange(R_cci, 2 * R_cci, dr)


# In[21]:


from scipy.integrate import odeint


def TOV(y, r, c_i, rho_drip):
    """TOV equation and continuity equation for pressure and mass"""
    P = y[0]
    M = y[1]

    # --- SAFETY CHECK: STOP INTEGRATION AT SURFACE ---
    # If Pressure is negative or zero, we are outside the star.
    # We return 0 derivatives so the solver 'coasts' through the vacuum
    # without crashing or doing excess work.
    if P <= 0:
        return [0.0, 0.0]

    # Calculate density only if P > 0
    try:
        rho = density(P, c_i, rho_drip)
    except (ValueError, RuntimeWarning):
        # Fallback if density function fails mathematically
        return [0.0, 0.0]

    dPdr = (
        -G
        * (rho + P / c**2)
        * (M + 4.0 * pi * r**3 * P / c**2)
        / (r * (r - 2.0 * G * M / c**2))
    )
    dMdr = 4.0 * pi * rho * r**2

    dydr = np.array([dPdr, dMdr])
    return dydr

def solve_TOV(radius, rho_cci, M_core, c_i, rho_drip):
    """function takes a radius array, crust-core interface density, core mass, EoS 
    fit parameters and neutron drip and solves the TOV equation"""

    # initial conditions
    P0 = pressure(rho_cci, c_i, rho_drip)
    M0 = M_core
    y0 = np.array([P0, M0])

    # integration
    # We can lower mxstep now because the 'Safety Check' makes the vacuum easy to solve
    y = odeint(TOV, y0, radius, args=(c_i, rho_drip), mxstep=500000)

    df = pd.DataFrame(y, columns=["Pressure", "Mass"])
    return df

# --- Execution ---
df_TOV = solve_TOV(r, rho_cci, M_core, fit_params_EoS, rho_drip)

# Apply density. We use a lambda to handle the negative pressures that 
# odeint might have left in the very last step before clamping.
df_TOV["Density"] = df_TOV["Pressure"].apply(
    lambda p: density(p, fit_params_EoS, rho_drip) if p > 0 else 0
)

# Filter out the vacuum (where Density is 0)
df_TOV = df_TOV[df_TOV["Density"] > 0]

# Assign radius (sliced to match the filtered dataframe length)
# This assumes 'r' is your input radius array defined elsewhere
df_TOV["Radius"] = r[: len(df_TOV)]


# Total neutron star mass in solar masses and radius in kilometers:

# In[22]:


M_NS, R_NS = df_TOV["Mass"].iloc[-1], df_TOV["Radius"].iloc[-1]
print(np.round(M_NS / Msun, 3), np.round(R_NS / 1e5, 3))


# Spline fits to interpolate between mass density and radius and vice versa. In the equatorial plane, the cylindrical radius and spherical radius coincide.

# In[23]:


fit_rho_radius = InterpolatedUnivariateSpline(
    np.flipud(df_TOV["Density"].values), np.flipud(df_TOV["Radius"]), k=1
)
fit_radius_rho = InterpolatedUnivariateSpline(
    df_TOV["Radius"].values, df_TOV["Density"].values, k=1
)


# To illustrate the mass dependence, calculate the overlying relative mass fraction and interpolate between density and $\Delta M /M$.

# In[24]:


rel_mass = (M_NS - df_TOV["Mass"].values) / M_NS
fit_rho_rel_mass = InterpolatedUnivariateSpline(
    np.flipud(df_TOV["Density"].values), np.flipud(rel_mass), k=1
)


# ## Mutual friction parameters
# ---

# We calculate the mutual friction coefficients via $\mathcal{B} = \mathcal{R} / (1 + \mathcal{R}^2)$ for the five inner crustal regions using the equations given in Section 2 of Graber et al. (2018), where $\Delta v$ is estimated via a mesoscopic force balance and we take $\delta \approx 10^{-2}$. We obtain three coupling profiles (A), (B) and (C) based on different microscopic assumptions.

# ### Microscopic vortex-nucleus velocities

# In[25]:


df_v_f = pd.DataFrame()
delta = 1e-2


# In[26]:


df_v_f["$\Delta v$ (A)"] = np.round(
    (df_EoS["E_s"] ** 2 + df_EoS["E_s"] * df_EoS["E_l"] + 0.5 * df_EoS["E_l"] ** 2)
    ** 0.5
    * delta
    * 1e6
    * eV
    / (df_EoS["R_N"] * fm * df_EoS["a"] * fm * m * df_EoS["n_s"] * fm ** (-3) * kappa)
)

df_v_f["$\Delta v$ (B)"] = np.round(
    abs(df_EoS["E_p"])
    * delta
    * 1e6
    * eV
    / (df_EoS["R_N"] * fm * df_EoS["a"] * fm * m * df_EoS["n_s"] * fm ** (-3) * kappa)
)

df_v_f["$\Delta v$ (C)"] = np.round(
    abs(df_EoS["E_p"])
    * delta
    * 1e6
    * eV
    / (
        df_EoS[r"$\xi$"]
        * fm
        * df_EoS["a"]
        * fm
        * m
        * df_EoS["n_s"]
        * fm ** (-3)
        * kappa
    )
)


# Relative velocities correspond to the following pinning forces per unit length:

# In[27]:


df_v_f["f (A) [$10^{15}$]"] = np.round(
    (
        (df_EoS["E_s"] ** 2 + df_EoS["E_s"] * df_EoS["E_l"] + 0.5 * df_EoS["E_l"] ** 2)
        ** 0.5
        * delta
        * 1e6
        * eV
        / (df_EoS["R_N"] * fm * df_EoS["a"] * fm)
    )
    * 1e-15,
    2,
)

df_v_f["f (B) [$10^{15}$]"] = np.round(
    (abs(df_EoS["E_p"]) * delta * 1e6 * eV / (df_EoS["R_N"] * fm * df_EoS["a"] * fm))
    * 1e-15,
    2,
)

df_v_f["f (C) [$10^{15}$]"] = np.round(
    (abs(df_EoS["E_p"]) * delta * 1e6 * eV / (df_EoS[r"$\xi$"] * fm * df_EoS["a"] * fm))
    * 1e-15,
    2,
)


# Corresponding pinning forces given by Seveso et al. (2016):

# In[28]:


df_v_f["f_S (L fixed) [$10^{15}$]"] = np.array([0.13, 0.29, 3.40, 2.35, 0.27])
df_v_f["f_S (L varied) [$10^{15}$]"] = np.array([0.32, 0.31, 8.55, 1.84, 0.06])


# In[29]:


df_v_f


# ### Mutual friction profiles

# In[30]:


R_A = (
    2.8
    * (mu / hbar) ** (1 / 2)
    * (
        (df_EoS["E_s"] ** 2 + df_EoS["E_s"] * df_EoS["E_l"] + 0.5 * df_EoS["E_l"] ** 2)
        ** 0.5
        * 1e6
        * eV
        * delta
        / (m * df_EoS["n_s"] * fm ** (-3) * kappa)
    )
    ** 0.5
    * df_EoS["R_N"]
    * fm
    / (df_EoS["a"] ** (3 / 2) * fm ** (3 / 2))
)


# In[31]:


R_B = (
    2.8
    * (mu / hbar) ** (1 / 2)
    * (abs(df_EoS["E_p"]) * 1e6 * eV * delta / (m * df_EoS["n_s"] * fm ** (-3) * kappa))
    ** 0.5
    * df_EoS["R_N"]
    * fm
    / (df_EoS["a"] ** (3 / 2) * fm ** (3 / 2))
)


# In[32]:


R_C = (
    1
    / (2 * pi ** (1 / 2))
    * (mu / hbar) ** (1 / 2)
    * (abs(df_EoS["E_p"]) * 1e6 * eV * delta / (m * df_EoS["n_s"] * fm ** (-3) * kappa))
    ** 0.5
    * (df_EoS["a"] * fm) ** 0.5
    / (df_EoS[r"$\xi$"] * fm)
)


# Fiducial microscopic parameters for domain III give

# In[33]:


R_EB_estimate = (
    2.8
    * (mu / hbar) ** (1 / 2)
    * (3 * 1e6 * eV * delta / (m * 2e-2 * fm ** (-3) * kappa)) ** 0.5
    * 7
    * fm
    / (56 ** (3 / 2) * fm ** (3 / 2))
)
print(np.round(R_EB_estimate, 4))


# In[34]:


R_J_estimate = (
    1
    / (2 * pi ** (1 / 2))
    * (mu / hbar) ** (1 / 2)
    * (3 * 1e6 * eV * delta / (m * 2e-2 * fm ** (-3) * kappa)) ** 0.5
    * (56 * fm) ** 0.5
    / (12 * fm)
)
print(np.round(R_J_estimate, 3))


# Their ratio is

# In[35]:


np.round((2 * pi ** (1 / 2)) ** (-1) / 2.8, 2)


# The mutual friction coefficients read

# In[36]:


label = [
    r"$\rho$",
    r"(A): $\mathcal{B}_{\rm EB}$ with $E_{\rm s,l}$",
    r"(B): $\mathcal{B}_{\rm EB}$ with $E_{\rm p}$",
    r"(C): $\mathcal{B}_{\rm J}$ with $E_{\rm p}$",
]
df_mf = pd.DataFrame(columns=label)


# In[37]:


df_mf[label[0]] = df_EoS[r"$\rho$ [$10^{12}$]"] * 1e12
df_mf[label[1]] = R_A / (1 + R_A**2)
df_mf[label[2]] = R_B / (1 + R_B**2)
df_mf[label[3]] = R_C / (1 + R_C**2)


# In[38]:


df_mf


# Using the TOV result, we calculate $\mathcal{B}$ as a function of radius $\tilde{r}$. We use a spline function to interpolate between the five crustal domains. The fit is performed in log-log space due to the change over several orders of magnitude. $\mathcal{B}$ is required inside the entire inner crust, between the neutron drip and the crust-core interface. Using the fit to extrapolate beyond the above regime works at higher densities but gives  unphysical results at lower densities ($\mathcal{B} = 0.5$ is the maximum value). For simplicity, we take the coefficients to be constant in the low-density region.

# In[39]:


def fit_radius_drag(r, label_index, rho_b):
    """fit for the mutual friction coefficient as a function of radius, while extrapolating the interpolation
    to the crust-core interface, we take B to be constant at lowest densities"""
    if r > fit_rho_radius(rho_b):
        return df_mf[label[label_index]].values[0]
    else:
        return 10 ** InterpolatedUnivariateSpline(
            np.log10(df_mf[label[0]]), (np.log10(df_mf[label[label_index]])), k=2
        )(np.log10(fit_radius_rho(r)))


# In[40]:


df_mf_fit = pd.DataFrame(columns=label)
x_radius = np.linspace(fit_rho_radius(rho_drip), R_cci, 1000)
df_mf_fit[label[0]] = fit_radius_rho(x_radius)

for i in range(1, 4):
    df_mf_fit[label[i]] = np.array(
        [fit_radius_drag(x, i, df_mf[r"$\rho$"][0]) for x in x_radius]
    )


# Mutual friction profiles as a function of density, relative mass fraction and radius:

# In[41]:


colors = ["#990099", "#0000CC", "#FFA319"]
linestyle = [[], [2, 1], [8, 1, 4, 1]]


# In[42]:


import matplotlib.pyplot as plt
import numpy as np
# Assuming 'mpl', 'df_mf', 'df_mf_fit', 'label', 'colors', 'linestyle' 
# 'R_cci', and 'M_coeff' are already defined.

fig = plt.figure()
fig.set_size_inches(10, 6)
mpl.rcParams.update({"font.size": 20})

# --- Main Plotting Loop ---
for i in range(1, 4):
    plt.loglog(
        df_mf[label[0]].values, df_mf[label[i]].values, ".", color=colors[i - 1], ms=12
    )
    plt.loglog(
        df_mf_fit[label[0]].values,
        df_mf_fit[label[i]].values,
        "-",
        linewidth="3",
        color=colors[i - 1],
        label=label[i],
        dashes=linestyle[i - 1],
    )

# --- Axes and Labels ---

# 1. Comment out explicit limits
# plt.xlim(4e11, 1.35e14)

# 2. Comment out explicit Y limits (from previous request)
# plt.ylim(1e-5, 1e-1) 

plt.xlabel(r"$\rho$ (g$\,$cm$^{-3}$)")
plt.ylabel(r"mutual friction coefficient")

ax = fig.get_axes()[0]

# 3. Comment out hardcoded ticks and labels so Matplotlib generates them automatically
# ax.set_xticks(np.array([5e11, 2e12, 1e13, 3e13, 1e14]))
# ax.set_xticklabels(
#     [
#         r"$5 \times 10^{11}$",
#         r"$2 \times 10^{12}$",
#         r"$1 \times 10^{13}$",
#         r"$3 \times 10^{13}$",
#         r"$1 \times 10^{14}$",
#     ]
# )

plt.legend(loc=3, handlelength=3, fontsize=20)

# --- Dynamic Title (Header) ---
plt.title(rf"$R_{{cci}} = {R_cci}$, $M_{{coeff}} = {M_coeff}$", pad=20)

# --- Save to File ---
filename = f"mutual_friction_Rcci_{R_cci}_M_coeff_{M_coeff}.png"

plt.savefig(filename, bbox_inches='tight', dpi=300)
print(f"Plot saved as: {filename}")

plt.show()


# In[43]:


import matplotlib.pyplot as plt
import numpy as np

# Assuming 'mpl', 'df_mf', 'df_mf_fit', 'label', 'colors', 'linestyle' 
# 'R_cci', 'M_coeff', and 'fit_rho_rel_mass' are defined.

fig = plt.figure()
fig.set_size_inches(10, 6)
mpl.rcParams.update({"font.size": 20})

# --- Main Plotting Loop ---
for i in range(1, 4):
    plt.semilogy(
        fit_rho_rel_mass(df_mf[label[0]].values),
        df_mf[label[i]].values,
        ".",
        color=colors[i - 1],
        ms=12,
    )
    plt.semilogy(
        fit_rho_rel_mass(df_mf_fit[label[0]].values),
        df_mf_fit[label[i]].values,
        "-",
        linewidth="3",
        color=colors[i - 1],
        label=label[i],
        dashes=linestyle[i - 1],
    )

# --- Axes and Labels ---
# 1. Comment out explicit X limits to make it automatic
plt.xlim(0.0, 0.07)

# 2. Y limits were already commented out
# plt.ylim(1e-5, 1e-1)

plt.ylabel(r"mutual friction coefficient")
plt.xlabel(r"$\Delta M / M$")

ax = fig.get_axes()[0]

# 3. Comment out hardcoded ticks and labels
# ax.set_xticks(np.arange(0.002, 0.009, 0.002))
# ax.set_xticklabels(np.round(np.arange(0.002, 0.009, 0.002), 3))

plt.legend(loc=3, handlelength=3, fontsize=20)

# --- Dynamic Title (Header) ---
plt.title(rf"$R_{{cci}} = {R_cci}$, $M_{{coeff}} = {M_coeff}$", pad=20)

# --- Save to File ---
filename = f"mutual_friction_rel_mass_Rcci_{R_cci}_Mcoeff_{M_coeff}.png"

plt.savefig(filename, bbox_inches='tight', dpi=300)
print(f"Plot saved as: {filename}")

plt.show()


# In[44]:


import matplotlib.pyplot as plt
import numpy as np

# Assuming 'mpl', 'df_mf', 'df_mf_fit', 'label', 'colors', 'linestyle' 
# 'R_cci', 'M_coeff', and 'fit_rho_radius' are defined.

fig = plt.figure()
fig.set_size_inches(10, 6)
mpl.rcParams.update({"font.size": 20})

# --- Main Plotting Loop ---
for i in range(1, 4):
    plt.semilogy(
        fit_rho_radius(df_mf[label[0]].values),
        df_mf[label[i]].values,
        ".",
        color=colors[i - 1],
        ms=12,
    )
    plt.semilogy(
        fit_rho_radius(df_mf_fit[label[0]].values),
        df_mf_fit[label[i]].values,
        "-",
        linewidth="3",
        color=colors[i - 1],
        label=label[i],
        dashes=linestyle[i - 1],
    )

# --- Axes and Labels ---

# 1. Comment out explicit X limits
plt.xlim(1e6, 1.6e6)

# Optional: If you want the X-axis reversed (high to low) automatically:
# plt.gca().invert_xaxis()

# 2. Y limits were already commented out
# plt.ylim(1e-5, 1e-1)

plt.xlabel(r"$r$ (km)")
plt.ylabel(r"mutual friction coefficient")

ax = fig.get_axes()[0]

# 3. Comment out hardcoded ticks and labels
# ax.set_xticks(np.arange(1e6, 1.05e6, 1e4))
# ax.set_xticklabels(np.round(np.arange(1e6, 1.05e6, 1e4) * 1e-5, 1))

plt.legend(loc=3, handlelength=3, fontsize=20)

# --- Dynamic Title (Header) ---
plt.title(rf"$R_{{cci}} = {R_cci}$, $M_{{coeff}} = {M_coeff}$", pad=20)

# --- Save to File ---
filename = f"mutual_friction_radius_Rcci_{R_cci}_Mcoeff_{M_coeff}.png"

plt.savefig(filename, bbox_inches='tight', dpi=300)
print(f"Plot saved as: {filename}")

plt.show()


# ## Rotational evolution
# ---

# To study the rotational evolution, we consider a three-component neutron star toy model. For simplicity, we work in the equatorial plane ($\tilde{r} = r$), integrate the equations of motion in 1D and rescale the results to match the total crustal moment of inertia in cylindrical coordinates with that in spherical geometry.

# ### Moment of inertias

# Total neutron star moment of inertia (see Lattimer & Prakash (2001)):

# In[45]:


I_total = 0.35 * M_NS * R_NS**2
print(I_total)


# Moment of inertia of the full crust (from crust-core interface to surface) in spherical coordinates:

# In[46]:


def integrand_spherical(r):
    """integrand of the moment of inertia integral in spherical coordinates"""
    return 8 * pi / 3 * fit_radius_rho(r) * r**4


I_crust_total = quad(integrand_spherical, R_cci, R_NS)[0]
print(I_crust_total, I_crust_total / I_total)


# Moment of inertia per unit height of the full crust (from crust-core interface to surface) in cylindrical coordinates:

# In[47]:


def integrand_cylindrical(r):
    """integrand of the moment of inertia integral in cylindrical coordinates"""
    return 2 * pi * fit_radius_rho(r) * r**3


I_crust_total_unitheight = quad(integrand_cylindrical, R_cci, R_NS)[0]
print(I_crust_total_unitheight)


# Match the two to obtain height ($2h$) of cylinder

# In[48]:


h = I_crust_total / (2 * I_crust_total_unitheight)
print(np.round(h))


# Moments of inertia of the neutron superfluid (95%) and charged particles (5%) in the neutron star core:

# In[49]:


I_core = 0.95 * (I_total - I_crust_total)
I_protons = 0.05 * (I_total - I_crust_total)
print(I_core, I_protons, I_core / I_total, I_protons / I_total)


# Moment of inertia of the crustal superfluid (integration between crust-core interface and neutron drip):

# In[50]:


R_drip = fit_rho_radius(rho_drip)
print(np.round(R_drip / 1e5, 3))


def integrand_I(r, h):
    """integrand of the moment of inertia integral in cylindrical oordinates"""
    return 4 * pi * h * fit_radius_rho(r) * r**3


I_sf = quad(integrand_I, R_cci, R_drip, args=(h,))[0]
print(I_sf, I_sf / I_total)


# Moment of inertia of the outer crust plus the charged particle conglomerate in the core (dominated by the core protons):

# In[51]:


I_crust = I_total - I_core - I_sf
print(I_crust, I_crust / I_total)


# ### Differential equations

# We illustrate how the density-dependent mutual friction parameters affect the glitch rise by numerically integrating the differential equations given in Section 4 of Graber et al. (2018). We take crust and core to rotate rigidly and coupled via a constant mutual friction coefficient (we choose two fiducial values ${\cal B}_{\rm core} \approx 5 \times 10^{-5}$ and ${\cal B}_{\rm core} \approx 10^{-2}$ associated with electron-vortex scattering and vortex-fluxtube coupling, respectively), while the superfluid can exhibit differential rotation.

# In[52]:


def equations_of_motion(
    y, t, x, n, dx, h, rho_x, R_drip, I_crust, I_core, B_crust, B_core
):
    """equations of motion of the three-component toy model"""

    # array of n sf values (due to differential rotation) plus one core and one crust value
    omega_sf = y[:-2]
    omega_core = y[-2]
    omega_crust = y[-1]

    # r-derivative, which we take to vanish at the boundaries
    domdr = (omega_sf[2:] - omega_sf[:-2]) / (2 * dx)
    domdr = np.append(0.0, domdr)
    domdr = np.append(domdr, 0.0)

    # change in superfluid and core angular velocities
    del_om_sf = B_crust * (2 * omega_sf + x * domdr) * (omega_crust - omega_sf)
    del_om_core = 2 * omega_core * B_core * (omega_crust - omega_core)

    # ignoring the external torque, crustal component is initially in equilibrium
    del_om_crust = 0.0
    for rho_n, do_n, x_n in zip(rho_x, del_om_sf, x):
        del_om_crust -= rho_n * do_n * x_n**3 * dx

    del_om_crust = del_om_crust * 4 * pi * h * R_drip**4 / I_crust

    del_om_crust -= I_core * del_om_core / I_crust

    derivatives = np.append(del_om_sf, del_om_core)
    derivatives = np.append(derivatives, del_om_crust)

    return derivatives


# In[53]:


def time_step(
    t,
    omega_sf,
    omega_core,
    omega_crust,
    x,
    n,
    dx,
    h,
    rho_x,
    R_drip,
    I_crust,
    I_core,
    B_crust,
    B_core,
):
    """integrating the equations of motion in time"""
    # initial conditions
    y0 = np.append(omega_sf, omega_core)
    y0 = np.append(y0, omega_crust)
    # integration
    y = odeint(
        equations_of_motion,
        y0,
        t,
        args=(x, n, dx, h, rho_x, R_drip, I_crust, I_core, B_crust, B_core),
        mxstep=500000,
        rtol=1e-10,
    )
    return y


# Setting up the 1D spatial grid between crust-core interface and neutron drip:

# In[54]:


n = 100
dx = (1 - R_cci / R_drip) / n
x = R_cci / R_drip + np.arange(n + 1) * dx
rho_x = fit_radius_rho(x * R_drip)


# Initial values typical for the Vela pulsar (core and crust are in equilibrium, whereas superfluid has a (positive) initial lag everywhere in the inner crust) are used. Specifically, we choose a critical lag that reproduces the glitch step size recorded by Palfreyman et al. (2018).

# In[55]:


om_Vela = 2 * pi * 11.195
omega_core_0 = 1 * om_Vela
omega_crust_0 = 1 * om_Vela


# In[56]:


om_crit = 6.3e-3
rel_om_crit = om_crit / omega_crust_0
omega_sf_0 = omega_crust_0 * np.ones(n + 1) * (1 + rel_om_crit)


# In[57]:


print(np.round(omega_core_0, 4), np.round(omega_sf_0[0], 4), rel_om_crit)


# Grid for time integration (stop at $120 \, {\rm s}$ to include current limit on glitch rise)

# In[58]:


t_end = 120
n_t = 20000
dt = t_end / n_t
t = np.arange(0, t_end + dt, dt)


# ### Mutual friction profiles

# $\mathcal{B}$ as a function of grid points $x$ and four constant coefficient $\mathcal{B} = 10^{-1}, 10^{-2}, 10^{-3}, 10^{-4}$ for comparison:

# In[59]:


label_x = [
    r"$x$",
    r"$10^{-1}$",
    r"$10^{-2}$",
    r"$10^{-3}$",
    r"$10^{-4}$",
    r"(A)",
    r"(B)",
    r"(C)",
]


# In[60]:


df_mf_x = pd.DataFrame(columns=label_x)
df_mf_x[label_x[0]] = x

for i in range(1, 5):
    df_mf_x[label_x[i]] = 10 ** (-i) * np.ones(n + 1)

for i in range(5, 8):
    df_mf_x[label_x[i]] = np.array(
        [fit_radius_drag(x0 * R_drip, i - 4, df_mf[r"$\rho$"][0]) for x0 in x]
    )


# ## Weak core mutual friction
# ---

# Integrating the equations of motion for the weak fiducial core mutual friction, i.e. ${\cal B}_{\rm core} \approx 5 \times 10^{-5}$ due to the scattering of electrons off the vortex magnetic field, and seven crustal friction profiles:

# In[61]:


B_core_weak = 5e-5


# In[62]:


integration_results_weak = {}

for i in range(7):
    integration_results_weak[i] = time_step(
        t,
        omega_sf_0,
        omega_core_0,
        omega_crust_0,
        x,
        n,
        dx,
        h,
        rho_x,
        R_drip,
        I_crust,
        I_core,
        df_mf_x[label_x[i + 1]],
        B_core_weak,
    )


# ### Superfluid differential rotation

# $\Omega_{\rm sf}$ as a function of position for several time steps for drag profile (A) (i.e. $\mathcal{B}_{\rm EB}$ calculated with $E_{\rm s,l}$) and (C) (i.e. $\mathcal{B}_{\rm J}$ calculated with $E_{\rm p}$:

# In[63]:


drag_A = 4
drag_C = 6


# In[64]:


imagelist_sf_weak_A = [
    (integration_results_weak[drag_A][i][:-2]) for i in range(len(t))
]
t_select_A = np.array([0, 50, 300, 1000, 3000, 8000])
t_passed_A = np.round(t_select_A * dt, 3)
print(t_passed_A)


# In[65]:


fig = plt.figure()
fig.set_size_inches(10, 6)
mpl.rcParams.update({"font.size": 20})

plt.plot(
    x * R_drip * 1e-5,
    imagelist_sf_weak_A[t_select_A[0]],
    "-",
    linewidth=3,
    color="#990099",
)
plt.plot(
    x * R_drip * 1e-5,
    imagelist_sf_weak_A[t_select_A[1]],
    ":",
    linewidth=2,
    color="black",
)
plt.plot(
    x * R_drip * 1e-5,
    imagelist_sf_weak_A[t_select_A[2]],
    ":",
    linewidth=2,
    color="black",
)
plt.plot(
    x * R_drip * 1e-5,
    imagelist_sf_weak_A[t_select_A[3]],
    ":",
    linewidth=2,
    color="black",
)
plt.plot(
    x * R_drip * 1e-5,
    imagelist_sf_weak_A[t_select_A[4]],
    ":",
    linewidth=2,
    color="black",
)
plt.plot(
    x * R_drip * 1e-5,
    imagelist_sf_weak_A[t_select_A[5]],
    "-",
    linewidth=3,
    color="#0000CC",
)

plt.xlim(R_drip * 1e-5, R_cci * 1e-5)
plt.ylim(70.3395, 70.3475)
plt.xlabel(r"$\tilde{r}$ (km)")
plt.ylabel(r"$\Omega_{\rm sf}(t,\tilde{r})$  $({\rm rad}/{\rm s})$")
ax = fig.get_axes()[0]
ax.set_xticks(np.arange(R_cci * 1e-5, R_drip * 1e-5, 0.1))
ax.set_xticklabels(np.round(np.arange(R_cci * 1e-5, R_drip * 1e-5, 0.1), 1))
ax.set_yticks(np.arange(70.340, 70.348, 0.001))
ax.set_yticklabels(np.round(np.arange(70.340, 70.348, 0.001), 3))

plt.annotate(r"$t = 0 \, {\rm s}$", xy=(10.41, 70.3468), fontsize=16, color="#990099")
plt.annotate(r"$t = 300 \, {\rm ms}$", xy=(10.37, 70.3455), fontsize=16)
plt.annotate(r"$t = 1.8 \, {\rm s}$", xy=(10.27, 70.34418), fontsize=16)
plt.annotate(r"$t = 6 \, {\rm s}$", xy=(10.15, 70.34235), fontsize=16)
plt.annotate(r"$t = 18 \, {\rm s}$", xy=(10.05, 70.34103), fontsize=16)
plt.annotate(r"$t = 48 \, {\rm s}$", xy=(10.41, 70.3398), fontsize=16, color="#0000CC")
plt.annotate(
    r"${\cal B}_{\rm core} \approx 5 \times 10^{-5}$",
    xycoords="axes fraction",
    xy=(0.72, 1 - 0.09),
    fontsize=22,
)
plt.annotate(r"profile (A)", xycoords="axes fraction", xy=(0.825, 0.035), fontsize=22)

plt.show()


# In[66]:


imagelist_sf_weak_C = [
    (integration_results_weak[drag_C][i][:-2]) for i in range(len(t))
]
t_select_C = np.array([0, 20, 100, 500, 5000, 20000])
t_passed_C = np.round(t_select_C * dt, 3)
print(t_passed_C)


# In[67]:


fig = plt.figure()
fig.set_size_inches(10, 6)
mpl.rcParams.update({"font.size": 20})

plt.plot(
    x * R_drip * 1e-5,
    imagelist_sf_weak_C[t_select_C[0]],
    "-",
    linewidth=3,
    color="#990099",
)
plt.plot(
    x * R_drip * 1e-5,
    imagelist_sf_weak_C[t_select_C[1]],
    ":",
    linewidth=2,
    color="black",
)
plt.plot(
    x * R_drip * 1e-5,
    imagelist_sf_weak_C[t_select_C[2]],
    ":",
    linewidth=2,
    color="black",
)
plt.plot(
    x * R_drip * 1e-5,
    imagelist_sf_weak_C[t_select_C[3]],
    ":",
    linewidth=2,
    color="black",
)
plt.plot(
    x * R_drip * 1e-5,
    imagelist_sf_weak_C[t_select_C[4]],
    ":",
    linewidth=2,
    color="black",
)
plt.plot(
    x * R_drip * 1e-5,
    imagelist_sf_weak_C[t_select_C[5]],
    "-",
    linewidth=3,
    color="#0000CC",
)

plt.xlim(R_drip * 1e-5, R_cci * 1e-5)
plt.ylim(70.3395, 70.3475)
plt.xlabel(r"$\tilde{r}$ (km)")
plt.ylabel(r"$\Omega_{\rm sf}(t,\tilde{r})$  $({\rm rad}/{\rm s})$")
ax = fig.get_axes()[0]
ax.set_xticks(np.arange(R_cci * 1e-5, R_drip * 1e-5, 0.1))
ax.set_xticklabels(np.round(np.arange(R_cci * 1e-5, R_drip * 1e-5, 0.1), 1))
ax.set_yticks(np.arange(70.340, 70.348, 0.001))
ax.set_yticklabels(np.round(np.arange(70.340, 70.348, 0.001), 3))

plt.annotate(r"$t = 0 \, {\rm s}$", xy=(10.41, 70.3468), fontsize=16, color="#990099")
plt.annotate(r"$t = 120 \, {\rm ms}$", xy=(10.38, 70.34575), fontsize=16)
plt.annotate(r"$t = 600 \, {\rm ms}$", xy=(10.37, 70.3431), fontsize=16)
plt.annotate(r"$t = 3 \, {\rm s}$", xy=(10.25, 70.3412), fontsize=16)
plt.annotate(r"$t = 30 \, {\rm s}$", xy=(10.15, 70.34075), fontsize=16)
plt.annotate(r"$t = 120 \, {\rm s}$", xy=(10.41, 70.3398), fontsize=16, color="#0000CC")
plt.annotate(
    r"${\cal B}_{\rm core} \approx 5 \times 10^{-5}$",
    xycoords="axes fraction",
    xy=(0.72, 1 - 0.09),
    fontsize=22,
)
plt.annotate(r"profile (C)", xycoords="axes fraction", xy=(0.825, 0.035), fontsize=22)

plt.show()


# Animation of the time evolution of the superfluid component and the non-superfluid 'crust' component:

# In[68]:


imagelist_crust_weak_A = np.array(
    [
        integration_results_weak[drag_A][:, -1][i] * np.ones(len(x))
        for i in range(len(t))
    ]
)


# In[69]:


Writer = animation.writers["ffmpeg"]
writer = Writer(fps=10, metadata=dict(artist="Graber et al. (2018)"))


# In[70]:


fig, ax = plt.subplots()
fig.set_size_inches(10, 6)
mpl.rcParams.update({"font.size": 20})
plt.title(r"Time evolution for {}".format(label_x[6]), fontsize=22)

plt.plot(
    x * R_drip * 1e-5,
    imagelist_sf_weak_A[t_select_A[0]],
    ":",
    linewidth=3,
    color="black",
)
plt.plot(
    x * R_drip * 1e-5,
    imagelist_sf_weak_A[t_select_A[5]],
    "-",
    linewidth=3,
    color=colors[1],
)

ax = fig.get_axes()[0]

(line1,) = ax.plot(
    x * R_drip * 1e-5,
    imagelist_sf_weak_A[0],
    "-",
    linewidth=3,
    color=colors[0],
    label=r"$\Omega_{\rm sf}$",
)
(line2,) = ax.plot(
    x * R_drip * 1e-5,
    imagelist_crust_weak_A[0],
    "-",
    linewidth=3,
    color=colors[2],
    label=r"$\Omega_{\rm crust}$",
)

plt.xlim(R_drip * 1e-5, R_cci * 1e-5)
plt.ylim(70.3395, 70.3475)
plt.xlabel(r"$r$ (km)")
plt.ylabel(r"$\Omega(t,\tilde{r})$  $({\rm rad}/{\rm s})$")
ax = fig.get_axes()[0]
ax.set_xticks(np.arange(R_cci * 1e-5, R_drip * 1e-5, 0.1))
ax.set_xticklabels(np.round(np.arange(R_cci * 1e-5, R_drip * 1e-5, 0.1), 1))
ax.set_yticks(np.arange(70.340, 70.348, 0.001))
ax.set_yticklabels(np.round(np.arange(70.340, 70.348, 0.001), 3))

# update the data


def animate(i):
    line1.set_ydata(imagelist_sf_weak_A[i])
    line2.set_ydata(imagelist_crust_weak_A[i])
    return line1, line2


plt.legend(loc=1, handlelength=1.5, fontsize=20)

ani = animation.FuncAnimation(fig, animate, np.arange(0, 5000, 2))


# In[71]:


#HTML(ani.to_html5_video())


# ### Crustal evolution and phase shift

# Change in crustal frequency over time, the equilibrium value expected from angular momentum conservation and fractional glitch step-size calculated for comparison with Palfreyman et al. (2018):

# In[72]:


df_delomega_crust_weak = pd.DataFrame(columns=label_x[1:])

for i in range(7):
    df_delomega_crust_weak[label_x[i + 1]] = (
        integration_results_weak[i][:, -1] - omega_crust_0
    ) / (2 * pi)


# In[73]:


del_nu_equilibrium = omega_crust_0 / (2 * pi) * rel_om_crit * I_sf / (I_total) * 1e6
print(np.round(del_nu_equilibrium, 2))


# In[74]:


frac_nu = del_nu_equilibrium * 1e-6 * 2 * pi / omega_crust_0
print(np.round(frac_nu, 9))


# In[75]:


fig = plt.figure()
fig.set_size_inches(10, 6)
mpl.rcParams.update({"font.size": 20})

plt.plot(
    t,
    1e6 * df_delomega_crust_weak[label_x[1]],
    ":",
    label=r"constant $\mathcal{B}$",
    linewidth=2,
    color="black",
)

for i in range(1, 4):
    plt.plot(
        t,
        1e6 * df_delomega_crust_weak[label_x[i + 1]],
        ":",
        label="_nolegend_",
        linewidth=2,
        color="black",
    )

for i in range(4, 7):
    plt.plot(
        t,
        1e6 * df_delomega_crust_weak[label_x[i + 1]],
        "-",
        label=label_x[i + 1],
        linewidth=3,
        color=colors[i - 4],
    )

plt.plot([-10, 100], [del_nu_equilibrium, del_nu_equilibrium], "black", alpha=0.3)

plt.xlim(-2, 60)
plt.ylim(1, 5e2)
plt.xlabel(r"t (s)")
plt.ylabel(r"$\Delta\nu\ (\mu{\rm Hz})$")
ax = fig.get_axes()[0]
ax.set_yscale("log")

plt.annotate(r"$10^{-1}$", xy=(-0.5, 2.8e2), fontsize=16)
plt.annotate(r"$10^{-2}$", xy=(6, 1.6e2), fontsize=16)
plt.annotate(r"$10^{-3}$", xy=(30, 4.3e1), fontsize=16)
plt.annotate(r"$10^{-4}$", xy=(3, 0.7e1), fontsize=16)
plt.annotate(r"equilibrium", xy=(15, 1.2e1), fontsize=10, alpha=0.5)
plt.annotate(
    r"${\cal B}_{\rm core} \approx 5 \times 10^{-5}$",
    xycoords="axes fraction",
    xy=(0.72, 0.02),
    fontsize=22,
)

plt.legend(loc=1, handlelength=1.5, fontsize=20)

plt.show()


# Illustrate crust-core coupling via phase shift accumulated after a glitch:

# In[76]:


def phi(delta_nu, dt):
    """phase shift accumulated after a glitch as function of time"""

    phase = np.array([])
    for i in range(len(delta_nu)):
        phase = np.append(phase, np.sum(dt * delta_nu[: i + 1]))
    return phase


# In[77]:


df_phi_weak = pd.DataFrame(columns=label_x[1:])

for i in range(7):
    df_phi_weak[label_x[i + 1]] = phi(df_delomega_crust_weak[label_x[i + 1]], dt)


# In[78]:


fig = plt.figure()
fig.set_size_inches(10, 6)
mpl.rcParams.update({"font.size": 20})

plt.plot(
    t,
    df_phi_weak[label_x[1]],
    ":",
    label=r"constant $\mathcal{B}$",
    linewidth=2,
    color="black",
)

for i in range(1, 4):
    plt.plot(
        t,
        df_phi_weak[label_x[i + 1]],
        ":",
        label="_nolegend_",
        linewidth=2,
        color="black",
    )

for i in range(4, 7):
    plt.plot(
        t,
        df_phi_weak[label_x[i + 1]],
        "-",
        label=label_x[i + 1],
        linewidth=3,
        color=colors[i - 4],
    )

plt.xlim(-1, 60)
plt.ylim(-0.0001, 0.0036)
plt.xlabel(r"t (s)")
plt.ylabel(r"$\phi$")

plt.annotate(r"$10^{-1}$", xy=(9, 1.78e-3), rotation=0, fontsize=16)
plt.annotate(r"$10^{-2}$", xy=(10, 1.3e-3), rotation=0, fontsize=16)
plt.annotate(r"$10^{-3}$", xy=(25, 1.86e-3), rotation=0, fontsize=16)
plt.annotate(r"$10^{-4}$", xy=(25, 0.67e-3), rotation=0, fontsize=16)
plt.annotate(
    r"${\cal B}_{\rm core} \approx 5 \times 10^{-5}$",
    xycoords="axes fraction",
    xy=(0.72, 0.02),
    fontsize=22,
)

plt.legend(loc=2, handlelength=1.5, fontsize=20)

plt.show()


# Crust-core coupling timescales for different core mutual friction strengths:

# In[79]:


print(
    np.round((I_crust / I_core) * 1 / (2 * omega_core_0 * 1e-2), 4),
    np.round((I_crust / I_core) * 1 / (2 * omega_core_0 * 5e-5), 1),
)


# ## Strong core mutual friction
# ---

# We repeat the above calculations to obtain glitch rise predictions for ${\cal B}_{\rm core} \approx 10^{-2}$ associated with Kelvin wave excitations as a result of vortex-fluxtube coupling. This estimate can be obtained by using typical values for the core interaction parameters, i.e. the interaction length scale is the penetration depth and the distance between pinning sites the interfluxtube spacing (see e.g. Haskell et al. (2014)).

# In[80]:


f_core = np.round(5 * 1e6 * eV / (100 * fm * 1e3 * fm) * 1e-15)
delta_v_core = np.round(5 * 1e6 * eV / (1e14 * kappa * 100 * fm * 1e3 * fm) * 1e-4)
print(f_core, delta_v_core)


# In[81]:


R_core = np.round(
    (0.5 * mu / hbar) ** 0.5
    * (5 * 1e6 * eV / (1e14 * kappa)) ** 2
    * 1
    / ((4e4) ** 1.5 * (100 * fm) ** 1.5 * 1e6 * fm**2),
    2,
)
print(R_core)


# In[82]:


B_core_strong = 1e-2


# In[83]:


integration_results_strong = {}

for i in range(7):
    integration_results_strong[i] = time_step(
        t,
        omega_sf_0,
        omega_core_0,
        omega_crust_0,
        x,
        n,
        dx,
        h,
        rho_x,
        R_drip,
        I_crust,
        I_core,
        df_mf_x[label_x[i + 1]],
        B_core_strong,
    )


# ### Superfluid differential rotation

# $\Omega_{\rm sf}$ as a function of position for several time steps for drag models (A) and (C):

# In[84]:


imagelist_sf_strong_A = [
    (integration_results_strong[drag_A][i][:-2]) for i in range(len(t))
]
imagelist_sf_strong_C = [
    (integration_results_strong[drag_C][i][:-2]) for i in range(len(t))
]


# In[85]:


fig = plt.figure()
fig.set_size_inches(10, 6)
mpl.rcParams.update({"font.size": 20})

plt.plot(
    x * R_drip * 1e-5,
    imagelist_sf_strong_A[t_select_A[0]],
    "-",
    linewidth=3,
    color=colors[0],
)
plt.plot(
    x * R_drip * 1e-5,
    imagelist_sf_strong_A[t_select_A[1]],
    ":",
    linewidth=2,
    color="black",
)
plt.plot(
    x * R_drip * 1e-5,
    imagelist_sf_strong_A[t_select_A[2]],
    ":",
    linewidth=2,
    color="black",
)
plt.plot(
    x * R_drip * 1e-5,
    imagelist_sf_strong_A[t_select_A[3]],
    ":",
    linewidth=2,
    color="black",
)
plt.plot(
    x * R_drip * 1e-5,
    imagelist_sf_strong_A[t_select_A[4]],
    "-",
    linewidth=3,
    color=colors[1],
)

plt.xlim(R_drip * 1e-5, R_cci * 1e-5)
plt.ylim(70.3395, 70.3475)
plt.xlabel(r"$\tilde{r}$ (km)")
plt.ylabel(r"$\Omega_{\rm sf}(t,\tilde{r})$  $({\rm rad}/{\rm s})$")
ax = fig.get_axes()[0]
ax.set_xticks(np.arange(R_cci * 1e-5, R_drip * 1e-5, 0.1))
ax.set_xticklabels(np.round(np.arange(R_cci * 1e-5, R_drip * 1e-5, 0.1), 1))
ax.set_yticks(np.arange(70.340, 70.348, 0.001))
ax.set_yticklabels(np.round(np.arange(70.340, 70.348, 0.001), 3))

plt.annotate(r"$t = 0 \, {\rm s}$", xy=(10.41, 70.3468), fontsize=16, color="#990099")
plt.annotate(r"$t = 300 \, {\rm ms}$", xy=(10.37, 70.3455), fontsize=16)
plt.annotate(r"$t = 1.8 \, {\rm s}$", xy=(10.27, 70.34397), fontsize=16)
plt.annotate(r"$t = 6 \, {\rm s}$", xy=(10.13, 70.34168), fontsize=16)
plt.annotate(r"$t = 18 \, {\rm s}$", xy=(10.41, 70.33985), fontsize=16, color="#0000CC")
plt.annotate(
    r"${\cal B}_{\rm core} \approx 10^{-2}$",
    xycoords="axes fraction",
    xy=(0.79, 1 - 0.09),
    fontsize=22,
)
plt.annotate(r"profile (A)", xycoords="axes fraction", xy=(0.825, 0.035), fontsize=22)

plt.show()


# In[86]:


fig = plt.figure()
fig.set_size_inches(10, 6)
mpl.rcParams.update({"font.size": 20})

plt.plot(
    x * R_drip * 1e-5,
    imagelist_sf_strong_C[t_select_C[0]],
    "-",
    linewidth=3,
    color="#990099",
)
plt.plot(
    x * R_drip * 1e-5,
    imagelist_sf_strong_C[t_select_C[1]],
    ":",
    linewidth=2,
    color="black",
)
plt.plot(
    x * R_drip * 1e-5,
    imagelist_sf_strong_C[t_select_C[2]],
    ":",
    linewidth=2,
    color="black",
)
plt.plot(
    x * R_drip * 1e-5,
    imagelist_sf_strong_C[t_select_C[3]],
    ":",
    linewidth=2,
    color="black",
)
plt.plot(
    x * R_drip * 1e-5,
    imagelist_sf_strong_C[t_select_C[4]],
    ":",
    linewidth=2,
    color="black",
)
plt.plot(
    x * R_drip * 1e-5,
    imagelist_sf_strong_C[t_select_C[5]],
    "-",
    linewidth=3,
    color="#0000CC",
)

plt.xlim(R_drip * 1e-5, R_cci * 1e-5)
plt.ylim(70.3395, 70.3475)
plt.xlabel(r"$\tilde{r}$ (km)")
plt.ylabel(r"$\Omega_{\rm sf}(t,\tilde{r})$  $({\rm rad}/{\rm s})$")
ax = fig.get_axes()[0]
ax.set_xticks(np.arange(R_cci * 1e-5, R_drip * 1e-5, 0.1))
ax.set_xticklabels(np.round(np.arange(R_cci * 1e-5, R_drip * 1e-5, 0.1), 1))
ax.set_yticks(np.arange(70.340, 70.348, 0.001))
ax.set_yticklabels(np.round(np.arange(70.340, 70.348, 0.001), 3))

plt.annotate(r"$t = 0 \, {\rm s}$", xy=(10.41, 70.3468), fontsize=16, color="#990099")
plt.annotate(r"$t = 120 \, {\rm ms}$", xy=(10.38, 70.34575), fontsize=16)
plt.annotate(r"$t = 600 \, {\rm ms}$", xy=(10.37, 70.34295), fontsize=16)
plt.annotate(r"$t = 3 \, {\rm s}$", xy=(10.22, 70.3407), fontsize=16)
plt.annotate(r"$t = 30 \, {\rm s}$", xy=(10.13, 70.3407), fontsize=16)
plt.annotate(r"$t = 120 \, {\rm s}$", xy=(10.41, 70.3398), fontsize=16, color="#0000CC")
plt.annotate(
    r"${\cal B}_{\rm core} \approx 10^{-2}$",
    xycoords="axes fraction",
    xy=(0.79, 1 - 0.09),
    fontsize=22,
)
plt.annotate(r"profile (C)", xycoords="axes fraction", xy=(0.825, 0.035), fontsize=22)

plt.show()


# ### Crustal evolution and phase shift

# Change in crustal frequency over time. Glitch step size remains the same.

# In[87]:


df_delomega_crust_strong = pd.DataFrame(columns=label_x[1:])

for i in range(7):
    df_delomega_crust_strong[label_x[i + 1]] = (
        integration_results_strong[i][:, -1] - omega_crust_0
    ) / (2 * pi)


# In[88]:


fig = plt.figure()
fig.set_size_inches(10, 6)
mpl.rcParams.update({"font.size": 20})

plt.plot(
    t,
    1e6 * df_delomega_crust_strong[label_x[1]],
    ":",
    label=r"constant $\mathcal{B}$",
    linewidth=2,
    color="black",
)

for i in range(1, 4):
    plt.plot(
        t,
        1e6 * df_delomega_crust_strong[label_x[i + 1]],
        ":",
        label="_nolegend_",
        linewidth=2,
        color="black",
    )

for i in range(4, 7):
    plt.plot(
        t,
        1e6 * df_delomega_crust_strong[label_x[i + 1]],
        "-",
        label=label_x[i + 1],
        linewidth=3,
        color=colors[i - 4],
    )

plt.plot([-10, 100], [del_nu_equilibrium, del_nu_equilibrium], "black", alpha=0.3)

plt.xlim(-0.2, 6)
plt.ylim(1e-1, 1.2e2)
plt.xlabel(r"t (s)")
plt.ylabel(r"$\Delta\nu\ (\mu{\rm Hz})$")
ax = fig.get_axes()[0]
ax.set_yscale("log")

plt.annotate(r"$10^{-1}$", xy=(0.2, 0.65e2), fontsize=16)
plt.annotate(r"$10^{-2}$", xy=(0.15, 1e1), fontsize=16)
plt.annotate(r"$10^{-3}$", xy=(0.9, 0.22e1), fontsize=16)
plt.annotate(r"$10^{-4}$", xy=(2, 4e-1), fontsize=16)
plt.annotate(
    r"${\cal B}_{\rm core} \approx 10^{-2}$",
    xycoords="axes fraction",
    xy=(0.79, 0.02),
    fontsize=22,
)

plt.legend(loc=1, handlelength=1.5, fontsize=20)

plt.show()


# Phase shift for the stronger core friction (break shifted to the left)

# In[89]:


df_phi_strong = pd.DataFrame(columns=label_x[1:])

for i in range(7):
    df_phi_strong[label_x[i + 1]] = phi(df_delomega_crust_strong[label_x[i + 1]], dt)


# In[90]:


fig = plt.figure()
fig.set_size_inches(10, 6)
mpl.rcParams.update({"font.size": 20})

plt.plot(
    t,
    df_phi_strong[label_x[1]],
    ":",
    label=r"constant $\mathcal{B}$",
    linewidth=2,
    color="black",
)

for i in range(4, 7):
    plt.plot(
        t,
        df_phi_strong[label_x[i + 1]],
        "-",
        label=label_x[i + 1],
        linewidth=3,
        color=colors[i - 4],
    )

for i in range(1, 4):
    plt.plot(
        t,
        df_phi_strong[label_x[i + 1]],
        ":",
        label="_nolegend_",
        linewidth=2,
        color="black",
    )

plt.xlim(-1, 60)
plt.ylim(-0.00005, 0.00105)
plt.xlabel(r"t (s)")
plt.ylabel(r"$\phi$")

plt.annotate(r"$10^{-1}$", xy=(10, 0.23e-3), rotation=0, fontsize=16)
plt.annotate(r"$10^{-2}$", xy=(25, 0.35e-3), rotation=0, fontsize=16)
plt.annotate(r"$10^{-3}$", xy=(51, 0.66e-3), rotation=0, fontsize=16)
plt.annotate(r"$10^{-4}$", xy=(55, 0.31e-3), rotation=0, fontsize=16)
plt.annotate(
    r"${\cal B}_{\rm core} \approx 10^{-2}$",
    xycoords="axes fraction",
    xy=(0.79, 0.02),
    fontsize=22,
)

plt.legend(loc=2, handlelength=1.5, fontsize=20)

plt.show()


# ## Timing residuals prediction
# ---

# To compare predicted glitch rises with recent pulse-to-pulse observations of a Vela glitch, timing residuals associated with the models are calculated. Provided that residuals and changes in pulsar rotation frequency are small, residuals are given by $- \phi /\nu_{\rm crust}(0) = - 2\pi  \phi / \Omega_{\rm crust}(0)$. Redo integration for different crust-core coupling strengths using a modified grid for the integration, i.e. a time step of $1\,$s.

# In[91]:


t_end_TOAs = 120
n_t_TOAs = 120
dt_TOAs = t_end_TOAs / n_t_TOAs
t_TOAs = np.arange(0, t_end_TOAs + dt_TOAs, dt_TOAs)
print(dt_TOAs)


# In[92]:


integration_results_TOAs = {}
df_delomega_crust_TOAs = pd.DataFrame(columns=label_x[1:])
df_phi_TOAs = pd.DataFrame(columns=label_x[1:])

for i in range(7):
    integration_results_TOAs[i] = time_step(
        t_TOAs,
        omega_sf_0,
        omega_core_0,
        omega_crust_0,
        x,
        n,
        dx,
        h,
        rho_x,
        R_drip,
        I_crust,
        I_core,
        df_mf_x[label_x[i + 1]],
        B_core_weak,
    )

for i in range(7):
    df_delomega_crust_TOAs[label_x[i + 1]] = (
        integration_results_TOAs[i][:, -1] - omega_crust_0
    ) / (2 * pi)

for i in range(7):
    df_phi_TOAs[label_x[i + 1]] = phi(df_delomega_crust_TOAs[label_x[i + 1]], dt_TOAs)


# Timing residuals in ms

# In[93]:


df_model_TOAs = pd.DataFrame(columns=label_x[1:])

df_model_TOAs = -df_phi_TOAs * 2 * pi * 1e3 / omega_crust_0
df_model_TOAs["t"] = t_TOAs


# Export results for data comparison in acconmpanying notebook:

# In[94]:


df_model_TOAs.to_csv("model_residuals_Bcore5e-5.txt", sep="&", index=False, header=True)


# In[95]:


fig = plt.figure()
fig.set_size_inches(10, 6)
mpl.rcParams.update({"font.size": 20})

plt.plot(
    t_TOAs,
    df_model_TOAs[label_x[1]],
    ":",
    label=r"constant $\mathcal{B}$",
    linewidth=2,
    color="black",
)

for i in range(1, 4):
    plt.plot(
        t_TOAs,
        df_model_TOAs[label_x[i + 1]],
        ":",
        label="_nolegend_",
        linewidth=2,
        color="black",
    )

for i in range(4, 7):
    plt.plot(
        t_TOAs,
        df_model_TOAs[label_x[i + 1]],
        "-",
        label=label_x[i + 1],
        linewidth=3,
        color=colors[i - 4],
    )

plt.xlim(-1, 61)
plt.ylim(-0.3, 0.01)
plt.xlabel(r"t (s)")
plt.ylabel(r"timing residuals (ms)")

plt.annotate(r"$10^{-1}$", xy=(8, -0.162), rotation=0, fontsize=16)
plt.annotate(r"$10^{-2}$", xy=(10.5, -0.135), rotation=0, fontsize=16)
plt.annotate(r"$10^{-3}$", xy=(35, -0.21), rotation=0, fontsize=16)
plt.annotate(r"$10^{-4}$", xy=(50, -0.1), rotation=0, fontsize=16)

plt.legend(loc=3, handlelength=1.5, fontsize=20)

plt.show()


# Comparison for a range of crust-core coupling strengths:

# In[96]:


Bcore_range = np.array([1e-5, 2e-5, 3e-5, 5e-5, 1e-4, 5e-4, 1e-2])
label_Bcore = ["{:.1e}".format(x) for x in Bcore_range]


# In[97]:


df_TOAs_Bcore_comparison = pd.DataFrame()

for i in range(7):
    integration_results_comparison = time_step(
        t_TOAs,
        omega_sf_0,
        omega_core_0,
        omega_crust_0,
        x,
        n,
        dx,
        h,
        rho_x,
        R_drip,
        I_crust,
        I_core,
        df_mf_x[label_x[drag_A + 1]],
        Bcore_range[i],
    )[:, -1]

    phase_shift_comparison = phi(
        (integration_results_comparison - omega_crust_0) / (2 * pi), dt_TOAs
    )

    df_TOAs_Bcore_comparison[label_Bcore[i]] = (
        -phase_shift_comparison * 2 * pi * 1e3 / omega_crust_0
    )


# In[98]:


df_TOAs_Bcore_comparison.to_csv(
    "model_residuals_Bcore_comparison.txt", sep="&", index=False, header=True
)


# <details>
# <summary>Zpracovani dat z grafu</summary>
# 
# %%script echo skipping
# import glob
# import re
# 
# # Find all png files matching the pattern
# png_files = glob.glob("mutual_friction_radius_Rcci_*_Mcoeff_*.png")
# 
# # Extract parameters
# data = []
# seen_params = set()
# 
# for f in png_files:
#     # Regex updated to stop capturing before the final dot if present
#     match = re.search(r"Rcci_(\d+)_Mcoeff_([\d.]+?)\.png", f)
#     if match:
#         r_cci_val = int(match.group(1))
#         m_coeff_val = float(match.group(2))
#         
#         if (r_cci_val, m_coeff_val) not in seen_params:
#             seen_params.add((r_cci_val, m_coeff_val))
#             
#             # Calculate M_core
#             m_core_val = m_coeff_val * Msun
#             
#             # Define r range for this R_cci
#             r_val = np.arange(r_cci_val, 2 * r_cci_val, dr)
#             
#             # Solve TOV
#             try:
#                 df_res = solve_TOV(r_val, rho_cci, m_core_val, fit_params_EoS, rho_drip)
#                 
#                 # Apply density to filter vacuum
#                 df_res["Density"] = df_res["Pressure"].apply(
#                     lambda p: density(p, fit_params_EoS, rho_drip) if p > 0 else 0
#                 )
#                 df_res = df_res[df_res["Density"] > 0]
#                 df_res["Radius"] = r_val[: len(df_res)]
#                 
#                 if not df_res.empty:
#                     m_ns_val = df_res["Mass"].iloc[-1]
#                     r_ns_val = df_res["Radius"].iloc[-1]
#                     
#                     data.append({
#                         "R_cci": r_cci_val,
#                         "M_core": m_core_val,
#                         "Mass": np.round(m_ns_val / Msun, 3),
#                         "Radius": np.round(r_ns_val / 1e5, 3)
#                     })
#             except Exception as e:
#                 print(f"Error for {f}: {e}")
# 
# # Create DataFrame
# df_results = pd.DataFrame(data)
# df_results = df_results.sort_values(by=["R_cci", "M_core"])
# df_results
# </details>

# In[110]:


def _make_unique_spline(x, y, k=1):
    """Vytvoří spline fit z dat, která mohou obsahovat duplicity.
    Seřadí data podle x a odstraní duplicitní hodnoty x."""
    order = np.argsort(x)
    x_sorted, y_sorted = x[order], y[order]
    mask = np.concatenate(([True], np.diff(x_sorted) > 0))
    return InterpolatedUnivariateSpline(x_sorted[mask], y_sorted[mask], k=k)


def _get_B_profile(prof_idx, rho_values):
    """Interpoluje profil B(rho) z tabulky df_mf pro daný index profilu (1=A, 2=B, 3=C).
    Vrátí pole hodnot B pro zadané hustoty rho_values."""
    rho_table = df_mf[label[0]].values
    B_table = df_mf[label[prof_idx]].values
    fit = InterpolatedUnivariateSpline(
        np.log10(rho_table), np.log10(B_table), k=2
    )
    rho_min = rho_table[0]
    B_min = B_table[0]
    return np.array([
        B_min if rho < rho_min else 10 ** fit(np.log10(rho))
        for rho in rho_values
    ])


def _simulate_one_set(R_cci_input, M_coeff_input):
    """Provede celou simulaci pro jednu sadu parametrů.
    Vrací list tří křivek del_nu(t) pro profily A, B, C a časový vektor."""

    # 1. TOV řešení
    M_core_val = M_coeff_input * Msun
    r_val = np.arange(R_cci_input, 20e5, 1e1)
    df_tov = solve_TOV(r_val, rho_cci, M_core_val, fit_params_EoS, rho_drip)

    df_tov["Density"] = df_tov["Pressure"].apply(
        lambda p: density(p, fit_params_EoS, rho_drip) if p > 0 else 0
    )
    df_tov = df_tov[df_tov["Density"] > 0]
    df_tov["Radius"] = r_val[:len(df_tov)]

    M_NS = df_tov["Mass"].iloc[-1]
    R_NS = df_tov["Radius"].iloc[-1]
    print(f"    M_NS = {np.round(M_NS / Msun, 3)} Msun, R_NS = {np.round(R_NS / 1e5, 3)} km")

    # 2. Spline fity hustota <-> poloměr
    rho_arr = df_tov["Density"].values
    r_arr = df_tov["Radius"].values
    fit_rho_to_r = _make_unique_spline(np.flipud(rho_arr), np.flipud(r_arr))
    fit_r_to_rho = _make_unique_spline(r_arr, rho_arr)

    # 3. Momenty setrvačnosti
    R_drip_val = fit_rho_to_r(rho_drip)
    I_total = 0.35 * M_NS * R_NS**2
    I_crust_total = quad(lambda r: 8*pi/3 * fit_r_to_rho(r) * r**4, R_cci_input, R_NS)[0]
    I_core = 0.95 * (I_total - I_crust_total)
    h = R_NS
    I_sf = quad(lambda r: 4*pi * h * fit_r_to_rho(r) * r**3, R_cci_input, R_drip_val)[0]
    I_crust = I_total - I_core - I_sf

    # 4. Prostorová mřížka a počáteční podmínky
    n_grid = 100
    dx = (1 - R_cci_input / R_drip_val) / n_grid
    x_grid = R_cci_input / R_drip_val + np.arange(n_grid + 1) * dx
    rho_x = fit_r_to_rho(x_grid * R_drip_val)

    om_Vela = 2 * pi * 11.195
    omega_core_0 = om_Vela
    omega_crust_0 = om_Vela
    om_crit = 6.3e-3
    omega_sf_0 = omega_crust_0 * np.ones(n_grid + 1) * (1 + om_crit / omega_crust_0)

    t_end = 120
    dt = t_end / 20000
    t_sim = np.arange(0, t_end + dt, dt)

    B_core = 5e-5  # Slabý coupling

    # 5. Simulace pro profily A, B, C
    results = []
    for prof_idx in [1, 2, 3]:
        B_crust = _get_B_profile(prof_idx, rho_x)
        y = time_step(
            t_sim, omega_sf_0, omega_core_0, omega_crust_0,
            x_grid, n_grid, dx, h, rho_x, R_drip_val,
            I_crust, I_core, B_crust, B_core
        )
        omega_crust_t = y[:, -1]
        del_nu = (omega_crust_t - omega_crust_0) / (2 * pi)
        results.append(del_nu)

    return t_sim, results


def run_glitch_simulation(R_cci_1, M_coeff_1, R_cci_2, M_coeff_2):
    """Porovná glitch rise křivky pro dvě sady parametrů.
    Set 1 = plné čáry, Set 2 = čárkované čáry."""

    profile_names = ["A", "B", "C"]

    # Nastavení grafu
    fig, ax = plt.subplots(figsize=(10, 6))
    mpl.rcParams.update({"font.size": 20})

    # Referenční křivky (konstantní B z původních dat)
    for i in range(4):
        ax.plot(
            t, 1e6 * df_delomega_crust_weak[label_x[i + 1]],
            ":", color="gray", alpha=0.5, linewidth=1,
            label=(r"constant $\mathcal{B}$" if i == 0 else "_nolegend_")
        )

    # Simulace obou sad
    for R_cci, M_coeff, ls, set_name in [
        (R_cci_1, M_coeff_1, "-", "Set 1"),
        (R_cci_2, M_coeff_2, "--", "Set 2"),
    ]:
        print(f"  -> Processing {set_name} (R={R_cci}, M={M_coeff})...")
        t_sim, curves = _simulate_one_set(R_cci, M_coeff)
        for i, del_nu in enumerate(curves):
            ax.plot(
                t_sim, 1e6 * del_nu, ls,
                linewidth=2, color=colors[i],
                label=f"{profile_names[i]} ({set_name})"
            )

    # Rovnovážná čára
    ax.axhline(del_nu_equilibrium, color="black", alpha=0.3)

    # Formátování
#     ax.set_xscale('log')
    ax.set_xlim(0, 60)
    ax.set_ylim(1, 1000)
    ax.set_yscale('log')
    ax.set_xlabel(r"t (s)")
    ax.set_ylabel(r"$\Delta\nu\ (\mu{\rm Hz})$")

    # Anotace konstantních B hodnot
    for val, pos in [
        (r"$\mathcal{B} = 10^{-1}$", (-0.5, 2.8e2)),
        (r"$\mathcal{B} = 10^{-2}$", (6, 1.6e2)),
        (r"$\mathcal{B} = 10^{-3}$", (30, 4.3e1)),
        (r"$\mathcal{B} = 10^{-4}$", (3, 0.7e1)),
    ]:
        ax.annotate(val, xy=pos, fontsize=14, color="gray", alpha=0.9,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

    import matplotlib.lines as mlines
    # Přidání vlastních popisků pro sady
    val_r1 = f"{R_cci_1/1e6:.1f}".replace('.', ',')
    label1 = f"Set 1 (R_cci = {val_r1} $\\cdot 10^6$, M_coeff = {M_coeff_1})"
    line_set1 = mlines.Line2D([], [], color='black', ls='-', linewidth=2, label=label1)
    val_r2 = f"{R_cci_2/1e6:.1f}".replace('.', ',')
    label2 = f"Set 2 (R_cci = {val_r2} $\\cdot 10^6$, M_coeff = {M_coeff_2})"
    line_set2 = mlines.Line2D([], [], color='black', ls='--', linewidth=2, label=label2)
    ax.legend(handles=[line_set1, line_set2], loc='upper right', fontsize=12)


# Example usage:
R_cci_1 = 1400000
M_coeff_1 = 2.2

R_cci_2 = 1200000
M_coeff_2 = 1.1

run_glitch_simulation(R_cci_1, M_coeff_1, R_cci_2, M_coeff_2)

