#!/usr/bin/python -i

# Plot DREAMOutput data

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import datetime
import math
import h5py
import os


# File and folder names
datafolder = input("Location of data files:\n")
plotfoldername = "plots_" + datafolder

if not os.path.exists("./" + plotfoldername):
    os.mkdir(plotfoldername)

for file in os.listdir(datafolder):
    if file.startswith("Output_injection_1"):
        init2 = datafolder + "/" + file
        sett_init2 = h5py.File(init2, "r")

# Set radial grid
a_grid = sett_init2["grid/r"][()]
a_grid = a_grid.transpose()

init_2_duration = sett_init2["settings/timestep/tmax"][()][0]
init_2_nt = sett_init2["settings/timestep/nsavesteps"][()][0] + 1

timegrid = np.linspace(0, init_2_duration, init_2_nt)

timegrid *= 1000


# PLOTS ##################

# matplotlib initial settings
plt.rcParams.update({'font.size': 18})
matplotlib.rc('image', cmap='inferno')
plt.rcParams['axes.facecolor'] = 'black'
formatter = matplotlib.ticker.LogFormatter(10, labelOnlyBase=False)
formatter2 = matplotlib.ticker.LogFormatterExponent(base=10)
fs = [10, 8]

f1 = h5py.File(init2, "r")
t1 = len(f1["eqsys/I_p"][()])


# Hot electron temperature profile
def plothotprof(f=f1, t=t1-1, s=False, r=0):
    output = f["eqsys/f_hot"][()][t,r,:,:]
    fig = plt.figure(figsize=fs)
    plt.contourf(np.linspace(0, 120, 120), np.linspace(0, 5, 5), output, levels=100)
    axes = plt.gca()
    axes.set_facecolor('black')
    cbar = plt.colorbar()
    cbar.set_label(r"n_e [m$^{-3}$]")
    plt.ylabel("Pitch grid")
    plt.xlabel("Momentum grid")
    if s:
        
        plt.savefig(plotfoldername + "/" + "f_hot_f" + str(f) + "_t" + str(t) + "_r" + str(r) + ".png", dpi=150)
    else:
        plt.show()



# I_p total plasma current
def I_p(s=False):
    I_p = f1["eqsys/I_p"][()]
    fig = plt.figure(figsize=fs)
    plot = plt.plot(timegrid[0:], I_p[0:] / 1e6)
    axes = plt.gca()
    axes.set_facecolor('white')
    axes.grid()
    plt.ylabel("Total plasma curret (MA)")
    plt.xlabel("Time [ms]")
    if s:
        now = datetime.datetime.now()
        date_time = now.strftime("_%Y.%m.%d_%Hh%Mm%Ss")
        plt.savefig(plotfoldername + "/" + "I_p" + date_time + ".png", dpi=150)
    else:
        plt.show()


# Electric field
def E_field(s=False):
    E_field = f1["eqsys/E_field"][()]
    fig = plt.figure(figsize=fs)
    plot_1 = plt.contourf(a_grid, timegrid, E_field, levels=100)
    axes = plt.gca()
    axes.set_facecolor('black')
    cbar = plt.colorbar()
    cbar.set_label("Electric field [V/m]")
    plt.xlabel("Minor radius [m]")
    plt.ylabel("Time [ms]")
    if s:
        now = datetime.datetime.now()
        date_time = now.strftime("_%Y.%m.%d_%Hh%Mm%Ss")
        plt.savefig(plotfoldername + "/" + "E_field" + date_time + ".png", dpi=150)
    else:
        plt.show()


def E_field_log(s=False, logdiff=10):
    j_re = f1["eqsys/E_field"][()]
    j_re[j_re < 1e-5] = 1e-5
    levels = 22
    logdiff = logdiff
    j_min = j_re.min()
    j_max = j_re.max()
    logmax = math.ceil(np.log10(j_max))
    logmin = logmax - logdiff
    numofticks = logdiff
    div = logdiff // numofticks
    power = np.arange((logmax - (numofticks * div)), logmax, div)
    array = np.zeros(len(power)) + 10.
    ticks = np.power(array, power)
    levels1 = np.logspace(logmin, logmax, levels, base=10.0)
    norm = matplotlib.colors.LogNorm(vmin=j_min, vmax=j_max)
    fig2 = plt.figure(figsize=fs)
    plot_2 = plt.contourf(a_grid, timegrid, j_re, levels=levels1, norm=norm)
    axes2 = plt.gca()
    axes2.set_facecolor('black')
    cbar2 = plt.colorbar(ticks=ticks, format=formatter2)
    cbar2.update_ticks()
    cbar2.set_label(r'log$_{10}$(E$_{||}$/[1V/m])')
    plt.xlabel("Minor radius [m]")
    plt.ylabel("Time [ms]")
    if s:
        now = datetime.datetime.now()
        date_time = now.strftime("_%Y.%m.%d_%Hh%Mm%Ss")
        plt.savefig(plotfoldername + "/" + "E_field" + "_LOG" + date_time + ".png", dpi=150)
    else:
        plt.show()


# Total current
def j_tot(s=False):
    j_tot = f1["eqsys/j_tot"][()]
    j_tot /= 1e6
    fig = plt.figure(figsize=fs)
    plot_1 = plt.contourf(a_grid, timegrid, j_tot, levels=100)
    axes = plt.gca()
    axes.set_facecolor('black')
    cbar = plt.colorbar()
    cbar.set_label(r"Total current density [MA/m$^2$]")
    plt.xlabel("Minor radius [m]")
    plt.ylabel("Time [ms]")
    if s:
        now = datetime.datetime.now()
        date_time = now.strftime("_%Y.%m.%d_%Hh%Mm%Ss")
        plt.savefig(plotfoldername + "/" + "j_tot" + date_time + ".png", dpi=150)
    else:
        plt.show()


def j_tot_log(s=False, logdiff=5):
    j_re = f1["eqsys/j_tot"][()]
    j_re[j_re < 1e-10] = 1e-10
    j_re /= 1e6
    levels = 12
    logdiff = logdiff
    j_min = j_re.min()
    j_max = j_re.max()
    logmax = math.ceil(np.log10(j_max))
    logmin = logmax - logdiff
    numofticks = logdiff
    div = logdiff // numofticks
    power = np.arange((logmax - (numofticks * div)), logmax, div)
    array = np.zeros(len(power)) + 10.
    ticks = np.power(array, power)
    levels1 = np.logspace(logmin, logmax, levels, base=10.0)
    norm = matplotlib.colors.LogNorm(vmin=j_min, vmax=j_max)
    fig2 = plt.figure(figsize=fs)
    plot_2 = plt.contourf(a_grid, timegrid, j_re, levels=levels1, norm=norm)
    axes2 = plt.gca()
    axes2.set_facecolor('black')
    cbar2 = plt.colorbar(ticks=ticks, format=formatter2)
    cbar2.update_ticks()
    cbar2.set_label(r'log$_{10}$(j_tot/[1MA/m$^2$])')
    plt.xlabel("Minor radius [m]")
    plt.ylabel("Time [ms]")
    if s:
        now = datetime.datetime.now()
        date_time = now.strftime("_%Y.%m.%d_%Hh%Mm%Ss")
        plt.savefig(plotfoldername + "/" + "j_tot" + "_LOG" + date_time + ".png", dpi=150)
    else:
        plt.show()


# Superthermal electron current
def j_hot(s=False):
    j_tot = f1["eqsys/j_hot"][()]
    j_tot /= 1e3
    fig = plt.figure(figsize=fs)
    plot_1 = plt.contourf(a_grid, timegrid, j_tot, levels=100)
    axes = plt.gca()
    axes.set_facecolor('black')
    cbar = plt.colorbar()
    cbar.set_label(r"Hot electron current density [kA/m$^2$]")
    plt.xlabel("Minor radius [m]")
    plt.ylabel("Time [ms]")
    if s:
        now = datetime.datetime.now()
        date_time = now.strftime("_%Y.%m.%d_%Hh%Mm%Ss")
        plt.savefig(plotfoldername + "/" + "j_hot" + date_time + ".png", dpi=150)
    else:
        plt.show()


def j_hot_log(s=False, logdiff=15):
    j_re = f1["eqsys/j_hot"][()]
    j_re[j_re < 1e-10] = 1e-10
    j_re /= 1e3
    levels = 16
    logdiff = logdiff
    j_min = j_re.min()
    j_max = j_re.max()
    logmax = math.ceil(np.log10(j_max))
    logmin = logmax - logdiff
    numofticks = logdiff
    div = logdiff // numofticks
    power = np.arange((logmax - (numofticks * div)), logmax, div)
    array = np.zeros(len(power)) + 10.
    ticks = np.power(array, power)
    levels1 = np.logspace(logmin, logmax, levels, base=10.0)
    norm = matplotlib.colors.LogNorm(vmin=j_min, vmax=j_max)
    fig2 = plt.figure(figsize=fs)
    plot_2 = plt.contourf(a_grid, timegrid, j_re, levels=levels1, norm=norm)
    axes2 = plt.gca()
    axes2.set_facecolor('black')
    cbar2 = plt.colorbar(ticks=ticks, format=formatter2)
    cbar2.update_ticks()
    cbar2.set_label(r'log$_{10}$(j_hot/[1MA/m$^2$])')
    plt.xlabel("Minor radius [m]")
    plt.ylabel("Time [ms]")
    if s:
        now = datetime.datetime.now()
        date_time = now.strftime("_%Y.%m.%d_%Hh%Mm%Ss")
        plt.savefig(plotfoldername + "/" + "j_hot" + "_LOG" + date_time + ".png", dpi=150)
    else:
        plt.show()


# Runaway electron current
def j_re(s=False):
    j_re = f1["eqsys/j_re"][()]
    j_re /= 1e3
    fig = plt.figure(figsize=fs)
    plot_1 = plt.contourf(a_grid, timegrid, j_re, levels=100)
    axes = plt.gca()
    axes.set_facecolor('black')
    cbar = plt.colorbar()
    cbar.set_label(r"Runaway current density [kA/m$^2$]")
    plt.xlabel("Minor radius [m]")
    plt.ylabel("Time [ms]")
    if s:
        now = datetime.datetime.now()
        date_time = now.strftime("_%Y.%m.%d_%Hh%Mm%Ss")
        plt.savefig(plotfoldername + "/" + "j_re" + date_time + ".png", dpi=150)
    else:
        plt.show()


def j_re_log(s=False, logdiff=15):
    j_re = f1["eqsys/j_re"][()]
    j_re[j_re < 1e-10] = 1e-10
    j_re /= 1e3
    levels = 16
    logdiff = logdiff
    j_min = j_re.min()
    j_max = j_re.max()
    logmax = math.ceil(np.log10(j_max))
    logmin = logmax - logdiff
    numofticks = logdiff
    div = logdiff // numofticks
    power = np.arange((logmax - (numofticks * div)), logmax, div)
    array = np.zeros(len(power)) + 10.
    ticks = np.power(array, power)
    levels1 = np.logspace(logmin, logmax, levels, base=10.0)
    norm = matplotlib.colors.LogNorm(vmin=j_min, vmax=j_max)
    fig2 = plt.figure(figsize=fs)
    plot_2 = plt.contourf(a_grid, timegrid, j_re, levels=levels1, norm=norm)
    axes2 = plt.gca()
    axes2.set_facecolor('black')
    cbar2 = plt.colorbar(ticks=ticks, format=formatter2)
    cbar2.update_ticks()
    cbar2.set_label(r'log$_{10}$(j_re/[1kA/m$^2$])')
    plt.xlabel("Minor radius [m]")
    plt.ylabel("Time [ms]")
    if s:
        now = datetime.datetime.now()
        date_time = now.strftime("_%Y.%m.%d_%Hh%Mm%Ss")
        plt.savefig(plotfoldername + "/" + "j_re" + "_LOG" + date_time + ".png", dpi=150)
    else:
        plt.show()


# Hot and RE current density on logarithmic scale
def j_h_and_re(s=False, logdiff=15):
    j_re = f1["eqsys/j_hot"][()]
    j_re += f1["eqsys/j_re"][()]
    j_re[j_re < 1e-10] = 1e-10
    j_re /= 1e3
    levels = 16
    logdiff = logdiff
    j_min = j_re.min()
    j_max = j_re.max()
    logmax = math.ceil(np.log10(j_max))
    logmin = logmax - logdiff
    numofticks = logdiff
    div = logdiff // numofticks
    power = np.arange((logmax - (numofticks * div)), logmax, div)
    array = np.zeros(len(power)) + 10.
    ticks = np.power(array, power)
    levels1 = np.logspace(logmin, logmax, levels, base=10.0)
    norm = matplotlib.colors.LogNorm(vmin=j_min, vmax=j_max)
    fig2 = plt.figure(figsize=fs)
    plot_2 = plt.contourf(a_grid, timegrid, j_re, levels=levels1, norm=norm)
    axes2 = plt.gca()
    axes2.set_facecolor('black')
    cbar2 = plt.colorbar(ticks=ticks, format=formatter2)
    cbar2.update_ticks()
    cbar2.set_label(r'log$_{10}$((j_hot+j_re)/[1kA/m$^2$])')
    plt.xlabel("Minor radius [m]")
    plt.ylabel("Time [ms]")
    if s:
        now = datetime.datetime.now()
        date_time = now.strftime("_%Y.%m.%d_%Hh%Mm%Ss")
        plt.savefig(plotfoldername + "/" + "j_hot_and_re" + "_LOG" + date_time + ".png", dpi=150)
    else:
        plt.show()


# logarithmic temperature contour plot
def T_cold_log(s=False, logdiff=5):
    T_cold = f1["eqsys/T_cold"][()]
    levels = 10
    logdiff = logdiff
    T_min = T_cold.min()
    T_max = T_cold.max()
    logmax = math.ceil(np.log10(T_max))
    logmin = logmax - logdiff
    numofticks = logdiff
    div = logdiff // numofticks
    power = np.arange((logmax - (numofticks * div)), logmax, div)
    array = np.zeros(len(power)) + 10.
    ticks = np.power(array, power)
    levels1 = np.logspace(logmin, logmax, levels, base=10.0)
    norm = matplotlib.colors.LogNorm(vmin=T_min, vmax=T_max)
    fig2 = plt.figure(figsize=fs)
    plot_2 = plt.contourf(a_grid, timegrid, T_cold, levels=levels1, norm=norm)
    axes2 = plt.gca()
    axes2.set_facecolor('black')
    cbar2 = plt.colorbar(ticks=ticks, format=formatter2)
    cbar2.update_ticks()
    cbar2.set_label(r'log$_{10}$(T/[1eV])')
    plt.xlabel("Minor radius [m]")
    plt.ylabel("Time [ms]")
    if s:
        now = datetime.datetime.now()
        date_time = now.strftime("_%Y.%m.%d_%Hh%Mm%Ss")
        plt.savefig(plotfoldername + "/" + "T_cold" + "_LOG" + date_time + ".png", dpi=150)
    else:
        plt.show()


# Deuterium densities
def n_D(s=False, inj=True, background=True):
    fig = plt.figure(figsize=fs)
    plot_1 = plt.contourf(a_grid, timegrid, n, levels=100)
    axes = plt.gca()
    axes.set_facecolor('black')
    cbar = plt.colorbar()
    cbar.set_label(naming + " n_D [1/m^3]")
    plt.xlabel("Minor radius [m]")
    plt.ylabel("Time [ms]")
    plt.title("nShard" + deut)
    if s:
        now = datetime.datetime.now()
        date_time = now.strftime("_%Y.%m.%d_%Hh%Mm%Ss")
        # plt.savefig(plotfoldername + "/" + "n_D_" + naming + date_time + ".png", dpi=150)
        plt.savefig(plotfoldername + "/" + "n_D_" + naming + "nShard" + deut + date_time + ".png", dpi=150)
    else:
        plt.show()


def n_D_log(s=False, logdiff=4, inj=True, background=True, cutoff=1e8):
    levels = 10
    logdiff = logdiff
    n_min = n.min()
    n_max = n.max()
    logmax = math.ceil(np.log10(n_max))
    logmin = logmax - logdiff
    numofticks = logdiff
    div = logdiff // numofticks
    power = np.arange((logmax - (numofticks * div)), logmax, div)
    array = np.zeros(len(power)) + 10.
    ticks = np.power(array, power)
    levels1 = np.logspace(logmin, logmax, levels, base=10.0)
    norm = matplotlib.colors.LogNorm(vmin=n_min, vmax=n_max)
    fig2 = plt.figure(figsize=fs)
    plot_2 = plt.contourf(a_grid, timegrid, n, levels=levels1, norm=norm)
    axes2 = plt.gca()
    axes2.set_facecolor('black')
    cbar2 = plt.colorbar(ticks=ticks, format=formatter2)
    cbar2.update_ticks()
    cbar2.set_label(naming + " " + r'log$_{10}$(n_D/[1/m^3])')
    plt.xlabel("Minor radius [m]")
    plt.ylabel("Time [ms]")
    plt.title("nShard" + deut)
    if s:
        now = datetime.datetime.now()
        date_time = now.strftime("_%Y.%m.%d_%Hh%Mm%Ss")
        plt.savefig(plotfoldername + "/" + "n_D_" + naming + "_LOG" + date_time + ".png", dpi=150)
    else:
        plt.show()


# Neon densities
def n_Ne_tot(s=False):
    n_i = appendiondensity(filenames, 2)
    for i in range(3, 12):
        n_i += appendiondensity(filenames, i)

    fig = plt.figure(figsize=fs)
    plot_1 = plt.contourf(a_grid, timegrid, n_i, levels=100)
    axes = plt.gca()
    axes.set_facecolor('black')
    cbar = plt.colorbar()
    cbar.set_label("n_Ne_tot [1/m^3]")
    plt.xlabel("Minor radius [m]")
    plt.ylabel("Time [ms]")
    if s:
        now = datetime.datetime.now()
        date_time = now.strftime("_%Y.%m.%d_%Hh%Mm%Ss")
        plt.savefig(plotfoldername + "/" + "n_Ne" + date_time + ".png", dpi=150)
    else:
        plt.show()


def n_Ne_log(s=False, logdiff=8, cutoff=1e8):
    n = appendiondensity(filenames, 2)
    for i in range(3, 12):
        n += appendiondensity(filenames, i)

    n[n < cutoff] = cutoff
    levels = 18
    logdiff = logdiff
    n_min = n.min()
    n_max = n.max()
    logmax = math.ceil(np.log10(n_max))
    logmin = logmax - logdiff
    numofticks = logdiff
    div = logdiff // numofticks
    power = np.arange((logmax - (numofticks * div)), logmax, div)
    array = np.zeros(len(power)) + 10.
    ticks = np.power(array, power)
    levels1 = np.logspace(logmin, logmax, levels, base=10.0)
    norm = matplotlib.colors.LogNorm(vmin=n_min, vmax=n_max)
    fig2 = plt.figure(figsize=fs)
    plot_2 = plt.contourf(a_grid, timegrid, n, levels=levels1, norm=norm)
    axes2 = plt.gca()
    axes2.set_facecolor('black')
    cbar2 = plt.colorbar(ticks=ticks, format=formatter2)
    cbar2.update_ticks()
    cbar2.set_label(r'log$_{10}$(n_Ne_tot/[1/m^3])')
    plt.xlabel("Minor radius [m]")
    plt.ylabel("Time [ms]")
    if s:
        now = datetime.datetime.now()
        date_time = now.strftime("_%Y.%m.%d_%Hh%Mm%Ss")
        plt.savefig(plotfoldername + "/" + "n_Ne_tot" + "_LOG" + date_time + ".png", dpi=150)
    else:
        plt.show()


# plot everything and copy geometry picture, DREAM revision ID and the run file
def all_plot_and_copy(s=True):
    src1 = "DEMO_geom_1.png"
    dst1 = datafolder + "/" + src1
    os.system(f"cp {src1} {dst1}")
    src2 = "DREAM_revision_id.md"
    dst2 = datafolder + "/" + src2
    os.system(f"cp {src2} {dst2}")
    src3 = datafolder + ".py"
    dst3 = datafolder + "/" + src3
    os.system(f"cp {src3} {dst3}")
    I_p(s)
    E_field(s)
    E_field_log(s)
    j_tot(s)
    j_tot_log(s)
    j_hot(s)
    j_hot_log(s)
    j_re(s)
    j_re_log(s)
    j_h_and_re(s)
    T_cold_log(s)
    # n_D(s)
    n_D(s, background=False)
    n_D_log(s)
    #n_D(s, inj=False)
    # n_D_log(s, inj=False)
    n_Ne_tot(s)
    n_Ne_log(s)
    
    
# just plot everything
def just_plot_all(s=True):
    I_p(s)
    E_field(s)
    E_field_log(s)
    j_tot(s)
    j_tot_log(s)
    j_hot(s)
    j_hot_log(s)
    j_re(s)
    j_re_log(s)
    j_h_and_re(s)
    T_cold_log(s)
    # n_D(s)
    # n_D(s, background=False)
    # n_D_log(s)
    # n_D(s, inj=False)
    # n_D_log(s, inj=False)
    # n_Ne_tot(s)
    # n_Ne_log(s)


def d_plot(s=True):
    # n_D(s, background=False)
    # n_D(s)
    n_D(s, background=False)


#all_plot_and_copy()

just_plot_all()

#plothotprof()