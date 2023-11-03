import numpy as np
import scipy.constants
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import datetime
import math
import h5py
import os


from DREAM.DREAMOutput import DREAMOutput
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import LogLocator
from matplotlib.ticker import LogFormatter

# matplotlib initial settings
# use plt.tight_layout() for matplotlib to not cut off labels at the edge of figures
plt.rcParams.update({'font.size': 14})
matplotlib.rc('image', cmap='inferno')


def auto_yscale(yscale):
    if yscale is None:
        datamin = data_to_plot.min()
        datamax = data_to_plot.max()
        if datamax > 0 and datamin > 0:
            log10_min = math.log10(datamin)
            log10_max = math.log10(datamax)
            if (log10_max - log10_min) > 2.5:
                yscale = "log"
            else:
                yscale = "linear"
        elif datamax > 0 and datamin < 0:
            yscale = "linear"
        elif datamax < 0 and datamin < 0:
            log10_min = math.log10(abs(datamin))
            log10_max = math.log10(abs(datamax))
            if (log10_min - log10_max) > 2.5:
                yscale = "log"
                print("\033[31mAbsolute values plotted!\033[m")
                data_to_plot = abs(data_to_plot)
            else:
                yscale = "linear"
        else:
            yscale = "linear"
    else:
        pass

    return yscale

class CustomDREAMOutput:
    """
    \033[32m
    Custom class for DREAMOutput files with plotting functions.
    Reads and stores *most* of the important values from a DREAM output HDF5 file.
    \033[31mThe syntax for reaching some attribute values:\033[m .attribute[0]
    \033[31mTo get the physical units:\033[m .attribute[1]
    \033[32m
    For the DREAM documentation visit https://ft.nephy.chalmers.se/dream/ 
    See the source code of DREAM at https://github.com/chalmersplasmatheory/DREAM\033[m
    """

    def __init__(self, filename, start_time):
        """
        \033[32m
        In addition to reading the HDF5 file, the initialization 
        also contains calculations with the different grids.\033[m
        """
        self.filename = filename
        self.dirname = os.path.dirname(filename)
        self.just_filename = os.path.basename(filename)
        self.filename_start = self.just_filename[:-3]
        f = h5py.File(filename, "r")

        # time related settings
        self.timegrid = f["grid/t"][()] + start_time
        self.timegrid_ms = self.timegrid * 1000
        self.number_of_timesteps = len(self.timegrid) - 1
        self.timestep = f["grid/t"][()][1]

        # spatial coordinates - [m]
        self.radialgrid = f["grid/r"][()]
        self.number_of_radial_cells = len(self.radialgrid)
        self.radialgrid_edges = f["grid/r_f"][()]
        self.radial_step = f["grid/dr"][()][1]
        self.major_radius = f["settings/radialgrid/R0"][()]
        self.minor_radius = f["settings/radialgrid/a"][()]

        # hottail and runawaygrid settings
        self.hottailgrid_enabled = f["settings/hottailgrid/enabled"][()]
        self.runawaygrid_enabled = f["settings/runawaygrid/enabled"][()]

        # Some constants
        m_e = scipy.constants.m_e  # electron mass
        e = scipy.constants.e  # elemental charge
        c = scipy.constants.c  # speed of light
        pi = scipy.constants.pi 

        # HOTTAILGRID
        if self.hottailgrid_enabled:
            # momentum and pitchgrid - hottail
            self.momentumgrid = f["grid/hottail/p1"][()]  # in m_e*c normalized momentum
            self.momentumgrid_edges = f["grid/hottail/p1_f"][()]  # in m_e*c normalized momentum - edges
            self.pitchgrid = f["grid/hottail/p2"][()]  # cosine of pitch angle - cell centers
            self.pitchgrid_edges = f["grid/hottail/p2_f"][()]  # cosine of pitch angle - cell edges
            self.pitchgrid_rad = np.arccos(self.pitchgrid)  # pitch angle in radians - cell centers
            self.pitchgrid_rad_edges = np.arccos(self.pitchgrid_edges)  # pitch angle in radians - cell edges
            self.pitchgrid_degrees = 180 * self.pitchgrid_rad / scipy.constants.pi  # pitch degrees - centers
            self.pitchgrid_degrees_edges = 180 * self.pitchgrid_rad_edges / scipy.constants.pi  # pitch degrees - edges

            # electron distribution - hottail
            self.distribution_t_r_xi_p = [f["eqsys/f_hot"][()], "1/m$^3$"]  # time; radii; pitch; momentum
            self.angle_avg_distr = [np.sum(self.distribution_t_r_xi_p[0], 2),
                                    "1/m$^3$"]  # angle averaged distribution - time; radii; momentum

        # RUNAWAYGRID
        if self.runawaygrid_enabled:
            # momentum and pitchgrid - runaway
            self.re_momentumgrid = f["grid/runaway/p1"][()]  # in m_e*c normalized momentum
            self.re_momentumgrid_edges = f["grid/runaway/p1_f"][()]  # in m_e*c normalized momentum
            self.re_pitchgrid = f["grid/runaway/p2"][()]  # cosine of pitch angle - cell centers
            self.re_pitchgrid_edges = f["grid/runaway/p2_f"][()]  # cosine of pitch angle - cell edges
            self.re_pitchgrid_rad = np.arccos(self.re_pitchgrid)  # pitch angle in radians - cell centers
            self.re_pitchgrid_rad_edges = np.arccos(self.re_pitchgrid_edges)  # pitch angle in radians - cell edges
            self.re_pitchgrid_degrees = 180 * self.re_pitchgrid_rad / (
                scipy.constants.pi)  # pitch angle in degreess - cell centers
            self.re_pitchgrid_degrees_edges = 180 * self.re_pitchgrid_rad_edges / (
                scipy.constants.pi)  # pitch angle in degrees - cell edges

            # energy grid from the momentum grid - runaway
            self.re_energy_grid_electronvolts = self.re_momentumgrid * 510998.95

            # electron distribution - runaway
            self.re_distribution_t_r_xi_p = [f["eqsys/f_re"][()], "1/m$^3$"]  # time; radii; pitch; momentum
            self.re_angle_avg_distr = [np.sum(self.re_distribution_t_r_xi_p[0], 2),
                                       "1/m$^3$"]  # angle averaged distribution - time; radii; momentum

        # other eqsys data
        self.electric_field = [f["eqsys/E_field"][()], "V/m"]  # V/m
        self.cold_temperature_eV = [f["eqsys/T_cold"][()], "eV"]  # eV
        self.cold_energy_density = [f["eqsys/W_cold"][()], "J/m$^3$"]  # J/m^3
        self.avg_cold_temp = [np.average(self.cold_temperature_eV[0], 1), "eV"]

        # current densities
        self.ohmic_current_density = [f["eqsys/j_ohm"][()], "A/m$^3$"]  # A/m^3
        self.hot_current_density = [f["eqsys/j_hot"][()], "A/m$^3$"]  # A/m^3
        self.runaway_current_density = [f["eqsys/j_re"][()], "A/m$^3$"]  # A/m^3
        self.total_current_density = [f["eqsys/j_tot"][()], "A/m$^3$"]  # A/m^3

        # particle densities
        self.cold_electron_density = [f["eqsys/n_cold"][()], "1/m$^3$"]  # 1/m^3
        self.hot_electron_density = [f["eqsys/n_hot"][()], "1/m$^3$"]  # 1/m^3
        self.runaway_density = [f["eqsys/n_re"][()], "1/m$^3$"]  # 1/m^3
        self.total_electron_density = [f["eqsys/n_tot"][()], "1/m$^3$"]  # 1/m^3
        self.ion_densities_t_i_r = [f["eqsys/n_i"][()], "1/m$^3$"]  # 1/m^3 - time; iontypes; radii

        # ion metadata
        self.ion_Z = f["ionmeta/Z"][()]  # atomic number of the ions
        self.ion_names = f["ionmeta/names"][()]  # user set names for ions
        
        self.real_volumes_of_cells = [f["grid/VpVol"][()], "m$^3$"]  # m^3
        self.for_integral = np.divide(f["grid/VpVol"][()], (self.major_radius * 2 * pi))

        # power densities
        try:
            self.collisional_power_density_RE = [f["other/fluid/Tcold_nre_coll"][()], "J/(s$\\cdot$m$^3$)"]  # J/(s*m^3)
            self.ohmic_heating_power_density = [f["other/fluid/Tcold_ohmic"][()], "J/(s$\\cdot$m$^3$)"]  # J/(s*m^3)
            self.radiated_power_density = [f["other/fluid/Tcold_radiation"][()], "J/(s$\\cdot$m$^3$)"]  # J/(s*m^3)
            
            self.collisional_power_RE = [np.multiply(self.collisional_power_density_RE[0], self.for_integral), "W"]
            self.ohmic_heating_power = [np.multiply(self.ohmic_heating_power_density[0], self.for_integral), "W"]
            self.radiated_power = [np.multiply(self.radiated_power_density[0], self.for_integral), "W"]
            
            # sums power values in the radial direction
            self.collisional_power_RE_sum = [np.sum(self.collisional_power_RE[0], 1), "W"]
            self.ohmic_heating_power_sum = [np.sum(self.ohmic_heating_power[0], 1), "W"]
            self.radiated_power_sum = [np.sum(self.radiated_power[0], 1), "W"]
        except:
            pass

        # densities integrated
        self.cold_energy = [np.multiply(self.cold_energy_density[0], self.for_integral), "J"]
        self.ohmic_current = [np.multiply(self.ohmic_current_density[0], self.for_integral), "A"]
        self.hot_current = [np.multiply(self.hot_current_density[0], self.for_integral), "A"]
        self.runaway_current = [np.multiply(self.runaway_current_density[0], self.for_integral), "A"]
        self.total_current = [np.multiply(self.total_current_density[0], self.for_integral), "A"]
        self.N_cold_electrons = [np.multiply(self.cold_electron_density[0], self.for_integral), "-"]
        self.N_hot_electrons = [np.multiply(self.hot_electron_density[0], self.for_integral), "-"]
        self.N_runaways = [np.multiply(self.runaway_density[0], self.for_integral), "-"]
        self.N_total_electrons = [np.multiply(self.total_electron_density[0], self.for_integral), "-"]

        self.N_ions = [self.ion_densities_t_i_r[0] * self.for_integral, "-"]

        # total values based on densities and real cell volumes - WORK IN PROGRESS!
        self.cold_energy_sum = [np.sum(self.cold_energy[0], 1), "J"]
        self.ohmic_current_sum = [np.sum(self.ohmic_current[0], 1), "A"]
        self.hot_current_sum = [np.sum(self.hot_current[0], 1), "A"]
        self.runaway_current_sum = [np.sum(self.runaway_current[0], 1), "A"]
        self.total_current_sum = [np.sum(self.total_current[0], 1), "A"]
        plasma_current = f["eqsys/I_p"][()]
        self.plasma_current = [plasma_current.reshape((plasma_current.shape[0])), "A"]
        self.N_cold_electrons_sum = [np.sum(self.N_cold_electrons[0], 1), "-"]
        self.N_hot_electrons_sum = [np.sum(self.N_hot_electrons[0], 1), "-"]
        self.N_runaways_sum = [np.sum(self.N_runaways[0], 1), "-"]
        self.N_total_electrons_sum = [np.sum(self.N_total_electrons[0], 1), "-"]

        # sums ions in the radial direction
        self.N_ions_sum = [np.sum(self.N_ions[0], 2), "-"]

        # Create folder for storing plot files
        self.plotfoldername = self.dirname + "/plots/"
        self.make_plot_folder()

    def __str__(self):
        """
        \033[32m
        Strings to print when print(*object_name*) is called\033[m
        """

        str1 = f"CustomDREAMOutput object from the file: {self.filename}\n"
        str2 = f".info() - basic information about this file\n"
        str3 = f".list_atrs() - list all attributes of the object\n"
        str4 = f"Use help(*object_name*) for further help on the class"
        return str1 + str2 + str3 + str4

    def info(self):
        """
        \033[32m
        Basic information about time and radial resolution and
        if the hottail or runaway grids are enabled\033[m
        """

        print("Simulation time from " + str(round(self.timegrid_ms[0], 4)) +" ms to "+ str(round(self.timegrid_ms[-1], 4)) + " ms")
        print("Number of saved timesteps: " + str(self.number_of_timesteps))
        print("Save timestep (dt): " + str(round(self.timestep * 1000, 4)) + " ms")

        print("Minor radius: " + str(self.minor_radius) + " m")
        print("Number of radial grid cells: " + str(len(self.radialgrid)))
        print("Radial step (dr): " + str(self.radial_step) + " m")

        print("Hottailgrid enabled: " + str(bool(self.hottailgrid_enabled)))
        print("Runawaygrid enabled: " + str(bool(self.runawaygrid_enabled)))

    def list_attrs(self):
        """
        \033[32m
        Lists all attributes with types and dimensions if relevant
        for help with plotting and for debugging purposes.
        The attribute names are used in plotting functions as the key for the data.\033[m
        """

        print("\nList of attributes\n")
        for key in self.__dict__.keys():
            element = self.__dict__[key]
            if type(element) == list:
                element = element[0]

            if type(element) == np.ndarray:
                details = element.shape
            else:
                details = element
            print(key + ": " + str(type(element)) + " - " + str(details))
        print("\n")

    def make_plot_folder(self):
        """
        \033[32m
        Makes a folder for plot files at the location of the datafile 
        with the name: "plots_" + filename\033[m
        """

        if not os.path.exists(self.plotfoldername):
            os.mkdir(self.plotfoldername)
            print("Created plot folder\n")
        else:
            pass
        
    def get_deut(self, timestep=0, which=None):
        temp = np.add(self.ion_densities_t_i_r[0][timestep, 0, :], self.ion_densities_t_i_r[0][timestep, 1, :])
        temp2 = np.add(self.ion_densities_t_i_r[0][timestep, 13, :], self.ion_densities_t_i_r[0][timestep, 14, :])
        temp3 = np.add(temp, temp2)
        
        if which == "inj":
            return temp
        elif which == "tot":
            return temp3
        elif which == "bg":
            return temp2
        else:
            return temp

    def check_dimensions_2D(self, data):
        """
        \033[32m
        Checks the dimensions of the data which the user wants to plot in 2D.
        Returns the data in proper shape for subsequent plotting.
        If something can be corrected (related to the dimensions),
        this method tries to correct it. (meaning cut or extend data by 1 row 
        to use cell edges instead of centers or vice versa)\033[m
        """

        try:
            shape = data.shape
        except:
            print("\033[31mERROR:\033[m data is not type numpy.ndarray\n")
            return 1

        if len(shape) != 2:
            print("\033[31mERROR:\033[m data is not 2D\n")
            return 2

        if shape[0] == self.timegrid.shape[0]:
            print("timegrid OK\n")
            if shape[1] == self.radialgrid.shape[0]:
                print("radialgrid OK\n")
                return data
            elif shape[1] == self.radialgrid.shape[0] + 1:
                print("radialgrid EDGES\n")
                return 9
        elif shape[0] == self.timegrid.shape[0] - 1:
            print("timegrid CENTERS\n")
            if shape[1] == self.radialgrid.shape[0]:
                print("radialgrid OK\n")
                corrected_data = np.append(data, [data[-1, :]], 0)
                print("duplicated the last row to be able to plot the data")
                return corrected_data
            elif shape[1] == self.radialgrid.shape[0] + 1:
                print("radialgrid EDGES\n")
                return 9

        else:
            print("\033[31mERROR:\033[m data is 2D but not in the correct dimensions\n")
            return 3

    # PLOTTING FUNCTIONS FOR HOTTAIL OR RUNAWAYGRID DISTRIBUTIONS #
    def plot_2D_momentum(self,figsize=[8, 5], at_timestep=0, at_radial_cell=0, ymin=None, ymax=None,
                         from_momentum_cell=0, to_momentum_cell=-1, save=False):
        """
        \033[32m
        Plots the 2D electron distribution on a normalized momentum grid 
        with pitch angles (in degrees) at a specified timestep and radial cell.
        Momentum cell boundaries for plotting can also be specified.
        Can be used on either a hottail or a runawaygrid.\033[m
        """

        fig = plt.figure(figsize=figsize)
        data_to_plot = self.distribution_t_r_xi_p[0][at_timestep, at_radial_cell, :,
                                                     from_momentum_cell:to_momentum_cell]

        if data_to_plot.min() < 1.0:
            vmin = 1.0
        else:
            vmin = data_to_plot.min()

        im = plt.pcolormesh(self.momentumgrid_edges[from_momentum_cell:to_momentum_cell], self.pitchgrid_degrees_edges,
                            data_to_plot, norm=colors.LogNorm(vmin=vmin, vmax=data_to_plot.max()))
        axes = plt.gca()
        axes.set_facecolor('black')
        axes.set_ylim(ymin, ymax)
        cbar = fig.colorbar(im, ax=axes)

        cbar.set_label(r"Electron distribution [1/m$^3$]")
        plt.title("Momentum and pitch distribution of electrons\nt=" 
                  + str(np.round(self.timegrid_ms[at_timestep], 3))
                  + "ms r=" + str(np.round(self.radialgrid[at_radial_cell], 3)) + "m")
        plt.ylabel("Pitch angle (degrees)")
        plt.xlabel(r"Momentum normalized to $m_ec$")

        if save:
            plt.savefig(self.plotfoldername + "2D_momentum_at_radius_" +
                        str(np.round(self.radialgrid[at_radial_cell], 3)) 
                        + "_at_time_" + str(np.round(self.timegrid_ms[at_timestep], 3)) 
                        + "ms.png", dpi=150)

    def plot_2D_energy(self, at_timestep=0, at_radial_cell=0, from_energy_cell=0, to_energy_cell=-1, save=False):
        """
        \033[32m
        Plots the 2D electron distribution on an energy grid with
        pitch angles (in degrees) at a specified timestep and radial cell.
        Energy cell boundaries for plotting can also be specified.
        Can be used on either a hottail or a runawaygrid.\033[m
        """

        fig = plt.figure(figsize=[15, 7])
        data_to_plot = self.distribution_t_r_xi_p[0][at_timestep, at_radial_cell, :, from_energy_cell:to_energy_cell]

        if data_to_plot.min() < 1.0:
            vmin = 1.0
        else:
            vmin = data_to_plot.min()

        im = plt.pcolormesh(self.energy_grid_electronvolts[from_energy_cell:to_energy_cell] / 1000, self.pitchgrid,
                            data_to_plot, norm=colors.LogNorm(vmin=vmin, vmax=data_to_plot.max()))
        axes = plt.gca()
        axes.set_facecolor('black')
        cbar = fig.colorbar(im, ax=axes)

        cbar.set_label(r"Electron distribution [1/m$^3$]")
        plt.title("Energy and pitch distribution of electrons")
        plt.ylabel("Pitch angle")
        plt.xlabel(r"Energy [keV]")

        if save:
            now = datetime.datetime.now()
            date_time = now.strftime("_%Y-%m-%d_%Hh-%Mm-%Ss")
            plt.savefig(self.plotfoldername + "2D_energy_at_radial_cell_" +
                        str(at_radial_cell) + "_at_timestep_" + str(at_timestep) +
                        date_time + ".png", dpi=150)

    def plot_angle_avg_momentum(self, at_timestep=0, at_radial_cell=0, from_momentum_cell=0, to_momentum_cell=-1,
                                save=False):
        """
        \033[32m
        1D angle averaged electron distribution on a normalized 
        momentum grid at a specified timestep and radial cell.
        Momentum cell boundaries for plotting can also be specified.
        Can be used on either a hottail or a runawaygrid.\033[m
        """

        plt.figure(figsize=[15, 7])
        data_to_plot = self.angle_avg_distr[0][at_timestep, at_radial_cell, from_momentum_cell:to_momentum_cell]
        plt.scatter(self.momentumgrid[from_momentum_cell:to_momentum_cell], data_to_plot)

        axes = plt.gca()
        axes.set_facecolor('white')
        axes.set_xticks(self.momentumgrid[from_momentum_cell:to_momentum_cell], minor=True)
        axes.set_axisbelow(True)
        axes.xaxis.grid(True, which='minor', linestyle="--")
        axes.yaxis.grid(True, which="both")
        axes.set_yscale("log")

        plt.title("Angle averaged momentum distribution")
        plt.ylabel(r"Electron distribution [1/m$^3$]")
        plt.xlabel(r"Momentum normalized to $m_ec$")

        if save:
            now = datetime.datetime.now()
            date_time = now.strftime("_%Y-%m-%d_%Hh-%Mm-%Ss")
            plt.savefig(self.plotfoldername + "angle_avg_momentum_at_radial_cell_" +
                        str(at_radial_cell) + "_at_timestep_" + str(at_timestep) +
                        date_time + ".png", dpi=150)

    def plot_momentum_at_specific_pitch(self, at_timestep=0, at_radial_cell=0,
                                        from_momentum_cell=0, to_momentum_cell=-1,
                                        pitch_cell=0, save=False):
        """
        \033[32m
        Scatter plot momentum distribution of electrons in a specific pitch cell,
        at a specific timestep and radial cell. 
        Momentum cell boundaries for plotting can also be specified.
        Can be used on either a hottail or a runawaygrid.\033[m
        """

        plt.figure(figsize=[15, 7])
        data_to_plot = self.distribution_t_r_xi_p[0][at_timestep, at_radial_cell, pitch_cell,
                                                     from_momentum_cell:to_momentum_cell]
        plt.scatter(self.momentumgrid[from_momentum_cell:to_momentum_cell], data_to_plot)

        axes = plt.gca()
        axes.set_facecolor('white')
        axes.set_xticks(self.momentumgrid[from_momentum_cell:to_momentum_cell], minor=True)
        axes.set_axisbelow(True)
        axes.xaxis.grid(True, which='minor', linestyle="--")
        axes.yaxis.grid(True, which="both")
        axes.set_yscale("log")

        plt.title("Momentum distribution at pitch cell #" + str(pitch_cell))
        plt.ylabel(r"Electron distribution [1/m$^3$]")
        plt.xlabel(r"Momentum normalized to $m_ec$")

        if save:
            now = datetime.datetime.now()
            date_time = now.strftime("_%Y-%m-%d_%Hh-%Mm-%Ss")
            plt.savefig(self.plotfoldername + "momentum_pitch_cell_" + str(pitch_cell) + "_at_radial_cell_" +
                        str(at_radial_cell) + "_at_timestep_" + str(at_timestep) + date_time + ".png", dpi=150)

    def plot_pitch_at_specific_momentum(self, at_timestep=0, at_radial_cell=0,
                                        momentum_cell=0, save=False):
        """
        \033[32m
        Scatter plot momentum distribution of electrons in a specific pitch cell,
        at a specific timestep and radial cell. 
        Momentum cell boundaries for plotting can also be specified.
        Can be used on either a hottail or a runawaygrid.\033[m
        """

        plt.figure(figsize=[15, 7])
        data_to_plot = self.distribution_t_r_xi_p[0][at_timestep, at_radial_cell, :, momentum_cell]
        plt.scatter(self.pitchgrid, data_to_plot)

        axes = plt.gca()
        axes.set_facecolor('white')
        axes.set_xticks(self.pitchgrid, minor=True)
        axes.set_axisbelow(True)
        axes.xaxis.grid(True, which='minor', linestyle="--")
        axes.yaxis.grid(True, which="both")
        axes.set_yscale("linear")

        plt.title("Pitch distribution at momentum cell #" + str(momentum_cell))
        plt.ylabel(r"Electron distribution [1/m$^3$]")
        plt.xlabel("Pitch angle")

        if save:
            now = datetime.datetime.now()
            date_time = now.strftime("_%Y-%m-%d_%Hh-%Mm-%Ss")
            plt.savefig(self.plotfoldername + "pitch_momentum_cell_" + str(momentum_cell) + "_at_radial_cell_" +
                        str(at_radial_cell) + "_at_timestep_" + str(at_timestep) +
                        date_time + ".png", dpi=150)    
    
    # GENERAL PLOTTING FUNCTIONS #
    def plot_2D(self, key, figsize=None, save=False, normalization=None, 
                datamin=None, datamax=None, logdiff=None, levels=None):
        """     
        \033[32m
        pcolormesh 2D plotting function for the whole time and radial grid without interpolation.
        Takes a dictionary key for the data to plot, use .list_attrs() method to get all keys.
        Check if the data you want to plot has the correct dimensions.
        Uses the above .check_dimensions_2D(self, data) method\033[m
        
        \033[31mIMPORTANT: \033[mIf a key is needed for a plotting function, pass it as a string.
        """

        if figsize is None:
            figsize = [8, 5]
            
        if normalization is None:
            normalization = "lin"
        
        data = self.__dict__[key][0]

        # cross-checking data dimensions with the timegrid and the radial grid
        data_to_plot = self.check_dimensions_2D(data)
        
        if type(data_to_plot) is not np.ndarray:
            return 1
        else:
            datamax = data_to_plot.max()
            datamin = data_to_plot.min()
            
        if logdiff is None:
            logdiff = math.log10(datamax) - 1

        sign = ""
        if datamax < 0:
            sign = "|"
            print("\033[31mThe displayed values are negative!\033[m")
            data_to_plot = abs(data_to_plot)
            datamax = data_to_plot.max()
        elif datamax > 0 and datamin < 0:
            print("Minimum data value is negative, maximum is positive.\n\033[31mChanged to linear plotting\033[m")
            normalization = "lin"
            
        # if levels is None:
        #     levels = 10

        fig = plt.figure(figsize=figsize)
        if normalization == "log":
            logmax = math.ceil(math.log10(datamax))
            # cmap = plt.cm.inferno  # define the colormap
            # # extract all colors from the map
            # cmaplist = [cmap(i) for i in range(cmap.N)]
            # # create the new map
            # cmap = matplotlib.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
            # # define the bins and normalize
            # bounds = np.logspace(0, logmax, levels)
            # norm = BoundaryNorm(bounds, cmap.N)
            print("log")
            datamin = 10 ** (math.log10(datamax) - logdiff)
            im = plt.pcolormesh(self.radialgrid, self.timegrid_ms, data_to_plot,
                                norm=colors.LogNorm(vmin=datamin, vmax=datamax))  # norm=norm, cmap=cmap
        elif normalization == "lin":
            print("lin")
            im = plt.pcolormesh(self.radialgrid, self.timegrid_ms, data_to_plot)
        else:
            print("\033[31mERROR:\033[m not valid normalization\n")

        axes = plt.gca()
        axes.set_facecolor('black')

        if normalization == "log":
            # l_f = LogFormatter(10, labelOnlyBase=True)
            cbar = fig.colorbar(im, ax=axes)
            # ticks = 10 ** np.linspace(0, logmax, logmax + 1)
            # cbar.set_ticks([])
            # cbar.set_ticks(ticks)
            # cbar.set_ticklabels(ticks)
        else:
            cbar = fig.colorbar(im, ax=axes)
            
        cbar.set_label(sign + " ".join(key.split("_")) + sign + " [" + self.__dict__[key][1] + "]")

        plt.title(" ".join(key.split("_")))
        plt.ylabel("time [ms]")
        plt.xlabel("Minor radius [m]")

        if save:
            now = datetime.datetime.now()
            date_time = now.strftime("_%Y-%m-%d_%Hh-%Mm-%Ss")
            plt.savefig(self.plotfoldername + "_" + key[0] + date_time + ".png", dpi=150)

    def plot_2D_pcolormesh(self, key, figsize=None, save=False):
        """     
        \033[32m
        2D plotting function for the whole time and radial grid without interpolation.
        Takes a dictionary key for the data to plot, use .list_attrs() method to get all keys.
        Check if the data you want to plot has the correct dimensions.
        Uses the above .check_dimensions_2D(self, data) method\033[m
        
        \033[31mIMPORTANT: \033[mIf a key is needed for a plotting function, pass it as a string.
        """

        if figsize is None:
            figsize = [8, 5]
        data = self.__dict__[key][0]

        # cross-checking data dimensions with the timegrid and the radial grid
        data_to_plot = self.check_dimensions_2D(data)
        if type(data_to_plot) is not np.ndarray:
            return 1
        else:
            datamax = data_to_plot.max()

        if datamax < 0:
            data_to_plot = abs(data_to_plot)
            datamax = data_to_plot.max()
            datamin = 10 ** (math.log10(datamax) - 10)
        # check minimum and maximum of data and decide on linear or logarithmic plotting

        fig = plt.figure(figsize=figsize)
        if abs(math.log10(datamax - datamin)) > 2.5:
            print("log")
            im = plt.pcolormesh(self.radialgrid, self.timegrid_ms, data_to_plot,
                                norm=colors.LogNorm(vmin=datamin, vmax=datamax))
        else:
            print("lin")
            im = plt.pcolormesh(self.radialgrid, self.timegrid_ms, data_to_plot)

        axes = plt.gca()
        axes.set_facecolor('black')

        cbar = fig.colorbar(im, ax=axes)
        cbar.set_label(" ".join(key.split("_")) + " [" + self.__dict__[key][1] + "]")

        plt.title(" ".join(key.split("_")))
        plt.ylabel("time [ms]")
        plt.xlabel("Minor radius [m]")

        if save:
            now = datetime.datetime.now()
            date_time = now.strftime("_%Y-%m-%d_%Hh-%Mm-%Ss")
            plt.savefig(self.plotfoldername + "_" + key[0] + date_time + ".png", dpi=150)        
    
    def plot_1D(self, key, figsize=None, save=False, yscale=None):
        """
        \033[32m
        1D plot of the data behind the key. Based on the dimensions 
        the x-axis is either the time or the radial grid.\033[m
        """

        if figsize is None:
            figsize = [15, 7]
        data_to_plot = self.__dict__[key][0]

        try:
            shape = data_to_plot.shape
        except:
            print("\033[31mERROR:\033[m data is not type numpy.ndarray\n")
            return 1

        if len(shape) != 1:
            print("\033[31mERROR:\033[m data is not 1D\n")
            return 2
        
        yscale = auto_yscale(yscale)
            
        plt.figure(figsize=figsize)

        if shape[0] == self.timegrid.shape[0]:
            plt.scatter(self.timegrid_ms, data_to_plot)
            plt.xlabel("time [ms]")
        elif shape[0] == self.timegrid.shape[0] - 1:
            data_to_plot = np.append(data_to_plot, [data_to_plot[-1]], 0)
            plt.scatter(self.timegrid_ms, data_to_plot)
            plt.xlabel("time [ms]")
        elif shape[0] == self.radialgrid.shape[0]:
            plt.scatter(self.radialgrid, data_to_plot)
            plt.xlabel("minor radius [m]")
        elif shape[1] == self.radialgrid.shape[0] + 1:
            print("radialgrid EDGES")
            return 9
        else:
            print("\033[31mERROR:\033[m data is 1D but not the correct length for the time or radial grid\n")
            return 3

        axes = plt.gca()
        axes.set_facecolor('white')
        axes.set_yscale(yscale)
        plt.grid()
        plt.title(" ".join(key.split("_")))
        plt.ylabel(" ".join(key.split("_")) + " [" + self.__dict__[key][1] + "]")

        if save:
            now = datetime.datetime.now()
            date_time = now.strftime("_%Y-%m-%d_%Hh-%Mm-%Ss")
            plt.savefig(self.plotfoldername + "_" + key[0] + date_time + ".png", dpi=150)
            
    def plot_1D_HU(self, key, figsize=None, save=False, yscale=None, ylabel=None):
        """
        \033[32m
        Hungarian 1D plot of the data behind the key. Based on the dimensions 
        the x-axis is either the time or the radial grid.\033[m
        """

        if figsize is None:
            figsize = [15, 7]
        data_to_plot = self.__dict__[key][0]

        try:
            shape = data_to_plot.shape
        except:
            print("\033[31mERROR:\033[m data is not type numpy.ndarray\n")
            return 1

        if len(shape) != 1:
            print("\033[31mERROR:\033[m data is not 1D\n")
            return 2
        
        yscale = auto_yscale(yscale)
        
        plt.figure(figsize=figsize)

        if shape[0] == self.timegrid.shape[0]:
            plt.scatter(self.timegrid_ms, data_to_plot)
            plt.xlabel("Idő [ms]")
        elif shape[0] == self.timegrid.shape[0] - 1:
            data_to_plot = np.append(data_to_plot, [data_to_plot[-1]], 0)
            plt.scatter(self.timegrid_ms, data_to_plot)
            plt.xlabel("Idő [ms]")
        elif shape[0] == self.radialgrid.shape[0]:
            plt.scatter(self.radialgrid, data_to_plot)
            plt.xlabel("Kissugár [m]")
        elif shape[1] == self.radialgrid.shape[0] + 1:
            print("radialgrid EDGES")
            return 9
        else:
            print("\033[31mERROR:\033[m data is 1D but not the correct length for the time or radial grid\n")
            return 3
        
        axes = plt.gca()
        axes.set_facecolor('white')
        axes.set_yscale(yscale)
        plt.grid()
        plt.ylabel(ylabel + " [" + self.__dict__[key][1] + "]")

        if save:
            now = datetime.datetime.now()
            date_time = now.strftime("_%Y-%m-%d_%Hh-%Mm-%Ss")
            plt.savefig(self.plotfoldername + "_" + key[0] + date_time + ".png", dpi=150)
            
    def plot_deut_HU(self, figsize=None, save=False, timestep=0):
        """
        \033[32m
        Hungarian 1D plot of the data behind the key. Based on the dimensions 
        the x-axis is either the time or the radial grid.\033[m
        """

        if figsize is None:
            figsize = [15, 7]
        data_to_plot = self.get_deut(timestep)
        
        plt.figure(figsize=figsize)

        # plt.scatter(self.radialgrid, data_to_plot)
        plt.plot(self.radialgrid, data_to_plot, drawstyle="steps-mid")
        plt.xlabel("Kissugár [m]")
        
        axes = plt.gca()
        axes.set_facecolor('white')
        axes.set_yscale("linear")
        plt.grid()
        plt.ylabel(r"Belőtt n$_D$ [m$^{-3}$]")

        if save:
            now = datetime.datetime.now()
            date_time = now.strftime("_%Y-%m-%d_%Hh-%Mm-%Ss")
            plt.savefig(self.plotfoldername + "_" + key[0] + date_time + ".png", dpi=150)

        
def open_files(txt_name):
    filenames = open(txt_name, "r").readlines()
    cdos = []
    dos = []
    for file in filenames:
        cdos.append(CustomDREAMOutput(file.rstrip()))
        dos.append(DREAMOutput(file.rstrip()))
        
    #print(cdos)
    return cdos, dos