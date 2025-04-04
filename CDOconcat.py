import numpy as np
import h5py
from bisect import bisect_left
from DREAM.DREAMOutput import DREAMOutput


class CustomConcat:
    def __init__(self, filenames):        
        mec2 = 510998.95  # eV
        # Set up grids from the last file - deliberately changed from first to last
        # in case there was a change in DREAM settings along the simulation
        f = h5py.File(filenames[-1], "r")

        self.radialgrid = np.asarray(f["grid/r"])
        self.radialgrid_edges = np.asarray(f["grid/r_f"])
        self.radialgrid_length = len(self.radialgrid)
        self.radial_step = np.asarray(f["grid/dr"])[1]
        try:
            self.major_radius = np.asarray(f["settings/radialgrid/R0"])
        except:
            print("Cylindrical grid")
        self.minor_radius = np.asarray(f["settings/radialgrid/a"])
        
        # Other constant parameters
        # Number of ion states, ion metadata and volumes
        self.ion_states = np.asarray(f["eqsys/n_i"]).shape[1]
        self.ion_Z = np.asarray(f["ionmeta/Z"])  # atomic number of the ions
        self.ion_names = np.asarray(f["ionmeta/names"])  # user set names for ions
        self.real_volumes_of_cells = np.asarray(f["grid/VpVol"]) * self.radial_step * self.major_radius  # m^3

        # hottail and runawaygrid settings
        self.hottailgrid_enabled = np.asarray(f["settings/hottailgrid/enabled"])
        self.runawaygrid_enabled = np.asarray(f["settings/runawaygrid/enabled"])
        
        if self.hottailgrid_enabled:
            # momentum and pitchgrid - hottail
            self.hot_momentumgrid = np.asarray(f["grid/hottail/p1"])  # in m_e*c normalized momentum
            self.hot_momentumgrid_edges = np.asarray(f["grid/hottail/p1_f"])  # in m_e*c normalized momentum - edges
            self.hot_pitchgrid = np.asarray(f["grid/hottail/p2"])  # cosine of pitch angle - cell centers
            self.hot_pitchgrid_edges = np.asarray(f["grid/hottail/p2_f"])  # cosine of pitch angle - cell edges
            self.hot_pitchgrid_rad = np.arccos(self.hot_pitchgrid)  # pitch angle in radians - cell centers
            self.hot_pitchgrid_rad_edges = np.arccos(self.hot_pitchgrid_edges)  # pitch angle in radians - cell edges
            self.hot_pitchgrid_degrees = 180 * self.hot_pitchgrid_rad / np.pi  # pitch degrees - centers
            self.hot_pitchgrid_degrees_edges = 180 * self.hot_pitchgrid_rad_edges / np.pi  # pitch degrees - edges
            self.hottailgrid_dimensions = (len(self.hot_momentumgrid), len(self.hot_pitchgrid))

        # RUNAWAYGRID
        if self.runawaygrid_enabled:
            # momentum and pitchgrid - runaway
            self.re_momentumgrid = np.asarray(f["grid/runaway/p1"])  # in m_e*c normalized momentum
            self.re_momentumgrid_edges = np.asarray(f["grid/runaway/p1_f"])  # in m_e*c normalized momentum
            self.re_pitchgrid = np.asarray(f["grid/runaway/p2"])  # cosine of pitch angle - cell centers
            self.re_pitchgrid_edges = np.asarray(f["grid/runaway/p2_f"])  # cosine of pitch angle - cell edges
            self.re_pitchgrid_rad = np.arccos(self.re_pitchgrid)  # pitch angle in radians - cell centers
            self.re_pitchgrid_rad_edges = np.arccos(self.re_pitchgrid_edges)  # pitch angle in radians - cell edges
            self.re_pitchgrid_degrees = 180 * self.re_pitchgrid_rad / np.pi  # pitch angle in degreess - cell centers
            self.re_pitchgrid_degrees_edges = 180 * self.re_pitchgrid_rad_edges / np.pi # pitch angle in degrees - cell edges
            self.runawaygrid_dimensions = (len(self.re_momentumgrid), len(self.re_pitchgrid))

            # energy grid from the momentum grid - runaway
            self.re_energy_grid_electronvolts = ((np.sqrt(self.re_momentumgrid ** 2 + 1) - 1) * mec2) + mec2
            
            # common momentumgrid
            self.full_momentumgrid = np.concatenate((self.hot_momentumgrid, self.re_momentumgrid))
        
        f.close()
        
        # First for loop to determine needed timegrid length
        self.timegrid_length = 0
        for file in filenames:
            f = h5py.File(file, "r")
            self.timegrid_length = self.timegrid_length + len(np.asarray(f["grid/t"])) - 1
            # Because of this previous -1 we will have throw away 
            # the first timestep data for almost everything
            f.close()  # NOTE
        
        # Space allocation for the concatenated object
        g_time = self.timegrid_length
        g_radii = self.radialgrid_length
        g_ions = self.ion_states
        
        if self.hottailgrid_enabled:
            g_p_hot = self.hottailgrid_dimensions[0]
            g_xi_hot = self.hottailgrid_dimensions[1]
        else:
            g_p_hot = None  # to avoid being unbound
            g_xi_hot = None

        if self.runawaygrid_enabled:    
            g_p_re = self.runawaygrid_dimensions[0]
            g_xi_re = self.runawaygrid_dimensions[1]
        else:
            g_p_re = None
            g_xi_re = None
        
        self.timegrid = np.zeros(g_time)
        self.timegrid_ms = np.zeros(g_time)
        self.E_field = np.zeros((g_time, g_radii))
        self.Ectot = np.zeros((g_time, g_radii))
        self.Ecfree = np.zeros((g_time, g_radii))
        self.I_p = np.zeros(g_time)
        self.I_wall = np.zeros(g_time)
        self.T_cold = np.zeros((g_time, g_radii))
        self.V_loop_w = np.zeros(g_time)
        self.W_cold = np.zeros((g_time, g_radii))
        self.W_cold_sum = np.zeros(g_time)
        
        # Additional data from other/fluid
        self.gammaAva = np.zeros((g_time, g_radii))
        self.Tcold_fhot_coll = np.zeros((g_time, g_radii))
        self.Tcold_fre_coll = np.zeros((g_time, g_radii))
        self.Tcold_ohmic = np.zeros((g_time, g_radii))
        self.Tcold_radiation = np.zeros((g_time, g_radii))
        self.Tcold_transport = np.zeros((g_time, g_radii))
        self.W_hot = np.zeros((g_time, g_radii))
        self.W_re = np.zeros((g_time, g_radii))
        self.Wcold_Tcold_Drr = np.zeros((g_time, g_radii + 1))
        self.gammaCompton = np.zeros((g_time, g_radii))
        self.gammaDreicer = np.zeros((g_time, g_radii))
        self.gammaHot = np.zeros((g_time, g_radii))
        self.gammaTritium = np.zeros((g_time, g_radii))
        self.runawayRate = np.zeros((g_time, g_radii))
        self.flux_to_RE = np.zeros((g_time, g_radii))
        
        # NOTE DREAMOutput class has to be used to get some integrated currents (I_hot, I_re, I_ohm)
        # and f_re_avg
        if g_p_hot is not None and g_xi_hot is not None:
            self.f_hot = np.zeros((g_time, g_radii, g_xi_hot, g_p_hot))
            self.j_hot = np.zeros((g_time, g_radii))
            self.I_hot = np.zeros(g_time)
            self.n_hot = np.zeros((g_time, g_radii))
            
        if g_p_re is not None and g_xi_re is not None:
            self.f_re = np.zeros((g_time, g_radii, g_xi_re, g_p_re))
            self.f_re_avg = np.zeros((g_time, g_radii, g_p_re))
            self.f_re_density = np.zeros((g_time, g_radii, g_p_re))

        if (g_p_hot is not None) and (g_p_re is not None):
            # put together the hottailgrid and the runawaygrid
            self.f_full_avg = np.zeros((g_time, g_radii, g_p_hot + g_p_re))
            
        self.j_re = np.zeros((g_time, g_radii))
        self.I_re = np.zeros(g_time)
        self.j_ohm = np.zeros((g_time, g_radii))
        self.I_ohm = np.zeros(g_time)
        self.j_tot = np.zeros((g_time, g_radii))
        self.n_cold = np.zeros((g_time, g_radii))
        self.n_i = np.zeros((g_time, g_ions, g_radii))
        self.avg_Ar_ionization = np.zeros((g_time, g_radii))
        self.n_re = np.zeros((g_time, g_radii))
        self.n_tot = np.zeros((g_time, g_radii))
        
        # NOTE for these next parameters there is no need to throw away data in time
        self.Eceff = np.zeros((g_time, g_radii))
        self.Zeff = np.zeros((g_time, g_radii))
        self.conductivity = np.zeros((g_time, g_radii))
        self.pCrit = np.zeros((g_time, g_radii))
        
        # Filling up the object with data from multiple files
        ti = 0
        rt = 0
        for file in filenames:
            print(file, end="\r")
            f = h5py.File(file, "r")
            
            # 1D
            temptime = np.asarray(f["grid/t"])[1:]
            end = len(temptime) + ti
            self.timegrid[ti:end] = temptime + rt  
            self.timegrid_ms[ti:end] = self.timegrid[ti:end] * 1000
            self.I_p[ti:end] = np.asarray(f["eqsys/I_p"])[1:, 0]
            self.I_wall[ti:end] = np.asarray(f["eqsys/I_wall"])[1:, 0]
            self.V_loop_w[ti:end] = np.asarray(f["eqsys/V_loop_w"])[1:, 0]
            
            # 2D
            self.E_field[ti:end, :] = np.asarray(f["eqsys/E_field"])[1:, :]
            self.Ectot[ti:end, :] = np.asarray(f["other/fluid/Ectot"])[:, :]
            self.Ecfree[ti:end, :] = np.asarray(f["other/fluid/Ecfree"])[:, :]
            self.T_cold[ti:end, :] = np.asarray(f["eqsys/T_cold"])[1:, :]
            self.W_cold[ti:end, :] = np.asarray(f["eqsys/W_cold"])[1:, :]  # cold electron energy density

            for i in range(ti, end):
                self.W_cold_sum[i] = 0
                for j in range(g_radii):
                    self.W_cold_sum[i] += self.W_cold[i, j] * self.real_volumes_of_cells[j]

            if self.hottailgrid_enabled:
                self.j_hot[ti:end, :] = np.asarray(f["eqsys/j_hot"])[1:, :]
                self.n_hot[ti:end, :] = np.asarray(f["eqsys/n_hot"])[1:, :]
                self.W_hot[ti:end, :] = np.asarray(f["other/fluid/W_hot"])

            self.j_re[ti:end, :] = np.asarray(f["eqsys/j_re"])[1:, :]
            self.j_ohm[ti:end, :] = np.asarray(f["eqsys/j_ohm"])[1:, :]
            self.j_tot[ti:end, :] = np.asarray(f["eqsys/j_tot"])[1:, :]
            self.n_cold[ti:end, :] = np.asarray(f["eqsys/n_cold"])[1:, :]
            
            self.n_re[ti:end, :] = np.asarray(f["eqsys/n_re"])[1:, :]
            self.n_tot[ti:end, :] = np.asarray(f["eqsys/n_tot"])[1:, :]
            
            self.Eceff[ti:end, :] = np.asarray(f["other/fluid/Eceff"])
            self.Zeff[ti:end, :] = np.asarray(f["other/fluid/Zeff"])
            self.conductivity[ti:end, :] = np.asarray(f["other/fluid/conductivity"])
            self.pCrit[ti:end, :] = np.asarray(f["other/fluid/pCrit"])
            
            self.gammaAva[ti:end, :] = np.asarray(f["other/fluid/GammaAva"])
            
            self.Tcold_ohmic[ti:end, :] = np.asarray(f["other/fluid/Tcold_ohmic"])
            self.Tcold_radiation[ti:end, :] = np.asarray(f["other/fluid/Tcold_radiation"])
            try:
                self.Tcold_transport[ti:end, :] = np.asarray(f["other/fluid/Tcold_transport"])
                self.Wcold_Tcold_Drr[ti:end, :] = np.asarray(f["other/fluid/Wcold_Tcold_Drr"])
            except:
                print("Tcold_transport + Wcold_Tcold_Drr not loaded")
            
            self.W_re[ti:end, :] = np.asarray(f["other/fluid/W_re"])
            self.gammaCompton[ti:end, :] = np.asarray(f["other/fluid/gammaCompton"])
            self.gammaDreicer[ti:end, :] = np.asarray(f["other/fluid/gammaDreicer"])

            try:
                self.gammaHot[ti:end, :] = np.asarray(f["other/fluid/gammaHottail"])
            except:
                print("Explicit hottail generation (gammaHot) not loaded.")
            
            self.gammaTritium[ti:end, :] = np.asarray(f["other/fluid/gammaTritium"])
            self.runawayRate[ti:end, :] = np.asarray(f["other/fluid/runawayRate"])
            self.n_i[ti:end, :, :] = np.asarray(f["eqsys/n_i"])[1:, :, :]

            if self.hottailgrid_enabled:
                self.Tcold_fhot_coll[ti:end, :] = np.asarray(f["other/fluid/Tcold_fhot_coll"])
                try:
                    self.f_hot[ti:end, :, :, :] = np.asarray(f["eqsys/f_hot"])[1:, :, :, :]  # time; radii; pitch; momentum
                except:
                    print("Hottail distribution not loaded from file", file, "\n There might be a change in hottailgrid settings between the last output file and this one.")

            if self.runawaygrid_enabled:
                self.Tcold_fre_coll[ti:end, :] = np.asarray(f["other/fluid/Tcold_fre_coll"])
                try:
                    self.f_re[ti:end, :, :, :] = np.asarray(f["eqsys/f_re"])[1:, :, :, :]  # time; radii; pitch; momentum
                except:
                    print("Runaway distribution not loaded from file", file, "\n There might be a change in runawaygrid settings between the last output file and this one.")
            
            # Close the HDF5 file
            f.close()
            
            # Finish with the DREAMOutput methods
            do = DREAMOutput(file)
            if self.hottailgrid_enabled:
                self.I_hot[ti:end] = do.eqsys.j_hot.current()[1:]
            self.I_re[ti:end] = do.eqsys.j_re.current()[1:]
            self.I_ohm[ti:end] = do.eqsys.j_ohm.current()[1:]
            if self.runawaygrid_enabled:
                try:
                    self.f_re_avg[ti:end, :, :] = do.eqsys.f_re.angleAveraged()[1:, :, :]
                    self.f_full_avg[ti:end, :, g_p_hot:] = self.f_re_avg[ti:end, :, :]
                    self.f_re_density[ti:end, :, :] = do.eqsys.f_re.angleAveraged(moment="density")[1:, :, :]
                except:
                    pass

                try:
                    self.f_full_avg[ti:end, :, :g_p_hot] = do.eqsys.f_hot.angleAveraged()[1:, :, :]
                except:
                    pass
                
                
            do.close()
            
            # Increase the indices
            rt += temptime[-1]
            ti += len(temptime)
            
        self.flux_to_RE = self.runawayRate - self.n_re * self.gammaAva - self.gammaTritium - self.gammaCompton
        self.density_D_0 = self.n_i[:, 0, :]
        self.density_D_1 = self.n_i[:, 1, :]
        self.density_T_0 = self.n_i[:, 2, :]
        self.density_T_1 = self.n_i[:, 3, :]
        self.density_Ar = self.n_i[:, 4:, :]
        temp1 = 0  # weighted sum
        temp2 = 0  # simple sum of all argon atoms
        if self.density_Ar.size > 0:
            for i in range(g_time):
                for j in range(19):
                    temp1 += j * self.density_Ar[i, j, :]
                    temp2 += self.density_Ar[i, j, :]

                self.avg_Ar_ionization[i] = temp1 / temp2
                
        print("\n")

    def get_data_at_time_ms(self, time, key):
        """Returns the data closest to the given time for the specified key."""
        idx = bisect_left(self.timegrid_ms, time)

        if hasattr(self, key):
            if getattr(self, key).shape[0] == self.timegrid_length:
                return getattr(self, key)[idx]  # Return the corresponding data at that index
            else:
                raise AttributeError(f"Attribute '{key}' does not have a time dimension.")
        else:
            raise KeyError(f"Key '{key}' not found in data.")
    
    def info(self):
        """
        \033[32m
        Basic information about time and radial resolution and
        if the hottail or runaway grids are enabled\033[m
        """
        print("Simulation time from " + str(round(self.timegrid[0] * 1000, 4)) +" ms to "
              + str(round(self.timegrid[-1] * 1000, 4)) + " ms")
        print("Minor radius: " + str(self.minor_radius) + " m")
        print("Number of radial grid cells: " + str(len(self.radialgrid)))
        print("Radial step (dr): " + str(self.radial_step) + " m")
        
        # HOTTAILGRID
        print("Hottailgrid enabled: " + str(bool(self.hottailgrid_enabled)))
        if self.hottailgrid_enabled:
            print("Resolution: " + str(self.hottailgrid_dimensions))
        
        # RUNAWAYGRID
        print("Runawaygrid enabled: " + str(bool(self.runawaygrid_enabled)))
        if self.runawaygrid_enabled:
            print("Resolution: " + str(self.runawaygrid_dimensions))
            
class CustomConcatTimeOnly:
    def __init__(self, filenames, start_time=0):
        self.timegrid_length = 0
        for file in filenames:
            f = h5py.File(file, "r")
            self.timegrid_length = self.timegrid_length + len(np.asarray(f["grid/t"])) - 1
            # Because of this previous -1 we will have throw away 
            # the first timestep data for almost everything
            f.close()  # NOTE
        
        g_time = self.timegrid_length
        
        self.timegrid = np.zeros(g_time)
        self.timegrid_ms = np.zeros(g_time)
        ti = 0
        rt = start_time
        for file in filenames:
            f = h5py.File(file, "r")
            
            temptime = np.asarray(f["grid/t"])[1:]
            end = len(temptime) + ti
            self.timegrid[ti:end] = temptime + rt  
            self.timegrid_ms[ti:end] = self.timegrid[ti:end] * 1000
 
            f.close()
            
            # Increase the indices
            rt += temptime[-1]
            print("File: " + file + " Time: " + str(rt * 1e3) + " ms", end="\n")
            ti += len(temptime)

