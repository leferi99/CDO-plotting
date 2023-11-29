import numpy as np
import h5py
from DREAM.DREAMOutput import DREAMOutput


class Distribution:
    def __init__(self, filenames, start_time=0):        
        # Set up grids from the first file
        f = h5py.File(filenames[0], "r")

        self.radialgrid = f["grid/r"][()]
        self.radialgrid_edges = f["grid/r_f"][()]
        self.radialgrid_length = len(self.radialgrid)
        self.dr = f["grid/dr"][()]
        self.major_radius = f["settings/radialgrid/R0"][()]
        self.minor_radius = f["settings/radialgrid/a"][()]
        self.real_volumes_of_cells = f["grid/VpVol"][()]  # m^3

        # runawaygrid setting
        self.runawaygrid_enabled = f["settings/runawaygrid/enabled"][()]

        # RUNAWAYGRID
        if self.runawaygrid_enabled:
            # momentum and pitchgrid - runaway
            self.re_momentumgrid = f["grid/runaway/p1"][()]  # in m_e*c normalized momentum
            self.re_momentumgrid_edges = f["grid/runaway/p1_f"][()]  # in m_e*c normalized momentum
            # energy grid in MeV
            self.re_energygrid_MeV = ((np.sqrt(self.re_momentumgrid ** 2 + 1) - 1) * 0.51099895)
        
        f.close()
        
        # First for loop to determine needed timegrid length
        self.timegrid_length = 0
        for file in filenames:
            f = h5py.File(file, "r")
            self.timegrid_length = self.timegrid_length + len(f["grid/t"][()]) - 1
            f.close()
        
        # Space allocation for the concatenated object
        g_time = self.timegrid_length
        g_radii = self.radialgrid_length
        g_p_re = len(self.re_momentumgrid)
        
        self.timegrid_ms = np.zeros(g_time)
        self.f_re_avg_density = np.zeros((g_time, g_radii, g_p_re))
        self.f_re_avg = np.zeros((g_time, g_radii, g_p_re))
        self.n_re = np.zeros((g_time, g_radii))
        self.n_tot = np.zeros((g_time, g_radii))
        
        # Filling up the object with data from multiple files
        ti = 0
        rt = 0
        for file in filenames:
            f = h5py.File(file, "r")
            
            # 1D
            temptime = f["grid/t"][()][1:] * 1000
            end = len(temptime) + ti
            
            self.timegrid_ms[ti:end] = temptime + rt
            self.n_re[ti:end, :] = f["eqsys/n_re"][()][1:, :]
            self.n_tot[ti:end, :] = f["eqsys/n_tot"][()][1:, :]
            
            # Close the HDF5 file
            f.close()
            
            # DREAMOutput methods
            do = DREAMOutput(file)
            if self.runawaygrid_enabled:
                self.f_re_avg_density[ti:end, :, :] = do.eqsys.f_re.angleAveraged(moment="density")[1:, :, :]  # f*p**2
                self.f_re_avg[ti:end, :, :] = do.eqsys.f_re.angleAveraged()[1:, :, :] # moment="distribution" integrates over xi0
                
            do.close()
            
            # Increase the indices
            rt += temptime[-1]
            ti += len(temptime)

            """
            (avg_density / (p**2)) * (mc**2 *p) / sqrt(p**2 + 1)
            """
            
        
        
            
            


