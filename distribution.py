import numpy as np
import h5py
import scipy.constants
from DREAM.DREAMOutput import DREAMOutput


class Distribution:
    def __init__(self, filenames, start_time=0.):        
        # Set up grids from the first file
        f = h5py.File(filenames[0], "r")
        m_e = scipy.constants.m_e
        c = scipy.constants.c
        mec2 = 510998.95  # eV

        self.radialgrid = np.asarray(f["grid/r"])
        self.radialgrid_edges = np.asarray(f["grid/r_f"])
        self.radialgrid_length = len(self.radialgrid)
        self.dr = np.asarray(f["grid/dr"])
        self.major_radius = np.asarray(f["settings/radialgrid/R0"])
        self.minor_radius = np.asarray(f["settings/radialgrid/a"])
        self.VpVol = np.asarray(f["grid/VpVol"])
        self.hottail_Vprime = np.asarray(f["grid/hottail/Vprime"])
        self.hottail_Vprime_VpVol = np.divide(self.hottail_Vprime, self.VpVol[:, None, None])
        self.hottail_momentumgrid = np.asarray(f["grid/hottail/p1"])
        self.hottail_pitchgrid_edges = np.asarray(f["grid/hottail/p2_f"])
        self.hottail_dxi = self.hottail_pitchgrid_edges[1:] - self.hottail_pitchgrid_edges[:-1]
        self.real_volumes_of_cells = self.VpVol * self.dr * self.major_radius

        # runawaygrid setting
        self.runawaygrid_enabled = np.asarray(f["settings/runawaygrid/enabled"])

        # RUNAWAYGRID
        if self.runawaygrid_enabled:
            self.re_Vprime = np.asarray(f["grid/runaway/Vprime"])
            self.re_Vprime_VpVol = np.divide(self.re_Vprime, self.VpVol[:, None, None])
            # momentum and pitchgrid - runaway
            self.re_momentumgrid = np.asarray(f["grid/runaway/p1"]) # in m_e*c normalized momentum
            self.re_pitchgrid = np.asarray(f["grid/runaway/p2"])
            self.re_pitchgrid_edges = np.asarray(f["grid/runaway/p2_f"])
            self.re_momentumgrid_edges = np.asarray(f["grid/runaway/p1_f"])  # in m_e*c normalized momentum
            self.re_dxi = self.re_pitchgrid_edges[1:] - self.re_pitchgrid_edges[:-1]
            # energy grid
            self.re_energygrid_eV = ((np.sqrt(self.re_momentumgrid ** 2 + 1) - 1) * mec2) + mec2
            self.re_energygrid_Joule = ((np.sqrt(self.re_momentumgrid ** 2 + 1) - 1) * m_e * c ** 2) + m_e * c ** 2
            self.re_energygrid_edges_Joule = ((np.sqrt(self.re_momentumgrid_edges ** 2 + 1) - 1) * m_e * c ** 2)
            self.velocity_grid = (self.re_momentumgrid * c)/np.sqrt(self.re_momentumgrid ** 2 + 1)
            self.momentum_with_KE = np.sqrt((self.re_energygrid_Joule ** 2 / c ** 2) + self.re_energygrid_Joule * m_e)
        
        f.close()
        
        # First for loop to determine needed timegrid length
        self.timegrid_length = 0
        for file in filenames:
            f = h5py.File(file, "r")
            self.timegrid_length = self.timegrid_length + len(np.asarray(f["grid/t"])) - 1
            f.close()
        
        # Space allocation for the concatenated object
        g_time = self.timegrid_length
        g_radii = self.radialgrid_length
        g_p_re = len(self.re_momentumgrid)
        g_xi_re = len(self.re_pitchgrid)
        
        self.timegrid_ms = np.zeros(g_time)
        self.f_re_avg_density = np.zeros((g_time, g_radii, g_p_re))
        self.f_re_density = np.zeros((g_time, g_radii, g_xi_re, g_p_re))
        self.f_re_avg = np.zeros((g_time, g_radii, g_p_re))
        self.f_re_current_density = np.zeros((g_time, g_radii, g_p_re))
        self.n_re = np.zeros((g_time, g_radii))
        self.j_re = np.zeros((g_time, g_radii))
        self.I_re = np.zeros((g_time))
        self.n_tot = np.zeros((g_time, g_radii))
        
        # Filling up the object with data from multiple files
        ti = 0
        rt = start_time
        for file in filenames:
            f = h5py.File(file, "r")
            do = DREAMOutput(file)
            
            # 1D
            temptime = np.asarray(f["grid/t"])[1:] * 1000
            end = len(temptime) + ti
            
            self.timegrid_ms[ti:end] = temptime + rt
            self.n_re[ti:end, :] = np.asarray(f["eqsys/n_re"])[1:, :]  # n_re is the runaway electron density
            self.j_re[ti:end, :] = np.asarray(f["eqsys/j_re"])[1:, :]  # j_re is the runaway electron current density 
            self.I_re[ti:end] = do.eqsys.j_re.current()[1:] # type: ignore # I_re is the total runaway current 
            self.n_tot[ti:end, :] = np.asarray(f["eqsys/n_tot"])[1:, :] # n_tot is the total electron density 
            
            # Close the HDF5 file
            f.close()
            
            # DREAMOutput methods
            do = DREAMOutput(file)
            if self.runawaygrid_enabled:
                self.f_re_avg_density[ti:end, :, :] = do.eqsys.f_re.angleAveraged(moment="density")[1:, :, :]  # type: ignore # this is supposedly dn/(drdp)
                self.f_re_density[ti:end, :, :, :] = do.eqsys.f_re[1:, :, :, :]  # type: ignore # this is supposedly dn/(drdpdxi)
                self.f_re_avg[ti:end, :, :] = do.eqsys.f_re.angleAveraged()[1:, :, :] # type: ignore # moment="distribution" integrates over xi0
                # Current density for comparison
                self.f_re_current_density[ti:end, :, :] = do.eqsys.f_re.angleAveraged(moment="current")[1:, :, :] # type: ignore
                
            do.close()
            
            # Increase the indices
            rt += temptime[-1]
            ti += len(temptime)

        # Calculate dn/(dr*dE) 
        self.dnOverdrdE = self.f_re_avg_density * (
            self.re_energygrid_Joule / (
                m_e * c ** 2 * np.sqrt(
                    self.re_energygrid_Joule ** 2 - (m_e * c ** 2) ** 2
                    )
                )
            )
        self.dnOverdr = np.zeros((g_time, g_radii))
        self.alternate_current = np.zeros((g_time, g_radii))
        for i in range(g_p_re):
            self.dnOverdr += (self.re_energygrid_edges_Joule[i+1] - self.re_energygrid_edges_Joule[i]) * self.dnOverdrdE[:, :, i]
            self.alternate_current += (self.re_energygrid_edges_Joule[i+1] - self.re_energygrid_edges_Joule[i]) * self.velocity_grid[i] * self.dnOverdrdE[:, :, i] * scipy.constants.e

        

