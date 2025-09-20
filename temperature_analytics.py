import numpy as np
from scipy.special import spherical_jn
from scipy.special import spherical_yn


class SingleDipole:
    def __init__(self, radius, nback, diel_data, nTOT):
        """Defines the different system parameters.
        Keyword arguments:
        radius -- [cm] radius of NP
        n -- [unitless] refractive index of background
        diel_data -- [wavelengths in cm, n_re, n_im]
        nTOT -- [unitless] number of multipoles when calculating cross-section
        """
        self.radius = radius
        self.nback = nback
        self.nTOT = nTOT
        self.wave = diel_data[:, 0]  # cm
        self.n_re = diel_data[:, 1]
        self.n_im = diel_data[:, 2]
        self.numwave = len(self.wave)

    def psi(self, n, rho):
        return rho * spherical_jn(n, rho)

    def psi_prime(self, n, rho):
        term1 = spherical_jn(n, rho)
        term2 = rho * spherical_jn(n, rho, derivative=True)
        return term1 + term2

    def hankel(self, n, rho):
        return spherical_jn(n, rho) + 1j * spherical_yn(n, rho)

    def hankel_prime(self, n, rho):
        term1 = spherical_jn(n, rho, derivative=True)
        term2 = 1j * spherical_yn(n, rho, derivative=True)
        return term1 + term2

    def xi(self, n, rho):
        return rho * self.hankel(n, rho)

    def xi_prime(self, n, rho):
        return self.hankel(n, rho) + rho * self.hankel_prime(n, rho)

    def mie_coefficent(self, j):
        n_re = self.n_re
        n_im = self.n_im
        m = (n_re + 1j * n_im) / self.nback
        k = 2 * np.pi * self.nback / self.wave
        x = k * self.radius
        numer_a = (m * self.psi(j, m * x) * self.psi_prime(j, x)
                   - self.psi(j, x) * self.psi_prime(j, m * x))
        denom_a = (m * self.psi(j, m * x) * self.xi_prime(j, x)
                   - self.xi(j, x) * self.psi_prime(j, m * x))
        numer_b = (self.psi(j, m * x) * self.psi_prime(j, x)
                   - m * self.psi(j, x) * self.psi_prime(j, m * x))
        denom_b = (self.psi(j, m * x) * self.xi_prime(j, x)
                   - m * self.xi(j, x) * self.psi_prime(j, m * x))
        aj = numer_a / denom_a
        bj = numer_b / denom_b
        return aj, bj

    def cross_sects(self):
        a_j = np.zeros((self.nTOT, self.numwave), dtype=complex)
        b_j = np.zeros((self.nTOT, self.numwave), dtype=complex)
        ext_insum = np.zeros((self.nTOT, self.numwave))
        sca_insum = np.zeros((self.nTOT, self.numwave))
        for idx, j in enumerate(range(1, self.nTOT + 1)):
            a_j[idx, :], b_j[idx, :] = self.mie_coefficent(j=j)
            ext_insum[idx, :] = (2 * j + 1) * (
                np.real(a_j[idx, :] + b_j[idx, :]))
            sca_insum[idx, :] = (2 * j + 1) * (
                np.abs(a_j[idx, :])**2 + np.abs(b_j[idx, :])**2)
        k = 2 * np.pi * self.nback / self.wave
        C_ext = 2 * np.pi / (k**2) * np.sum(ext_insum, axis=0)
        C_sca = 2 * np.pi / (k**2) * np.sum(sca_insum, axis=0)
        C_abs = C_ext - C_sca
        return C_abs, C_sca  # cm


class CalculateTemperatures:
    def __init__(self, abs_cross, kappa, radius, P0):
        self.abs_cross = abs_cross  # um^2
        self.kappa = kappa  # W / (m K)
        self.radius = radius  # m
        self.P0 = P0  # W

    def intens_at_centerofbeam(self, which_inten, D=None, w0=None):
        if which_inten == 'uni_circular':
            I0 = 4 * self.P0 / (np.pi * (D)**2)
        if which_inten == 'gauss':
            I0 = 2 * self.P0 / (np.pi * w0**2)
        return I0   # W / m^2

    def P_j(self, abscross_j, I_j):
        return abscross_j * (1E-6)**2 * I_j  # W

    def T_j_singleNP(self, P_j):
        return P_j / (4 * np.pi * self.kappa * self.radius)  # C

    def hexagonal_array(self,
                        full_window,
                        array_size,
                        intpart_dist=None,
                        part_per_row=None):
        """Define locations of NPs from full window.

        """
        # Define full window to calculate temperatures.
        wind_range = np.arange(-full_window/2, full_window/2+1,
                                0.25)*1E-6
        x_all, y_all = np.meshgrid(wind_range, wind_range)
        r_all = np.column_stack((x_all.ravel(), y_all.ravel())) # [pnts, 2]

        # Define grid either by interparticle distance or parts per row
        if intpart_dist is None:
            intpart_dist = array_size / part_per_row
        # Note, grids must be approx. centered at origin
        intpart_height = 0.5 * np.sqrt(3) * intpart_dist
        x_nps = np.arange(-array_size / 2,
                           array_size / 2, intpart_dist)*1E-6
        y_nps = np.arange(-array_size / 2,
                           array_size / 2, intpart_height)*1E-6
        X, Y = np.meshgrid(x_nps, y_nps)  # rectangular grid
        for rowi in range(int(np.ceil(len(y_nps) / 2))):  # offset rows
            X[2 * rowi, :] = X[2 * rowi, :] + intpart_dist*1E-6 / 2
        r_allNPs = np.column_stack((X.ravel(), Y.ravel()))  # [num_part, 2]

        # Now, I'd like to find where r_allNPs is in r_all
        idx_allNPs = []
        for pt in r_allNPs:
            matches = np.where(np.all(np.isclose(r_all, pt, atol=1e-6), axis=1))[0]
            if len(matches) > 0:
                idx_allNPs.append(matches[0])  # or all matches if more than one
        idx_allNPs = np.array(idx_allNPs)
        if idx_allNPs.shape[0] != r_allNPs.shape[0]:
            print('ERROR: Did not find correct number of NPs.')
        return r_all, idx_allNPs, r_allNPs

    def all_abscross(self, j, r_all, idx_allNPs):
        abs_cross_all = r_all[:,0] * 0
        abs_cross_all[idx_allNPs] = self.abs_cross

        abs_cross_at_j = abs_cross_all[j]
        abs_cross_at_allk_neq_j = np.delete(abs_cross_all, j)
        return abs_cross_all, abs_cross_at_j, abs_cross_at_allk_neq_j

    # def uni_circ(self, r, j, thresh):
    #     # For uniform circular illumination ONLY
    #     weight = np.linalg.norm(r, axis=-1) - thresh / 2
    #     weight[weight > 0] = 0
    #     weight[weight < 0] = 1
    #     I0 = self.intens_at_centerofbeam(which_inten='uni_circular', D=thresh)
    #     I_at_j = I0 * weight[j]
    #     weight = np.delete(weight, j)
    #     I_at_allk_neq_j = weight * I0
    #     return I_at_j, I_at_allk_neq_j
#
    def gauss_inten(self, r, j, gauss_xy0, w0):
        I0 = self.intens_at_centerofbeam(which_inten='gauss', w0=w0)
        x0, y0 = gauss_xy0
        r_r0 = np.sqrt((r[:, 0] - x0)**2 + (r[:, 1] - y0)**2)
        I_r = I0 * np.exp(-2 * (r_r0)**2 / w0**2)
        I_at_j = I_r[j]
        I_at_allk_neq_j = np.delete(I_r, j)
        return I_r, I_at_j, I_at_allk_neq_j

    def T_tot_at_j(self, j, r, idx_allNPs, which_inten,
                   gauss_xy0=None, thresh=None, w0=None):
        # Total temperature calculated from array, just at point j.
        rj = r[j, :]
        rj_minus_rk = np.linalg.norm(r - rj, axis=-1)
        rj_minus_rk_jneqk = np.delete(rj_minus_rk, j)

        # Position dependent intensity of illumination
        # if which_inten == 'uni_circular':
        #     I_at_j, I_at_allk = self.uni_circ(r=r,
        #                                       j=j,
        #                                       thresh=thresh)
        if which_inten == 'gauss':
            _, I_at_j, I_at_allk = self.gauss_inten(r=r,
                                                    j=j,
                                                    gauss_xy0=gauss_xy0,
                                                    w0=w0)

        _, abs_at_j, abs_at_allk = self.all_abscross(j=j,
                                                     r_all=r,
                                                     idx_allNPs=idx_allNPs)

        # Self term
        P_j = self.P_j(abscross_j=abs_at_j,
                       I_j=I_at_j)
        T_self = self.T_j_singleNP(P_j)

        # Particles k neq j
        P_allk = self.P_j(abscross_j=abs_at_allk,
                          I_j=I_at_allk)
        T_ext = P_allk / (4 * np.pi * self.kappa * rj_minus_rk_jneqk)
        T_tot = T_self + np.sum(T_ext)
        return T_self, T_ext, T_tot







    # def T_int(self, kap_out, kap_in, I0, mie_or_dda, Cabs):
    #     ''' Temperature of a single sphere
    #     kap_out: thermal conductivity of background [W/(m*K)]
    #     kap_in: thermal conductivity of sphere [W/(m*K)]
    #     I0: incident intensity [W/m^2]
    #     '''
    #     r = 1E-9  # evaluate T at center of sphere
    #     if mie_or_dda == 'mie':
    #         Q = Cabs * (1 / 100)**2 * I0  # [W]



    #     radius_m = self.radius * (1 / 100)  # [m]
    #     T1 = Q / (4 * np.pi * kap_out * radius_m)
    #     T2 = Q / (8 * np.pi * kap_in * radius_m**3) * (radius_m**2 - r**2)
    #     T = T1 + T2
    #     return T, Q
