import numpy as np
from numpy import nan

from scipy.interpolate import interp2d


from pyfme.aircrafts.aircraft import Aircraft
from pyfme.utils.coordinates import wind2body


# from StackOverflow
def pad(data):
    bad_indexes = np.isnan(data)
    good_indexes = np.logical_not(bad_indexes)
    good_data = data[good_indexes]
    interpolated = np.interp(bad_indexes.nonzero()[0], good_indexes.nonzero()[0], good_data)
    data[bad_indexes] = interpolated
    return data


class MyAircraft(Aircraft):

    def __init__(self):
        super().__init__()

        # Mass & Inertia
        self.mass = 4200  # kg
        self.inertia = np.diag([
            14932.18,
            42147.14,
            30376.11
        ])

        # Geometry
        self.Sw = 37.66  # m2
        self.chord = 7.84  # m

        # Thr, t coefficients
        self.T = 6500  # N

        heights = np.array([
            0.2, 0.25, 0.3, 0.42, 0.5, 0.6, 1.0, 1.5, 10.0
        ])

        pitch_angles = np.deg2rad(
            np.array([-4, -2, 0, 2, 4, 6, 8, 10, 12])
        )

        CD_data = np.array([
            [.0275, .0269, .0304, .0383, .0511, .0696, .0952, .1304, .1806],
            [.0275, .027, .0306, .0384, .0506, .0678, .0903, .1192, .1559],
            [.0275, .0271, nan, nan, nan, nan, nan, .1146, .1469],
            [.0275, .0272, nan, nan, nan, nan, nan, .1109, .1398],
            [.0275, .0273, nan, nan, nan, nan, nan, .1095, .1372],
            [.0274, .0273, nan, nan, nan, nan, nan, .1088, .136],
            [.0274, .0274, nan, nan, nan, nan, nan, .1083, .1348],
            [.0274, .0274, .0313, .0392, .0508, .0663, .0855, .1084, .1349],
            [.0274, .0274, .0314, .0392, .051, .0366, .0859, .109, .1356]
        ])

        CL_data = np.array([
            [-.0677, .1491, .3701, .5957, .8274, 1.0673, 1.3188, 1.5873, 1.8849],
            [-.0563, .1507, .3603, .5727, .7882, 1.0075, 1.2315, 1.4619, 1.701],
            [-.0493, .1511, nan, nan, nan, nan, nan, 1.3943, 1.611],
            [-.0418, .1502, nan, nan, nan, nan, nan, 1.3187, 1.515],
            [-.0382, .1487, nan, nan, nan, nan, nan, 1.2763, 1.4632],
            [-.0364, .1472, nan, nan, nan, nan, nan, 1.2492, 1.4307],
            [-.0344, .01432, nan, nan, nan, nan, nan, 1.2008, 1.3734],
            [-.0341, .1411, .3161, .4909, .6651, .8385, 1.0109, 1.1821, 1.3517],
            [-.0341, .1389, .3117, .4842, .6561, .8272, .9972, 1.1661, 1.3335]
        ])

        Cm_data = np.array([
            [.0597, .0511, .039, .023, .0021, -.0255, -.0627, -.1143, -.19],
            [.0615, .0512, .0389, .0239, .0056, -.0168, -.0448, -.0803, -.1264],
            [.0619, .0513, nan, nan, nan, nan, nan, -.0621, -.0964],
            [.0621, .0515, nan, nan, nan, nan, nan, -.042, -.0654],
            [.0621, .0515, nan, nan, nan, nan, nan, -.0309, -.0493],
            [.0619, .0516, nan, nan, nan, nan, nan, -.0239, -.0396],
            [.0615, .0517, nan, nan, nan, nan, nan, -.012, -.0234],
            [.0613, .0518, .0421, .0323, .0224, .0124, .0023, -.0079, -.0181],
            [.0612, .0519, .0425, .033, .0235, .014, .0044, -.0051, -.0147]
        ])

        np.apply_along_axis(pad, 0, CD_data)
        np.apply_along_axis(pad, 0, CL_data)
        np.apply_along_axis(pad, 0, Cm_data)

        self.CD_values = self._interp(pitch_angles, heights, CD_data)
        self.CL_values = self._interp(pitch_angles, heights, CL_data)
        self.Cm_values = self._interp(pitch_angles, heights, Cm_data)

        # CONTROLS
        self.controls = {'delta_elevator': 0,
                         'delta_aileron': 0,
                         'delta_rudder': 0,
                         'delta_t': 0}

    @staticmethod
    def _interp(x, y, data):
        xx, yy = np.meshgrid(x, y)
        f = interp2d(xx, yy, data, kind='linear')
        return f

    def relative_height(self, state):
        return state.position.height / self.chord

    def _calculate_aero_lon_forces_moments_coeffs(self, state):
        pitch = state.attitude.theta
        rel_height = self.relative_height(state)

        self.CD = self.CD_values(pitch, rel_height)[0]
        self.CL = self.CL_values(pitch, rel_height)[0]
        self.Cm = self.Cm_values(pitch, rel_height)[0]

    def _calculate_aero_lat_forces_moments_coeffs(self, state):
        self.CY = 0.0
        self.Cl = 0.0
        self.Cn = 0.0

    def _calculate_aero_forces_moments(self, state):
        q = self.q_inf
        Sw = self.Sw
        c = self.chord
        b = self.span

        self._calculate_aero_lon_forces_moments_coeffs(state)
        self._calculate_aero_lat_forces_moments_coeffs(state)

        D = q * Sw * self.CD
        Y = q * Sw * self.CY
        L = q * Sw * self.CL
        l = q * Sw * b * self.Cl
        m = q * Sw * c * self.Cm
        n = q * Sw * b * self.Cn

        return D, Y, L, l, m, n

    def _calculate_thrust_forces_moments(self, environment):
        # delta_t = self.controls['delta_t']
        # rho = environment.rho
        #
        # T = 0.5 * rho * self.Ct * delta_t  # N

        T = self.T

        # We will consider that the engine is aligned along the OX (body) axis
        Ft = np.array([T, 0, 0])

        return Ft

    def calculate_forces_and_moments(self, state, environment, controls):
        # Update controls and aerodynamics
        # super().calculate_forces_and_moments(state, environment, controls)
        self._calculate_aerodynamics(state, environment)

        Ft = self._calculate_thrust_forces_moments(environment)

        D, Y, L, l, m, n = self._calculate_aero_forces_moments(state)
        Fg = environment.gravity_vector * self.mass

        Fa_wind = np.array([-D, Y, -L])
        Fa_body = wind2body(Fa_wind, self.alpha, self.beta)
        Fa = Fa_body

        self.total_forces = Ft + Fg + Fa
        self.total_moments = np.array([l, m, n])

        return self.total_forces, self.total_moments
