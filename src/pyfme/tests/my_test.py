import numpy as np
import matplotlib.pyplot as plt

from pyfme.environment.environment import Environment
from pyfme.environment.atmosphere import ISA1976
from pyfme.environment.gravity import VerticalConstant
from pyfme.environment.wind import NoWind

from pyfme.simulator import Simulation

from pyfme.models.state import EarthPosition, AircraftState, \
                               EulerAttitude, BodyVelocity

from pyfme.aircrafts.my_aircraft import MyAircraft
from pyfme.models import EulerFlatEarth

from pyfme.utils.input_generator import Constant

if __name__ == '__main__':
    aircraft = MyAircraft()

    atmosphere = ISA1976()
    gravity = VerticalConstant()
    wind = NoWind()

    environment = Environment(atmosphere, gravity, wind)

    position = EarthPosition(
        x=0,
        y=0,
        height=1.568
    )
    attitude = EulerAttitude(
        theta=np.deg2rad(6.2),
        phi=0,
        psi=0
    )
    velocity = BodyVelocity(
        u=40.8 * 1.5,
        v=0,
        w=0,
        attitude=attitude
    )

    full_state = AircraftState(position, attitude, velocity)

    system = EulerFlatEarth(t0=0, full_state=full_state)

    controls = {
        'delta_elevator': Constant(0),
        'delta_aileron': Constant(0),
        'delta_rudder': Constant(0),
        'delta_t': Constant(0),
    }

    t_end = 1000

    simulation = Simulation(aircraft, system, environment, controls)
    results = simulation.run(t_end)

    kwargs = {
        'marker': None,
        'sharex': True,  # same x axis in case subplots
    }

    data_columns = [
        ['x_earth', 'y_earth', 'z_earth'],
        ['psi', 'theta', 'phi'],
        # ['v_north', 'v_east', 'v_down'],
        # ['alpha', 'beta', 'TAS'],
        ['u', 'v', 'w'],
        # ['Fx', 'Fy', 'Fz'],
        # ['Mx', 'My', 'Mz']
    ]

    colors = ['red', 'green', 'blue']

    fig, axes = plt.subplots(nrows=3, ncols=len(data_columns))

    for idx, data in enumerate(data_columns):
        for idy, (column, color) in enumerate(zip(data, colors)):
            ax = axes[idy, idx]
            ax.axhline(color='black')
            results.plot(y=column, ax=ax, color=color, **kwargs)
            ax.grid()

    plt.subplots_adjust(left=0.05, bottom=0.05,
                        right=0.95, top=0.95, hspace=0.1)
    plt.show()
