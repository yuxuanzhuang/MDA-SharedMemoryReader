from multiprocessing.shared_memory import SharedMemory
from functools import cached_property
from MDAnalysis.coordinates.memory import MemoryReader 
import numpy as np
import os
import sys
import mmap
import ctypes
import logging
from MDAnalysis.lib.log import ProgressBar

class SharedMemoryReader(MemoryReader):
    """
    Shared memory reader for MDAnalysis trajectories.

    The trajectory data is stored in shared memory supported by
    the `multiprocessing.shared_memory.SharedMemory` object.
    The access is similar to the `MemoryReader` object, but
    the data is stored in shared memory and can be accessed
    by multiple processes.
    """
    format = 'SHARED_MEMORY'
    def __init__(self, coordinate_array, order='fac',
                 dimensions=None, dt=1, filename=None,
                 velocities=None, forces=None,
                 **kwargs):
        """
        Parameters
        ----------
        coordinate_array : numpy.ndarray
            The underlying array of coordinates. The MemoryReader now
            necessarily requires a np.ndarray
        order : {"afc", "acf", "caf", "fac", "fca", "cfa"} (optional)
            the order/shape of the return data array, corresponding
            to (a)tom, (f)rame, (c)oordinates all six combinations
            of 'a', 'f', 'c' are allowed ie "fac" - return array
            where the shape is (frame, number of atoms,
            coordinates).
        dimensions: [A, B, C, alpha, beta, gamma] (optional)
            unitcell dimensions (*A*, *B*, *C*, *alpha*, *beta*, *gamma*)
            lengths *A*, *B*, *C* are in the MDAnalysis length unit (Å), and
            angles are in degrees. An array of dimensions can be given,
            which must then be shape (nframes, 6)
        dt: float (optional)
            The time difference between frames (ps).  If :attr:`time`
            is set, then `dt` will be ignored.
        filename: string (optional)
            The name of the file from which this instance is created. Set to ``None``
            when created from an array
        velocities : numpy.ndarray (optional)
            Atom velocities.  Must match shape of coordinate_array.  Will share order
            with coordinates.
        forces : numpy.ndarray (optional)
            Atom forces.  Must match shape of coordinate_array  Will share order
            with coordinates
        """
        super().__init__(coordinate_array, order, dimensions, dt, filename,
                            velocities, forces, **kwargs)

    def set_array(self, coordinate_array, order='fac'):
        """
        Set underlying array in desired column order in shared memory.

        Parameters
        ----------
        coordinate_array : :class:`~numpy.ndarray` object
            The underlying array of coordinates
        order : {"afc", "acf", "caf", "fac", "fca", "cfa"} (optional)
            the order/shape of the return data array, corresponding
            to (a)tom, (f)rame, (c)oordinates all six combinations
            of 'a', 'f', 'c' are allowed ie "fac" - return array
            where the shape is (frame, number of atoms,
            coordinates).
        """
        # Only make copy if not already in float32 format
        self._coordinate_array = SharedMemoryArray(coordinate_array,
                                                   dtype='float32')
        self.stored_format = order

    @property
    def coordinate_array(self):
        """
        The underlying array of coordinates.
        """
        return self._coordinate_array.array
    
    @coordinate_array.setter
    def coordinate_array(self, value):
        """
        Set the underlying array of coordinates.
        """
        self.set_array(value)

    @property
    def velocity_array(self):
        """
        The underlying array of velocities.
        """
        if self._velocity_array is None:
            return None
        else:
            return self._velocity_array.array

    @velocity_array.setter
    def velocity_array(self, value):
        if value is None:
            self._velocity_array = None
        else:
            self._velocity_array = SharedMemoryArray(value)

    @property
    def force_array(self):
        """
        The underlying array of forces.
        """
        if self._force_array is None:
            return None
        else:
            return self._force_array.array

    @force_array.setter
    def force_array(self, value):
        if value is None:
            self._force_array = None
        else:
            self._force_array = SharedMemoryArray(value)


class SharedMemoryArray(object):
    """
    A shared memory array that can be pickled and unpickled.
    """

    def __init__(self, array, dtype=None):
        """
        Parameters
        ----------
        array : :class:`~numpy.ndarray` object
            The array to be shared in memory
        dtype : :class:`~numpy.dtype` object (optional)
            The data type of the array. If not provided, the data type
            of the input array will be used.
        """
        self.shared_memory = SharedMemory(create=True, size=array.nbytes)
        self.shape = array.shape
        if dtype is None:
            self.dtype = array.dtype
        else:
            self.dtype = dtype
        self.array[:] = array

    def __getstate__(self):
        return self.dtype, self.shape, self.shared_memory.name

    def __setstate__(self, state):
        self.dtype, self.shape, name = state
        self.shared_memory = SharedMemory(name=name)

    @cached_property
    def array(self):
        return np.ndarray(self.shape, dtype=self.dtype, buffer=self.shared_memory.buf)

    def __del__(self):
        self.shared_memory.close()

    def copy(self):
        # create a new SharedMemoryArray with a copy of the data
        return SharedMemoryArray(self.array.copy())


def transfer_to_shared_memory(universe,
                        start=None, stop=None, step=None,
                        verbose=False, **kwargs):
    """Transfer the trajectory to in shared memory representation.

    Replaces the current trajectory reader object with one of type
    :class:`SharedMemoryReader` to support in-place
    editing of coordinates.

    Parameters
    ----------
    universe: :class:`Universe`
        The Universe object
    start: int, optional
        start reading from the nth frame.
    stop: int, optional
        read upto and excluding the nth frame.
    step: int, optional
        Read in every nth frame. [1]
    verbose: bool, optional
        Will print the progress of loading trajectory to memory, if
        set to True. Default value is False.
    """
    if not isinstance(universe.trajectory, SharedMemoryReader):
        n_frames = len(range(
            *universe.trajectory.check_slice_indices(start, stop, step)
        ))
        n_atoms = len(universe.atoms)
        coordinates = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)
        ts = universe.trajectory.ts
        has_vels = ts.has_velocities
        has_fors = ts.has_forces
        has_dims = ts.dimensions is not None

        velocities = np.zeros_like(coordinates) if has_vels else None
        forces = np.zeros_like(coordinates) if has_fors else None
        dimensions = (np.zeros((n_frames, 6), dtype=np.float32)
                        if has_dims else None)

        for i, ts in enumerate(ProgressBar(universe.trajectory[start:stop:step],
                                            verbose=verbose,
                                            desc="Loading frames")):
            np.copyto(coordinates[i], ts.positions)
            if has_vels:
                np.copyto(velocities[i], ts.velocities)
            if has_fors:
                np.copyto(forces[i], ts.forces)
            if has_dims:
                np.copyto(dimensions[i], ts.dimensions)

        # Overwrite trajectory in universe with an SharedMemoryReader
        # object, to provide fast access and allow the coordinates to be
        # accessed by multiple processes.
        if step is None:
            step = 1
        universe.trajectory = SharedMemoryReader(
            coordinates,
            dimensions=dimensions,
            dt=universe.trajectory.ts.dt * step,
            filename=universe.trajectory.filename,
            velocities=velocities,
            forces=forces, **kwargs)