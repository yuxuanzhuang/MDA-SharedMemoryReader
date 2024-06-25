"""
SharedMemoryReader
SharedMemoryReader for MDAnalysis
"""

# Add imports here
from importlib.metadata import version

__version__ = version("sharedmemoryreader")

from .sharedmemoryreader import SharedMemoryReader, transfer_to_shared_memory