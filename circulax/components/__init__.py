# Copyright (c) 2026 Circulax contributors
"""Reusable electronic, photonic and fitted circuit models."""

from circulax.components.angelov import Angelov
from circulax.components.angelov_gan import AngelovGaN
from circulax.components.bipolar import BJT_PNP, BJT_PNP_Dynamic
from circulax.components.comparator import Comparator
from circulax.components.diode_cmc import DIODE_CMC
from circulax.components.ekv import EKV26
from circulax.components.electronic import CoupledInductors, DifferentialAdapter, IdealTransformer, TransmissionLine
from circulax.components.filters import Bessel3, Biquad, LowPass1
from circulax.components.juncap import JUNCAP200
from circulax.components.modulators import (
    QuadratureCombiner,
    electrode_netlist,
    loaded_group_index,
    lumped_mzm,
    mach_zehnder,
    optical_netlist,
    traveling_wave_mzm,
)
from circulax.components.mosfet import PMOSDynamic
from circulax.components.mosvar import MOSVAR
from circulax.components.opamp import OpAmp
from circulax.components.optical_filters import OpticalFilter
from circulax.components.optical_passive import (
    NonlinearOpticalWaveguide,
    OpticalAttenuator,
    OpticalInterface,
    OpticalPhaseShifter,
    OpticalTermination,
)
from circulax.components.phase_shifters import ElectricalPhaseShifter, PNJunctionPhaseShifter, PNPhaseShifter, loaded_phase_shifter
from circulax.components.photodetectors import (
    CarrierPhotodiode,
    IdealPhotodetector,
    OpticalCW,
    OpticalGaussianPulse,
    Photodiode,
    make_photodetector_subcircuit,
)
from circulax.components.photonic import OpticalDelayLine, delay_line_fdomain
from circulax.components.r2 import R2CMC, R2ETCMC, r2_thermal_subcircuit
from circulax.components.r3 import R3_CMC
from circulax.components.resonators import (
    AddDropRing,
    AllPassRing,
    AllPassRingEO,
    RingModSetup,
    RingModulator,
    RingModulatorEO,
    loaded_ring,
)
from circulax.components.thermo_optic import JouleHeater, ThermalPhaseShifter, thermo_optic_phase_shifter
from circulax.components.transmission_line import RLGCTransmissionLine, mode_transmission_line, rlgc_transmission_line

__all__ = [
    "BJT_PNP",
    "DIODE_CMC",
    "EKV26",
    "JUNCAP200",
    "MOSVAR",
    "R2CMC",
    "R2ETCMC",
    "R3_CMC",
    "AddDropRing",
    "AllPassRing",
    "AllPassRingEO",
    "Angelov",
    "AngelovGaN",
    "BJT_PNP_Dynamic",
    "Bessel3",
    "Biquad",
    "CarrierPhotodiode",
    "Comparator",
    "CoupledInductors",
    "DifferentialAdapter",
    "ElectricalPhaseShifter",
    "IdealPhotodetector",
    "IdealTransformer",
    "JouleHeater",
    "LowPass1",
    "NonlinearOpticalWaveguide",
    "OpAmp",
    "OpticalAttenuator",
    "OpticalCW",
    "OpticalDelayLine",
    "OpticalFilter",
    "OpticalGaussianPulse",
    "OpticalInterface",
    "OpticalPhaseShifter",
    "OpticalTermination",
    "PMOSDynamic",
    "PNJunctionPhaseShifter",
    "PNPhaseShifter",
    "Photodiode",
    "QuadratureCombiner",
    "RLGCTransmissionLine",
    "RingModSetup",
    "RingModulator",
    "RingModulatorEO",
    "ThermalPhaseShifter",
    "TransmissionLine",
    "delay_line_fdomain",
    "electrode_netlist",
    "loaded_group_index",
    "loaded_phase_shifter",
    "loaded_ring",
    "lumped_mzm",
    "mach_zehnder",
    "make_photodetector_subcircuit",
    "mode_transmission_line",
    "optical_netlist",
    "r2_thermal_subcircuit",
    "rlgc_transmission_line",
    "thermo_optic_phase_shifter",
    "traveling_wave_mzm",
]
