33 stage ring oscillator (generated)

.include "models.inc"

.subckt inverter in out vdd vss w=1u l=0.2u pfact=2
  xmp out in vdd vdd pmos w={w*pfact} l={l}
  xmn out in vss vss nmos w={w} l={l}
.ends

i0 0 1 dc 0 pulse 0 10u 1n 1n 1n 1n
xu1 1 2 vdd 0  inverter w={10u} l={1u}
xu2 2 3 vdd 0  inverter w={10u} l={1u}
xu3 3 4 vdd 0  inverter w={10u} l={1u}
xu4 4 5 vdd 0  inverter w={10u} l={1u}
xu5 5 6 vdd 0  inverter w={10u} l={1u}
xu6 6 7 vdd 0  inverter w={10u} l={1u}
xu7 7 8 vdd 0  inverter w={10u} l={1u}
xu8 8 9 vdd 0  inverter w={10u} l={1u}
xu9 9 10 vdd 0  inverter w={10u} l={1u}
xu10 10 11 vdd 0  inverter w={10u} l={1u}
xu11 11 12 vdd 0  inverter w={10u} l={1u}
xu12 12 13 vdd 0  inverter w={10u} l={1u}
xu13 13 14 vdd 0  inverter w={10u} l={1u}
xu14 14 15 vdd 0  inverter w={10u} l={1u}
xu15 15 16 vdd 0  inverter w={10u} l={1u}
xu16 16 17 vdd 0  inverter w={10u} l={1u}
xu17 17 18 vdd 0  inverter w={10u} l={1u}
xu18 18 19 vdd 0  inverter w={10u} l={1u}
xu19 19 20 vdd 0  inverter w={10u} l={1u}
xu20 20 21 vdd 0  inverter w={10u} l={1u}
xu21 21 22 vdd 0  inverter w={10u} l={1u}
xu22 22 23 vdd 0  inverter w={10u} l={1u}
xu23 23 24 vdd 0  inverter w={10u} l={1u}
xu24 24 25 vdd 0  inverter w={10u} l={1u}
xu25 25 26 vdd 0  inverter w={10u} l={1u}
xu26 26 27 vdd 0  inverter w={10u} l={1u}
xu27 27 28 vdd 0  inverter w={10u} l={1u}
xu28 28 29 vdd 0  inverter w={10u} l={1u}
xu29 29 30 vdd 0  inverter w={10u} l={1u}
xu30 30 31 vdd 0  inverter w={10u} l={1u}
xu31 31 32 vdd 0  inverter w={10u} l={1u}
xu32 32 33 vdd 0  inverter w={10u} l={1u}
xu33 33 1 vdd 0  inverter w={10u} l={1u}

vdd vdd 0 1.2

.options method=trap
.options klu

.control
  pre_osdi psp103v4.osdi
  tran 0.05n 1u 0 0.05n
  rusage all
  set
  set noaskquit
  quit
.endc

.end
