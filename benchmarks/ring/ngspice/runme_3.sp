3 stage ring oscillator (generated)

.include "models.inc"

.subckt inverter in out vdd vss w=1u l=0.2u pfact=2
  xmp out in vdd vdd pmos w={w*pfact} l={l}
  xmn out in vss vss nmos w={w} l={l}
.ends

i0 0 1 dc 0 pulse 0 10u 1n 1n 1n 1n
xu1 1 2 vdd 0  inverter w={10u} l={1u}
xu2 2 3 vdd 0  inverter w={10u} l={1u}
xu3 3 1 vdd 0  inverter w={10u} l={1u}

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
