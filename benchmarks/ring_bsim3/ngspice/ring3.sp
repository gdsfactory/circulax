3-stage BSIM3v3 ring oscillator, VACASK c6288 0.35um, VDD=1.8V

.include "bsim3_models.inc"

.subckt inv in out vdd vss
  mn out in vss vss nmos w=2u l=350n ad=1p pd=2.5u as=1p ps=2.5u
  mp out in vdd vdd pmos w=4u l=350n ad=2p pd=4.5u as=2p ps=4.5u
.ends

i0 0 1 dc 0 pulse 0 100u 1n 1n 1n 1n
x1 1 2 vdd 0 inv
x2 2 3 vdd 0 inv
x3 3 1 vdd 0 inv

vdd vdd 0 1.8

.options method=trap
.options klu

.control
  tran 0.05n 1u 0 0.05n
  meas tran freq TRIG v(1) val=0.9 rise=3 TARG v(1) val=0.9 rise=5
  echo "FREQ_MHz=$&{freq/1e6}"
  quit
.endc

.end
