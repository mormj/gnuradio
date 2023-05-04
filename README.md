<p align="center">
<img src="https://github.com/gnuradio/gnuradio/blob/main/docs/gnuradio.png" width="75%" />
</p>

The `fair-graph` branch is where GR 4.0 development will be taking place utilizing
the `graph-prototype` library developed by the team at FAIR (https://github.com/fair-acc/graph-prototype).
It is expected that API and functionality will change rapidly until 4.0 is released

GNU Radio is a free & open-source software development toolkit that 
provides signal processing blocks to implement software radios. It can 
be used with readily-available, low-cost external RF hardware to create 
software-defined radios, or without hardware in a simulation-like 
environment. It is widely used in hobbyist, academic, and commercial 
environments to support both wireless communications research and real-world 
radio systems.

## Building

```
meson setup build
cd build
ninja
```
