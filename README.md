# low-freq-real-time-proc
This is a temporary package for low-frequency processing of a spool of distributed acoustic sensing (DAS) data using true low-pass filtering alongside rolling mean processing. Additionally, it facilitates real-time DAS data processing, enabling either low-frequency or rolling mean processing directly on site.

This repository may be merged into the [SpoolProcessing](https://github.com/DASDAE/SpoolProcessing) package. It uses [DASCore](https://dascore.org/) library and the ```lf_das.py``` script, and it is the modified version of Dr. Ge Jin's [DASLowFreqProcessing](https://github.com/DASDAE/DASLowFreqProcessing).

Examples are available through Jupyter notebooks.

Tested using DASCore v0.0.13

**Deprecation note:**

This package is deprecated and no longer maintained. All of its functionality has been integrated into DASCore. Instructions can be found under DASCore recipes linked below:

[Low-Frequency Processing](https://dascore.org/recipes/low_freq_proc.html)

[Real-Time Processing](https://dascore.org/recipes/real_time_proc.html)
