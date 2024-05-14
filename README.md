# Quantum Machine Control (QMC)

<!-- start intro -->

Python package for qubit experiments, based on Quantum Machines hardware.

<!-- end intro -->

## Installation

<!-- start installation -->

Install the latest version of QMC using pip:

```
Todo: package this project
```

<!-- end installation -->

## Usage

QMC has three main components:

- [[**ExperimentClass**]]: Base class used to talk to the hardware and run experiments. 

- [**AnalysisClass**]: Base class to analyze experimental data.

- [**DataLoggingClass**]: Base class to log data. Currently built into **ExperimentClass**.

And a few sub-components

- [**ExperimentClass_Labber**] An auxiliary class to set non-QM equipments (currently controlled through Labber).

- [**ExperimentClass_Octave**] An auxiliary class to directly set Octave (many through configuration, and is not implemented until the next qmm communication).


See the [api reference] for more information.

[**ExperimentClass**]: 
[**AnalysisClass**]: 
[api reference]: 
