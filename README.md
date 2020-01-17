# EEGnet-Wolf

This project implements the EEGnet on Mr. Wolf. It was created during a Semester Thesis.

## Requirements

- [PULP-SDK](https://github.com/pulp-platform/pulp-sdk "PULP-SDK repository")
- [PULP-DSP](https://github.com/pulp-platform/pulp-dsp "PULP-DSP repository")

Both the SDK (including the [PULP-RISCV toolchain](https://github.com/pulp-platform/pulp-riscv-gnu-toolchain "PULP RISCV Toolchain Repository")) and the DSP library needs to be build and installed.

## Usage

This program requires the quantized EEGnet. For this, visit [QuantLab-RPR](https://iis-git.ee.ethz.ch/sctibor/quantlab-rpr "Quantlab-RPR including EEGNet")
1. Setup [QuantLab-RPR](https://iis-git.ee.ethz.ch/sctibor/quantlab-rpr "Quantlab-RPR including EEGNet").
2. In QuantLab, execute `python export-net-data.py --exp_id xxx --train`. This trains and quantizes the network, and exports the necessary data to into the folder `export`.
3. Copy all files (`net.npz`, `input.npz`, `verification.npz`, `config.json`) from `[quantlab_root]/export/` into this project at `[project_root]/data/`.
4. Run `run.sh` to generate the necessary header files and run the code.
