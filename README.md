# GOES-SOSMAG-Mag-Subtraction

Visualize and analyze GOES and SOSMAG magnetic field data for space weather
research

# Usage

- Run `main.py` from root with argparse for paths
- spacecraft are optional, but at lease ONE must be provided
- to get help with argparse inputs, give:

 ```commandline
python .\src\main.py -h
```

example for 8-01-22:

```commandline
python .\src\main.py --g17-file .\data\08\dn_magn-l2-avg1m_g17_d20220801_v2-0-2.nc --g18-file .\data\08\dn_magn-l2-avg1m_g18_d20220801_v2-0-2.nc --gk2a-file .\data\08\SOSMAG_20220801_b_gse.nc
```

- Optional args for longitudinal degrees to add local noon/midnight times to
  plots.

- example:

```commandline
python .\src\main.py --gk2a-file .\data\08\SOSMAG_20220801_b_gse.nc --g17-file .\data\08\dn_magn-l2-avg1m_g17_d20220801_v2-0-2.nc --g17-deg 105 --gk2a-deg 128.2
```

# Unit tests

- must be run from ./test/unit with:

```commandline
python -m unittest .\test_data_loader.py
```