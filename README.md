# neural-analysis

Using Generealized Additive Models with neuroscience based priors to explore the relation between high-dimensional enviroments (multiple bats) to neural activity.
Interpretability is accomplished via:
- novel method of dealing with multicollinearity
- PDPs
- Shapley Values
<br> example: ![An example of PDPs](72_AlloModel_11111111111_0.png)
<hr> Code-Flow: ![A description of what the code is (supposedly) doing](CodeFlow.png)
<hr> <br>
Files and Directories:

.
<br>├── inputs/
<br>│   ├── behavior/
<br>│   │   ├── day_1.csv
<br>│   │   ├── ...
<br>│   │   └── day_k.csv
<br>│   ├── cells/
<br>│   │   ├── cell_1_day_1.csv
<br>│   │   ├── ...
<br>│   │   └── cell_n_day_k.csv
<br>│   ├── shuffles/
<br>│   │   ├── cell_1/
<br>│   │   │   ├── shuffle_0.csv
<br>│   │   │   ├── ...
<br>│   │   │   └── shuffle_100.csv
<br>│   │   ├── ...
<br>│   │   └── cell_n/
<br>│   │       └── ...
<br>│   └── subsets/
<br>│       ├── day_1/
<br>│       │   ├── Allo/
<br>│       │   │   ├── subset_00000000000.csv
<br>│       │   │   ├── ...
<br>│       │   │   └── subset_11111111111.csv
<br>│       │   └── Ego/
<br>│       │       ├── subset_00000000.csv
<br>│       │       ├── ...
<br>│       │       └── subset_11111111.csv
<br>│       └── day_k/
<br>│           └── ...
<br>├── notebooks
<br>├── scripts
<br>└── outputs/
<br>    ├── cell_1/
<br>    │   ├── Allo/
<br>    │   │   ├── Shuffles/
<br>    │   │   │   └── <see "output directory" format>
<br>    │   │   └── Shapley/
<br>    │   │       └── <see "output directory" format>
<br>    │   ├── Ego/
<br>    │   │   ├── Shuffles/
<br>    │   │   │   └── <see "output directory" format>
<br>    │   │   └── Shapley/
<br>    │   │       └── <see "output directory" format>
<br>    │   └── MegaPlot
<br>    ├── ...
<br>    └── cell_n/
<br>        └── ...

<!--
inputs
  behavior
    day_1.csv
    ...
    day_k.csv
  cells
    cell_1_day_1.csv
    ...
    cell_n_day_k.csv
  shuffles
    cell_1
      shuffle_0.csv
      ...
      shuffle_100.csv
    ...
    cell_n
      ...
  subsets
    day_1
      Allo
        subset_00000000000.csv
        ...
        subset_11111111111.csv
      Ego
        subset_00000000.csv
        ...
        subset_11111111.csv
    day_k
      ...
notebooks
scripts
outputs
  cell_1
    Allo
      Shuffles
        <see "output directory" format>
      Shapley
        <see "output directory" format>
    Ego
      Shuffles
        <see "output directory" format>
      Shapley
        <see "output directory" format>
    MegaPlot
  ...
  cell_n
    ...
-->