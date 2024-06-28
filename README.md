
# IRGames: An algorithm for two-player repeated games with imperfect public monitoring

  

  

IRGames is designed to compute equilibrium payoff sets for two-player discounted repeated games with imperfect public monitoring and public randomization. It provides an upper bound on perfect public equilibrium payoffs (PPE).

  

  

## Key Features

  

  

-  **Computation of PPE payoffs**: Advanced mathematical models and computational techniques used to characterize the set of PPE payoff pairs.

  

  

  

-  **Support for imperfect public monitoring**: Tailored to games with imperfect public monitoring.

  

  

  

-  **Public randomization**: Integrates mechanisms that allow for a wider range of equilibria.

  

  

  

- **Two-player games**: Full functionality limited to games with two players.

  

  

  

- **Visualization**: Provides visual representations of the payoff set.

  

  

  

- **User-friendly**: Easy definition of game parameters and result visualization.

  

  

  

- **Extensive documentation**: Accompanied by a scientific article and detailed usage guidelines.

  

  

  

-  **Open source**: Users can modify and enhance the software as needed.

  

  

  

-  **Computational efficiency:**  Optimized for games with up to three actions per player and four signals.

  

  

  

- **Platform Compatibility:**  Developed and tested on Linux and macOS.

  

  

  

  

## Installation

  

  

  

Before you install and run IRGames, ensure you have Python 3.x installed. Check your Python version by typing:

  

  

  

`python3 --version`

  

  

  

in your command prompt or terminal, or:

  

  

  

`python --version`

  

  

  

into the command line interface. If the output of the terminal is not

  

  

  

  

`Python 3.x.x`,

  

  

  

  

install it by following the guide on the Python website:

  

  

  

  

  

`https://www.python.org/downloads/`.

  

  

  

  

## Dependencies

  

  

  

  

Following modules and packages need to be installed to run the code:

  

  

  

  

`numpy (1.22.4)`; no requirements needed

  

  

  

  

`matplotlib (3.5.1)`; requires cycler, fonttools, kiwisolver, numpy, packaging, pillow, pyparsing, python-dateutil

  

  

  

  

`pycddlib (2.1.4)`; no requirements needed

  

  

  

  

`scipy (1.8.0)`; requires numpy

  

  

  

  

`sympy (1.9)`; requires: mpmath

  

  

  

  

`shapley (1.0.3)`; requires: numba, numpy, scipy, six

  

  

  

  

`openpyxl (3.0.10)`; requires: et-xmlfile

  

  

  

  

`nashpy (0.0.32)`; requires: numpy, scipy

  

  

  

  

To install modules and packages for mac OS, type the following commands in the terminal:

  

  

  

  

`brew install package-name`

  

  

  

  

or for Linux and Windows Subsystem for Linux 2 (WSL 2):

  

  

  

  

`pip3 install package-name`.

  

  

  

  

## Setup

  

  

  

  

Here is how to get ready to run the calculations:

  

  

  

  

1. **Script**: Download the script from the repo.

  

  

  

  

2. **Input folder**: Create a directory named `Input_folder`,  in the script's location and download `input_data.xlsm` from the repository to this folder.

  

  

  

  

3. **Output path**: Define a working directory for saving outputs. If unspecified, files are saved in the current directory.

  

  

  

  

4. **Output folders**: Ensure folders `Output_vertices`, `Output_halfspaces`, and `Output_png` exist post-run or create them manually.

  

  

  

  

5. **Verification**: Run `python3 irgames.py -verify` to check proper installation and functioning. If not called, the data will be automatically retrieved from the Excel file.

  

  

  

  

## Usage

  

  

  

  

### Basic Example

  

  

  

  

Run the algorithm with the following command:

  

  

  

  

`python3 irgames.py`

  

  

  

  

The algorithm can accept multiple inputs necessary to define the game. The verification example below is an infinitely repeated prisoners' dilemma  with the stage game payoff matrix as follows:

  

  

  

  

  


  
|                | Player 2: C | Player 2: D |
|----------------|-------------|-------------|
| **Player 1: C**| 2, 2        | -1, 3       |
| **Player 1: D**| 3, -1       | 0, 0        |
  

  

  

Players share a common discount factor δ, expressed as a fraction. In the example, the discount factor is 9/10.

  

  

  

  

The game is one of imperfect public monitoring, with two signals: the good signal y_1 and the bad signal y_2. The signal structure is captured by the following matrix:

  | *Action Combination* | *Good Signal (y_1)* | *Bad Signal (y_2)* |
|----------------------|---------------------|--------------------|
| **CC**               | 2/3                 | 1/3                |
| **CD**               | 1/2                 | 1/2                |
| **DC**               | 1/2                 | 1/2                |
| **DD**               | 1/4                 | 3/4                |

  
  

  

This matrix shows the probabilities of each signal being realized for each possible combination of the player's actions.

  

  

  

  

### Expanded Usage

  

  

  

  

The algorithm is adjustable to user preferences. Possible adjustments are in the list of arguments that can be accessed as follows:

  

  

  

  

`python3 irgames.py -h`

  

  

  

  

or

  

  

  

  

`python3 irgames.py --help`

  

  

  

  

Below is the list of optional arguments:

  

  

  

- `-h`, `--help`: Shows the help message and exits.

  

  

  

- `-verify`: Verifies proper instalation, the code will run the hard-coded example incorporated in the script.

  

  

  

- `-output`: Directory where the output files should be created. If not provided the output path will automatically be set in the location of the current working directory.

  

  

  

- `-number_type {float,fraction}`: The decision of how fast and accurate the script should work. The user can opt for `float` (fast) or `fraction` (accurate) number type. Default: `float`.

  

  

  

- `-difference`: Pre-defined error bound between polytope areas. Default = 0.005.

  

  

  

- `-rounds`: Enter the number of rounds you would like the algorithm to compute. Default: 2.

  

  

- `-mp`: Specifies the number of processes for parallel processing (parallel computing). If opted to use the maximum amount of computational resources on the machine, type `-mp 0`. By default, the algorithm uses a single process.

  

  

  

- `-output_vertices`: Saves the output in the text format in the `Output_vertices` folder; increases computation time, but makes the availability of the output after the script finishes. Additionally, it saves the H-representation of all iterations. If disabled, the tool will only save the final round. Default: off. Call `-output_vertices` to enable it.

  

  

  

- `-s_from`: Continues iterating from the specified (and previously completed)  round. Specify with an integer what round you wish to continue the analysis from.

  

  

  

- `-log`: Log of activities: the payoff matrix, the signal structure, the discount factor, number of rounds, computational time needed (depends on values that have been used: float or fraction), etc. Default: off. Call  `-log` to turn on.

  

  

  

- `-plot`: If chosen gives the possibility to plot the desired round. Specify the integer of the round to be visualized.

  

  

  

- `-sp`: Simplifying polytopes using the Ramer-Douglas-Peucker (RDP) line simplification algorithm. Suggested error round, the epsilon value = 0.000001.

  

  

  

  

### Input from excel

  
The payoff matrix, signal structure and the discount factor are retrieved from the `input_data.xlsm`  uploaded to this repository. The Excel file uses macros, that can be either enabled or disabled. If enabled, macros will automatically generate matrices for the stage game payoffs and probability distributions. Either way, the algorithm will retrieve the information from the excel file necessary for further computations.

  
The fields in the Excel file should be filled out as indicated in the file. After the script is called, all necessary information is retrieved. The user should make sure all entries are in the number format. Otherwise, errors might occur.

  

  

### Number Type

  

  

  

`-number_type {fraction, float} `

  

  

  

  

It is possible to choose between fast or intense computation. The script is set to fast computation (floating-point numbers) by default. In that case, accuracy is traded for speed. If the user opts for precision (intense computation), choose fraction:

  

  

  

  

`python3 irgames.py -number_type fraction`

  

  

  

  

### Polytope Area Difference

  

  

  

  

`-difference`

  

  

  

  

An error bound has been pre-defined to check for convergence of polytopes (the difference is set to 0.005 by default). This error bound can be adjusted to meet the accuracy requirements:

  

  

  

  

`python3 irgames.py -difference 0.02`

  

  

  

  

### Number of rounds

  

  

  

  

Specify the preferred number of rounds using the flag

  

  

  

  

`-rounds`

  

  

  

  

By default, the number of rounds is set to 2 and can be changed as follows:

  

  

  

  

`python3 irgames.py -rounds 5`

  

  

  

  

### Parallel Computing

  

  

  

  

`-mp`

  

  

  

In this algorithm, the computation of continuation payoff sets for each action profile is distributed across different cores. Upon completion of all tasks, the outcomes are aggregated and used as input for the subsequent round.

  

  

  

The current version of the algorithm available in the repository includes a multiprocessing feature, but further optimizations are planned. Note that there may be issues with the current multiprocessing implementation on operating systems other than Linux.

  

  

  

`python3 irgames.py -mp`

  

  

  

By default, multiprocessing is disabled, and the algorithm will use a single process to perform computations. To specify the number of processes, enter the following command instead:

  

  

  

  

`python3 irgames.py -mp some_integer`

  

  

  

  

Replace `some_integer` with the number of processes to be used. If `some_integer` is  zero, the algorithm will use the maximum number of processes available.

  

  

  

  

### Save vertices to text

  

  

  

  

` -output_vertices`

  

  

  

  

allows to save the extreme points in text format. This option is off by default to prevent a potential increase in computation time. To enable it, use the following command in the terminal:

  

  

  

  

`python3 irgames.py -output_vertices`

  

  

  

  

Thus, the output is saved in text format. If enabled, the half-space representations are also saved in text format. This is important if we want to visualize desired round or continue refining the set from the desired round. Note that even if the text output option is disabled, the last round of vertices is still saved automatically.

  

  

  

  

### Continue from round n

  

  

  

  

`-s_from`

  

  

  

  

Allows to continue refining the set from a selected (previously completed) round n. The half-space representation of each iteration is automatically saved to  `Output_halfspaces` (if `output_vertices` is enabled) such that the output is used as input in future iterations if the set is iterated from a round n. This can be adjusted using the following command:

  

  

  

`python3 irgames.py -s_from 5`

  

  

  

Thus, the iterations will continue from round 5. If the number of rounds after the iteration is not adjusted, i.e., if the number of rounds after calling `-s_from` is not specified, the number of iterations will be the default number; in this case, it would be 2. If we would like to have five additional rounds from round 5, call the following command:

  

  

  

`python3 irgames.py -s_from 5 -rounds 5`

  

  

  

  

To further streamline the process, the algorithm automatically saves the areas of the polytopes for each iteration. If the set is iterated from some round n, the polytope area from that round is automatically retrieved, which can save some computation time.

  

  

  

### Creating a log file

  

  

  

`-log`

  

  

  

The actions can be logged. By default, the argument is off. If enabled, the log file contains the following: the number type (float or fraction);  the number of processes used (if multiprocessing is chosen); the polygon area error bound; the number of actions of each player; the number of signals; probability distributions of the signals; stage game payoff matrix; the discount factor; the error bound on the simplification technique (RDP); incentive compatibility constraints; the minimax payoffs; the number of rounds; and time spent needed for each round. To log the actions, call the following command line:

  

  

  

`python3 irgames.py -log`

  

  

  

### Plot Round n

  

  

  

`-plot`

  

  

  

  

Allows us to visualize a round of  choice. To specify which round to visualize, type the desired round as an integer in the terminal as follows:

  

  

  

  

`python3 irgames.py -plot 3`

  

  

  

  

The png file is than saved to the folder in the working directory under the name `Iteration_3_number_type`.

  

  

  

  

### Simplifying Polytopes

  

  

  

`-sp`

  

  

  

  

The RDP line simplification algorithm is used to simplify the lines of the payoff set,  maintaining the shape of the polytope that defines it. An inherent issue of the algorithm is an ever increasing number of points, which in return increases computation time, especially in later iterations. Enabling this simplification technique, points that fall short of a user-defined threshold are eliminated, and the shape of the polytope is preserved. For maximum precision, we recommend a threshold of 0.00001, though it can be adjusted to meet the users precision requirements.

  

  

  

  

### Summary of features

  

  

  

  

Here is a summary of the features discussed so far:

  

  

  

  

• Input of game parameters (number of actions, discount factor, signals, probabilities)

  

  

  

  

• Input via Excel file upload

  

  

  

  

• Option to use fast or intense computation (floating-point arithmetic or fractions)

  

  

  

  

• Error bound on polytope area for convergence

  

  

  

  

• Adjustable number of rounds

  

  

  

• Option to save vertices to text format

  

  

  

  

• Possibility to continue iterations from a selected round

  

  

  

  

• Simplifying the polytope, by preserving its shape

  

  

  

• Parallel computing option with adjustable number of processes (to be advanced)

  

  

The above-mentioned arguments do not have to be applied separately -  they can be applied simultaneously. Imagine, we would like to run the verification example, opt for accurate results (fractions), log our actions, save the vertices to the text file, five rounds, and enable the simplification feature with the error threshold of 0.0001. The command typed in the terminal would look like this:

  

  

  

  

`python3 irgames.py -verify -number_type fraction -log - output_vertices -round 5 -sp 0.0001`

  

  

  

  

## License

  

  

  

  

IRGames is released under the BSD License. For more details, see the LICENSE file in the repository.

  

  

  

  

## Citation

  

  

  

  

If you use this algorithm in your research, please cite it as follows:

  

  

  

  

  

  

[Karabegovic, Jasmina], IRGames: An algorithm for two-player repeated games with imperfect public monitoring, Version [1], [2024]. Available at: [https://github.com/jasmina-karabegovic/IRGames.git]

  

  

  

  

  

  

## Contact

  

  

  

  

  

  

For help and support, please contact:

  

  

  

  

  

  

- **Name**: Jasmina Karabegovic

  

  

  

  

  

  

- **Email**: [karabegovic.jasmina95@gmail.com](mailto:karabegovic.jasmina95@gmail.com)
