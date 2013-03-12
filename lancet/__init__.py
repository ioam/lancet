"""
Lancet consists of three fundamental class types: Argument Specifiers, Command
Templates and Launchers. Launch Helpers are useful helper functions for using
Lancet.

Argument Specifiers
-------------------

Argument specifiers are intended to be clear, succinct and composable way of
specifying large parameter sets. High-dimensional parameter sets are typical of
large-scale scientific models and can make managing such models and simulations
difficult. Using argument specifiers, you can document how the parameters of
your model vary across runs without long, flat lists of arguments or requiring
deeply nested loops.

Argument specifiers can be freely intermixed with Python code, simplifying the
use of scientific software with a Python interface. They also invoke commandline
programs using command template objects to build commands and launchers to
execute them. In this way they can help simplify the management of simulation,
analysis and visualisation tools with Python.

Typical usage includes specification of constant arguments, description of
parameter ranges or even search procedures (such as hillclimbing, bisection
optimisation etc). Argument specifiers can compose together using Cartesian
Products or Concatenation to express huge numbers of argument sets concisely.

Command Templates
------------------

When working with external tools, a command template is needed to turn an
argument specifier into an executable command. These objects are designed to be
customisable with sensible defaults to handle as much boilerplate as
possible. They may require argument specifiers with a certain structure (certain
argument names and value types) to operate correctly.

Launchers
---------

A launcher is designed to execute commands on a given computational platform,
making use of as much concurrency where possible. The default Launcher class
executes commands locally while the QLauncher class is designed to launch
commands on Sun Grid Engine clusters.

Launch Helpers
--------------

These helper functions help verify correct usage of lancet for specific
use-cases. They check for consistent useage automatically and offer an in-depth
review of all settings specified before launch. The goal is to help users
identify mistakes early before consuming computational time and resources.
"""

try:
    import IPython
    from lancet.ipython import *
except:
    from lancet.core import *

