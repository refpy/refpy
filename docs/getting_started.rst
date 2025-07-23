###############
Getting Started
###############

.. raw:: html

   <hr style="height:6px; background-color:#888; border:none; margin:1.5em 0;" />

***********************************
About RefPy: A Foundational Library
***********************************

RefPy is designed as a foundational library for pipeline and riser design. A foundational library provides core building blocks and reusable components that can be used directly in your own scripts, applications, or larger frameworks. It is not a standalone application, but rather a toolkit that enables rapid development and robust analysis by providing well-tested, well-documented, and efficient functions and classes.

What is a Foundational Library?
-------------------------------

A foundational library is a collection of essential tools and abstractions that serve as the basis for more complex workflows and applications. It is intended to be imported and used as a dependency in your own Python code, rather than run as a program by itself. Foundational libraries are typically:

- Modular and extensible
- Well-documented and tested
- Focused on core functionality, not user interfaces
- Designed for integration into other projects

Current Modules in RefPy
------------------------

RefPy currently includes the following modules:

- ``linepipe_tools.py``: Provides the ``Pipe`` class for geometric and material property calculations of pipeline sections, including diameters, areas, stiffness, and moments of inertia.
- ``dnv_tools.py``: Contains classes and functions for DNV pipeline code calculations, including general utilities and limit state checks (``DNVGeneral``, ``DNVLimitStates``).
- ``lateral_buckling_tools.py``: Implements the ``LBDistributions`` class for lateral buckling reliability analysis and friction factor distribution fitting.
- ``pipe_soil_interaction_tools.py``: Provides the ``PSI`` class for pipe-soil interaction calculations, including resistance and displacement models.
- ``oos_tools.py``: Contains the ``OOSAnonymisation`` and ``OOSSmoother`` classes for processing, anonymising, and smoothing Out-Of-Straightness (OOS) survey data.
- ``abaqus_tools.py``: Includes the ``AbaqusPy`` class for generating sensitivity files and automating input preparation for Abaqus finite element analysis.

Each module is designed to be imported and used independently or in combination, depending on your analysis needs. For more details, see the API Reference section of the documentation.

.. raw:: html

   <hr style="height:6px; background-color:#888; border:none; margin:1.5em 0;" />

*************************
Testing the RefPy Package
*************************

The repository includes a file called ``test_refpy.py`` that provides automated testing for the main classes and methods in RefPy. This script is designed to help you verify that your installation and code changes are working as expected.

How to Run the Tests
--------------------

To run all tests, simply execute the following command in your terminal from the root of the repository:

.. code-block:: bash

   $ python test_refpy.py

What Does It Test?
------------------

The ``test_refpy.py`` script will automatically run doctests for the following classes:

- ``Pipe``
- ``DNVGeneral``
- ``DNVLimitStates``
- ``LBDistributions``
- ``PSI``

Each method in these classes that includes a docstring example will be tested. The script will print a summary of the number of tests attempted, passed, and failed.

This makes it easy to check that all core functionality is working as intended after installation or modification.

.. raw:: html

   <hr style="height:6px; background-color:#888; border:none; margin:1.5em 0;" />

***********************
Running Example Scripts
***********************

The repository also includes a file called ``example_refpy.py`` that demonstrates how to use the main features of RefPy with real input data. This script is a practical starting point for new users.

How to Run the Example
----------------------

To run the example script, execute the following command in your terminal from the root of the repository:

.. code-block:: bash

   $ python example_1_2_refpy.py

What Does It Demonstrate?
-------------------------

The ``example_1_2_refpy.py`` script contains two worked examples:

- **Pipe, DNV, PSI, LBDistributions Example:** Demonstrates how to create and use the ``Pipe``, ``DNVLimitStates``, ``PSI``, ``LBDistributions`` classes for pipeline property calculations.
- **OOSSmoother and OOSAnonymisation Example:** Shows how to use the ``OOSSmoother`` and ``OOSAnonymisation`` classes for group-wise signal processing and survey data anonymisation.

.. figure:: _static/example_1.png
   :alt: Example 1
   :width: 400
   :align: center

   Example 1: Pipe, DNV, PSI and LB Calculations

.. figure:: _static/example_2_plot1.png
   :alt: Example 2 Plot 1
   :width: 600
   :align: center

   Example 2, Plot 1: Survey Route Overview. This plot is useful to visualise the out-of-straightness per survey section. It splits the survey route into individual sections and displays the out-of-straightness for each section, distinguishing between nominally straight and curved sections.

.. figure:: _static/example_2_plot2.png
   :alt: Example 2 Plot 2
   :width: 600
   :align: center

.. figure:: _static/example_2_plot2b.png
   :alt: Example 2 Plot 2
   :width: 600
   :align: center

   Example 2, Plot 2: Anonymised Smoothing. These plots intend to visualise how the anonymisation and smoothing processes affect the survey data at the pipeline group level.

.. figure:: _static/example_2_plot3.png
   :alt: Example 2 Plot 3
   :width: 600
   :align: center

   Example 2, Plot 3: Spectral Analysis. This plot visualises the wavelength distribution of the survey data, highlighting the noise associated with the survey.