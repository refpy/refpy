############################################
RefPy
############################################

.. raw:: html

   <hr style="height:6px; background-color:#888; border:none; margin:1.5em 0;" />

******************
Project Philosophy
******************

 **RefPy** is an open-source code for pipeline and riser design.

 **RefPy** is a personal project developed by Ismael Ripoll to consolidate and automate tasks frequently encountered in his daily engineering routine, with the aim of having them version-controlled and accessible through a structured repository.
 
 The repository has been made publicly accessible to encourage collaboration and knowledge sharing among colleagues, while maintaining that all proprietary or confidential information remains excluded. It has been made available to allow colleagues to review, build upon, and make use of it in a collaborative and open manner, in the hope that this may give the repository a chance to continue improving beyond its original scope.

  - RefPy has made available with the intend of allowing the community to review, expand and use it under a `MIT License <https://github.com/refpy/refpy/blob/main/LICENSE>`_. The MIT license is one of the most permissive open-source licenses, allowing for reuse and modification with minimal restrictions.

  - Source code: The source code is saved in `GitHub <https://github.com/refpy/refpy>`_.

  - PyPi Wheel: A downloadable wheel of RefPy is found in `PyPi <https://pypi.org/project/refpy>`_. To install RefPy, execute pip install refpy in a command terminal.

********************
Project Team Members
********************

 The management of this open-source project is carried out by `Ismael Ripoll <https://github.com/ ismael-ripoll>`_.

 While RefPy is currently a personal initiative led by Ismael Ripoll, the project is hosted under the  dedicated refpy GitHub organisation rather than a personal account. Although there is no immediate  intention to formally announce or release the project or expand the team, this structure has been adopted to maintain flexibility for any future developments and to facilitate transparent and manageable   collaboration if it occurs.

********************************
How to Contribute to the Project
********************************

 Interested in contributing to this personal project?
 
  - If you have a contribution in mind, please add it in `Discussions <https://github.com/refpy/refpy/discussions>`_.
  - If you have identified an issue with the code, please add it in `Issues <https://github.com/refpy/refpy/issues>`_.

******************************
RefPy Python Library Versions
******************************
 THE REQUIREMENTS.txt file contains the libraries for which the script has been developed and shown to run.

 If some of libraries imported by RefPy lead to issues, before running RefPy for first time, follow these steps to activate a virtual environment from the REQUIREMENTS.TXT file:

  1. Open a Command Prompt or PowerShell and navigate to the folder where RefPy is cloned and copy REQUIREMENTS.txt file.

     `$ cd path/to/refpy`

  2. In the folder where the code that imports RefPy is saved, to create the virtual environment, run:

     `$ python -m venv env`

  3. To activate the virtual environement, run: [This command works in the VSCode's PowerShell. Command may vary in other shells]

     `$ env/Scripts/Activate.ps1`

  4. To install the necessary libraries, run:

     `$ pip install -r ./REQUIREMENTS.txt`

 Afterwards, the virtual environement must be activated before running RefPy by using the command in Step 3.

########
Contents
########
.. toctree::
   :maxdepth: 1

   getting_started
   api_reference
