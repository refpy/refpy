'''
Batch file
'''

# import os
from refpy import AbaqusPy

sens_dict = {'PARAM_sens': [0]}
TEMPLATE_FILENAME = 'example_3'

for sens_nb in sens_dict['PARAM_sens']:
    SENSITIVITY_FILENAME = f"{TEMPLATE_FILENAME}_sens{sens_nb}"
    abaqus_input_file_writer = AbaqusPy(
        template_filename=TEMPLATE_FILENAME,
        sensitivity_filename=SENSITIVITY_FILENAME,
        param_dict=sens_dict,
        isens=sens_nb
    )
    abaqus_input_file_writer.run()
    # os.system( f'abaqus int j={file_name} ask_delete=off')
