import argparse

class Argparser:
    """
    The actual argparser
    """
    def __init__(self):
        self.args = self.prepare_arg_parser().parse_args()

    def prepare_arg_parser(self):
        """
        Add all args to the argparser
        """

        arg_parser = argparse.ArgumentParser()

        # Hardware specifications

        arg_parser.add_argument('--model_name', type=str, default='san_ortho_sec_data_L1_new',
                                help='Name of Model you want to generate IPF map')
        arg_parser.add_argument('--dataset_type', type=str, default='Val',
                                help='Val or Test Dataset')
        arg_parser.add_argument('--data', type=str, default='Ti64',
                                help='type of material')

        arg_parser.add_argument('--model_to_load', type=str, default= 'model_best',
                                help='which model to load')
        arg_parser.add_argument('--file_type', type=str, default='SR',
                                help='[SR, HR, LR]')
        arg_parser.add_argument('--section', type=str,
                                default= 'X_Block',
                                help='[X_Block, Y_Block, Z_Block]')
        
        return arg_parser
