import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='EEG Model Training Script')
    parser.add_argument('--data_path', type=str, default='/userhome2/liweile/EEG_Image_decode/THINGS/Preprocessed_data_250Hz', help='Path to data')
    parser.add_argument('--output_dir', type=str, default='./outputs/contrast', help='Directory to save output results')
    parser.add_argument('--project', type=str, default='train_pos_img_text_rep', help='Project name for logging')
    parser.add_argument('--entity', type=str, default="sustech_rethinkingbci", help='WandB entity name')
    parser.add_argument('--name', type=str, default="lr=3e-4_img_pos_pro_eeg", help='Experiment name')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--insubject', default=True, help='Flag to indicate within-subject training')
    parser.add_argument('--encoder_type', type=str, default='encoder_low_level',
                        choices=['EEGNetv4_Encoder', 'ATCNet_Encoder', 'EEGConformer_Encoder', 'EEGITNet_Encoder', 'ShallowFBCSPNet_Encoder', 'encoder_low_level'],
                        help='Encoder type')
    parser.add_argument('--img_encoder', type=str, default='Proj_img', help='Image encoder type')
    parser.add_argument('--logger', default=True, help='Enable logging')
    parser.add_argument('--gpu', type=str, default='cuda:1', help='GPU device to use')
    parser.add_argument('--subjects', nargs='+', default=['sub-01'], help='List of subject IDs')

    return parser
