import argparse

def get_parser():
    # Use argparse to parse the command-line arguments
    parser = argparse.ArgumentParser(description='EEG Transformer Training Script')
    parser.add_argument('--data_path', type=str,
                        default="/userhome2/liweile/EEG_Image_decode/THINGS/Preprocessed_data_250Hz",
                        help='Path to the EEG dataset')
    parser.add_argument('--output_dir', type=str, default='./outputs/contrast', help='Directory to save output results')
    parser.add_argument('--project', type=str, default="EEG_Image_decode", help='WandB project name')
    parser.add_argument('--entity', type=str, default="lisavila-shanghaitech-university", help='WandB entity name')
    parser.add_argument('--name', type=str, default="lr=3e-4_img_pos_pro_eeg", help='Experiment name')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--logger', type=bool, default=True, help='Enable WandB logging')
    parser.add_argument('--gpu', type=str, default='cuda:1', help='GPU device to use')
    parser.add_argument('--insubject', type=bool, default=True, help='In-subject mode or cross-subject mode')
    parser.add_argument('--encoder_type', type=str, default='ATMS', help='Encoder type')
    parser.add_argument('--subjects', nargs='+',
                        default=['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08',
                                 'sub-09', 'sub-10'], help='List of subject IDs (default: sub-01 to sub-10)')
    return parser
