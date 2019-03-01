import argparse


ap = argparse.ArgumentParser(prog='arg_parse.py',
                             description="Parameters of Deep Learning.",
                             epilog="Default values have been set.")


# parameters
ap.add_argument('--batch_size', type=int, default=32)
ap.add_argument('--learning_rate', type=float, default=0.001)
ap.add_argument('--drop_out', type=float, default=0.5)
ap.add_argument('--epochs', type=int, default=5)
ap.add_argument('--print_every', type=int, default=30)

# model
ap.add_argument('--arch', choices=['vgg16', 'alexnet'], default="vgg16") # default type: str
ap.add_argument('--input_size', type=int,  choices=[25088, 9216], default=25088)
ap.add_argument('--output_size', type=int, default=102)
ap.add_argument('--hidden_layers', type=int, default=4096)

# paths
ap.add_argument('--checkpoint_path', default='checkpoint.pth')
ap.add_argument('--data_dir', default='flowers')
ap.add_argument('--image_path', default='flowers/test/1/image_06743.jpg')



# other
ap.add_argument('--topk', type=int, default=5)



args, _ = parser.parse_known_args()
