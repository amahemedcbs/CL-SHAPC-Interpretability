import Saliency.DGR.utils
from Saliency.DGR.utils import checkattr

##-------------------------------------------------------------------------------------------------------------------##

def define_classifier(args, config, device, depth=0, stream=False):
    model = define_standard_classifier(args=args, config=config, device=device, depth=depth)
    return model


##-------------------------------------------------------------------------------------------------------------------##

## Function for defining discriminative classifier model
def define_standard_classifier(args, config, device, depth=0):
    # Import required model
    from Saliency.DGR.models.classifier import Classifier
    # Specify model
    model = Classifier(
        image_size=config['size'],
        image_channels=config['channels'],
        classes=config['output_units'],
        # -conv-layers
        depth=depth,
        conv_type=args.conv_type if depth>0 else None,
        start_channels=args.channels if depth>0 else None,
        reducing_layers=args.rl if depth>0 else None,
        num_blocks=args.n_blocks if depth>0 else None,
        conv_bn=(True if args.conv_bn=="yes" else False) if depth>0 else None,
        conv_nl=args.conv_nl if depth>0 else None,
        no_fnl=True if depth>0 else None,
        global_pooling=checkattr(args, 'gp') if depth>0 else None,
        # -fc-layers
        fc_layers=args.fc_lay,
        fc_units=args.fc_units,
        fc_drop=args.fc_drop,
        fc_bn=True if args.fc_bn=="yes" else False,
        fc_nl=args.fc_nl,
        excit_buffer=True,
        phantom=checkattr(args, 'fisher_kfac')
    ).to(device)
    # Return model
    return model

##-------------------------------------------------------------------------------------------------------------------##