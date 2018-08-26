import sys
from utils.utils import get_args, process_config, create_dirs
from data_loader.wikiart_data_loader import WikiArtDataLoader
from models.resnet50_attr_model import ResNet50AttrModel
from trainers.resnet_50_trainer import ResNet50ModelTrainer

# TODO: Add proper logging

def main():
    # get json configuration filepath from the run argument
    # process the json configuration file
    try:
        args = get_args()
        config, log_dir, checkpoint_dir = process_config(args.config)
    except:
        print('missing or invalid arguments')
        print('Unexpected error:', sys.exc_info()[0])

    # create the experiment directories
    create_dirs([log_dir, checkpoint_dir])

    print('Create the data generator')
    data_loader = WikiArtDataLoader(config)

    print('Create the model')
    model = ResNet50AttrModel(config)
    print('model ready loading data now')

    print('Create the trainer')
    trainer = ResNet50ModelTrainer(model.model, data_loader.get_train_data(), data_loader.get_val_data(), config, log_dir, checkpoint_dir)

    print('Start training the model.')
    trainer.train()

def infer():
    # get json configuration filepath from the run argument
    # process the json configuration file
    try:
        config = 'input_params_for_inference.json'
        config, _, _ = process_config(config)
    except:
        print('missing or invalid arguments')
        print('Unexpected error:', sys.exc_info()[0])

    print('Create the data generator')
    data_loader = WikiArtDataLoader(config)

    print('Create the model')
    model = ResNet50AttrModel(config)
    print('model ready loading data now')

    print('Create the trainer')
    trainer = ResNet50ModelTrainer(model.model, data_loader.get_train_data(), data_loader.get_val_data(), config, '', '')

    print('Infer.')
    trainer.predict()


if __name__ == '__main__':
    main()