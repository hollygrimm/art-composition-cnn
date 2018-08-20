import numpy as np
from keras.utils import Sequence
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input

class WikiArtDataGenerator(Sequence):
    """Generates data for Keras."""
    def __init__(self, img_filenames, labels, batch_size=32, target_size=(300, 300), n_channels=3,
                 shuffle=True):
        """Initialization.
        
        Args:
            img_filenames: A list of path to image files.
            labels: A dictionary of corresponding labels.
        """
        self.img_filenames = img_filenames
        self.labels = labels
        self.batch_size = batch_size
        self.target_size = target_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch."""
        return int(np.floor(len(self.img_filenames) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data."""
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Get filenames and labels for batch
        img_filenames_temp = [self.img_filenames[k] for k in indexes]
        labels_temp = [self.labels[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(img_filenames_temp, labels_temp)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch."""
        self.indexes = np.arange(len(self.img_filenames))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def prepare_image(self, image_path, target_size):
        """Loads image from filepath and scales RGB values from -1 to 1

        Args:
            image_path (str): Image Path
            target_size (tuple): scale to x, y dimensions

        Returns:
            image_array
        """
        img = image.load_img(image_path, target_size = target_size)
        x = image.img_to_array(img)
        x = preprocess_input(x, mode='tf')
        return x         

    def __data_generation(self, X_filenames_batch, Y_batch):
        """Generates data containing batch_size samples."""
        X_batch = []

        # Read image data in for batch
        for img_filename in X_filenames_batch:
            img_data = self.prepare_image(img_filename, self.target_size)
            X_batch.append(img_data)
        X = np.array(X_batch)

        Y_batch_transposed = list(map(list, zip(*Y_batch)))

        return X, [np.array(Y_batch_transposed[i]) for i in range(len(Y_batch_transposed))]  