
from base.base_model import BaseModel
import keras.backend as K
from keras.applications.resnet50 import ResNet50
from keras.layers import Lambda, GlobalAveragePooling2D, Input, Dense, Reshape
from keras.layers.embeddings import Embedding
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.optimizers import Adagrad
from keras.optimizers import Adam

class ResNet50AttrModel(BaseModel):
    def __init__(self, config):
        super(ResNet50AttrModel, self).__init__(config)
        self.img_size = config['img_size']
        self.weights_path = config['weights_path']
        self.base_lr = config['base_lr']
        self.optimizer = config['optimizer']        
        self.numerical_loss_weights = config['numerical_loss_weights']
        self.categorical_loss_weights = config['categorical_loss_weights']
        self.colors = config['colors']
        self.harmonies = config['harmonies']
        self.styles = config['styles']
        self.build_model()

    def l2_normalize(self, x):
        """Apply L2 Normalization

        Args:
            x (tensor): output of convolution layer
        """        
        return K.l2_normalize(x, 0)

    def l2_normalize_output_shape(self, input_shape):
        return input_shape

    def global_average_pooling(self, x):
        """Apply global average pooling

        Args:
            x (tensor): output of convolution layer
        """
        x = GlobalAveragePooling2D()(x)
        return x

    def global_average_pooling_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def build_model(self):
        _input = Input(shape=(self.img_size, self.img_size, 3))
        resnet = ResNet50(include_top=False, weights='imagenet', input_tensor=_input)
        activation_layers = []
        layers = resnet.layers
        for layer in layers:
            if 'activation' in layer.name:
                activation_layers.append(layer)

        activations = 0
        activations_gap_plus_lastactivation_gap_l2 = []
        # Create GAP layer for the activation layer at the end of each ResNet block, except for last one
        nlayers = len(activation_layers) - 1
        for i in range(1, nlayers):
            layer = activation_layers[i]
            # three activations per block, select only the last one
            if layer.output_shape[-1] > activation_layers[i - 1].output_shape[-1]:
                # print(layer.name, layer.input_shape, layer.output_shape[-1], activation_layers[i - 1].output_shape[-1])
                activations += layer.output_shape[-1]
                _out = Lambda(self.global_average_pooling,
                            output_shape=self.global_average_pooling_output_shape, name=layer.name + '_gap')(layer.output)
                activations_gap_plus_lastactivation_gap_l2.append(_out)

        print("sum of all activations should be 13056: {}".format(activations))

        last_layer_output = GlobalAveragePooling2D()(activation_layers[-1].output)

        last_layer_output = Lambda(self.l2_normalize, output_shape=self.l2_normalize_output_shape,
                                name=activation_layers[-1].name+'_gap')(last_layer_output)

        activations_gap_plus_lastactivation_gap_l2.append(last_layer_output)

        merged = Concatenate(axis=1)(activations_gap_plus_lastactivation_gap_l2)
        print("merged shape should be (?, 15104): ", merged.shape)
        merged = Lambda(self.l2_normalize, output_shape=self.l2_normalize_output_shape, name='merge')(merged)

        # create an output for each attribute
        outputs = []

        attrs = [k for k in self.numerical_loss_weights]
        for attr in attrs:
            outputs.append(Dense(1, kernel_initializer='glorot_uniform', activation='tanh', name=attr)(merged))

        # output length of encoded color, 4 values: sin(hue), cos(hue), saturation, value
        outputs.append(Dense(4, activation='sigmoid', name='pri_color')(merged))

        # TODO: Train on harmony and style
        # outputs.append(Dense(len(self.harmonies), activation='softmax', name='harmony')(merged))
        # TODO: use embedding
        # color_harmony_embedding_size = 10
        # color_harmony_embedding = Embedding(len(self.harmonies) + 1, color_harmony_embedding_size, input_length = 1, name='harmony_emb')(merged)
        # outputs.append(Reshape(target_shape=(1, 1, color_harmony_embedding_size), name='harmony')(color_harmony_embedding))
        # outputs.append(Dense(len(self.styles), activation='softmax', name='style')(merged))

        non_negative_attrs = []
        for attr in non_negative_attrs:
            outputs.append(Dense(1, kernel_initializer='glorot_uniform', activation='sigmoid', name=attr)(merged))

        self.model = Model(inputs=_input, outputs=outputs)
        if self.weights_path:
            self.model.load_weights(self.weights_path)

        # TODO: Add weight Decay
        if self.optimizer == 'adagrad':
            optimizer = Adagrad(lr=self.base_lr)
        elif self.optimizer == 'adam':
            optimizer = Adam(lr=self.base_lr, beta_1=0.9, beta_2=0.999, epsilon=0.00000001, amsgrad=True)
            
        loss = {}
        metrics = {}

        for attr in attrs:
            loss[attr] = 'mean_squared_error'
            metrics[attr] = 'mean_squared_error'

        for attr in self.categorical_loss_weights:
            # use mean_squared_error for loss color encoded by three values, cyan-magenta-yellow
            if attr == 'pri_color':
                loss[attr] = 'mean_squared_error'
                metrics[attr] = 'mean_squared_error'
            else:
                loss[attr] = 'categorical_crossentropy'
                metrics[attr] = 'categorical_crossentropy'

        loss_weights = dict(self.numerical_loss_weights)
        loss_weights.update(self.categorical_loss_weights)
        
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics,
                    loss_weights=loss_weights)