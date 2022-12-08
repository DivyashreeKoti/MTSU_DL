#List of hyper parameters for tuning purpose
class Hyperparameters:
    def __init__(self, learning_rate = 0.001,
                 weight_decay = 0.0001,
                 batch_size = 32,
                 num_epochs = 100,
                 image_size = 200,
                 patch_size = 32,
                 projection_dim = 64,
                 num_heads = 4,
                 transformer_layers = 8,
                 mlp_head_units = [2048, 2048],
                 conv_layers = 6):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.image_size = image_size  # We'll resize input images to this size
        self.patch_size = patch_size  # Size of the patches to be extract from the input images
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.transformer_units = [
            self.projection_dim * 2,
            self.projection_dim,
        ]  # Size of the transformer layers
        self.transformer_layers = transformer_layers
        self.mlp_head_units = mlp_head_units
        self.conv_layers = conv_layers
        