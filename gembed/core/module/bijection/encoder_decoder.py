from gembed.core.module import InvertibleModule
import lightning as pl

class EncoderDecoder(pl.LightningModule, InvertibleModule):
    """The `EncoderDecoder` class is a PyTorch Lightning module that combines an encoder and decoder to
    perform forward and inverse operations.
    """

    def __init__(self, encoder, decoder):
        """
        Defines an EncoderDecoder that takes in an encoder and decoder as
        parameters and assigns them to instance variables.
        
        :param encoder: The `encoder` parameter is an object that is responsible for encoding data. It takes
        in raw data and converts it into a new (typically lower dimensional) vector.
        :param decoder: The `decoder` parameter is an object that is responsible for decoding data. It takes
        encoded data and converts it back to its original format.
        """
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, Z, **kwargs):
        """
        The forward function returns the output of the decoder given an input Z.
        
        :param Z: The parameter Z is a tensor representing the encoded representation
        :return: The output of the `decoder` function with the input `Z` and any additional keyword
        arguments passed in.
        """
        return self.decoder(Z, **kwargs)

    def inverse(self, X, **kwargs):
        """
        The function "inverse" returns the result of calling the "encoder" function with the given
        arguments.
        
        :param X: The parameter `X` represents the input data that you want to encode.
        :return: the result of calling the `encoder` method with the arguments `X` and `**kwargs`.
        """
        return self.encoder(X, **kwargs)

    def __str__(self):
        return str(self.__class__.str())
