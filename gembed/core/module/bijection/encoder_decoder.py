from gembed.core.module import InvertibleModule

class EncoderDecoder(InvertibleModule):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, Z, **kwargs):
        return self.decoder(Z)

    def inverse(self, X, **kwargs):
        return self.encoder(X)

    def __str__(self):
        return str(self.__class__.str())
