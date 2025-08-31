from sf_examples.nethack.models.chaotic_dwarf import ChaoticDwarvenGPT5
from sf_examples.nethack.models.scaled import ScaledNet
from sf_examples.nethack.models.symbolic import SymbolicGlyphNet
from sf_examples.nethack.models.symbolic_nle_tokenizer import SymbolicGlyphTokenNet
from sf_examples.nethack.models.torchbeast import TorchBeastMessageEncoder

MODELS = [
    ChaoticDwarvenGPT5,
    ScaledNet,
    SymbolicGlyphNet,
    SymbolicGlyphTokenNet,
    TorchBeastMessageEncoder,
]
MODELS_LOOKUP = {c.__name__: c for c in MODELS}
