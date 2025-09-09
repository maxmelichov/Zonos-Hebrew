import torch
import torchaudio
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from zonos.utils import DEFAULT_DEVICE as device
from phonikud import phonemize

model = Zonos.from_pretrained("checkpoints/Zonos-v0.1-hybrid-Hebrew.pt", device=device)



wav, sampling_rate = torchaudio.load("assets/exampleaudio.mp3")
speaker = model.make_speaker_embedding(wav, sampling_rate)

torch.manual_seed(421)

text = "שלום עולם"

phonemes = phonemize(text, "he")
cond_dict = make_cond_dict(text=phonemes, speaker=speaker, language="he")
conditioning = model.prepare_conditioning(cond_dict)

codes = model.generate(conditioning)

wavs = model.autoencoder.decode(codes).cpu()
torchaudio.save("sample.wav", wavs[0], model.autoencoder.sampling_rate)
