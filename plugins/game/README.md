# pymss-plugin-game

OpenVPI GAME (General Audio Manipulation and Estimation v1.0.0) vocal to MIDI transcription plugin for [pymss](https://github.com/pymss-project/pymss).

## Features

- High accuracy vocal melody extraction into standard MIDI format (.mid).
- Dynamic chunking with integrated silence slicer.
- Registers `vocal_to_midi` capability into pymss plugin pool.

## Installation

```sh
pymss install game
```

## Usage in Python

```python
from pymss.plugins import get_registry

vocal_to_midi = get_registry().get_capability("vocal_to_midi")
result = vocal_to_midi(
    audio_path="vocal.wav",
    model_path="model.pt",
    output_dir="./output",
    tempo=120.0,
)
print("MIDI exported to:", result.output_path)
```
