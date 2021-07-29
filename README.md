# ASRecognition

ASRecognition is just an easy-to-use library for Automatic Speech Recognition.

The currently supported languages are German (de), Greek (el), English (en), Spanish (es), Persian (fa), Finnish (fi), French (fr), Hungarian (hu), Italian (it), Japanese (ja), Dutch (nl), Polish (pl), Portuguese (pt), Russian (ru), Chinese (zh-CN).

# Requirements

- Python 3.7+

# Installation

```console
$ pip install asrecognition
```

# How to use it?

```python
from asrecognition import ASREngine

# 1 - Load the ASR engine by a given language (may take a while for the first time)
asr = ASREngine("en")

# 2 - Use the ASR engine to transcribe a list of audio files
audio_paths = ["/path/to/sagan.mp3", "/path/to/asimov.wav"]
transcriptions = asr.transcribe(audio_paths)

# 3 - Voilà!
print(transcriptions)

# [
#  {"path": "/path/to/sagan.mp3", "transcription": "EXTRAORDINARY CLAIMS REQUIRE EXTRAORDINARY EVIDENCE"},
#  {"path": "/path/to/asimov.wav", "transcription": "VIOLENCE IS THE LAST REFUGE OF THE INCOMPETENT"}
# ]
```
# Want to help?

See the [contribution guidelines](https://github.com/jonatasgrosman/asrecognition/blob/master/CONTRIBUTING.md)
if you'd like to contribute to ASRecognition project.

You don't even need to know how to code to contribute to the project. Even the improvement of our documentation is an outstanding contribution.

If this project has been useful for you, please share it with your friends. This project could be helpful for them too.

If you like this project and want to motivate the maintainers, give us a :star:. This kind of recognition will make us very happy with the work that we've done with :heart:

# Citation
If you want to cite the tool you can use this:

```bibtex
@misc{grosman2021asrecognition,
  title={ASRecognition},
  author={Grosman, Jonatas},
  publisher={GitHub},
  journal={GitHub repository},
  howpublished={\url{https://github.com/jonatasgrosman/asrecognition}},
  year={2021}
}
```