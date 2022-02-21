**This tool is deprecated in favor of https://github.com/jonatasgrosman/huggingsound**

# ASRecognition

ASRecognition: just an easy-to-use library for Automatic Speech Recognition.

I have no intention of building a very complex toolkit for speech recognition here. 
In fact, this library uses a lot of things already built by the [Hugging Face](https://huggingface.co/) (Thank you guys!). 
I just wanna have an easy-to-use interface to add speech recognition features to my apps.
I hope this library could be useful for someone else too :)

The currently supported languages are Arabic (ar), German (de), Greek (el), English (en), Spanish (es), Persian (fa), Finnish (fi), French (fr), Hungarian (hu), Italian (it), Japanese (ja), Dutch (nl), Polish (pl), Portuguese (pt), Russian (ru), Chinese (zh-CN).

# Requirements

- Python 3.7+

# Installation

```console
$ pip install asrecognition
```

# How to use it?

```python
from asrecognition import ASREngine

# 1 - Load the ASR engine for a given language (on the first run it may take a while)
asr = ASREngine("en")

# 2 - Use the loaded ASR engine to transcribe a list of audio files
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
