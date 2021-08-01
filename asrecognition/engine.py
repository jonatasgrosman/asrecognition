import torch
import librosa
import datasets
import warnings
import logging
from typing import List, Dict, Optional
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

MODEL_PATH_BY_LANGUAGE = {
    "ar": "jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
    "de": "jonatasgrosman/wav2vec2-large-xlsr-53-german",
    "el": "jonatasgrosman/wav2vec2-large-xlsr-53-greek",
    "en": "jonatasgrosman/wav2vec2-large-xlsr-53-english",
    "es": "jonatasgrosman/wav2vec2-large-xlsr-53-spanish",
    "fa": "jonatasgrosman/wav2vec2-large-xlsr-53-persian",
    "fi": "jonatasgrosman/wav2vec2-large-xlsr-53-finnish",
    "fr": "jonatasgrosman/wav2vec2-large-xlsr-53-french",
    "hu": "jonatasgrosman/wav2vec2-large-xlsr-53-hungarian",
    "it": "jonatasgrosman/wav2vec2-large-xlsr-53-italian",
    "ja": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
    "nl": "jonatasgrosman/wav2vec2-large-xlsr-53-dutch",
    "pl": "jonatasgrosman/wav2vec2-large-xlsr-53-polish",
    "pt": "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese",
    "ru": "jonatasgrosman/wav2vec2-large-xlsr-53-russian",
    "zh-CN": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
}

class ASREngine():

    def __init__(self, language: str, device: Optional[str]="cpu", number_of_workers: Optional[int] = None, 
                 inference_batch_size: Optional[int] = 8, model_path: Optional[str]=None):
        """
        ASR Engine constructor.
        
        Parameters
        ----------
        language : str 
            Language code of the speech to be transcribed. 
            The supported languages are ar, de, el, en, es, fa, fi, fr, hu, it, ja, nl, pl, pt, ru, zh-CN

        device : str 
            Device to use for inference, default is "cpu". If you want to use a GPU for that, 
            you'll probably need to specify the device as "cuda"

        number_of_workers : int 
            Number of processes to use for multiprocessing data handling. By default it doesn't use multiprocessing.

        inference_batch_size : int 
            Number of items per inference batch, default is 8. You can speedup the inference time by increasing this value,
            but it'll increase the memmory usage too.
        
        model_path : str 
            A model path to use for the transcription. You only need to specify this if you want to use a
            custom model. If you don't specify this, the engine will use the default model for the given language.
        """
        
        self.language = language
        self.device = device
        self.number_of_workers = number_of_workers
        self.inference_batch_size = inference_batch_size
        self.model_path = model_path

        if self.model_path is None:
            self.model_path = MODEL_PATH_BY_LANGUAGE.get(self.language, None)
        
        if self.model_path is None:
            raise AttributeError(f"Language '{self.language}'' is not supported yet. Use one of the supported languages ({list(MODEL_PATH_BY_LANGUAGE.keys())}) or provide a valid model_path")

        self.processor = Wav2Vec2Processor.from_pretrained(self.model_path)
        self.model = Wav2Vec2ForCTC.from_pretrained(self.model_path)
        self.model.to(self.device)

    def transcribe(self, audio_paths: List[str]) -> List[Dict]:
        """
        Transcribe the given audio files.
        
        Parameters
        ----------
        audio_paths : List[str] 
            List of audio paths to transcribe.

        Returns
        -------
        List[Dict]
            A list of dictionaries containing the transcription for each audio file.
            The format of each list element is:
            {
                "path": str,
                "transcription": str
            }
        """

        data = datasets.Dataset.from_dict({"path": audio_paths})

        def _load_audio(item):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                waveform, sampling_rate = librosa.load(item["path"], sr=16_000)
                item["waveform"] = waveform
                item["duration"] = len(waveform) / sampling_rate
                return item
        
        data = data.map(_load_audio, num_proc=self.number_of_workers)

        data_more_than_20s = data.filter(
            lambda example: example["duration"] > 20,
            num_proc=self.number_of_workers
        )

        if len(data_more_than_20s) > 0:
            logging.warn(f"Some files ({data_more_than_20s['path']}) have more than 20 seconds.\n" \
                f"To prevent performance issues we highly recommend you to split them into smaller chunks of less than 20 seconds."
            )

        def _predict(batch, model=self.model, processor=self.processor, device=self.device):            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                inputs = processor(batch["waveform"], sampling_rate=16_000, return_tensors="pt", padding=True)
                with torch.no_grad():
                    if hasattr(inputs, "attention_mask"):
                        logits = model(inputs.input_values.to(device), attention_mask=inputs.attention_mask.to(device)).logits
                    else:
                        logits = model(inputs.input_values.to(device)).logits
                pred_ids = torch.argmax(logits, dim=-1)
                batch["transcription"] = processor.batch_decode(pred_ids)
                return batch

        data = data.map(_predict, batched=True, batch_size=self.inference_batch_size)

        transcriptions = []

        for item in data:
            transcriptions.append({
                "path": item["path"],
                "transcription": item["transcription"]
            })

        return transcriptions
