from tempfile import NamedTemporaryFile
from typing import Optional

import torch
import torchaudio
from ovos_plugin_manager.templates.stt import STT
from ovos_utils import classproperty
from ovos_utils.lang import standardize_lang_tag
from speech_recognition import AudioData
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


class Wav2VecSTT(STT):
    LANG2MODEL = {
        "gl": "proxectonos/Nos_ASR-wav2vec2-large-xlsr-53-gl-with-lm",

        "ca": "PereLluis13/wav2vec2-xls-r-1b-ca-lm",
        # "ca": "PereLluis13/wav2vec2-xls-r-300m-ca-lm", # <- smaller
        # "ca": "PereLluis13/Wav2Vec2-Large-XLSR-53-catalan",

        "fi": "GetmanY1/wav2vec2-large-fi-lp-cont-pt-1500h",

        "de": "facebook/wav2vec2-large-xlsr-53-german",
        "nl": "facebook/wav2vec2-large-xlsr-53-dutch",
        "it": "facebook/wav2vec2-large-xlsr-53-italian",
        "es": "facebook/wav2vec2-large-xlsr-53-spanish",
        "pl": "facebook/wav2vec2-large-xlsr-53-polish",
        "fr": "facebook/wav2vec2-large-xlsr-53-french",
        # "pt": "facebook/wav2vec2-large-xlsr-53-portuguese",

        "pt": "jonatasgrosman/wav2vec2-xls-r-1b-portuguese",  # <- bigger but better
        "en": "jonatasgrosman/wav2vec2-xls-r-1b-english",  # <- bigger but better
        "zh": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
        "ru": "jonatasgrosman/wav2vec2-large-xlsr-53-russian",
        "jp": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
        "hu": "jonatasgrosman/wav2vec2-large-xlsr-53-hungarian",
        "ar": "jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
        "fa": "jonatasgrosman/wav2vec2-large-xlsr-53-persian",
        "el": "jonatasgrosman/wav2vec2-large-xlsr-53-greek",
        # "pt": "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese", # <- smaller
        # "en": "jonatasgrosman/wav2vec2-large-xlsr-53-english", # <- smaller
        # "en": "jonatasgrosman/wav2vec2-large-english"
        # "it": "jonatasgrosman/wav2vec2-large-xlsr-53-italian",
        # "es": "jonatasgrosman/wav2vec2-large-xlsr-53-spanish",
        # "fr": "jonatasgrosman/wav2vec2-large-xlsr-53-french",
        # "de": "jonatasgrosman/wav2vec2-large-xlsr-53-german",
        # "nl": "jonatasgrosman/wav2vec2-large-xlsr-53-dutch",
        # "pl": "jonatasgrosman/wav2vec2-large-xlsr-53-polish",
        # "fi": "jonatasgrosman/wav2vec2-large-xlsr-53-finnish",

        "ia": "infinitejoy/wav2vec2-large-xls-r-300m-interlingua",
        "ta": "infinitejoy/wav2vec2-large-xls-r-300m-tamil-cv8",
        "or": "infinitejoy/wav2vec2-large-xls-r-300m-odia",
        "gn": "infinitejoy/wav2vec2-large-xls-r-300m-guarani",
        "cy": "infinitejoy/wav2vec2-large-xls-r-300m-welsh",
        "ur": "infinitejoy/wav2vec2-large-xls-r-300m-urdu",
        "ab": "infinitejoy/wav2vec2-large-xls-r-300m-abkhaz",
        "ba": "infinitejoy/wav2vec2-large-xls-r-300m-bashkir",
        "mr": "infinitejoy/wav2vec2-large-xls-r-300m-marathi-cv8",
        "lt": "infinitejoy/wav2vec2-large-xls-r-300m-lithuanian",
        "ha": "infinitejoy/wav2vec2-large-xls-r-300m-hausa",
        "br": "infinitejoy/wav2vec2-large-xls-r-300m-breton",
        "cv": "infinitejoy/wav2vec2-large-xls-r-300m-chuvash",
        "hy": "infinitejoy/wav2vec2-large-xls-r-300m-armenian",
        "mt": "infinitejoy/wav2vec2-large-xls-r-300m-maltese",
        "as": "infinitejoy/wav2vec2-large-xls-r-300m-assamese",
        "tt": "infinitejoy/wav2vec2-large-xls-r-300m-tatar",
        "ky": "infinitejoy/wav2vec2-large-xls-r-300m-kyrgyz",
        "ga": "infinitejoy/wav2vec2-large-xls-r-300m-irish",
        "ka": "infinitejoy/wav2vec2-large-xls-r-300m-georgian",
        "sk": "infinitejoy/wav2vec2-large-xls-r-300m-slovak",
        "sl": "infinitejoy/wav2vec2-large-xls-r-300m-slovenian",
        "mn": "infinitejoy/wav2vec2-large-xls-r-300m-mongolian",
        "ro": "infinitejoy/wav2vec2-large-xls-r-300m-romanian",
        "ku": "infinitejoy/wav2vec2-large-xls-r-300m-kurdish",
        "bg": "infinitejoy/wav2vec2-large-xls-r-300m-bulgarian",
        "hi": "infinitejoy/wav2vec2-large-xls-r-300m-hindi",
        "bas": "infinitejoy/wav2vec2-large-xls-r-300m-basaa",
        "sah": "infinitejoy/wav2vec2-large-xls-r-300m-sakha",
        "cnh": "infinitejoy/wav2vec2-large-xls-r-300m-hakha-chin",
        "rm-vallader": "infinitejoy/wav2vec2-large-xls-r-300m-romansh-vallader",
        "rm-sursilv": "infinitejoy/wav2vec2-large-xls-r-300m-romansh-sursilvan",
        # "fi": "infinitejoy/wav2vec2-large-xls-r-300m-finnish",
        # "hu": "infinitejoy/wav2vec2-large-xls-r-300m-hungarian",
        # "el": "infinitejoy/wav2vec2-large-xls-r-300m-greek",
        # "gl": "infinitejoy/wav2vec2-large-xls-r-300m-galician",
        # "ar": "infinitejoy/wav2vec2-large-xls-r-300m-arabic",
        # "id": "infinitejoy/wav2vec2-large-xls-r-300m-indonesian",

        "cz": "arampacha/wav2vec2-large-xlsr-czech",
        "uk": "arampacha/wav2vec2-large-xlsr-ukrainian",

        # "sv": "KBLab/wav2vec2-base-voxpopuli-sv-swedish",  # <- smaller
        "sv": "KBLab/wav2vec2-large-voxpopuli-sv-swedish",

        "id": "indonesian-nlp/wav2vec2-large-xlsr-indonesian",
        "lg": "indonesian-nlp/wav2vec2-luganda",
        "jv": "indonesian-nlp/wav2vec2-indonesian-javanese-sundanese",
        "su": "indonesian-nlp/wav2vec2-indonesian-javanese-sundanese",

        "da": "vachonni/wav2vec2-large-xls-r-300m-dansk-CV-80",
        "lb": "Lemswasabi/wav2vec2-large-xlsr-53-842h-luxembourgish-14h-with-lm",
        "ko": "kresnik/wav2vec2-large-xlsr-korean",
        "vi": "nguyenvulebinh/wav2vec2-base-vietnamese-250h",
        "ml": "Bluecast/wav2vec2-Malayalam",
        "hk": "voidful/wav2vec2-large-xlsr-53-hk",
        "bn": "arijitx/wav2vec2-large-xlsr-bengali",
        "eo": "cpierse/wav2vec2-large-xlsr-53-esperanto",
        "te": "anuragshas/wav2vec2-large-xlsr-53-telugu",
        "tr": "mpoyraz/wav2vec2-xls-r-300m-cv7-turkish",
        "ne": "gagan3012/wav2vec2-xlsr-nepali",
        "he": "imvladikon/wav2vec2-large-xlsr-53-hebrew",
        "sw": "alokmatta/wav2vec2-large-xlsr-53-sw",
        "pa": "kingabzpro/wav2vec2-large-xlsr-53-punjabi",
        # "sv": "KBLab/wav2vec2-large-xlsr-53-swedish",
        # "mr": "sumedh/wav2vec2-large-xlsr-marathi",
        # "ta": "Amrrs/wav2vec2-large-xlsr-53-tamil",
        # "or": "theainerd/wav2vec2-large-xlsr-53-odia",
        # "mt": "Akashpb13/xlsr_maltese_wav2vec2",
        # "ur": "kingabzpro/wav2vec2-urdu",
        # "lt": "m3hrdadfi/wav2vec2-large-xlsr-lithuanian",
        # "sl": "mrshu/wav2vec2-large-xlsr-slovene",
        # "mn": "bayartsogt/wav2vec2-large-xlsr-mongolian-v1",
        # "ro": "anton-l/wav2vec2-large-xlsr-53-romanian",
        # "ky": "aismlv/wav2vec2-large-xlsr-kyrgyz",
        # "pt": "lgris/wav2vec2-large-xlsr-open-brazilian-portuguese",
        # "pt": "lgris/wav2vec2-large-xlsr-open-brazilian-portuguese-v2",
        # "pt": "Edresson/wav2vec2-large-xlsr-coraa-portuguese"
    }

    def __init__(self, config: dict = None):
        super().__init__(config)
        model = self.config.get("model")
        lang = self.lang.split("-")[0]
        if not model and lang in self.LANG2MODEL:
            model = self.LANG2MODEL[lang]
        if not model:
            raise ValueError(f"'lang' {lang} not supported, a 'model' needs to be explicitly set in config file")
        self.processor = Wav2Vec2Processor.from_pretrained(model)
        self.asr_model = Wav2Vec2ForCTC.from_pretrained(model)
        if self.config.get("use_cuda"):
            self.asr_model.to("cuda")

    @classproperty
    def available_languages(cls) -> set:
        return set(standardize_lang_tag(t) for t in cls.LANG2MODEL.keys())

    def transcribe_file(self, file_path: str) -> str:
        waveform, sample_rate = torchaudio.load(file_path)
        # Resample if the audio is not at 16kHz
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        inputs = self.processor(waveform.squeeze().numpy(), sampling_rate=16_000, return_tensors="pt", padding=True)
        if self.config.get("use_cuda"):
            inputs = inputs.to("cuda")
        with torch.no_grad():
            logits = self.asr_model(inputs.input_values, attention_mask=inputs.attention_mask).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        return self.processor.batch_decode(predicted_ids)[0]

    def execute(self, audio: AudioData, language: Optional[str] = None):
        language = language or self.lang
        if language.split("-")[0] not in self.available_languages:
            raise ValueError(f"'lang' {language} not supported")
        with NamedTemporaryFile("wb", suffix=".wav") as f:
            f.write(audio.get_wav_data())
            transcription = self.transcribe_file(f.name)
        return transcription


if __name__ == "__main__":
    b = Wav2VecSTT({"lang": "pt", "model": "jonatasgrosman/wav2vec2-xls-r-1b-portuguese"})
    print(sorted(list(b.available_languages)))
    from speech_recognition import Recognizer, AudioFile

    eu = "/home/miro/PycharmProjects/ovos-stt-wav2vec-plugin/9ooDUDs5.wav"
    with AudioFile(eu) as source:
        audio = Recognizer().record(source)

    a = b.execute(audio, language="pt")
    print(a)
    # ten en conta que as funcionarlidades incluídas nesta páxino ofrécense unicamente con fins de demostración se tes algún comentario subxestión ou detectas algún problema durante a demostración ponte en contacto con nosco
