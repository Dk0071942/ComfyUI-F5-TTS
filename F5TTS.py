from pathlib import Path
import os.path
from .Install import Install
import subprocess
import wave
import torchaudio
import hashlib
import folder_paths
import tempfile
import soundfile as sf
import shutil
import sys
import numpy as np
import re
import io
import torch
from comfy.utils import ProgressBar
from cached_path import cached_path
sys.path.append(Install.f5TTSPath)
from model import DiT,UNetT,CFM # noqa E402
from model.utils_infer import ( # noqa E402
    #    load_model,
    preprocess_ref_audio_text,
    infer_process,
)
from model.utils import (
    load_checkpoint,
    get_tokenizer,
)
sys.path.pop()



class F5TTSCreate:
    voice_reg = re.compile(r"\{(\w+)\}")
    model_types = ["F5", "E2"]
    ode_methods = [
        "euler",
        "midpoint",
        "rk4",
        "explicit_adams",
        "implicit_adams",

        # adaptive ode methods gets "underflow in dt" error
        # https://github.com/rtqichen/torchdiffeq/issues/57
        # "dopri8",
        # "dopri5",
        # "bosh3",
        # "fehlberg2",
        # "adaptive_heun",

        # scipy_solvers just uses GPU, doesn't finish
        # 'scipy_solver:RK45',
        # 'scipy_solver:RK23',
        # 'scipy_solver:DOP853',
        # 'scipy_solver:Radau',
        # 'scipy_solver:BDF',
        # 'scipy_solver:LSODA',
    ]

    tooltip_rtol = "Relative tolerance. Leave at zero for default."
    tooltip_atol = "Relative tolerance. Leave at zero for default."

    def is_voice_name(self, word):
        return self.voice_reg.match(word.strip())

    def get_voice_names(self, chunks):
        voice_names = {}
        for text in chunks:
            match = self.is_voice_name(text)
            if match:
                voice_names[match[1]] = True
        return voice_names

    def split_text(self, speech):
        reg1 = r"(?=\{\w+\})"
        return re.split(reg1, speech)

    @staticmethod
    def load_voice(ref_audio, ref_text):
        main_voice = {"ref_audio": ref_audio, "ref_text": ref_text}

        main_voice["ref_audio"], main_voice["ref_text"] = preprocess_ref_audio_text( # noqa E501
            ref_audio, ref_text
        )
        return main_voice

    def load_model(self, model, ode_method, rtol, atol):
        models = {
            "F5": self.load_f5_model,
            "E2": self.load_e2_model,
        }
        return models[model](ode_method, rtol, atol)

    def load_e2_model(self, ode_method, rtol, atol):
        model_cls = UNetT
        model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)
        repo_name = "E2-TTS"
        exp_name = "E2TTS_Base"
        ckpt_step = 1200000
        ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{exp_name}/model_{ckpt_step}.safetensors")) # noqa E501
        vocab_file = self.get_vocab_file()
        ode_options = {}
        ode_arr = ode_method.split(":")
        ode_method2 = ode_method
        if len(ode_arr) > 1:
            ode_method2 = ode_arr[0]
            ode_options["options"] = {}
            ode_options["options"]["solver"] = ode_arr[1]
        if rtol != 0:
            ode_options["rtol"] = rtol
        if atol != 0:
            ode_options["atol"] = atol
        ema_model = self.load_model2(
            model_cls, model_cfg,
            ckpt_file, vocab_file, ode_method2, ode_options=ode_options
            )
        return ema_model

    def get_vocab_file(self):
        return os.path.join(
            Install.f5TTSPath, "data/Emilia_ZH_EN_pinyin/vocab.txt"
            )

    def load_f5_model(self, ode_method, rtol, atol):
        model_cls = DiT
        model_cfg = dict(
            dim=1024, depth=22, heads=16,
            ff_mult=2, text_dim=512, conv_layers=4
            )
        repo_name = "F5-TTS"
        exp_name = "F5TTS_Base"
        ckpt_step = 1200000
        ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{exp_name}/model_{ckpt_step}.safetensors")) # noqa E501
        vocab_file = self.get_vocab_file()
        ode_options = {}
        ode_arr = ode_method.split(":")
        ode_method2 = ode_method
        if len(ode_arr) > 1:
            ode_method2 = ode_arr[0]
            ode_options["options"] = {}
            ode_options["options"]["solver"] = ode_arr[1]
        if rtol != 0:
            ode_options["rtol"] = rtol
        if atol != 0:
            ode_options["atol"] = atol
        ema_model = self.load_model2(
            model_cls, model_cfg,
            ckpt_file, vocab_file, ode_method2, ode_options=ode_options
            )
        return ema_model

    def load_model2(self, model_cls, model_cfg, ckpt_path, vocab_file="", ode_method="euler", use_ema=True, ode_options={}):
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        target_sample_rate = 24000
        n_mel_channels = 100
        hop_length = 256
        if vocab_file == "":
            vocab_file = "Emilia_ZH_EN"
            tokenizer = "pinyin"
        else:
            tokenizer = "custom"

        print("\nvocab : ", vocab_file)
        print("tokenizer : ", tokenizer)
        print("model : ", ckpt_path, "\n")

        vocab_char_map, vocab_size = get_tokenizer(vocab_file, tokenizer)
        ode_base_options=dict(
            method=ode_method,
        )
        print(ode_base_options)
        odeint_kwargs={**ode_base_options, **ode_options}
        print(odeint_kwargs)
        model = CFM(
            transformer=model_cls(**model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
            mel_spec_kwargs=dict(
                target_sample_rate=target_sample_rate,
                n_mel_channels=n_mel_channels,
                hop_length=hop_length,
            ),
            odeint_kwargs=odeint_kwargs,
    #        dict(
    #            method=ode_method,
    #        ),
            vocab_char_map=vocab_char_map,
        ).to(device)

        model = load_checkpoint(model, ckpt_path, device, use_ema=use_ema)

        return model 

    def generate_audio(self, voices, model_obj, chunks):
        frame_rate = 44100
        generated_audio_segments = []
        pbar = ProgressBar(len(chunks))
        for text in chunks:
            print("text:"+text)
            match = self.is_voice_name(text)
            if match:
                voice = match[1]
            else:
                print("No voice tag found, using main.")
                voice = "main"
            if voice not in voices:
                print(f"Voice {voice} not found, using main.")
                voice = "main"
            text = F5TTSCreate.voice_reg.sub("", text)
            gen_text = text.strip()
            ref_audio = voices[voice]["ref_audio"]
            ref_text = voices[voice]["ref_text"]
            print(f"Voice: {voice}")
            print("text:"+text)
            audio, final_sample_rate, spectragram = infer_process(
                ref_audio, ref_text, gen_text, model_obj
                )
            generated_audio_segments.append(audio)
            frame_rate = final_sample_rate
            pbar.update(1)

        if generated_audio_segments:
            final_wave = np.concatenate(generated_audio_segments)
        wave_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(wave_file.name, final_wave, frame_rate)
        wave_file.close()

        waveform, sample_rate = torchaudio.load(wave_file.name)
        audio = {
            "waveform": waveform.unsqueeze(0),
            "sample_rate": sample_rate
            }
        os.unlink(wave_file.name)
        return audio

    def create(self, voices, chunks, model, ode_method, rtol, atol):
        model_obj = self.load_model(model, ode_method, rtol, atol)
        return self.generate_audio(voices, model_obj, chunks)


class F5TTSAudioInputs:
    def __init__(self):
        self.wave_file = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sample_audio": ("AUDIO",),
                "sample_text": ("STRING", {"default": "Text of sample_audio"}),
                "speech": ("STRING", {
                    "multiline": True,
                    "default": "This is what I want to say"
                }),
                "model": (F5TTSCreate.model_types,),
                "ode_method": (
                    F5TTSCreate.ode_methods,
                    {"tooltip": "The Ordinary Differential Equation"},
                    ),
                "relative_tolerance": ("FLOAT", {
                    "display": "number", "step": 0.00001,
                    "tooltip": F5TTSCreate.tooltip_rtol,
                    }),
                "absolute_tolerance": ("FLOAT", {
                    "display": "number", "step": 0.00001,
                    "tooltip": F5TTSCreate.tooltip_atol,
                }),
            },
        }

    CATEGORY = "audio"

    RETURN_TYPES = ("AUDIO", )
    FUNCTION = "create"

    def load_voice_from_input(self, sample_audio, sample_text):
        self.wave_file = tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False
            )
        for (batch_number, waveform) in enumerate(
                sample_audio["waveform"].cpu()):
            buff = io.BytesIO()
            torchaudio.save(
                buff, waveform, sample_audio["sample_rate"], format="WAV"
                )
            with open(self.wave_file.name, 'wb') as f:
                f.write(buff.getbuffer())
            break
        r = F5TTSCreate.load_voice(self.wave_file.name, sample_text)
        return r

    def remove_wave_file(self):
        if self.wave_file is not None:
            try:
                os.unlink(self.wave_file.name)
                self.wave_file = None
            except Exception as e:
                print("F5TTS: Cannot remove? "+self.wave_file.name)
                print(e)

    def create(
        self,
        sample_audio, sample_text, speech,
        model="F5", ode_method="euler",
        relative_tolerance=0, absolute_tolerance=0
    ):
        try:
            main_voice = self.load_voice_from_input(sample_audio, sample_text)

            f5ttsCreate = F5TTSCreate()

            voices = {}
            chunks = f5ttsCreate.split_text(speech)
            voices['main'] = main_voice

            audio = f5ttsCreate.create(
                voices, chunks, model, ode_method,
                relative_tolerance, absolute_tolerance
                )
        finally:
            self.remove_wave_file()
        return (audio, )

    @classmethod
    def IS_CHANGED(s, sample_audio,
                   sample_text, speech, ode_method,
                   relative_tolerance, absolute_tolerance,
                   ):
        m = hashlib.sha256()
        m.update(sample_text)
        m.update(sample_audio)
        m.update(speech)
        m.update(ode_method)
        m.update(relative_tolerance)
        m.update(absolute_tolerance)
        return m.digest().hex()


class F5TTSAudio:
    def __init__(self):
        self.use_cli = False

    @staticmethod
    def get_txt_file_path(file):
        p = Path(file)
        return os.path.join(os.path.dirname(file), p.stem + ".txt")

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = folder_paths.filter_files_content_types(
            os.listdir(input_dir), ["audio", "video"]
            )
        filesWithTxt = []
        for file in files:
            txtFile = F5TTSAudio.get_txt_file_path(file)
            if os.path.isfile(os.path.join(input_dir, txtFile)):
                filesWithTxt.append(file)
        filesWithTxt = sorted(filesWithTxt)

        return {
            "required": {
                "sample": (filesWithTxt, {"audio_upload": True}),
                "speech": ("STRING", {
                    "multiline": True,
                    "default": "This is what I want to say"
                }),
                "model": (F5TTSCreate.model_types,),
                "ode_method": (
                    F5TTSCreate.ode_methods,
                    {"tooltip": "The Ordinary Differential Equation"},
                    ),
                "relative_tolerance": ("FLOAT", {
                    "display": "number", "step": 0.00001,
                    "tooltip": F5TTSCreate.tooltip_rtol,
                    }),
                "absolute_tolerance": ("FLOAT", {
                    "display": "number", "step": 0.00001,
                    "tooltip": F5TTSCreate.tooltip_atol,
                }),
                "seed": ("INT", {
                    "display": "number", "step": 1,
                }),



            }
        }

    CATEGORY = "audio"

    RETURN_TYPES = ("AUDIO", )
    FUNCTION = "create"

    def create_with_cli(self, audio_path, audio_text, speech, output_dir):
        subprocess.run(
            [
                "python", "inference-cli.py", "--model", "F5-TTS",
                "--ref_audio", audio_path, "--ref_text", audio_text,
                "--gen_text", speech,
                "--output_dir", output_dir
            ],
            cwd=Install.f5TTSPath
        )
        output_audio = os.path.join(output_dir, "out.wav")
        with wave.open(output_audio, "rb") as wave_file:
            frame_rate = wave_file.getframerate()

        waveform, sample_rate = torchaudio.load(output_audio)
        audio = {"waveform": waveform.unsqueeze(0), "sample_rate": frame_rate}
        return audio

    def load_voice_from_file(self, sample):
        input_dir = folder_paths.get_input_directory()
        txt_file = os.path.join(
            input_dir,
            F5TTSAudio.get_txt_file_path(sample)
            )
        audio_text = ''
        with open(txt_file, 'r') as file:
            audio_text = file.read()
        audio_path = folder_paths.get_annotated_filepath(sample)
        return F5TTSCreate.load_voice(audio_path, audio_text)

    def load_voices_from_files(self, sample, voice_names):
        voices = {}
        p = Path(sample)
        for voice_name in voice_names:
            if voice_name == "main":
                continue
            sample_file = os.path.join(
                os.path.dirname(sample),
                "{stem}.{voice_name}{suffix}".format(
                    stem=p.stem,
                    voice_name=voice_name,
                    suffix=p.suffix
                    )
                )
            print("voice:"+voice_name+","+sample_file+','+sample)
            voices[voice_name] = self.load_voice_from_file(sample_file)
        return voices

    def create(
        self,
        sample, speech,
        model="F5", ode_method="euler",
        relative_tolerance=0, absolute_tolerance=0,
        seed=-1
    ):
        # Install.check_install()
        main_voice = self.load_voice_from_file(sample)

        f5ttsCreate = F5TTSCreate()
        if self.use_cli:
            # working...
            output_dir = tempfile.mkdtemp()
            audio_path = folder_paths.get_annotated_filepath(sample)
            audio = self.create_with_cli(
                audio_path, main_voice["ref_text"],
                speech, output_dir
                )
            shutil.rmtree(output_dir)
        else:
            chunks = f5ttsCreate.split_text(speech)
            voice_names = f5ttsCreate.get_voice_names(chunks)
            voices = self.load_voices_from_files(sample, voice_names)
            voices['main'] = main_voice

            if seed >= 0:
                torch.manual_seed(seed)
            else:
                torch.random.seed()
            audio = f5ttsCreate.create(
                voices, chunks, model, ode_method,
                relative_tolerance, absolute_tolerance
            )
        return (audio, )

    @classmethod
    def IS_CHANGED(s, sample, speech, ode_method,
                   relative_tolerance, absolute_tolerance,
                   ):
        m = hashlib.sha256()
        audio_path = folder_paths.get_annotated_filepath(sample)
        audio_txt_path = F5TTSAudio.get_txt_file_path(audio_path)
        last_modified_timestamp = os.path.getmtime(audio_path)
        txt_last_modified_timestamp = os.path.getmtime(audio_txt_path)
        m.update(audio_path)
        m.update(str(last_modified_timestamp))
        m.update(str(txt_last_modified_timestamp))
        m.update(speech)
        m.update(ode_method)
        m.update(relative_tolerance)
        m.update(absolute_tolerance)
        return m.digest().hex()
