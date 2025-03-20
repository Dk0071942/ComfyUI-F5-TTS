from pathlib import Path
import os.path
from .Install import Install
import subprocess
import math
import wave
import torch
import torchaudio
import hashlib
import folder_paths
import tempfile
import soundfile as sf
import sys
import numpy as np
import re
import io
from comfy.utils import ProgressBar
import comfy
from cached_path import cached_path

# check_install will download the f5-tts if the submodule wasn't downloaded.
Install.check_install()

f5tts_path = os.path.join(Install.f5TTSPath, "src")
sys.path.insert(0, f5tts_path)
from f5_tts.model import DiT,UNetT # noqa E402
from f5_tts.infer.utils_infer import ( # noqa E402
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    infer_process,
    remove_silence_edges,
)
sys.path.remove(f5tts_path)


class F5TTSCreate:
    voice_reg = re.compile(r"\{([^\}]+)\}")
    model_types = [
        "F5",
        "F5-HI",
        "F5-JP",
        "F5-FR",
        "F5-DE",
        "F5-IT",
        "F5-ES",
        "E2",
    ]
    vocoder_types = ["vocos", "bigvgan"]
    tooltip_seed = "Seed. -1 = random"
    tooltip_speed = "Speed. >1.0 slower. <1.0 faster"
    DEFAULT_MAX_CHARS = 135  # Default maximum characters per chunk
    
    # Default model configurations
    DEFAULT_MODEL_CFG = dict(
        dim=1024,
        depth=22,
        heads=16,
        ff_mult=2,
        text_dim=512,
        text_mask_padding=False,
        conv_layers=4,
        pe_attn_head=1,
    )
    
    HINDI_MODEL_CFG = dict(
        dim=768,
        depth=18,
        heads=12,
        ff_mult=2,
        text_dim=512,
        text_mask_padding=False,
        conv_layers=4,
        pe_attn_head=1,
        checkpoint_activations=False,
    )
    
    E2_MODEL_CFG = dict(
        dim=1024,
        depth=24,
        heads=16,
        ff_mult=4,
        text_mask_padding=False,
        pe_attn_head=1,
    )

    def get_model_types():
        model_types = F5TTSCreate.model_types[:]
        models_path = folder_paths.get_folder_paths("checkpoints")
        for model_path in models_path:
            f5_model_path = os.path.join(model_path, 'F5-TTS')
            if os.path.isdir(f5_model_path):
                for file in os.listdir(f5_model_path):
                    p = Path(file)
                    if (
                        p.suffix in folder_paths.supported_pt_extensions
                        and os.path.isfile(os.path.join(f5_model_path, file))
                    ):
                        txtFile = F5TTSCreate.get_txt_file_path(
                            os.path.join(f5_model_path, file)
                        )

                        if (
                            os.path.isfile(txtFile)
                        ):
                            model_types.append("model://"+file)
        return model_types

    @staticmethod
    def get_txt_file_path(file):
        p = Path(file)
        return os.path.join(os.path.dirname(file), p.stem + ".txt")

    def is_voice_name(self, word):
        return self.voice_reg.match(word.strip())

    def get_voice_names(self, chunks):
        voice_names = {}
        for text in chunks:
            match = self.is_voice_name(text)
            if match:
                voice_names[match[1]] = True
        return voice_names

    def smart_chunk_text(self, text, max_chars=DEFAULT_MAX_CHARS):
        """
        Splits the input text into chunks, each with a maximum number of characters.
        Respects UTF-8 encoding and natural sentence boundaries.

        Args:
            text (str): The text to be split.
            max_chars (int): The maximum number of characters per chunk.

        Returns:
            List[str]: A list of text chunks.
        """
        chunks = []
        current_chunk = ""
        
        # Split the text into sentences based on punctuation followed by whitespace
        # Handles both English and Chinese punctuation
        sentences = re.split(r"(?<=[;:,.!?])\s+|(?<=[；：，。！？])", text)
        
        for sentence in sentences:
            # Calculate UTF-8 encoded length of current chunk and sentence
            current_length = len(current_chunk.encode("utf-8"))
            sentence_length = len(sentence.encode("utf-8"))
            
            # Check if adding this sentence would exceed max_chars
            if current_length + sentence_length <= max_chars:
                # Add sentence to current chunk
                # Add space after sentence if it ends with a single-byte character
                if sentence and len(sentence[-1].encode("utf-8")) == 1:
                    current_chunk += sentence + " "
                else:
                    current_chunk += sentence
            else:
                # Current chunk is full, save it and start a new one
                if current_chunk:
                    chunks.append(current_chunk.strip())
                # Start new chunk with current sentence
                current_chunk = sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

    def split_text(self, speech):
        """
        Splits text into chunks based on voice tags and then applies smart chunking
        for long text segments.
        """
        # First split by voice tags
        voice_chunks = re.split(r"(?=\{[^\}]+\})", speech)
        
        # Process each voice chunk
        final_chunks = []
        for chunk in voice_chunks:
            if not chunk.strip():
                continue
                
            # If chunk contains a voice tag, process it separately
            if self.is_voice_name(chunk):
                final_chunks.append(chunk)
            else:
                # Apply smart chunking to non-voice-tagged text
                final_chunks.extend(self.smart_chunk_text(chunk))
        
        return final_chunks

    @staticmethod
    def load_voice(ref_audio, ref_text):
        # First get the original audio duration
        audio, sr = torchaudio.load(ref_audio)
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        original_duration = audio.shape[-1] / sr
        
        # Preprocess the audio
        main_voice = {"ref_audio": ref_audio, "ref_text": ref_text}
        main_voice["ref_audio"], main_voice["ref_text"] = preprocess_ref_audio_text(
            ref_audio, ref_text
        )
        
        # Get the preprocessed audio duration
        audio, sr = torchaudio.load(main_voice["ref_audio"])
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        preprocessed_duration = audio.shape[-1] / sr
        
        # Calculate the duration ratio
        main_voice["duration_ratio"] = original_duration / preprocessed_duration
        
        return main_voice

    def get_model_funcs(self):
        return {
            "F5": self.load_f5_model,
            "F5-HI": self.load_f5_model_hi,
            "F5-JP": self.load_f5_model_jp,
            "F5-FR": self.load_f5_model_fr,
            "F5-DE": self.load_f5_model_de,
            "F5-IT": self.load_f5_model_it,
            "F5-ES": self.load_f5_model_es,
            "E2": self.load_e2_model,
        }

    def get_vocoder(self, vocoder_name):
        if vocoder_name == "vocos":
            os.path.join(Install.f5TTSPath, "checkpoints/vocos-mel-24khz")
        elif vocoder_name == "bigvgan":
            os.path.join(Install.f5TTSPath, "checkpoints/bigvgan_v2_24khz_100band_256x") # noqa E501

    def load_vocoder(self, vocoder_name):
        sys.path.insert(0, f5tts_path)
        vocoder = load_vocoder(vocoder_name=vocoder_name)
        sys.path.remove(f5tts_path)
        return vocoder

    def load_model(self, model, vocoder_name):
        model_funcs = self.get_model_funcs()
        if model in model_funcs:
            return model_funcs[model](vocoder_name)
        else:
            return self.load_f5_model_url(model, vocoder_name)

    def get_vocab_file(self):
        return os.path.join(
            Install.f5TTSPath, "data/Emilia_ZH_EN_pinyin/vocab.txt"
            )

    def load_e2_model(self, vocoder):
        model_cls = UNetT
        repo_name = "E2-TTS"
        exp_name = "E2TTS_Base"
        ckpt_step = 1200000
        ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{exp_name}/model_{ckpt_step}.safetensors")) # noqa E501
        vocab_file = self.get_vocab_file()
        vocoder_name = "vocos"
        ema_model = load_model(
            model_cls, 
            self.E2_MODEL_CFG,
            ckpt_file, 
            vocab_file=vocab_file,
            mel_spec_type=vocoder_name,
        )
        vocoder = self.load_vocoder(vocoder_name)
        return (ema_model, vocoder, vocoder_name)

    def load_f5_model(self, vocoder):
        repo_name = "F5-TTS"
        extension = "safetensors"
        if vocoder == "bigvgan":
            exp_name = "F5TTS_Base_bigvgan"
            ckpt_step = 1250000
            extension = "pt"
        else:
            exp_name = "F5TTS_Base"
            ckpt_step = 1200000
        return self.load_f5_model_url(
            f"hf://SWivid/{repo_name}/{exp_name}/model_{ckpt_step}.{extension}", # noqa E501
            vocoder,
            model_cfg=self.DEFAULT_MODEL_CFG,
        )

    def load_f5_model_jp(self, vocoder):
        return self.load_f5_model_url(
            "hf://Jmica/F5TTS/JA_8500000/model_8499660.pt",
            vocoder,
            "hf://Jmica/F5TTS/JA_8500000/vocab_updated.txt"
            )

    def load_f5_model_fr(self, vocoder):
        return self.load_f5_model_url(
            "hf://RASPIAUDIO/F5-French-MixedSpeakers-reduced/model_1374000.pt", # noqa E501
            vocoder,
            "hf://RASPIAUDIO/F5-French-MixedSpeakers-reduced/vocab.txt" # noqa E501
            )

    def load_f5_model_de(self, vocoder):
        return self.load_f5_model_url(
            "hf://aihpi/F5-TTS-German/F5TTS_Base/model_420000.safetensors", # noqa E501
            vocoder,
            "hf://aihpi/F5-TTS-German/vocab.txt" # noqa E501
            )

    def load_f5_model_it(self, vocoder):
        return self.load_f5_model_url(
            "hf://alien79/F5-TTS-italian/model_159600.safetensors", # noqa E501
            vocoder,
            "hf://alien79/F5-TTS-italian/vocab.txt" # noqa E501
            )

    def load_f5_model_es(self, vocoder):
        return self.load_f5_model_url(
            "hf://jpgallegoar/F5-Spanish/model_1200000.safetensors", # noqa E501
            vocoder,
            "hf://jpgallegoar/F5-Spanish/vocab.txt" # noqa E501
            )

    def cached_path(self, url):
        if url.startswith("model:"):
            path = re.sub("^model:/*", "", url)
            models_path = folder_paths.get_folder_paths("checkpoints")
            for model_path in models_path:
                f5_model_path = os.path.join(model_path, 'F5-TTS')
                model_file = os.path.join(f5_model_path, path)
                if os.path.isfile(model_file):
                    return model_file
            raise FileNotFoundError("No model found: " + url)
            return None
        return str(cached_path(url)) # noqa E501

    def load_f5_model_hi(self, vocoder):
        return self.load_f5_model_url(
            "hf://SPRINGLab/F5-Hindi-24KHz/model_2500000.safetensors",
            "vocos",
            "hf://SPRINGLab/F5-Hindi-24KHz/vocab.txt",
            model_cfg=self.HINDI_MODEL_CFG,
        )

    def load_f5_model_url(
        self, url, vocoder_name, vocab_url=None, model_cfg=None
    ):
        vocoder = self.load_vocoder(vocoder_name)
        model_cls = DiT
        
        # Use default model configuration if none provided
        if model_cfg is None:
            model_cfg = self.DEFAULT_MODEL_CFG

        ckpt_file = str(self.cached_path(url)) # noqa E501

        if vocab_url is None:
            if url.startswith("model:"):
                vocab_file = F5TTSCreate.get_txt_file_path(ckpt_file)
            else:
                vocab_file = self.get_vocab_file()
        else:
            vocab_file = str(self.cached_path(vocab_url))

        # Load model with proper configuration
        ema_model = load_model(
            model_cls, 
            model_cfg,
            ckpt_file, 
            vocab_file=vocab_file,
            mel_spec_type=vocoder_name,
        )
        return (ema_model, vocoder, vocoder_name)

    def generate_audio(
        self, voices, model_obj, chunks, seed, vocoder, mel_spec_type,
        speed, cross_fade_duration=0.15, nfe_step=32, max_retries=3,
        cfg_strength=2.0, sway_sampling_coef=-1.0
    ):
        if seed >= 0:
            torch.manual_seed(seed)
        else:
            torch.random.seed()

        frame_rate = 44100
        generated_audio_segments = []
        pbar = ProgressBar(len(chunks))
        
        print("\n=== F5TTS Audio Generation Debug Info ===")
        print(f"Total chunks to process: {len(chunks)}")
        print(f"Cross-fade duration: {cross_fade_duration}")
        print(f"NFE steps: {nfe_step}")
        print(f"Speed: {speed}")
        print(f"CFG strength: {cfg_strength}")
        print(f"Sway sampling coefficient: {sway_sampling_coef}")
        
        # Get reference audio info
        ref_audio = None
        ref_text = None
        voice = "main"
        
        for text in chunks:
            match = self.is_voice_name(text)
            if match:
                voice = match[1]
            if voice not in voices:
                voice = "main"
            if ref_audio is None:
                ref_audio = voices[voice]["ref_audio"]
                ref_text = voices[voice]["ref_text"]
                # Check input audio duration
                audio, sr = torchaudio.load(ref_audio)
                if audio.shape[0] > 1:
                    audio = torch.mean(audio, dim=0, keepdim=True)
                duration = audio.shape[-1] / sr
                if duration > 12:
                    print(f"\nERROR: Input audio duration ({duration:.2f}s) exceeds maximum allowed duration (12s)")
                    print("Please use a shorter input audio or trim it to 12 seconds or less")
                    return None
        
        # Process each chunk
        for i, text in enumerate(chunks):
            print(f"\n--- Processing chunk {i+1}/{len(chunks)} ---")
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
            if gen_text == "":
                print(f"No text for {voice}, skip")
                continue
                
            ref_audio = voices[voice]["ref_audio"]
            ref_text = voices[voice]["ref_text"]
            print(f"Voice: {voice}")
            print(f"Text to generate: {gen_text}")
            
            # Generate audio with retry logic
            retry_count = 0
            while retry_count < max_retries:
                try:
                    audio, final_sample_rate, spectragram = infer_process(
                        ref_audio, ref_text, gen_text, model_obj,
                        vocoder=vocoder, mel_spec_type=mel_spec_type,
                        device=comfy.model_management.get_torch_device(),
                        cross_fade_duration=cross_fade_duration,
                        nfe_step=nfe_step,
                        speed=speed,
                        cfg_strength=cfg_strength,
                        sway_sampling_coef=sway_sampling_coef
                    )
                    
                    print(f"Generated audio length: {len(audio)/final_sample_rate:.2f}s")
                    generated_audio_segments.append(audio)
                    frame_rate = final_sample_rate
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    print(f"Error generating audio: {str(e)}")
                    retry_count += 1
                    if retry_count >= max_retries:
                        print(f"Failed to generate audio after {max_retries} attempts")
                        return None
                    # Adjust parameters for retry
                    cross_fade_duration *= 0.8
                    nfe_step *= 2
                    print(f"Retrying with adjusted parameters (attempt {retry_count + 1}/{max_retries})")
            
            pbar.update(1)

        if generated_audio_segments:
            print("\n=== Final Audio Generation ===")
            final_wave = np.concatenate(generated_audio_segments)
            total_duration = len(final_wave) / frame_rate
            print(f"Total audio duration: {total_duration:.2f}s")
            print(f"Total segments: {len(generated_audio_segments)}")
        else:
            print("\nERROR: No audio segments were generated")
            return None

        waveform = torch.from_numpy(final_wave).unsqueeze(0)
        audio = {
            "waveform": waveform.unsqueeze(0),
            "sample_rate": frame_rate
        }
        print("\n=== Generation Complete ===")
        return audio

    def create(
        self, voices, chunks, seed=-1, model="F5",
        vocoder_name="vocos", speed=1, cross_fade_duration=0.15, nfe_step=32,
        cfg_strength=2.0, sway_sampling_coef=-1.0
    ):
        (
            model_obj,
            vocoder,
            mel_spec_type
        ) = self.load_model(model, vocoder_name)
        return self.generate_audio(
            voices,
            model_obj,
            chunks, seed,
            vocoder, mel_spec_type=mel_spec_type,
            speed=speed,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef
        )

    def time_shift(self, audio, speed):
        import torch_time_stretch
        rate = audio['sample_rate']
        waveform = audio['waveform']

        new_waveform = torch_time_stretch.time_stretch(
            waveform,
            torch_time_stretch.Fraction(math.floor(speed*100), 100),
            rate
        )

        return {"waveform": new_waveform, "sample_rate": rate}


class F5TTSAudioInputs:
    def __init__(self):
        self.wave_file_name = None

    @classmethod
    def INPUT_TYPES(s):
        model_types = F5TTSCreate.get_model_types()
        return {
            "required": {
                "sample_audio": ("AUDIO",),
                "sample_text": ("STRING", {"default": "Text of sample_audio"}),
                "speech": ("STRING", {
                    "multiline": True,
                    "default": "This is what I want to say"
                }),
                "seed": ("INT", {
                    "display": "number", "step": 1,
                    "default": 1, "min": -1,
                    "tooltip": F5TTSCreate.tooltip_seed,
                }),
                "model": (model_types,),
                "vocoder": (F5TTSCreate.vocoder_types, {
                    "tooltip": "Most models are usally vocos",
                }),
                "speed": ("FLOAT", {
                    "default": 1.0,
                    "step": 0.01,
                    "tooltip": F5TTSCreate.tooltip_speed,
                }),
            },
            "optional": {
                "cfg_strength": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Classifier-free guidance strength. Higher values amplify the characteristics of the reference voice.",
                }),
                "sway_sampling_coef": ("FLOAT", {
                    "default": -1.0,
                    "min": -5.0,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Sway sampling coefficient. Affects the stability of the generated audio.",
                }),
            }
        }

    CATEGORY = "audio"

    RETURN_TYPES = ("AUDIO", )
    FUNCTION = "create"

    def load_voice_from_input(self, sample_audio, sample_text):
        wave_file = tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False
            )
        self.wave_file_name = wave_file.name
        wave_file.close()

        hasAudio = False
        for (batch_number, waveform) in enumerate(
            sample_audio["waveform"].cpu()
        ):
            buff = io.BytesIO()
            torchaudio.save(
                buff, waveform, sample_audio["sample_rate"], format="WAV"
                )
            with open(self.wave_file_name, 'wb') as f:
                f.write(buff.getbuffer())
            hasAudio = True
            break
        if not hasAudio:
            raise FileNotFoundError("No audio input")
        r = F5TTSCreate.load_voice(self.wave_file_name, sample_text)
        return r

    def remove_wave_file(self):
        if self.wave_file_name is not None:
            try:
                os.unlink(self.wave_file_name)
                self.wave_file_name = None
            except Exception as e:
                print("F5TTS: Cannot remove? "+self.wave_file_name)
                print(e)

    def create(
        self,
        sample_audio, sample_text,
        speech, seed=-1, model="F5", vocoder="vocos",
        speed=1, cfg_strength=2.0, sway_sampling_coef=-1.0
    ):
        try:
            main_voice = self.load_voice_from_input(sample_audio, sample_text)

            f5ttsCreate = F5TTSCreate()

            voices = {}
            chunks = f5ttsCreate.split_text(speech)
            voices['main'] = main_voice

            audio = f5ttsCreate.create(
                voices, chunks, seed, model, vocoder, speed,
                cfg_strength=cfg_strength, sway_sampling_coef=sway_sampling_coef
            )
            if speed != 1:
                audio = f5ttsCreate.time_shift(audio, speed)
        finally:
            self.remove_wave_file()
        return (audio, )

    @classmethod
    def IS_CHANGED(
        s, sample_audio, sample_text,
        speech, seed, model, vocoder, speed,
        cfg_strength=2.0, sway_sampling_coef=-1.0
    ):
        m = hashlib.sha256()
        m.update(str(sample_text).encode('utf-8'))
        m.update(str(sample_audio).encode('utf-8'))
        m.update(str(speech).encode('utf-8'))
        m.update(str(seed).encode('utf-8'))
        m.update(str(model).encode('utf-8'))
        m.update(str(vocoder).encode('utf-8'))
        m.update(str(speed).encode('utf-8'))
        m.update(str(cfg_strength).encode('utf-8'))
        m.update(str(sway_sampling_coef).encode('utf-8'))
        return m.digest().hex()


class F5TTSAudio:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        input_dirs = [
                "",
                'audio',
                'F5-TTS',
        ]
        files = []
        for dir_short in input_dirs:
            d = os.path.join(input_dir, dir_short)
            if os.path.exists(d):
                dir_files = folder_paths.filter_files_content_types(
                    os.listdir(d), ["audio", "video"]
                    )
                dir_files = [os.path.join(dir_short, s) for s in dir_files]
                files.extend(dir_files)
        filesWithTxt = []
        for file in files:
            txtFile = F5TTSCreate.get_txt_file_path(file)
            if os.path.isfile(os.path.join(input_dir, txtFile)):
                filesWithTxt.append(file)
        filesWithTxt = sorted(filesWithTxt)

        model_types = F5TTSCreate.get_model_types()

        return {
            "required": {
                "sample": (filesWithTxt, {"audio_upload": True}),
                "speech": ("STRING", {
                    "multiline": True,
                    "default": "This is what I want to say"
                }),
                "seed": ("INT", {
                    "display": "number", "step": 1,
                    "default": 1, "min": -1,
                    "tooltip": F5TTSCreate.tooltip_seed,
                }),
                "model": (model_types,),
                "vocoder": (F5TTSCreate.vocoder_types, {
                    "tooltip": "Most models are usally vocos",
                }),
                "speed": ("FLOAT", {
                    "default": 1.0,
                    "tooltip": F5TTSCreate.tooltip_speed,
                }),
            },
            "optional": {
                "cfg_strength": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Classifier-free guidance strength. Higher values amplify the characteristics of the reference voice.",
                }),
                "sway_sampling_coef": ("FLOAT", {
                    "default": -1.0,
                    "min": -5.0,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Sway sampling coefficient. Affects the stability of the generated audio.",
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
            F5TTSCreate.get_txt_file_path(sample)
            )
        audio_text = ''
        with open(txt_file, 'r', encoding='utf-8') as file:
            audio_text = file.read()
        audio_path = folder_paths.get_annotated_filepath(sample)
        print("audio_text")
        print(audio_text)
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
       sample, speech, seed=-2, model="F5", vocoder="vocos",
       speed=1, cfg_strength=2.0, sway_sampling_coef=-1.0
    ):
        # vocoder = "vocos"
        # Install.check_install()
        main_voice = self.load_voice_from_file(sample)

        f5ttsCreate = F5TTSCreate()

        chunks = f5ttsCreate.split_text(speech)
        voice_names = f5ttsCreate.get_voice_names(chunks)
        voices = self.load_voices_from_files(sample, voice_names)
        voices['main'] = main_voice

        audio = f5ttsCreate.create(
            voices, chunks, seed, model, vocoder, speed,
            cfg_strength=cfg_strength, sway_sampling_coef=sway_sampling_coef
        )
        if speed != 1:
            audio = f5ttsCreate.time_shift(audio, speed)
        return (audio, )

    @classmethod
    def IS_CHANGED(s, sample, speech, seed, model, vocoder, speed,
                  cfg_strength=2.0, sway_sampling_coef=-1.0):
        m = hashlib.sha256()
        audio_path = folder_paths.get_annotated_filepath(sample)
        audio_txt_path = F5TTSCreate.get_txt_file_path(audio_path)
        last_modified_timestamp = os.path.getmtime(audio_path)
        txt_last_modified_timestamp = os.path.getmtime(audio_txt_path)
        m.update(audio_path.encode('utf-8'))
        m.update(str(last_modified_timestamp).encode('utf-8'))
        m.update(str(txt_last_modified_timestamp).encode('utf-8'))
        m.update(str(speech).encode('utf-8'))
        m.update(str(seed).encode('utf-8'))
        m.update(str(model).encode('utf-8'))
        m.update(str(vocoder).encode('utf-8'))
        m.update(str(speed).encode('utf-8'))
        m.update(str(cfg_strength).encode('utf-8'))
        m.update(str(sway_sampling_coef).encode('utf-8'))
        return m.digest().hex()
