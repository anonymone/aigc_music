# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Updated to account for UI changes from https://github.com/rkfg/audiocraft/blob/long/app.py
# also released under the MIT license.

import argparse
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
import subprocess as sp
from tempfile import NamedTemporaryFile
import time
import typing as tp
import warnings
import logging
from flask import Flask, request, jsonify, send_from_directory, render_template
app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

import torch
import gradio as gr

from audiocraft.data.audio_utils import convert_audio
from audiocraft.data.audio import audio_write
from audiocraft.models import MusicGen



MODEL = None  # Last used model
IS_BATCHED = "facebook/MusicGen" in os.environ.get('SPACE_ID', '')
MAX_BATCH_SIZE = 12
BATCHED_DURATION = 15
INTERRUPTING = False
# We have to wrap subprocess call to clean a bit the log when using gr.make_waveform
_old_call = sp.call


def _call_nostderr(*args, **kwargs):
    # Avoid ffmpeg vomitting on the logs.
    kwargs['stderr'] = sp.DEVNULL
    kwargs['stdout'] = sp.DEVNULL
    _old_call(*args, **kwargs)


sp.call = _call_nostderr
# Preallocating the pool of processes.
pool = ProcessPoolExecutor(4)
pool.__enter__()


def interrupt():
    global INTERRUPTING
    INTERRUPTING = True


class FileCleaner:
    def __init__(self, file_lifetime: float = 3600):
        self.file_lifetime = file_lifetime
        self.files = []

    def add(self, path: tp.Union[str, Path]):
        self._cleanup()
        self.files.append((time.time(), Path(path)))

    def _cleanup(self):
        now = time.time()
        for time_added, path in list(self.files):
            if now - time_added > self.file_lifetime:
                if path.exists():
                    path.unlink()
                self.files.pop(0)
            else:
                break


file_cleaner = FileCleaner()


def make_waveform(*args, **kwargs):
    # Further remove some warnings.
    be = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = gr.make_waveform(*args, **kwargs)
        print("Make a video took", time.time() - be)
        return out


def load_model(version='melody'):
    global MODEL
    print("Loading model", version)
    if MODEL is None or MODEL.name != version:
        MODEL = MusicGen.get_pretrained(version)


def _do_predictions(texts, melodies, duration, progress=False, **gen_kwargs):
    MODEL.set_generation_params(duration=duration, **gen_kwargs)
    print("new batch", len(texts), texts, [None if m is None else (m[0], m[1].shape) for m in melodies])
    be = time.time()
    processed_melodies = []
    target_sr = 32000
    target_ac = 1
    for melody in melodies:
        if melody is None:
            processed_melodies.append(None)
        else:
            sr, melody = melody[0], torch.from_numpy(melody[1]).to(MODEL.device).float().t()
            if melody.dim() == 1:
                melody = melody[None]
            melody = melody[..., :int(sr * duration)]
            melody = convert_audio(melody, sr, target_sr, target_ac)
            processed_melodies.append(melody)

    if any(m is not None for m in processed_melodies):
        outputs = MODEL.generate_with_chroma(
            descriptions=texts,
            melody_wavs=processed_melodies,
            melody_sample_rate=target_sr,
            progress=progress,
        )
    else:
        outputs = MODEL.generate(texts, progress=progress)

    outputs = outputs.detach().cpu().float()
    out_files = []
    for output in outputs:
        with NamedTemporaryFile("wb", suffix=".wav", delete=False) as file:
            audio_write(
                file.name, output, MODEL.sample_rate, strategy="loudness",
                loudness_headroom_db=16, loudness_compressor=True, add_suffix=False)
            out_files.append(pool.submit(make_waveform, file.name))
            file_cleaner.add(file.name)
    res = [out_file.result() for out_file in out_files]
    for file in res:
        file_cleaner.add(file)
    print("batch finished", len(texts), time.time() - be)
    print("Tempfiles currently stored: ", len(file_cleaner.files))
    return res


def predict_batched(texts, melodies):
    max_text_length = 512
    texts = [text[:max_text_length] for text in texts]
    load_model('melody')
    res = _do_predictions(texts, melodies, BATCHED_DURATION)
    return [res]


def predict_full(model, text, melody, duration, topk, topp, temperature, cfg_coef, progress=gr.Progress()):
    global INTERRUPTING
    INTERRUPTING = False
    if temperature < 0:
        raise gr.Error("Temperature must be >= 0.")
    if topk < 0:
        raise gr.Error("Topk must be non-negative.")
    if topp < 0:
        raise gr.Error("Topp must be non-negative.")

    topk = int(topk)
    load_model(model)

    def _progress(generated, to_generate):
        progress((generated, to_generate))
        if INTERRUPTING:
            raise gr.Error("Interrupted.")
    if progress is not None:
        MODEL.set_custom_progress_callback(_progress)

    outs = _do_predictions(
        [text], [melody], duration, progress=True,
        top_k=topk, top_p=topp, temperature=temperature, cfg_coef=cfg_coef)
    return outs[0]


def toggle_audio_src(choice):
    if choice == "mic":
        return gr.update(source="microphone", value=None, label="Microphone")
    else:
        return gr.update(source="upload", value=None, label="File")

# /var/folders/b1/0fd1b6hs7lz0fm_mh346lybm0000gn/T/gradio/fccba26d1a8b1a2c10f338eba922eb8dde157bc7/tmp7fjm7ml4.mp4
def server_run(textContetn, audio_file, progress):
    outputs = predict_full("small", textContetn, None, 5, 25, 0, 1.0, 3.0, progress=progress)
    app.logger.info("music is saved at " + outputs)
    return outputs

# flask web service
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/music')
def gen_music():
    app.logger.info("generating music...")
    prompt = request.args.get('prompt')
    if not prompt:
        jsonify({
            'success': False,
            'error_code': 5,
            'error_msg': 'Invalid params, prompt should not be empty'
        })
        return
    outs = server_run(prompt, None, None)
    response_data = {
        'success': True,
        'download_url': '/files/' + outs
    }
    return jsonify(response_data)

@app.route('/files/<string:filename>')
def download_music(filename):
    return send_from_directory(None, path=filename, as_attachment=False)

if __name__ == "__main__":
    # server_run("123", None, None)
    app.run(host='0.0.0.0', port=8000, debug=True)
