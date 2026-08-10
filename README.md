# TAGARELA Dataset

**A Large-Scale Portuguese Speech Dataset from Podcasts**

[Web Page](https://freds0.github.io/TAGARELA/) | [Dataset](https://huggingface.co/datasets/freds0/TAGARELA) | [Paper](https://arxiv.org/abs/2603.15326) | [Code](https://github.com/DosAnjos-AI/katube-2026)

## Abstract

**TAGARELA** is a large-scale Portuguese speech dataset containing more than **8,972 hours of podcast audio** collected from the *Cem Mil Podcasts* repository. Covering both Brazilian and European Portuguese, the dataset was designed to support research in Automatic Speech Recognition (ASR) and Text-to-Speech (TTS), helping address the lack of large public speech resources for Portuguese.

To ensure high-quality data, we developed a preprocessing pipeline including audio standardization, segmentation, speaker diarization, overlapping speech detection, speech enhancement, and large-scale transcription generation. Transcriptions were produced using a bootstrap strategy that combines commercial ASR systems and fine-tuned open-source models.

The resulting corpus is divided into a full ASR subset and a clean-speech subset optimized for TTS. Experiments with state-of-the-art models demonstrate the dataset's effectiveness for building robust Portuguese speech technologies.

## Dataset Overview

TAGARELA was created to reduce the gap between Portuguese and high-resource languages such as English, which benefit from massive speech corpora like LibriSpeech and GigaSpeech. The dataset was derived from the **Cem Mil Podcasts** collection and transformed into a high-quality speech resource through a dedicated processing pipeline.

### Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Hours | 8,972+ |
| Podcast Episodes | 16,806 |
| Podcast Shows | 2,094 |
| Distinct Speakers | ~13,368 |

### Dialect Distribution

- **Brazilian Portuguese (pt-BR):** 8,130 hours (91%)
- **European Portuguese (pt-PT):** 842 hours (9%)

## Dataset Subsets

### Full Dataset (ASR)
The full subset contains **8,972 hours of speech** and preserves natural conversational characteristics commonly found in podcasts. It is intended for training robust Automatic Speech Recognition systems under real-world conditions.

### Clean-Speech Dataset (TTS)
The **Clean-Speech** subset contains approximately **2,800 hours of filtered speech**, providing cleaner and more consistent audio for speech synthesis, voice cloning, and speech generation tasks.

## Processing Pipeline

TAGARELA was built through a multi-stage pipeline designed to transform raw podcast recordings into a high-quality research dataset. Source code for this pipeline is available at [DosAnjos-AI/katube-2026](https://github.com/DosAnjos-AI/katube-2026).

1. **Audio Standardization** - All recordings were converted to a common format (FLAC, 16 kHz, 16-bit, mono) to ensure consistency across the corpus.
2. **Segmentation** - Long recordings were split into 5–20 second segments, prioritizing natural pauses to preserve speech coherence.
3. **Speaker Diarization** - We used pyannote.audio to separate speakers and generate single-speaker segments, an essential requirement for TTS applications.
4. **Overlapping Speech Detection** - A classifier based on Wav2Vec2-XLS-R was trained to identify and remove segments containing overlapping speech.
5. **Bootstrap Transcription** - A 1,000-hour seed corpus transcribed with a commercial ASR service was used to fine-tune Whisper Large V3. A second Wav2Vec2-XLS-R model was trained to validate generated transcriptions, allowing low-quality samples to be filtered automatically.
6. **Speech Enhancement and Labeling** - Audio quality was improved using a speech enhancement model based on Vocos. Additionally, speaker identities and dialect labels (Brazilian vs. European Portuguese) were automatically generated, enriching the dataset with valuable metadata.

## Benchmark Results

To validate the dataset, several state-of-the-art ASR and TTS models were trained using TAGARELA.

### Automatic Speech Recognition

**Parakeet v2 Fine-Tuned** achieved the best performance, reaching **15.18% WER** and **7.09% CER**, outperforming Whisper Large V3, Distil-Whisper, and Wav2Vec2 Large. Models marked with **FT** were fine-tuned on TAGARELA data, while others are pre-trained baselines.

| Model | WER (%) | CER (%) |
|-------|---------|---------|
| Whisper Large V3 | 20.91 | 12.42 |
| Wav2Vec Large FT | 21.85 | 8.55 |
| Distil-Whisper FT | 20.02 | 11.18 |
| Parakeet v3 | 23.30 | 14.86 |
| Parakeet v2 FT | **15.18** | **7.09** |

### Text-to-Speech

Using the 2,800-hour Clean-Speech subset, both **Orpheus-TTS** and **Chatterbox** achieved MOS scores above **4.15**, demonstrating the suitability of TAGARELA for high-quality speech synthesis.

| Model | WER | CER | MOS |
|-------|-----|-----|-----|
| Chatterbox | 0.3111 ± 0.442 | 0.268 ± 0.423 | **4.176 ± 0.983** |
| Orpheus-TTS | **0.095 ± 0.100** | **0.046 ± 0.051** | 4.155 ± 1.001 |
| Ground Truth | 0.010 ± 0.033 | 0.006 ± 0.018 | 4.231 ± 1.001 |

### Audio Quality Metrics

The audio quality of the dataset was assessed using three objective metrics: STOI, PESQ, and SI-SDR.

## Models

The ASR model fine-tuned on TAGARELA is publicly available on the Hugging Face Hub in two formats: the original **NeMo** checkpoint and an **ONNX** export for lightweight, framework-independent inference. Both models expect **16 kHz mono** audio.

### Parakeet TDT 0.6B v3 pt-BR (NeMo)

`ASR` `NVIDIA NeMo` `FP32`

NVIDIA Parakeet TDT 0.6B v3 fine-tuned for Brazilian Portuguese on TAGARELA podcast data, with punctuation and synthetic data augmentation.

Download on Hugging Face: https://huggingface.co/alexandreacff/parakeet-tdt-0.6b-v3-ptBR-plus

**Installation**
```bash
pip install -U "nemo_toolkit[asr]" huggingface_hub
```

**Inference**
```python
import nemo.collections.asr as nemo_asr
from huggingface_hub import hf_hub_download

# Download the .nemo checkpoint from the Hub
checkpoint = hf_hub_download(
    repo_id="alexandreacff/parakeet-tdt-0.6b-v3-ptBR-plus",
    filename="parakeet-tdt-0.6b-v3-datasets-ptbr-e-podcasts-pontuados-e-sintetico.nemo",
)

# Load the model
asr_model = nemo_asr.models.ASRModel.restore_from(restore_path=checkpoint)
asr_model.eval()

# Transcribe a 16 kHz mono audio file
hypotheses = asr_model.transcribe(["audio.wav"])
print(hypotheses[0].text)
```

### Parakeet TDT 0.6B v3 pt-BR TAGARELA (ONNX)

`ASR` `ONNX` `CPU friendly`

ONNX export of the fine-tuned checkpoint above, ready for fast inference with [onnx-asr](https://github.com/istupakov/onnx-asr) and ONNX Runtime, without a deep learning framework installed. Released under CC BY 4.0.

Download on Hugging Face: https://huggingface.co/alefiury/parakeet-tdt-0.6b-v3-ptBR-TAGARELA-onnx

**Installation**
```bash
pip install "onnx-asr[cpu,hub]"
```

**Inference**
```python
import onnx_asr
from huggingface_hub import snapshot_download

# Download the ONNX model files
model_dir = snapshot_download(
    repo_id="alefiury/parakeet-tdt-0.6b-v3-ptBR-TAGARELA-onnx",
    local_dir="./parakeet-tdt-0.6b-v3-ptBR-TAGARELA-onnx",
)

# Load the model with the NeMo Conformer TDT runtime
model = onnx_asr.load_model("nemo-conformer-tdt", model_dir)

# Transcribe a 16 kHz mono audio file
print(model.recognize("audio.wav", language="pt"))
```

## Conclusion

TAGARELA provides a large-scale, publicly available Portuguese speech corpus that supports both ASR and TTS research. By combining advanced preprocessing, transcription, and quality-control techniques, the dataset offers a strong foundation for developing the next generation of Portuguese speech technologies.

## License

The TAGARELA dataset is released under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.

## Citation

If you use the TAGARELA dataset in your research, please cite:

```bibtex
@inproceedings{oliveira2026tagarela,
  title={TAGARELA - A Portuguese Speech Dataset from Podcasts},
  author={Oliveira, Frederico Santos de and Gris, Lucas Rafael Stefanel and
          Ferreira, Alef Iury Siqueira and Rosa, Augusto Seben da and
          Ferro Filho, Alexandre Costa and Casanova, Edresson and
          Shulby, Christopher Dane and Sousa, Rafael Teixeira and
          Silva, Diogo Fernandes Costa and Soares, Anderson da Silva and
          Galv{\~a}o Filho, Arlindo Rodrigues},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2026}
}
```

## Authors

- **Frederico Santos de Oliveira** - Federal University of Mato Grosso (UFMT)
- **Lucas Rafael Stefanel Gris** - Federal University of Goias (UFG)
- **Alef Iury Siqueira Ferreira** - Federal University of Goias (UFG)
- **Augusto Seben da Rosa** - Paulista State University (UNESP)
- **Alexandre Costa Ferro Filho** - Federal University of Goias (UFG)
- **Edresson Casanova** - NVIDIA
- **Christopher Dane Shulby** - Elsa Speak
- **Rafael Teixeira Sousa** - Federal University of Mato Grosso (UFMT)
- **Diogo Fernandes Costa Silva** - Federal University of Goias (UFG)
- **Anderson da Silva Soares** - Federal University of Goias (UFG)
- **Arlindo Rodrigues Galvão Filho** - Federal University of Goias (UFG)

## Acknowledgements

This work has been fully funded by the project *Research and Development of Algorithms for Construction of Digital Human Technological Components* supported by the **Advanced Knowledge Center in Immersive Technologies (AKCIT)** in partnership with the Federal University of Goiás (UFG).
