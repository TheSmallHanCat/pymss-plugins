# asr

Speech-to-text plugin for pymss and Studio. Exposes `pymss.asr.transcribe` as an Operation and `asr_transcribe` as a Python capability.

Install this directory through Studio's plugin manager to add a **Speech to Text** entry to Tools. The page accepts an audio file and the model/language parameters defined by the Operation. Results contain recognized text and timestamps when the backend provides them.

Apple Silicon uses funasr-mlx when available. Other supported environments use FunASR on CUDA or CPU. Model availability and language support depend on the selected backend and checkpoint. Plugin installation does not download all model weights.

The current page displays the returned report. Studio's existing dedicated ASR tool still owns its richer model-management and subtitle-export interface; migrating those features requires the host SDK changes described in the repository README.
