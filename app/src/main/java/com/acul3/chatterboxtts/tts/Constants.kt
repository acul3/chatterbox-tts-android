package com.acul3.chatterboxtts.tts

object Constants {
    // Vocabulary
    const val SPEECH_VOCAB = 8194
    const val SOT_SPEECH = 6561
    const val EOT_SPEECH = 6562
    const val SOT_TEXT = 255
    const val EOT_TEXT = 0

    // Sequence lengths
    const val MAX_TEXT_LEN = 256
    const val PREFILL_LEN = 293
    const val MAX_KV_LEN = 1293

    // Model architecture
    const val N_LAYERS = 30
    const val N_HEADS = 16
    const val HEAD_DIM = 64

    // Audio
    const val HIFI_T_MEL = 300
    const val T_MEL_FIXED = 2200
    const val UPSAMPLE = 480
    const val S3GEN_SR = 24000

    // Generation parameters
    const val MAX_DECODE_STEPS = 750
    const val DEFAULT_TEMPERATURE = 0.8f
    const val DEFAULT_MIN_P = 0.05f
    const val DEFAULT_TOP_P = 0.95f
    const val DEFAULT_REP_PENALTY = 1.05f

    // HuggingFace model repo
    const val HF_REPO_BASE = "https://huggingface.co/acul3/chatterbox-executorch/resolve/main"

    // Model filenames
    val MODEL_FILES = listOf(
        "cond_model.pte",
        "t3_prefill.pte",
        "t3_decode.pte",
        "s3gen_encoder.pte",
        "cfm_step1.pte",
        "cfm_step2.pte",
        "hifigan.pte",
        "tokenizer_encoder.pte",
        "ve_model.pte"
    )
}
