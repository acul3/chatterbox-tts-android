package com.acul3.chatterboxtts.models

import android.util.Log
import com.acul3.chatterboxtts.tts.Constants
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Tensor

/**
 * Vocoder pipeline: S3Gen encoder → CFM (2 steps) → HiFiGAN
 * Converts speech tokens to audio waveform.
 */
class VocoderPipeline(
    private val s3genEncoder: PteModel,
    private val cfmStep1: PteModel,
    private val cfmStep2: PteModel,
    private val hifigan: PteModel
) {

    companion object {
        private const val TAG = "VocoderPipeline"
    }

    /**
     * Convert speech tokens to audio samples.
     *
     * @param speechTokens Array of speech token IDs from T3 decoder
     * @param condSpeechTokens Conditioning speech tokens (from default voice)
     * @param speakerEmb Speaker embedding tensor
     * @param onProgress Progress callback
     * @return Float array of audio samples at 24kHz
     */
    fun synthesize(
        speechTokens: IntArray,
        condSpeechTokens: Tensor,
        speakerEmb: Tensor,
        onProgress: (Float, String) -> Unit = { _, _ -> }
    ): FloatArray {
        Log.i(TAG, "Starting vocoder pipeline with ${speechTokens.size} speech tokens")

        // Step 1: S3Gen encoder - convert speech tokens to mel spectrogram
        onProgress(0f, "Running S3Gen encoder...")
        val tokenTensor = Tensor.fromBlob(
            speechTokens,
            longArrayOf(1, speechTokens.size.toLong())
        )

        val s3genOutputs = s3genEncoder.forward(
            EValue.from(tokenTensor),
            EValue.from(condSpeechTokens),
            EValue.from(speakerEmb)
        )
        val melInput = s3genOutputs[0].toTensor()
        Log.i(TAG, "S3Gen encoder done")

        // Step 2: CFM denoising (2 steps)
        onProgress(0.25f, "Running CFM step 1/2...")
        val cfm1Outputs = cfmStep1.forward(EValue.from(melInput))
        val cfm1Out = cfm1Outputs[0].toTensor()
        Log.i(TAG, "CFM step 1 done")

        onProgress(0.5f, "Running CFM step 2/2...")
        val cfm2Outputs = cfmStep2.forward(EValue.from(cfm1Out))
        val melSpectrogram = cfm2Outputs[0].toTensor()
        Log.i(TAG, "CFM step 2 done")

        // Step 3: HiFiGAN vocoder - mel to waveform
        onProgress(0.75f, "Running HiFiGAN vocoder...")

        // Process in chunks for memory efficiency
        val melData = melSpectrogram.dataAsFloatArray
        val melFrames = melData.size / 100  // Approximate mel channels
        Log.i(TAG, "Mel data size: ${melData.size}, estimated frames: $melFrames")

        val higanOutputs = hifigan.forward(EValue.from(melSpectrogram))
        val audioTensor = higanOutputs[0].toTensor()
        val audioSamples = audioTensor.dataAsFloatArray

        Log.i(TAG, "HiFiGAN done: ${audioSamples.size} samples (${audioSamples.size.toFloat() / Constants.S3GEN_SR}s)")
        onProgress(1f, "Vocoder complete: ${audioSamples.size} samples")

        return audioSamples
    }
}
