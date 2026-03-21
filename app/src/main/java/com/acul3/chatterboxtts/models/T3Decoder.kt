package com.acul3.chatterboxtts.models

import android.util.Log
import com.acul3.chatterboxtts.tts.Constants
import com.acul3.chatterboxtts.tts.SpeechSampler
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Tensor

/**
 * T3 prefill + autoregressive decode loop.
 * Produces speech tokens from text + conditioning embeddings.
 */
class T3Decoder(
    private val prefillModel: PteModel,
    private val decodeModel: PteModel
) {

    companion object {
        private const val TAG = "T3Decoder"
        private const val KV_SIZE = Constants.N_LAYERS * Constants.N_HEADS * Constants.MAX_KV_LEN * Constants.HEAD_DIM
    }

    /**
     * Run the full T3 decode pipeline.
     *
     * @param condEmbedding Conditioning embedding tensor from cond_model
     * @param textTokens Encoded text token IDs
     * @param onProgress Callback with (step, message)
     * @return List of speech token IDs
     */
    fun decode(
        condEmbedding: Tensor,
        textTokens: Tensor,
        onProgress: (Float, String) -> Unit = { _, _ -> }
    ): IntArray {
        Log.i(TAG, "Starting T3 prefill...")
        onProgress(0f, "Running T3 prefill...")

        // Prefill: get initial logits and KV cache
        val prefillOutputs = prefillModel.forward(
            EValue.from(condEmbedding),
            EValue.from(textTokens)
        )

        var currentLogits = prefillOutputs[0].toTensor()
        val kvFlat = prefillOutputs[1].toTensor()

        // Split KV cache into keys and values
        // kvFlat shape: flat array of size (N_LAYERS * 2 * 1 * N_HEADS * MAX_KV_LEN * HEAD_DIM)
        val kvData = kvFlat.dataAsFloatArray
        val halfSize = kvData.size / 2
        val kvKData = kvData.copyOfRange(0, halfSize)
        val kvVData = kvData.copyOfRange(halfSize, kvData.size)

        var kvK = Tensor.fromBlob(
            kvKData,
            longArrayOf(
                Constants.N_LAYERS.toLong(), 1, Constants.N_HEADS.toLong(),
                Constants.MAX_KV_LEN.toLong(), Constants.HEAD_DIM.toLong()
            )
        )
        var kvV = Tensor.fromBlob(
            kvVData,
            longArrayOf(
                Constants.N_LAYERS.toLong(), 1, Constants.N_HEADS.toLong(),
                Constants.MAX_KV_LEN.toLong(), Constants.HEAD_DIM.toLong()
            )
        )

        Log.i(TAG, "Prefill done, starting decode loop...")
        onProgress(0.05f, "Prefill complete. Decoding speech tokens...")

        // Autoregressive decode
        val speechTokens = mutableListOf<Int>()
        val maxSteps = Constants.MAX_DECODE_STEPS

        for (step in 1..maxSteps) {
            // Sample token from logits
            val logitsData = currentLogits.dataAsFloatArray
            // Take last position logits (vocab_size slice)
            val vocabLogits = if (logitsData.size > Constants.SPEECH_VOCAB) {
                logitsData.copyOfRange(
                    logitsData.size - Constants.SPEECH_VOCAB,
                    logitsData.size
                )
            } else {
                logitsData
            }

            val token = SpeechSampler.sampleToken(
                logits = vocabLogits,
                previousTokens = speechTokens
            )

            // Check for EOS
            if (token == Constants.EOT_SPEECH) {
                Log.i(TAG, "EOS token at step $step")
                break
            }

            speechTokens.add(token)

            // Progress
            val progress = 0.05f + (step.toFloat() / maxSteps) * 0.95f
            onProgress(progress, "Decode step $step: ${speechTokens.size} tokens generated")

            // Prepare decode inputs
            val tokenTensor = Tensor.fromBlob(
                intArrayOf(token),
                longArrayOf(1, 1)
            )
            val stepTensor = Tensor.fromBlob(
                intArrayOf(Constants.PREFILL_LEN + step),
                longArrayOf(1)
            )

            // Run decode step
            val decodeOutputs = decodeModel.forward(
                EValue.from(tokenTensor),
                EValue.from(stepTensor),
                EValue.from(kvK),
                EValue.from(kvV)
            )

            currentLogits = decodeOutputs[0].toTensor()
            kvK = decodeOutputs[1].toTensor()
            kvV = decodeOutputs[2].toTensor()
        }

        Log.i(TAG, "Decode complete: ${speechTokens.size} speech tokens")
        onProgress(1f, "Generated ${speechTokens.size} speech tokens")

        return speechTokens.toIntArray()
    }
}
