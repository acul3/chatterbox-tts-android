package com.acul3.chatterboxtts.tts

import android.content.Context
import android.util.Log
import com.acul3.chatterboxtts.models.PteModel
import com.acul3.chatterboxtts.models.T3Decoder
import com.acul3.chatterboxtts.models.VocoderPipeline
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Tensor
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Main TTS engine class. Orchestrates the full Chatterbox pipeline:
 * Text → Tokenize → Condition → T3 Decode → Vocoder → Audio
 */
class ChatterboxTTS(private val context: Context) {

    companion object {
        private const val TAG = "ChatterboxTTS"
    }

    private val modelManager = ModelManager(context)
    private var tokenizer: TextTokenizer? = null

    // Models
    private var condModel: PteModel? = null
    private var veModel: PteModel? = null
    private var tokenizerEncoder: PteModel? = null
    private var t3Decoder: T3Decoder? = null
    private var vocoderPipeline: VocoderPipeline? = null

    // Cached conditioning tensors
    private var speakerEmb: Tensor? = null
    private var condSpeechTokens: Tensor? = null

    suspend fun ensureModelsReady(onProgress: (Float, String) -> Unit) {
        if (!modelManager.allModelsReady()) {
            modelManager.ensureModelsDownloaded(onProgress)
        } else {
            onProgress(1f, "All models cached.")
        }
    }

    fun loadModels(onProgress: (Float, String) -> Unit) {
        if (t3Decoder != null) {
            onProgress(1f, "Models already loaded")
            return
        }

        // Initialize tokenizer
        onProgress(0f, "Loading tokenizer...")
        tokenizer = TextTokenizer(context)

        // Load conditioning model
        onProgress(0.1f, "Loading conditioning model...")
        condModel = PteModel(modelManager.getModelPath("cond_model.pte")).also { it.load() }

        // Load VE model
        onProgress(0.2f, "Loading voice encoder...")
        veModel = PteModel(modelManager.getModelPath("ve_model.pte")).also { it.load() }

        // Load tokenizer encoder
        onProgress(0.3f, "Loading tokenizer encoder...")
        tokenizerEncoder = PteModel(modelManager.getModelPath("tokenizer_encoder.pte")).also { it.load() }

        // Load T3 models
        onProgress(0.4f, "Loading T3 prefill...")
        val prefill = PteModel(modelManager.getModelPath("t3_prefill.pte")).also { it.load() }
        onProgress(0.6f, "Loading T3 decode...")
        val decode = PteModel(modelManager.getModelPath("t3_decode.pte")).also { it.load() }
        t3Decoder = T3Decoder(prefill, decode)

        // Load vocoder models
        onProgress(0.7f, "Loading S3Gen encoder...")
        val s3gen = PteModel(modelManager.getModelPath("s3gen_encoder.pte")).also { it.load() }
        onProgress(0.8f, "Loading CFM models...")
        val cfm1 = PteModel(modelManager.getModelPath("cfm_step1.pte")).also { it.load() }
        val cfm2 = PteModel(modelManager.getModelPath("cfm_step2.pte")).also { it.load() }
        onProgress(0.9f, "Loading HiFiGAN...")
        val hifigan = PteModel(modelManager.getModelPath("hifigan.pte")).also { it.load() }
        vocoderPipeline = VocoderPipeline(s3gen, cfm1, cfm2, hifigan)

        // Load default voice conditioning
        onProgress(0.95f, "Loading default voice...")
        loadDefaultConditioning()

        onProgress(1f, "All models loaded!")
        Log.i(TAG, "All models loaded successfully")
    }

    /**
     * Generate speech audio from text.
     *
     * @param text Input text to speak
     * @param language Language code (e.g., "en", "fr", "ms")
     * @param onProgress Progress callback (0-1, message)
     * @return WAV byte array
     */
    fun generate(
        text: String,
        language: String,
        onProgress: (Float, String) -> Unit
    ): ByteArray {
        val tok = tokenizer ?: throw IllegalStateException("Tokenizer not loaded")

        // Step 1: Tokenize text
        onProgress(0f, "Tokenizing text...")
        val textTokenIds = tok.encode(text, language)
        Log.i(TAG, "Tokenized: ${textTokenIds.size} tokens")

        val textTensor = Tensor.fromBlob(
            textTokenIds,
            longArrayOf(1, textTokenIds.size.toLong())
        )

        // Step 2: Run conditioning
        onProgress(0.05f, "Running conditioning...")
        val spkEmb = speakerEmb ?: throw IllegalStateException("Speaker embedding not loaded")
        val condTokens = condSpeechTokens ?: throw IllegalStateException("Conditioning tokens not loaded")

        val condOutputs = condModel!!.forward(
            EValue.from(textTensor),
            EValue.from(spkEmb),
            EValue.from(condTokens)
        )
        val condEmbedding = condOutputs[0].toTensor()
        Log.i(TAG, "Conditioning done")

        // Step 3: T3 decode (autoregressive speech token generation)
        onProgress(0.1f, "Starting T3 decode...")
        val speechTokens = t3Decoder!!.decode(
            condEmbedding = condEmbedding,
            textTokens = textTensor
        ) { progress, message ->
            onProgress(0.1f + progress * 0.5f, message)
        }
        Log.i(TAG, "T3 decode produced ${speechTokens.size} speech tokens")

        // Step 4: Vocoder (speech tokens → audio)
        onProgress(0.6f, "Running vocoder...")
        val audioSamples = vocoderPipeline!!.synthesize(
            speechTokens = speechTokens,
            condSpeechTokens = condTokens,
            speakerEmb = spkEmb
        ) { progress, message ->
            onProgress(0.6f + progress * 0.35f, message)
        }

        // Step 5: Convert to WAV
        onProgress(0.95f, "Encoding WAV...")
        val wavData = AudioPlayer.floatToWav(audioSamples, Constants.S3GEN_SR)
        val durationSec = audioSamples.size.toFloat() / Constants.S3GEN_SR

        onProgress(1f, "Done! ${String.format("%.1f", durationSec)}s of audio generated.")
        Log.i(TAG, "Generated ${wavData.size} bytes WAV (${durationSec}s)")

        return wavData
    }

    /**
     * Load default voice conditioning from bundled binary assets.
     * Falls back to generating a dummy conditioning if files aren't found.
     */
    private fun loadDefaultConditioning() {
        try {
            // Try loading from assets
            speakerEmb = loadTensorFromAsset("speaker_emb.bin", longArrayOf(1, 256))
            condSpeechTokens = loadIntTensorFromAsset(
                "cond_speech_tokens.bin",
                longArrayOf(1, Constants.PREFILL_LEN.toLong())
            )
            Log.i(TAG, "Loaded conditioning from assets")
        } catch (e: Exception) {
            Log.w(TAG, "Conditioning assets not found, using defaults: ${e.message}")
            // Generate default conditioning tensors
            // Speaker embedding: zeros (will produce neutral voice)
            speakerEmb = Tensor.fromBlob(
                FloatArray(256) { 0f },
                longArrayOf(1, 256)
            )
            // Default conditioning speech tokens: SOT + padding
            val condTokens = IntArray(Constants.PREFILL_LEN) { 0 }
            condTokens[0] = Constants.SOT_SPEECH
            condSpeechTokens = Tensor.fromBlob(
                condTokens,
                longArrayOf(1, Constants.PREFILL_LEN.toLong())
            )
        }
    }

    private fun loadTensorFromAsset(name: String, shape: LongArray): Tensor {
        val data = context.assets.open(name).use { it.readBytes() }
        val buffer = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN)
        val floats = FloatArray(data.size / 4)
        buffer.asFloatBuffer().get(floats)
        return Tensor.fromBlob(floats, shape)
    }

    private fun loadIntTensorFromAsset(name: String, shape: LongArray): Tensor {
        val data = context.assets.open(name).use { it.readBytes() }
        val buffer = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN)
        val ints = IntArray(data.size / 4)
        buffer.asIntBuffer().get(ints)
        return Tensor.fromBlob(ints, shape)
    }

    fun close() {
        condModel?.close()
        veModel?.close()
        tokenizerEncoder?.close()
        // T3Decoder models closed implicitly
        // VocoderPipeline models closed implicitly
    }
}
