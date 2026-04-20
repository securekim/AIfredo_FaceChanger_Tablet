package com.example.aifredo_facechanger.ui.body

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.*
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.util.Size
import android.view.LayoutInflater
import android.view.TextureView
import android.view.View
import android.view.ViewGroup
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.OptIn
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.media3.common.MediaItem
import androidx.media3.common.Player
import androidx.media3.common.util.UnstableApi
import androidx.media3.exoplayer.ExoPlayer
import androidx.media3.exoplayer.DefaultLoadControl
import androidx.media3.exoplayer.rtsp.RtspMediaSource
import com.example.aifredo_facechanger.databinding.FragmentBodyChangerBinding
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.segmentation.Segmentation
import com.google.mlkit.vision.segmentation.Segmenter
import com.google.mlkit.vision.segmentation.selfie.SelfieSegmenterOptions
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import org.tensorflow.lite.support.common.FileUtil
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.DataType
import com.google.mediapipe.framework.image.BitmapImageBuilder

class BodyChangerFragment : Fragment() {

    private var _binding: FragmentBodyChangerBinding? = null
    private val binding get() = _binding!!

    private var cameraExecutor: ExecutorService? = null

    private var mlKitSegmenter: Segmenter? = null
    @Volatile private var poseLandmarker: PoseLandmarker? = null
    @Volatile private var yolactInterpreter: Interpreter? = null
    @Volatile private var modnetInterpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null
    private var nnApiDelegate: NnApiDelegate? = null

    private var yolactImageProcessor: ImageProcessor? = null
    private var modnetImageProcessor: ImageProcessor? = null
    private var yolactTensorImage: TensorImage? = null
    private var modnetTensorImage: TensorImage? = null

    private var yolactOutputBoxes: ByteBuffer? = null
    private var yolactOutputScores: ByteBuffer? = null
    private var yolactOutputCoeffs: ByteBuffer? = null
    private var yolactOutputProtos: ByteBuffer? = null
    private var modnetOutputBuffer: ByteBuffer? = null

    private var reusableMaskPixels: ByteBuffer? = null
    private var reusableProtosArray: FloatArray? = null
    private var reusableScoresArray: FloatArray? = null

    private val segmenterLock = Any()

    private var startColor: Int = Color.RED
    private var endColor: Int = Color.BLUE
    private var selectedModel: String = "MediaPipe Pose"
    private var selectedDelegate: String = "CPU"
    private var actualDelegate: String = "CPU"
    private var rtspQuality: String = "High"

    private val sdf = SimpleDateFormat("HH:mm:ss", Locale.getDefault())
    private val TAG = "BodyChanger"

    private val YOLACT_MODEL_FILE = "yolact_550x550_model_float16_quant.tflite"
    private val MODNET_MODEL_FILE = "MODNet_256x256_model_float16_quant.tflite"

    // RTSP
    private var exoPlayer: ExoPlayer? = null
    private var isRtspMode = false
    private val rtspFrameHandler = Handler(Looper.getMainLooper())
    private val rtspFrameRunnable = object : Runnable {
        override fun run() {
            if (isRtspMode && exoPlayer?.isPlaying == true) {
                extractFrameFromPlayer()
            }
            rtspFrameHandler.postDelayed(this, 33)
        }
    }

    companion object {
        private val sharedSegmenterExecutor: ExecutorService = Executors.newSingleThreadExecutor()
        @Volatile private var isInitializing = false
    }

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { startStream() }

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View {
        _binding = FragmentBodyChangerBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        cameraExecutor = Executors.newSingleThreadExecutor()

        if (allPermissionsGranted()) startStream()
        else requestPermissionLauncher.launch(arrayOf(Manifest.permission.CAMERA))
    }

    override fun onResume() {
        super.onResume()
        loadSettings()
        startStream()
        setupSegmenter()
    }

    override fun onPause() {
        super.onPause()
        stopRtsp()
        stopCamera()
        sharedSegmenterExecutor.execute { synchronized(segmenterLock) { closeCurrentSegmenter() } }
    }

    private fun loadSettings() {
        val sharedPref = activity?.getSharedPreferences("AIfredoPrefs", Context.MODE_PRIVATE) ?: return

        selectedModel = sharedPref.getString("body_model", "MediaPipe Pose") ?: "MediaPipe Pose"
        selectedDelegate = sharedPref.getString("body_delegate", "CPU") ?: "CPU"
        val startColorStr = sharedPref.getString("body_start_color", "#FF0000") ?: "#FF0000"
        val endColorStr = sharedPref.getString("body_end_color", "#0000FF") ?: "#0000FF"
        isRtspMode = sharedPref.getString("cam_source", "Embedded") == "RTSP"
        rtspQuality = sharedPref.getString("rtsp_quality", "High") ?: "High"

        try {
            startColor = Color.parseColor(startColorStr)
            endColor = Color.parseColor(endColorStr)
        } catch (e: Exception) {
            startColor = Color.RED
            endColor = Color.BLUE
        }
    }

    private fun setupSegmenter() {
        if (isInitializing) return
        isInitializing = true
        sharedSegmenterExecutor.execute {
            try {
                synchronized(segmenterLock) {
                    closeCurrentSegmenter()
                    System.gc()
                    try { Thread.sleep(300) } catch (e: Exception) {}
                    if (!isAdded) return@synchronized
                    when (selectedModel) {
                        "MediaPipe Pose" -> initMediaPipePose()
                        "YOLACT" -> initYolact()
                        "MODNet" -> initModNet()
                        "ML Kit" -> initMlKit()
                        else -> initMediaPipePose()
                    }
                }
            } finally {
                isInitializing = false
            }
        }
    }

    private fun closeCurrentSegmenter() {
        try {
            mlKitSegmenter?.close(); mlKitSegmenter = null
            poseLandmarker?.close(); poseLandmarker = null
            yolactInterpreter?.close(); yolactInterpreter = null
            modnetInterpreter?.close(); modnetInterpreter = null
            gpuDelegate?.close(); gpuDelegate = null
            nnApiDelegate?.close(); nnApiDelegate = null
        } catch (e: Exception) {}
    }

    private fun initMediaPipePose() {
        try {
            val baseOptionsBuilder = BaseOptions.builder().setModelAssetPath("pose_landmarker_lite.task")
            if (selectedDelegate.uppercase() == "GPU") baseOptionsBuilder.setDelegate(Delegate.GPU)
            else baseOptionsBuilder.setDelegate(Delegate.CPU)

            val optionsBuilder = PoseLandmarker.PoseLandmarkerOptions.builder()
                .setBaseOptions(baseOptionsBuilder.build())
                .setRunningMode(RunningMode.LIVE_STREAM)
                .setOutputSegmentationMasks(true)
                .setResultListener { result, image ->
                    processMediaPipePoseResult(result, image.width, image.height)
                }

            poseLandmarker = PoseLandmarker.createFromOptions(requireContext(), optionsBuilder.build())
            actualDelegate = selectedDelegate
            addLog(">> MediaPipe Pose Ready ($actualDelegate)")
        } catch (e: Exception) {
            addLog("MediaPipe Pose Init Error: ${e.message}")
        }
    }

    private fun processMediaPipePoseResult(result: PoseLandmarkerResult, originalWidth: Int, originalHeight: Int) {
        val segmentationMasks = result.segmentationMasks()
        if (segmentationMasks.isPresent && segmentationMasks.get().isNotEmpty()) {
            val mask = segmentationMasks.get()[0]
            try {
                val width = mask.width
                val height = mask.height
                
                val byteBuffer = try {
                    com.google.mediapipe.framework.image.ByteBufferExtractor.extract(mask)
                } catch (e: Exception) {
                    Log.e(TAG, "MediaPipe mask extraction failed: ${e.message}")
                    return
                }
                
                byteBuffer.rewind()
                val maskBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ALPHA_8)
                val pixels = ByteArray(width * height)
                if (byteBuffer.capacity() >= width * height * 4) {
                    for (i in 0 until width * height) {
                        val alpha = if (byteBuffer.float > 0.5f) 255 else 0
                        pixels[i] = alpha.toByte()
                    }
                } else {
                    byteBuffer.get(pixels)
                }
                maskBitmap.copyPixelsFromBuffer(ByteBuffer.wrap(pixels))
                val finalMask = Bitmap.createScaledBitmap(maskBitmap, originalWidth, originalHeight, true)
                maskBitmap.recycle()
                activity?.runOnUiThread { _binding?.bodyOverlay?.updateMaskOnly(finalMask, startColor, endColor) }
            } catch (e: Exception) {}
        }
    }

    private fun initMlKit() {
        actualDelegate = "N/A (ML Kit)"
        try {
            val options = SelfieSegmenterOptions.Builder().setDetectorMode(SelfieSegmenterOptions.STREAM_MODE).build()
            mlKitSegmenter = Segmentation.getClient(options)
            addLog(">> ML Kit Ready")
        } catch (e: Exception) { addLog("ML Kit Error: ${e.message}") }
    }

    private fun getInterpreterOptions(): Interpreter.Options {
        val options = Interpreter.Options()
        actualDelegate = "CPU"
        if (selectedDelegate.uppercase() == "GPU") {
            try {
                if (CompatibilityList().isDelegateSupportedOnThisDevice) {
                    gpuDelegate = GpuDelegate(GpuDelegate.Options().apply { setInferencePreference(GpuDelegate.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED) })
                    options.addDelegate(gpuDelegate); actualDelegate = "GPU"
                }
            } catch (e: Exception) {}
        } else if (selectedDelegate.uppercase() == "NNAPI") {
            try { nnApiDelegate = NnApiDelegate(); options.addDelegate(nnApiDelegate); actualDelegate = "NNAPI" } catch (e: Exception) {}
        }
        return options
    }

    private fun initYolact() {
        try {
            val interpreter = Interpreter(FileUtil.loadMappedFile(requireContext(), YOLACT_MODEL_FILE), getInterpreterOptions())
            yolactInterpreter = interpreter
            yolactImageProcessor = ImageProcessor.Builder().add(ResizeOp(550, 550, ResizeOp.ResizeMethod.BILINEAR)).add(NormalizeOp(floatArrayOf(123.675f, 116.28f, 103.53f), floatArrayOf(58.395f, 57.12f, 57.375f))).build()
            yolactTensorImage = TensorImage(DataType.FLOAT32)
            yolactOutputBoxes = ByteBuffer.allocateDirect(19248 * 4 * 4).order(ByteOrder.nativeOrder())
            yolactOutputScores = ByteBuffer.allocateDirect(19248 * 81 * 4).order(ByteOrder.nativeOrder())
            yolactOutputCoeffs = ByteBuffer.allocateDirect(19248 * 32 * 4).order(ByteOrder.nativeOrder())
            yolactOutputProtos = ByteBuffer.allocateDirect(138 * 138 * 32 * 4).order(ByteOrder.nativeOrder())
            reusableMaskPixels = ByteBuffer.allocateDirect(256 * 256).order(ByteOrder.nativeOrder())
            reusableProtosArray = FloatArray(138 * 138 * 32)
            reusableScoresArray = FloatArray(19248 * 81)
            addLog(">> YOLACT Ready ($actualDelegate)")
        } catch (e: Exception) { addLog("YOLACT Init Error: ${e.message}") }
    }

    private fun initModNet() {
        try {
            modnetInterpreter = Interpreter(FileUtil.loadMappedFile(requireContext(), MODNET_MODEL_FILE), getInterpreterOptions())
            modnetImageProcessor = ImageProcessor.Builder().add(ResizeOp(256, 256, ResizeOp.ResizeMethod.BILINEAR)).add(NormalizeOp(floatArrayOf(127.5f, 127.5f, 127.5f), floatArrayOf(127.5f, 127.5f, 127.5f))).build()
            modnetTensorImage = TensorImage(DataType.FLOAT32)
            modnetOutputBuffer = ByteBuffer.allocateDirect(1 * 256 * 256 * 1 * 4).order(ByteOrder.nativeOrder())
            reusableMaskPixels = ByteBuffer.allocateDirect(256 * 256).order(ByteOrder.nativeOrder())
            addLog(">> MODNet Ready ($actualDelegate)")
        } catch (e: Exception) { addLog("MODNet Init Error: ${e.message}") }
    }

    private fun startStream() {
        if (isRtspMode) startRtsp() else startCamera()
    }

    private fun startCamera() {
        stopRtsp()
        _binding?.viewFinder?.visibility = View.VISIBLE
        _binding?.playerView?.visibility = View.GONE
        val cameraProviderFuture = ProcessCameraProvider.getInstance(requireContext())
        cameraProviderFuture.addListener({
            val cameraProvider = try { cameraProviderFuture.get() } catch (e: Exception) { return@addListener }
            val preview = Preview.Builder().build().also { it.setSurfaceProvider(binding.viewFinder.surfaceProvider) }
            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .setTargetResolution(Size(640, 480))
                .build().also { it.setAnalyzer(cameraExecutor!!) { proxy -> processFrame(proxy) } }
            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(viewLifecycleOwner, CameraSelector.DEFAULT_FRONT_CAMERA, preview, imageAnalyzer)
                addLog("Camera started")
            } catch (e: Exception) {}
        }, ContextCompat.getMainExecutor(requireContext()))
    }

    @OptIn(UnstableApi::class)
    private fun startRtsp() {
        stopRtsp()
        stopCamera()
        _binding?.viewFinder?.visibility = View.GONE
        _binding?.playerView?.visibility = View.VISIBLE
        val sharedPref = activity?.getSharedPreferences("AIfredoPrefs", Context.MODE_PRIVATE) ?: return
        val ip = (sharedPref.getString("rtsp_ip", "") ?: "").trim().removePrefix("rtsp://")
        val id = sharedPref.getString("rtsp_id", "") ?: ""
        val pw = sharedPref.getString("rtsp_pw", "") ?: ""
        
        if (ip.isEmpty()) { addLog("RTSP IP is empty."); return }

        val streamPath = if (rtspQuality == "Low") "stream2" else "stream1"
        
        val rtspUrl = buildString {
            append("rtsp://")
            if (id.isNotEmpty() && pw.isNotEmpty()) {
                append("$id:$pw@")
            }
            append(ip)
            if (!ip.contains("/")) {
                if (!ip.contains(":")) {
                    append(":554")
                }
                append("/$streamPath")
            }
        }

        addLog("Connecting RTSP: $rtspUrl")

        val loadControl = DefaultLoadControl.Builder()
            .setBufferDurationsMs(500, 1000, 250, 500)
            .build()
        
        val mediaItem = MediaItem.Builder()
            .setUri(rtspUrl)
            .setLiveConfiguration(MediaItem.LiveConfiguration.Builder().setTargetOffsetMs(500).build())
            .build()

        exoPlayer = ExoPlayer.Builder(requireContext())
            .setLoadControl(loadControl)
            .build().apply {
            trackSelectionParameters = trackSelectionParameters.buildUpon()
                .setTrackTypeDisabled(androidx.media3.common.C.TRACK_TYPE_AUDIO, true)
                .build()

            addListener(object : Player.Listener {
                override fun onPlayerError(error: androidx.media3.common.PlaybackException) {
                    addLog("Playback Error: ${error.message}")
                }
                override fun onPlaybackStateChanged(playbackState: Int) {
                    if (playbackState == Player.STATE_READY && isRtspMode) addLog("RTSP Connected")
                }
            })

            setMediaSource(RtspMediaSource.Factory()
                .setForceUseRtpTcp(false)
                .createMediaSource(mediaItem))
            prepare(); playWhenReady = true
        }
        binding.playerView.player = exoPlayer
        rtspFrameHandler.post(rtspFrameRunnable)
    }

    private fun extractFrameFromPlayer() {
        val b = _binding ?: return
        val textureView = b.playerView.videoSurfaceView as? TextureView ?: return
        val originalBitmap = textureView.getBitmap() ?: return

        cameraExecutor?.execute {
            val targetWidth = 640
            val targetHeight = (originalBitmap.height * (targetWidth.toFloat() / originalBitmap.width)).toInt()
            val bitmap = try {
                Bitmap.createScaledBitmap(originalBitmap, targetWidth, targetHeight, true)
            } catch (e: Exception) {
                originalBitmap
            }
            
            processFrameBitmap(bitmap)
            
            if (bitmap != originalBitmap) {
                originalBitmap.recycle()
            }
            // Always recycle the original bitmap from textureView
            // Wait, if bitmap == originalBitmap, it's recycled above. 
            // If bitmap != originalBitmap, bitmap is a scaled version, and originalBitmap is NOT recycled in the block above.
            // Let's fix this properly:
            if (bitmap != originalBitmap) {
                originalBitmap.recycle()
            }
            // But who recycles 'bitmap'? processFrameBitmap should handle it or it should be handled here.
            // In MediaPipe detectAsync, we can't recycle immediately.
            // For others, we can.
            if (selectedModel != "MediaPipe Pose") {
                bitmap.recycle()
            }
        }
    }

    private fun stopCamera() {
        try { ProcessCameraProvider.getInstance(requireContext()).get().unbindAll() } catch (e: Exception) {}
    }

    private fun stopRtsp() {
        rtspFrameHandler.removeCallbacks(rtspFrameRunnable)
        exoPlayer?.release(); exoPlayer = null
        _binding?.playerView?.player = null
    }

    private fun processFrame(imageProxy: ImageProxy) {
        val bitmap = processImageProxy(imageProxy)
        if (bitmap != null) {
            processFrameBitmap(bitmap)
            if (selectedModel != "MediaPipe Pose") {
                bitmap.recycle()
            }
        }
        imageProxy.close()
    }

    private fun processFrameBitmap(bitmap: Bitmap) {
        synchronized(segmenterLock) {
            when (selectedModel) {
                "MediaPipe Pose" -> poseLandmarker?.detectAsync(BitmapImageBuilder(bitmap).build(), System.currentTimeMillis())
                "YOLACT" -> processYolact(bitmap)
                "MODNet" -> processModNet(bitmap)
                "ML Kit" -> processMlKit(bitmap)
                else -> {}
            }
        }
    }

    private fun processMlKit(bitmap: Bitmap) {
        mlKitSegmenter?.process(InputImage.fromBitmap(bitmap, 0))?.addOnSuccessListener { result ->
            val maskBuffer = result.buffer; maskBuffer.rewind()
            val width = result.width; val height = result.height
            val pixels = ByteArray(width * height)
            for (i in 0 until width * height) {
                if (maskBuffer.hasRemaining()) {
                    pixels[i] = (if (maskBuffer.float > 0.45f) 255 else 0).toByte()
                }
            }
            val maskBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ALPHA_8)
            maskBitmap.copyPixelsFromBuffer(ByteBuffer.wrap(pixels))
            activity?.runOnUiThread { _binding?.bodyOverlay?.updateData(maskBitmap, bitmap, startColor, endColor) }
        }
    }

    private fun processYolact(bitmap: Bitmap) {
        val interpreter = yolactInterpreter ?: return
        val tImage = yolactTensorImage ?: return
        tImage.load(bitmap)
        val inputBuffer = yolactImageProcessor?.process(tImage)?.buffer ?: return
        val outputs = mutableMapOf<Int, Any>()
        yolactOutputBoxes?.clear(); yolactOutputScores?.clear(); yolactOutputCoeffs?.clear(); yolactOutputProtos?.clear()
        outputs[0] = yolactOutputBoxes!!; outputs[1] = yolactOutputScores!!
        outputs[2] = yolactOutputCoeffs!!; outputs[3] = yolactOutputProtos!!
        try { interpreter.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputs) } catch (e: Exception) { return }
        yolactOutputScores?.rewind(); val scoresArray = reusableScoresArray ?: return
        yolactOutputScores?.asFloatBuffer()?.get(scoresArray)
        var bestIdx = -1; var maxScore = 0f
        for (i in 0 until 19248) { val score = scoresArray[i * 81 + 1]; if (score > maxScore) { maxScore = score; bestIdx = i } }
        if (bestIdx != -1 && maxScore > 0.15f) {
            val coeffs = FloatArray(32); yolactOutputCoeffs?.rewind()
            val fb = yolactOutputCoeffs?.asFloatBuffer(); fb?.position(bestIdx * 32); fb?.get(coeffs)
            val mask = generateYolactMask(coeffs, yolactOutputProtos!!)
            val finalMask = Bitmap.createScaledBitmap(mask, bitmap.width, bitmap.height, true)
            activity?.runOnUiThread { _binding?.bodyOverlay?.updateData(finalMask, bitmap, startColor, endColor) }
        }
    }

    private fun generateYolactMask(coeffs: FloatArray, protos: ByteBuffer): Bitmap {
        val width = 138; val height = 138
        val pixels = reusableMaskPixels!!; pixels.clear()
        val protosArray = reusableProtosArray ?: return Bitmap.createBitmap(1, 1, Bitmap.Config.ALPHA_8)
        protos.rewind(); protos.asFloatBuffer().get(protosArray)
        for (y in 0 until height) {
            for (x in 0 until width) {
                var sum = 0f; val offset = (y * width + x) * 32
                for (k in 0 until 32) sum += coeffs[k] * protosArray[offset + k]
                pixels.put((if (sum > 0.5f) 255 else 0).toByte())
            }
        }
        val mask = Bitmap.createBitmap(width, height, Bitmap.Config.ALPHA_8)
        pixels.rewind()
        mask.copyPixelsFromBuffer(pixels); return mask
    }

    private fun processModNet(bitmap: Bitmap) {
        val interpreter = modnetInterpreter ?: return
        val tImage = modnetTensorImage ?: return
        tImage.load(bitmap)
        val inputBuffer = modnetImageProcessor?.process(tImage)?.buffer ?: return
        modnetOutputBuffer?.clear()
        try { interpreter.run(inputBuffer, modnetOutputBuffer) } catch (e: Exception) { return }
        modnetOutputBuffer?.rewind(); val fb = modnetOutputBuffer?.asFloatBuffer() ?: return
        val pixels = reusableMaskPixels!!; pixels.clear()
        for (i in 0 until (256 * 256)) {
            if (fb.hasRemaining()) {
                pixels.put((if (fb.get() > 0.4f) 255 else 0).toByte())
            }
        }
        val mask = Bitmap.createBitmap(256, 256, Bitmap.Config.ALPHA_8)
        pixels.rewind()
        mask.copyPixelsFromBuffer(pixels)
        val finalMask = Bitmap.createScaledBitmap(mask, bitmap.width, bitmap.height, true)
        activity?.runOnUiThread { _binding?.bodyOverlay?.updateData(finalMask, bitmap, startColor, endColor) }
    }

    private fun processImageProxy(imageProxy: ImageProxy): Bitmap? {
        return try {
            val buffer = imageProxy.planes[0].buffer; buffer.rewind()
            val bitmap = Bitmap.createBitmap(imageProxy.width, imageProxy.height, Bitmap.Config.ARGB_8888)
            bitmap.copyPixelsFromBuffer(buffer)
            val matrix = Matrix().apply { postRotate(imageProxy.imageInfo.rotationDegrees.toFloat()); postScale(-1f, 1f) }
            Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        } catch (e: Exception) { null }
    }

    private fun addLog(message: String) {
        activity?.runOnUiThread { _binding?.let { b ->
            val timestamp = sdf.format(Date())
            b.eventLog.text = "[$timestamp] $message\n${b.eventLog.text.toString().take(1000)}"
        } }
    }

    private fun allPermissionsGranted() = ContextCompat.checkSelfPermission(requireContext(), Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED

    override fun onDestroyView() {
        super.onDestroyView()
        cameraExecutor?.shutdown(); _binding = null
    }
}
