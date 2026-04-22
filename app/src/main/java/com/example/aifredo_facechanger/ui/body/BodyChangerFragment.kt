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
    @Volatile private var rvmInterpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null
    private var nnApiDelegate: NnApiDelegate? = null

    private var yolactImageProcessor: ImageProcessor? = null
    private var modnetImageProcessor: ImageProcessor? = null
    private var rvmImageProcessor: ImageProcessor? = null
    private var yolactTensorImage: TensorImage? = null
    private var modnetTensorImage: TensorImage? = null
    private var rvmTensorImage: TensorImage? = null

    private var yolactOutputBoxes: ByteBuffer? = null
    private var yolactOutputScores: ByteBuffer? = null
    private var yolactOutputCoeffs: ByteBuffer? = null
    private var yolactOutputProtos: ByteBuffer? = null
    private var modnetOutputBuffer: ByteBuffer? = null
    
    private var yolactIdxBoxes = 0
    private var yolactIdxScores = 1
    private var yolactIdxCoeffs = 2
    private var yolactIdxProtos = 3

    // RVM Buffers
    private var rvmOutputPha: ByteBuffer? = null
    private var rvmOutputFgr: ByteBuffer? = null
    private var r1i: ByteBuffer? = null
    private var r2i: ByteBuffer? = null
    private var r3i: ByteBuffer? = null
    private var r4i: ByteBuffer? = null
    private var r1o: ByteBuffer? = null
    private var r2o: ByteBuffer? = null
    private var r3o: ByteBuffer? = null
    private var r4o: ByteBuffer? = null
    private var rvmRatioBuffer: ByteBuffer? = null

    private var rvmH: Int = 480
    private var rvmW: Int = 640
    private var rvmIdxSrc = 0
    private var rvmIdxR1i = 1; private var rvmIdxR2i = 2; private var rvmIdxR3i = 3; private var rvmIdxR4i = 4
    private var rvmIdxRatio = -1
    private var rvmIdxPha = 0; private var rvmIdxFgr = 1
    private var rvmIdxR1o = 2; private var rvmIdxR2o = 3; private var rvmIdxR3o = 4; private var rvmIdxR4o = 5
    private var isRvmNCHW = false

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
    private var isMirrorMode: Boolean = false

    private val sdf = SimpleDateFormat("HH:mm:ss", Locale.getDefault())
    private val TAG = "BodyChanger"

    private val YOLACT_MODEL_FILE = "yolact_550x550_model_float16_quant.tflite"
    private val MODNET_MODEL_FILE = "MODNet_256x256_model_float16_quant.tflite"
    private val RVM_MODEL_FILE = "rvm_resnet50_480x640_model_float16_quant.tflite"

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
        isMirrorMode = sharedPref.getBoolean("body_mirror_mode", false)

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
                        "RVM" -> initRvm()
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
            rvmInterpreter?.close(); rvmInterpreter = null
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
                .setMinPoseDetectionConfidence(0.2f)
                .setMinPosePresenceConfidence(0.4f)
                .setMinTrackingConfidence(0.4f)
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
                
                val finalMask = try {
                    Bitmap.createScaledBitmap(maskBitmap, originalWidth, originalHeight, true)
                } catch (e: Exception) { maskBitmap }
                
                if (finalMask != maskBitmap) {
                    maskBitmap.recycle()
                }
                
                activity?.runOnUiThread { _binding?.bodyOverlay?.updateMaskOnly(finalMask, startColor, endColor, isMirrorMode) }
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
                    gpuDelegate = GpuDelegate(GpuDelegate.Options().apply { 
                        setInferencePreference(GpuDelegate.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED)
                    })
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
            
            for (i in 0 until interpreter.outputTensorCount) {
                val shape = interpreter.getOutputTensor(i).shape()
                val name = interpreter.getOutputTensor(i).name().lowercase()
                if (name.contains("box")) yolactIdxBoxes = i
                else if (name.contains("score") || name.contains("conf")) yolactIdxScores = i
                else if (name.contains("coeff") || name.contains("mask")) yolactIdxCoeffs = i
                else if (name.contains("proto")) yolactIdxProtos = i
                else if (shape.size == 3) {
                    when (shape[2]) {
                        4 -> yolactIdxBoxes = i
                        81 -> yolactIdxScores = i
                        32 -> yolactIdxCoeffs = i
                    }
                } else if (shape.size == 4) {
                    if (shape[1] == 32 || shape[3] == 32) yolactIdxProtos = i
                }
            }

            yolactImageProcessor = ImageProcessor.Builder().add(ResizeOp(550, 550, ResizeOp.ResizeMethod.BILINEAR)).add(NormalizeOp(floatArrayOf(123.675f, 116.28f, 103.53f), floatArrayOf(58.395f, 57.12f, 57.375f))).build()
            yolactTensorImage = TensorImage(DataType.FLOAT32)
            
            yolactOutputBoxes = ByteBuffer.allocateDirect(interpreter.getOutputTensor(yolactIdxBoxes).numBytes()).order(ByteOrder.nativeOrder())
            yolactOutputScores = ByteBuffer.allocateDirect(interpreter.getOutputTensor(yolactIdxScores).numBytes()).order(ByteOrder.nativeOrder())
            yolactOutputCoeffs = ByteBuffer.allocateDirect(interpreter.getOutputTensor(yolactIdxCoeffs).numBytes()).order(ByteOrder.nativeOrder())
            yolactOutputProtos = ByteBuffer.allocateDirect(interpreter.getOutputTensor(yolactIdxProtos).numBytes()).order(ByteOrder.nativeOrder())
            
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
            modnetOutputBuffer = ByteBuffer.allocateDirect(modnetInterpreter!!.getOutputTensor(0).numBytes()).order(ByteOrder.nativeOrder())
            reusableMaskPixels = ByteBuffer.allocateDirect(256 * 256).order(ByteOrder.nativeOrder())
            addLog(">> MODNet Ready ($actualDelegate)")
        } catch (e: Exception) { addLog("MODNet Init Error: ${e.message}") }
    }

    private fun initRvm() {
        try {
            val modelFile = FileUtil.loadMappedFile(requireContext(), RVM_MODEL_FILE)
            var interpreter: Interpreter? = null
            
            if (selectedDelegate.uppercase() == "GPU") {
                try {
                    val options = Interpreter.Options()
                    gpuDelegate = GpuDelegate(GpuDelegate.Options().apply {
                        setInferencePreference(GpuDelegate.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED)
                        setPrecisionLossAllowed(true)
                    })
                    options.addDelegate(gpuDelegate)
                    interpreter = Interpreter(modelFile, options)
                    actualDelegate = "GPU"
                    addLog("RVM: GPU Delegate applied")
                } catch (e: Exception) {
                    addLog("RVM: GPU Delegate failed: ${e.message}")
                    gpuDelegate?.close(); gpuDelegate = null
                }
            }
            
            if (interpreter == null) {
                val options = Interpreter.Options().setNumThreads(4)
                interpreter = Interpreter(modelFile, options)
                actualDelegate = "CPU"
                addLog("RVM: Using CPU")
            }
            rvmInterpreter = interpreter!!

            rvmIdxRatio = -1
            for (i in 0 until rvmInterpreter!!.inputTensorCount) {
                val t = rvmInterpreter!!.getInputTensor(i)
                val name = t.name().lowercase()
                if (name.contains("src")) rvmIdxSrc = i
                else if (name.contains("r1i")) rvmIdxR1i = i
                else if (name.contains("r2i")) rvmIdxR2i = i
                else if (name.contains("r3i")) rvmIdxR3i = i
                else if (name.contains("r4i")) rvmIdxR4i = i
                else if (name.contains("ratio") || name.contains("downsample")) rvmIdxRatio = i
                Log.d(TAG, "RVM Input $i: $name, Shape: ${t.shape().contentToString()}, Type: ${t.dataType()}")
            }
            for (i in 0 until rvmInterpreter!!.outputTensorCount) {
                val t = rvmInterpreter!!.getOutputTensor(i)
                val name = t.name().lowercase()
                if (name.contains("fgr")) rvmIdxFgr = i
                else if (name.contains("pha")) rvmIdxPha = i
                else if (name.contains("r1o")) rvmIdxR1o = i
                else if (name.contains("r2o")) rvmIdxR2o = i
                else if (name.contains("r3o")) rvmIdxR3o = i
                else if (name.contains("r4o")) rvmIdxR4o = i
                Log.d(TAG, "RVM Output $i: $name, Shape: ${t.shape().contentToString()}, Type: ${t.dataType()}")
            }

            val srcShape = rvmInterpreter!!.getInputTensor(rvmIdxSrc).shape()
            isRvmNCHW = srcShape[1] == 3
            if (isRvmNCHW) {
                rvmH = srcShape[2]; rvmW = srcShape[3]
            } else {
                rvmH = srcShape[1]; rvmW = srcShape[2]
            }
            
            rvmImageProcessor = ImageProcessor.Builder()
                .add(ResizeOp(rvmH, rvmW, ResizeOp.ResizeMethod.BILINEAR))
                .add(NormalizeOp(0f, 255f))
                .build()
            rvmTensorImage = TensorImage(DataType.FLOAT32)
            
            fun allocateDirectBuffer(size: Int): ByteBuffer = ByteBuffer.allocateDirect(size).order(ByteOrder.nativeOrder())

            rvmOutputFgr = allocateDirectBuffer(rvmInterpreter!!.getOutputTensor(rvmIdxFgr).numBytes())
            rvmOutputPha = allocateDirectBuffer(rvmInterpreter!!.getOutputTensor(rvmIdxPha).numBytes())
            r1i = allocateDirectBuffer(rvmInterpreter!!.getInputTensor(rvmIdxR1i).numBytes())
            r2i = allocateDirectBuffer(rvmInterpreter!!.getInputTensor(rvmIdxR2i).numBytes())
            r3i = allocateDirectBuffer(rvmInterpreter!!.getInputTensor(rvmIdxR3i).numBytes())
            r4i = allocateDirectBuffer(rvmInterpreter!!.getInputTensor(rvmIdxR4i).numBytes())
            r1o = allocateDirectBuffer(rvmInterpreter!!.getOutputTensor(rvmIdxR1o).numBytes())
            r2o = allocateDirectBuffer(rvmInterpreter!!.getOutputTensor(rvmIdxR2o).numBytes())
            r3o = allocateDirectBuffer(rvmInterpreter!!.getOutputTensor(rvmIdxR3o).numBytes())
            r4o = allocateDirectBuffer(rvmInterpreter!!.getOutputTensor(rvmIdxR4o).numBytes())
            
            if (rvmIdxRatio != -1) {
                rvmRatioBuffer = allocateDirectBuffer(rvmInterpreter!!.getInputTensor(rvmIdxRatio).numBytes())
                if (rvmRatioBuffer!!.capacity() >= 4) rvmRatioBuffer!!.asFloatBuffer().put(1.0f)
            }
            
            reusableMaskPixels = allocateDirectBuffer(rvmW * rvmH)
            addLog(">> RVM Ready ($actualDelegate) ${rvmW}x${rvmH}")
        } catch (e: Exception) {
            addLog("RVM Init Error: ${e.message}")
            Log.e(TAG, "RVM Init Error", e)
        }
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

    @OptIn(UnstableApi::class)
    private fun extractFrameFromPlayer() {
        val b = _binding ?: return
        val textureView = b.playerView.videoSurfaceView as? TextureView ?: return
        val originalBitmap = try { textureView.getBitmap() } catch (e: Exception) { null } ?: return

        cameraExecutor?.execute {
            val targetWidth = 640
            val targetHeight = (originalBitmap.height * (targetWidth.toFloat() / originalBitmap.width)).toInt()
            val bitmap = try {
                Bitmap.createScaledBitmap(originalBitmap, targetWidth, targetHeight, true)
            } catch (e: Exception) {
                originalBitmap
            }
            
            if (bitmap != originalBitmap) {
                originalBitmap.recycle()
            }
            
            processFrameBitmap(bitmap)
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
        }
        imageProxy.close()
    }

    private fun processFrameBitmap(bitmap: Bitmap) {
        synchronized(segmenterLock) {
            when (selectedModel) {
                "MediaPipe Pose" -> {
                    poseLandmarker?.detectAsync(BitmapImageBuilder(bitmap).build(), System.currentTimeMillis())
                }
                "YOLACT" -> {
                    processYolact(bitmap)
                    bitmap.recycle()
                }
                "MODNet" -> {
                    processModNet(bitmap)
                    bitmap.recycle()
                }
                "RVM" -> {
                    processRvm(bitmap)
                    bitmap.recycle()
                }
                "ML Kit" -> {
                    processMlKit(bitmap)
                }
                else -> {
                    bitmap.recycle()
                }
            }
        }
    }

    private fun processMlKit(bitmap: Bitmap) {
        val segmenter = mlKitSegmenter
        if (segmenter == null || bitmap.isRecycled) {
            bitmap.recycle()
            return
        }
        segmenter.process(InputImage.fromBitmap(bitmap, 0))
            .addOnCompleteListener {
                bitmap.recycle()
            }
            .addOnSuccessListener { result ->
                val maskBuffer = result.buffer
                val width = result.width
                val height = result.height
                val pixels = ByteArray(width * height)
                maskBuffer.rewind()
                for (i in 0 until width * height) {
                    if (maskBuffer.hasRemaining()) {
                        pixels[i] = (if (maskBuffer.float > 0.45f) 255 else 0).toByte()
                    }
                }
                val maskBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ALPHA_8)
                maskBitmap.copyPixelsFromBuffer(ByteBuffer.wrap(pixels))
                activity?.runOnUiThread { _binding?.bodyOverlay?.updateMaskOnly(maskBitmap, startColor, endColor, isMirrorMode) }
            }
    }

    private fun processYolact(bitmap: Bitmap) {
        val interpreter = yolactInterpreter ?: return
        val tImage = yolactTensorImage ?: return
        tImage.load(bitmap)
        val inputBuffer = yolactImageProcessor?.process(tImage)?.buffer ?: return
        
        yolactOutputBoxes?.clear(); yolactOutputScores?.clear(); yolactOutputCoeffs?.clear(); yolactOutputProtos?.clear()
        val outputs = mutableMapOf<Int, Any>()
        outputs[yolactIdxBoxes] = yolactOutputBoxes!!
        outputs[yolactIdxScores] = yolactOutputScores!!
        outputs[yolactIdxCoeffs] = yolactOutputCoeffs!!
        outputs[yolactIdxProtos] = yolactOutputProtos!!
        
        try { 
            interpreter.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputs) 
        } catch (e: Exception) { 
            Log.e(TAG, "YOLACT Run Error: ${e.message}")
            return 
        }
        
        yolactOutputScores?.rewind(); val scoresArray = reusableScoresArray ?: return
        yolactOutputScores?.asFloatBuffer()?.get(scoresArray)
        var bestIdx = -1; var maxScore = 0f
        for (i in 0 until 19248) { val score = scoresArray[i * 81 + 1]; if (score > maxScore) { maxScore = score; bestIdx = i } }
        if (bestIdx != -1 && maxScore > 0.15f) {
            val coeffs = FloatArray(32); yolactOutputCoeffs?.rewind()
            val fb = yolactOutputCoeffs?.asFloatBuffer(); fb?.position(bestIdx * 32); fb?.get(coeffs)
            val mask = generateYolactMask(coeffs, yolactOutputProtos!!)
            
            val finalMask = try {
                Bitmap.createScaledBitmap(mask, bitmap.width, bitmap.height, true)
            } catch (e: Exception) { mask }
            
            if (finalMask != mask) {
                mask.recycle()
            }
            activity?.runOnUiThread { _binding?.bodyOverlay?.updateMaskOnly(finalMask, startColor, endColor, isMirrorMode) }
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
        
        val finalMask = try {
            Bitmap.createScaledBitmap(mask, bitmap.width, bitmap.height, true)
        } catch (e: Exception) { mask }
        
        if (finalMask != mask) {
            mask.recycle()
        }
        activity?.runOnUiThread { _binding?.bodyOverlay?.updateMaskOnly(finalMask, startColor, endColor, isMirrorMode) }
    }

    private fun processRvm(bitmap: Bitmap) {
        val interpreter = rvmInterpreter ?: return
        val tImage = rvmTensorImage ?: return
        tImage.load(bitmap)
        val inputBuffer = rvmImageProcessor?.process(tImage)?.buffer ?: return
        
        r1i!!.rewind(); r2i!!.rewind(); r3i!!.rewind(); r4i!!.rewind()
        val inputs = arrayOfNulls<Any>(interpreter.inputTensorCount)
        inputs[rvmIdxSrc] = inputBuffer
        inputs[rvmIdxR1i] = r1i!!
        inputs[rvmIdxR2i] = r2i!!
        inputs[rvmIdxR3i] = r3i!!
        inputs[rvmIdxR4i] = r4i!!
        if (rvmIdxRatio != -1) inputs[rvmIdxRatio] = rvmRatioBuffer!!
        
        for (i in inputs.indices) {
            if (inputs[i] == null) {
                val t = interpreter.getInputTensor(i)
                inputs[i] = ByteBuffer.allocateDirect(t.numBytes()).order(ByteOrder.nativeOrder())
            }
        }

        rvmOutputFgr?.clear(); rvmOutputPha?.clear()
        r1o?.clear(); r2o?.clear(); r3o?.clear(); r4o?.clear()
        
        val outputs = mutableMapOf<Int, Any>()
        outputs[rvmIdxFgr] = rvmOutputFgr!!
        outputs[rvmIdxPha] = rvmOutputPha!!
        outputs[rvmIdxR1o] = r1o!!
        outputs[rvmIdxR2o] = r2o!!
        outputs[rvmIdxR3o] = r3o!!
        outputs[rvmIdxR4o] = r4o!!
        
        try {
            interpreter.runForMultipleInputsOutputs(inputs, outputs)
            
            r1i!!.rewind(); r1o!!.rewind()
            if (r1i!!.capacity() == r1o!!.capacity()) r1i!!.put(r1o!!)
            r2i!!.rewind(); r2o!!.rewind()
            if (r2i!!.capacity() == r2o!!.capacity()) r2i!!.put(r2o!!)
            r3i!!.rewind(); r3o!!.rewind()
            if (r3i!!.capacity() == r3o!!.capacity()) r3i!!.put(r3o!!)
            r4i!!.rewind(); r4o!!.rewind()
            if (r4i!!.capacity() == r4o!!.capacity()) r4i!!.put(r4o!!)
        } catch (e: Exception) {
            val msg = "${e.javaClass.simpleName}: ${e.message ?: "Unknown Error"}"
            Log.e(TAG, "RVM Run Error: $msg")
            addLog("RVM Run Error: $msg")
            return
        }
        
        rvmOutputPha?.rewind()
        val fb = rvmOutputPha?.asFloatBuffer() ?: return
        val pixels = reusableMaskPixels!!; pixels.clear()
        for (i in 0 until (rvmH * rvmW)) {
            if (fb.hasRemaining()) {
                val alpha = fb.get()
                pixels.put((if (alpha > 0.5f) 255 else 0).toByte())
            }
        }
        val mask = Bitmap.createBitmap(rvmW, rvmH, Bitmap.Config.ALPHA_8)
        pixels.rewind()
        mask.copyPixelsFromBuffer(pixels)
        
        val finalMask = try {
            Bitmap.createScaledBitmap(mask, bitmap.width, bitmap.height, true)
        } catch (e: Exception) { mask }
        
        if (finalMask != mask) {
            mask.recycle()
        }
        activity?.runOnUiThread { _binding?.bodyOverlay?.updateMaskOnly(finalMask, startColor, endColor, isMirrorMode) }
    }

    private fun processImageProxy(imageProxy: ImageProxy): Bitmap? {
        return try {
            val buffer = imageProxy.planes[0].buffer; buffer.rewind()
            val bitmap = Bitmap.createBitmap(imageProxy.width, imageProxy.height, Bitmap.Config.ARGB_8888)
            bitmap.copyPixelsFromBuffer(buffer)
            val matrix = Matrix().apply { 
                postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())
                postScale(-1f, 1f) 
            }
            val rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
            if (rotated != bitmap) {
                bitmap.recycle()
            }
            rotated
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