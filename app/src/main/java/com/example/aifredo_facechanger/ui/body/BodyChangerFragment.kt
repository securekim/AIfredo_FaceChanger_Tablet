package com.example.aifredo_facechanger.ui.body

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.RectF
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.os.Looper
import android.util.Log
import android.util.Size
import android.view.LayoutInflater
import android.view.TextureView
import android.view.View
import android.view.ViewGroup
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.OptIn
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
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
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import org.tensorflow.lite.support.common.FileUtil
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.DataType
import com.google.mediapipe.framework.image.BitmapImageBuilder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlin.math.exp

class BodyChangerFragment : Fragment() {

    private var _binding: FragmentBodyChangerBinding? = null
    private val binding get() = _binding!!

    private var cameraExecutor: ExecutorService? = null
    private val isProcessing = AtomicBoolean(false)

    private var mlKitSegmenter: Segmenter? = null
    @Volatile private var poseLandmarker: PoseLandmarker? = null
    @Volatile private var yolactInterpreter: Interpreter? = null
    @Volatile private var yolo26nInterpreter: Interpreter? = null
    @Volatile private var modnetInterpreter: Interpreter? = null
    @Volatile private var rvmInterpreter: Interpreter? = null
    @Volatile private var yoloxInterpreter: Interpreter? = null

    private var gpuDelegate: GpuDelegate? = null
    private var yoloxGpuDelegate: GpuDelegate? = null
    private var nnApiDelegate: NnApiDelegate? = null

    private var yolactImageProcessor: ImageProcessor? = null
    private var yolo26nImageProcessor: ImageProcessor? = null
    private var modnetImageProcessor: ImageProcessor? = null
    private var rvmImageProcessor: ImageProcessor? = null

    private var yolactTensorImage: TensorImage? = null
    private var yolo26nTensorImage: TensorImage? = null
    private var modnetTensorImage: TensorImage? = null
    private var rvmTensorImage: TensorImage? = null

    private var rvmNchwBuffer: ByteBuffer? = null
    private var rvmInputArr: FloatArray? = null
    private var rvmNchwFloatArray: FloatArray? = null
    private var rvmOutputFloatArray: FloatArray? = null
    private var yoloxFloatArray: FloatArray? = null

    private var yolactOutputBoxes: ByteBuffer? = null
    private var yolactOutputScores: ByteBuffer? = null
    private var yolactOutputCoeffs: ByteBuffer? = null
    private var yolactOutputProtos: ByteBuffer? = null

    private var yolo26nOutput0: ByteBuffer? = null
    private var yolo26nOutput1: ByteBuffer? = null
    private var yolo26nIdxDetect = 0
    private var yolo26nIdxProto = 1

    private var modnetOutputBuffer: ByteBuffer? = null
    private var yoloxOutputBuffer: ByteBuffer? = null

    private var yolactIdxBoxes = 0
    private var yolactIdxScores = 1
    private var yolactIdxCoeffs = 2
    private var yolactIdxProtos = 3

    private val rvmStateBuffers = Array(4) { arrayOfNulls<ByteBuffer>(2) }
    private var rvmStateToggle = 0
    private var rvmOutputPha: ByteBuffer? = null
    private var rvmOutputFgr: ByteBuffer? = null
    private var rvmRatioBuffer: ByteBuffer? = null
    private var rvmByteArray: ByteArray? = null
    private var rvmMaskBitmap: Bitmap? = null

    private var rvmH: Int = 192
    private var rvmW: Int = 320
    private var yoloxW: Int = 320
    private var yoloxH: Int = 256
    private var isYoloxNchw: Boolean = true
    private var rvmIdxSrc = -1
    private var rvmIdxR1i = -1; private var rvmIdxR2i = -1; private var rvmIdxR3i = -1; private var rvmIdxR4i = -1
    private var rvmIdxRatio = -1
    private var rvmIdxPha = -1; private var rvmIdxFgr = -1
    private var rvmIdxR1o = -1; private var rvmIdxR2o = -1; private var rvmIdxR3o = -1; private var rvmIdxR4o = -1
    private var isRvmNCHW = false

    private var rvmInputs: Array<Any?>? = null
    private var rvmOutputs: MutableMap<Int, Any>? = null

    private var reusableMaskPixels: ByteBuffer? = null
    private var reusableProtosArray: FloatArray? = null
    private var reusableScoresArray: FloatArray? = null

    private var rtspBitmapBuffer: Bitmap? = null
    private var frameExtractionThread: HandlerThread? = null
    private var backgroundRtspHandler: Handler? = null

    private val segmenterLock = Any()

    private var startColor: Int = Color.RED
    private var endColor: Int = Color.BLUE
    private var selectedModel: String = "MediaPipe Pose"
    private var selectedDelegate: String = "CPU"
    private var actualDelegate: String = "CPU"
    private var rtspQuality: String = "High"
    private var isMirrorMode: Boolean = false
    private var lastMediaPipeTimestamp = -1L

    private val sdf = SimpleDateFormat("HH:mm:ss", Locale.getDefault())
    private val tagStr = "BodyChanger"

    private val yolactModelFile = "yolact_550x550_model_float16_quant.tflite"
    private val yolo26nModelFile = "yolo26n-seg_float16.tflite"
    private val modnetModelFile = "MODNet_256x256_model_float16_quant.tflite"
    private var rvmModelFile = "rvm_resnet50_192x320_model_float16_quant.tflite"
    private val yoloxHybridModelFile = "yolox_n_body_head_hand_post_0461_0.4428_1x3x256x320_float16.tflite"
    private val yoloxTinyModelFile = "yolox_tiny_320x320_model_float16_quant.tflite"

    private var maxMem = 0L
    private var maxCpu = 0.0
    private var lastCpuTime = 0L
    private var lastSampleTime = 0L
    private val perfHandler = Handler(Looper.getMainLooper())
    private val perfRunnable = object : Runnable {
        override fun run() {
            updatePerformanceMetrics()
            perfHandler.postDelayed(this, 1000)
        }
    }

    private var exoPlayer: ExoPlayer? = null
    private var isRtspMode = false

    private val rtspFrameHandler = Handler(Looper.getMainLooper())
    private val rtspFrameRunnable = object : Runnable {
        override fun run() {
            if (isRtspMode && exoPlayer?.isPlaying == true) {
                if (isProcessing.compareAndSet(false, true)) {
                    cameraExecutor?.execute { extractFrameFromPlayer() }
                }
            }
            rtspFrameHandler.postDelayed(this, 33)
        }
    }

    companion object {
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
        cameraExecutor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors())
        frameExtractionThread = HandlerThread("RTSP_FrameExtractor").apply { start() }
        backgroundRtspHandler = Handler(frameExtractionThread!!.looper)

        if (allPermissionsGranted()) startStream()
        else requestPermissionLauncher.launch(arrayOf(Manifest.permission.CAMERA))
    }

    override fun onResume() {
        super.onResume()
        loadSettings()
        startStream()
        setupSegmenter()
        perfHandler.post(perfRunnable)
    }

    override fun onPause() {
        super.onPause()
        stopRtsp()
        stopCamera()
        perfHandler.removeCallbacks(perfRunnable)
        lifecycleScope.launch(Dispatchers.Default) { synchronized(segmenterLock) { closeCurrentSegmenter() } }
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

        rvmModelFile = if (selectedModel == "RVM 720x1280") {
            "rvm_resnet50_720x1280_model_float16_quant.tflite"
        } else {
            "rvm_resnet50_192x320_model_float16_quant.tflite"
        }
    }

    private fun setupSegmenter() {
        if (isInitializing) return
        isInitializing = true
        lifecycleScope.launch(Dispatchers.Default) {
            try {
                System.gc()
                delay(300)

                synchronized(segmenterLock) {
                    closeCurrentSegmenter()
                    if (!isAdded) return@synchronized
                    when (selectedModel) {
                        "MediaPipe Pose" -> initMediaPipePose()
                        "YOLACT" -> initYolact()
                        "yolo26n-seg" -> initYolo26nSeg()
                        "YOLOX + RVM" -> {
                            initYolox(yoloxHybridModelFile)
                            initRvm()
                        }
                        "YOLOX tiny" -> initYolox(yoloxTinyModelFile)
                        "MODNet" -> initModNet()
                        "RVM 192x320", "RVM 720x1280", "RVM" -> initRvm()
                        "ML Kit" -> initMlKit()
                        else -> initMediaPipePose()
                    }
                }
            } finally {
                isInitializing = false
            }
        }
    }

    private fun updatePerformanceMetrics() {
        if (_binding == null) return
        val runtime = Runtime.getRuntime()
        val usedMem = (runtime.totalMemory() - runtime.freeMemory()) / 1024 / 1024
        if (usedMem > maxMem) maxMem = usedMem

        val currentCpuTime = android.os.Process.getElapsedCpuTime()
        val currentTime = System.currentTimeMillis()
        var cpuUsage = 0.0
        if (lastSampleTime > 0) {
            val cpuDiff = currentCpuTime - lastCpuTime
            val timeDiff = currentTime - lastSampleTime
            if (timeDiff > 0) {
                cpuUsage = (cpuDiff.toDouble() / timeDiff.toDouble() / Runtime.getRuntime().availableProcessors()) * 100.0
                if (cpuUsage > maxCpu) maxCpu = cpuUsage
            }
        }
        lastCpuTime = currentCpuTime
        lastSampleTime = currentTime

        binding.perfText.text = String.format(
            Locale.getDefault(),
            "CPU: %.1f%% (Peak: %.1f%%)\nMEM: %dMB (Peak: %dMB)\nGPU: %s",
            cpuUsage, maxCpu, usedMem, maxMem, if (gpuDelegate != null || yoloxGpuDelegate != null) "Active" else if (nnApiDelegate != null) "NNAPI" else "Off"
        )
    }

    private fun closeCurrentSegmenter() {
        try {
            mlKitSegmenter?.close(); mlKitSegmenter = null
            poseLandmarker?.close(); poseLandmarker = null
            yolactInterpreter?.close(); yolactInterpreter = null
            yolo26nInterpreter?.close(); yolo26nInterpreter = null
            modnetInterpreter?.close(); modnetInterpreter = null
            rvmInterpreter?.close(); rvmInterpreter = null
            yoloxInterpreter?.close(); yoloxInterpreter = null
            gpuDelegate?.close(); gpuDelegate = null
            yoloxGpuDelegate?.close(); yoloxGpuDelegate = null
            nnApiDelegate?.close(); nnApiDelegate = null
            rtspBitmapBuffer?.recycle(); rtspBitmapBuffer = null
            rvmMaskBitmap?.recycle(); rvmMaskBitmap = null
            rvmInputs = null; rvmOutputs = null
            rvmImageProcessor = null; rvmTensorImage = null
            rvmNchwBuffer = null; rvmInputArr = null
            rvmNchwFloatArray = null; rvmOutputFloatArray = null
            yoloxFloatArray = null
            for (i in 0..3) { rvmStateBuffers[i][0] = null; rvmStateBuffers[i][1] = null }
        } catch (e: Exception) {}
    }

    private fun convertBitmapToByteBuffer(bitmap: Bitmap, width: Int, height: Int, isNchw: Boolean): ByteBuffer {
        val scaled = if (bitmap.width != width || bitmap.height != height) {
            Bitmap.createScaledBitmap(bitmap, width, height, true)
        } else bitmap

        val buffer = ByteBuffer.allocateDirect(1 * 3 * width * height * 4).order(ByteOrder.nativeOrder())
        val pixels = IntArray(width * height)
        scaled.getPixels(pixels, 0, width, 0, 0, width, height)

        if (isNchw) {
            for (i in pixels.indices) buffer.putFloat(((pixels[i] shr 16) and 0xFF) / 255f)
            for (i in pixels.indices) buffer.putFloat(((pixels[i] shr 8) and 0xFF) / 255f)
            for (i in pixels.indices) buffer.putFloat((pixels[i] and 0xFF) / 255f)
        } else {
            for (i in pixels.indices) {
                buffer.putFloat(((pixels[i] shr 16) and 0xFF) / 255f)
                buffer.putFloat(((pixels[i] shr 8) and 0xFF) / 255f)
                buffer.putFloat((pixels[i] and 0xFF) / 255f)
            }
        }
        if (scaled != bitmap) scaled.recycle()
        buffer.rewind()
        return buffer
    }

    private fun initMediaPipePose() {
        try {
            val baseOptionsBuilder = BaseOptions.builder().setModelAssetPath("pose_landmarker_lite.task")
            if (selectedDelegate.uppercase() == "GPU") baseOptionsBuilder.setDelegate(Delegate.GPU)
            else baseOptionsBuilder.setDelegate(Delegate.CPU)

            val optionsBuilder = PoseLandmarker.PoseLandmarkerOptions.builder()
                .setBaseOptions(baseOptionsBuilder.build())
                .setRunningMode(RunningMode.VIDEO)
                .setOutputSegmentationMasks(true)
                .setMinPoseDetectionConfidence(0.2f)
                .setMinPosePresenceConfidence(0.4f)
                .setMinTrackingConfidence(0.4f)

            poseLandmarker = PoseLandmarker.createFromOptions(requireContext(), optionsBuilder.build())
            actualDelegate = selectedDelegate
            addLog(">> MediaPipe Pose Ready ($actualDelegate)")
        } catch (e: Exception) { addLog("MediaPipe Pose Init Error: ${e.message}") }
    }

    private fun initMlKit() {
        actualDelegate = "N/A"
        try {
            val options = SelfieSegmenterOptions.Builder().setDetectorMode(SelfieSegmenterOptions.STREAM_MODE).build()
            mlKitSegmenter = Segmentation.getClient(options)
            addLog(">> ML Kit Ready")
        } catch (e: Exception) { addLog("ML Kit Error: ${e.message}") }
    }

    private fun getInterpreterOptions(): Interpreter.Options {
        val options = Interpreter.Options()
        actualDelegate = "CPU"
        options.setNumThreads(Runtime.getRuntime().availableProcessors())
        options.setUseXNNPACK(true)
        if (selectedDelegate.uppercase() == "GPU") {
            try {
                val gpuOptions = GpuDelegate.Options().apply {
                    setInferencePreference(GpuDelegate.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED)
                    setPrecisionLossAllowed(true)
                }
                gpuDelegate = GpuDelegate(gpuOptions)
                options.addDelegate(gpuDelegate)
                actualDelegate = "GPU"
            } catch (e: Exception) {
                try {
                    nnApiDelegate = NnApiDelegate()
                    options.addDelegate(nnApiDelegate)
                    actualDelegate = "NNAPI"
                } catch (e2: Exception) { actualDelegate = "CPU (Fallback)" }
            }
        }
        return options
    }

    private fun getYoloxGpuInterpreterOptions(): Interpreter.Options {
        val options = Interpreter.Options()
        options.setNumThreads(Runtime.getRuntime().availableProcessors())
        try {
            val gpuOptions = GpuDelegate.Options().apply {
                setInferencePreference(GpuDelegate.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED)
                setPrecisionLossAllowed(true)
            }
            yoloxGpuDelegate = GpuDelegate(gpuOptions)
            options.addDelegate(yoloxGpuDelegate)
        } catch (e: Exception) { options.setUseXNNPACK(true) }
        return options
    }

    private fun initYolact() {
        try {
            val interpreter = Interpreter(FileUtil.loadMappedFile(requireContext(), yolactModelFile), getInterpreterOptions())
            yolactInterpreter = interpreter
            for (i in 0 until interpreter.outputTensorCount) {
                val name = interpreter.getOutputTensor(i).name().lowercase()
                if (name.contains("box")) yolactIdxBoxes = i
                else if (name.contains("score") || name.contains("conf")) yolactIdxScores = i
                else if (name.contains("coeff") || name.contains("mask")) yolactIdxCoeffs = i
                else if (name.contains("proto")) yolactIdxProtos = i
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

    private fun initYolo26nSeg() {
        try {
            val interpreter = Interpreter(FileUtil.loadMappedFile(requireContext(), yolo26nModelFile), getInterpreterOptions())
            yolo26nInterpreter = interpreter
            yolo26nImageProcessor = ImageProcessor.Builder().add(ResizeOp(640, 640, ResizeOp.ResizeMethod.BILINEAR)).add(NormalizeOp(0f, 255f)).build()
            yolo26nTensorImage = TensorImage(DataType.FLOAT32)

            val out0Size = interpreter.getOutputTensor(0).numBytes()
            val out1Size = interpreter.getOutputTensor(1).numBytes()

            if (out0Size > out1Size) {
                yolo26nIdxDetect = 0; yolo26nIdxProto = 1
            } else {
                yolo26nIdxDetect = 1; yolo26nIdxProto = 0
            }

            yolo26nOutput0 = ByteBuffer.allocateDirect(out0Size).order(ByteOrder.nativeOrder())
            yolo26nOutput1 = ByteBuffer.allocateDirect(out1Size).order(ByteOrder.nativeOrder())

            reusableMaskPixels = ByteBuffer.allocateDirect(640 * 640).order(ByteOrder.nativeOrder())
            addLog(">> yolo26n-seg Ready ($actualDelegate)")
        } catch (e: Exception) { addLog("yolo26n-seg Init Error: ${e.message}") }
    }

    private fun initModNet() {
        try {
            modnetInterpreter = Interpreter(FileUtil.loadMappedFile(requireContext(), modnetModelFile), getInterpreterOptions())
            modnetImageProcessor = ImageProcessor.Builder().add(ResizeOp(256, 256, ResizeOp.ResizeMethod.BILINEAR)).add(NormalizeOp(floatArrayOf(127.5f, 127.5f, 127.5f), floatArrayOf(127.5f, 127.5f, 127.5f))).build()
            modnetTensorImage = TensorImage(DataType.FLOAT32)
            modnetOutputBuffer = ByteBuffer.allocateDirect(modnetInterpreter!!.getOutputTensor(0).numBytes()).order(ByteOrder.nativeOrder())
            reusableMaskPixels = ByteBuffer.allocateDirect(256 * 256).order(ByteOrder.nativeOrder())
            addLog(">> MODNet Ready ($actualDelegate)")
        } catch (e: Exception) { addLog("MODNet Init Error: ${e.message}") }
    }

    private fun initYolox(modelFile: String) {
        lifecycleScope.launch(Dispatchers.Default) {
            val mappedFile = FileUtil.loadMappedFile(requireContext(), modelFile)
            var inter: Interpreter? = null
            try {
                inter = Interpreter(mappedFile, getYoloxGpuInterpreterOptions())
                addLog(">> YOLOX Ready: $modelFile (GPU)")
            } catch (e: Exception) {
                inter = Interpreter(mappedFile, Interpreter.Options().setNumThreads(Runtime.getRuntime().availableProcessors()).setUseXNNPACK(true))
                addLog(">> YOLOX Ready: $modelFile (CPU)")
            }
            inter?.let {
                val shape = it.getInputTensor(0).shape()
                if (shape[1] == 3) { isYoloxNchw = true; yoloxH = shape[2]; yoloxW = shape[3] }
                else { isYoloxNchw = false; yoloxH = shape[1]; yoloxW = shape[2] }
                yoloxOutputBuffer = ByteBuffer.allocateDirect(it.getOutputTensor(0).numBytes()).order(ByteOrder.nativeOrder())
                yoloxInterpreter = it
            }
        }
    }

    private fun initRvm() {
        try {
            val modelFile = FileUtil.loadMappedFile(requireContext(), rvmModelFile)
            rvmInterpreter = Interpreter(modelFile, getInterpreterOptions())
            rvmIdxSrc = -1; rvmIdxRatio = -1; rvmIdxFgr = -1; rvmIdxPha = -1
            val inputStates = mutableListOf<Pair<Int, Int>>()
            for (i in 0 until rvmInterpreter!!.inputTensorCount) {
                val shape = rvmInterpreter!!.getInputTensor(i).shape()
                if (shape.size == 4) {
                    if (shape[1] == 3 || shape[3] == 3) rvmIdxSrc = i
                    else inputStates.add(Pair(i, shape[1] * shape[2] * shape[3]))
                } else if (shape.size == 1) rvmIdxRatio = i
            }
            inputStates.sortByDescending { it.second }
            rvmIdxR1i = inputStates.getOrNull(0)?.first ?: -1
            rvmIdxR2i = inputStates.getOrNull(1)?.first ?: -1
            rvmIdxR3i = inputStates.getOrNull(2)?.first ?: -1
            rvmIdxR4i = inputStates.getOrNull(3)?.first ?: -1

            val outputStates = mutableListOf<Pair<Int, Int>>()
            for (i in 0 until rvmInterpreter!!.outputTensorCount) {
                val shape = rvmInterpreter!!.getOutputTensor(i).shape()
                if (shape.size == 4) {
                    if (shape[1] == 1 || shape[3] == 1) rvmIdxPha = i
                    else if (shape[1] == 3 || shape[3] == 3) rvmIdxFgr = i
                    else outputStates.add(Pair(i, shape[1] * shape[2] * shape[3]))
                }
            }
            outputStates.sortByDescending { it.second }
            rvmIdxR1o = outputStates.getOrNull(0)?.first ?: -1
            rvmIdxR2o = outputStates.getOrNull(1)?.first ?: -1
            rvmIdxR3o = outputStates.getOrNull(2)?.first ?: -1
            rvmIdxR4o = outputStates.getOrNull(3)?.first ?: -1

            val srcShape = rvmInterpreter!!.getInputTensor(rvmIdxSrc).shape()
            isRvmNCHW = srcShape[1] == 3
            if (isRvmNCHW) { rvmH = srcShape[2]; rvmW = srcShape[3] } else { rvmH = srcShape[1]; rvmW = srcShape[2] }

            val statesIn = intArrayOf(rvmIdxR1i, rvmIdxR2i, rvmIdxR3i, rvmIdxR4i)
            for (i in 0..3) if (statesIn[i] != -1) {
                val size = rvmInterpreter!!.getInputTensor(statesIn[i]).numBytes()
                rvmStateBuffers[i][0] = ByteBuffer.allocateDirect(size).order(ByteOrder.nativeOrder())
                rvmStateBuffers[i][1] = ByteBuffer.allocateDirect(size).order(ByteOrder.nativeOrder())
            }
            rvmInputs = arrayOfNulls<Any>(rvmInterpreter!!.inputTensorCount)
            for (i in 0..3) if (statesIn[i] != -1) rvmInputs!![statesIn[i]] = rvmStateBuffers[i][0]!!
            if (rvmIdxRatio != -1) {
                rvmRatioBuffer = ByteBuffer.allocateDirect(rvmInterpreter!!.getInputTensor(rvmIdxRatio).numBytes()).order(ByteOrder.nativeOrder())
                rvmRatioBuffer!!.asFloatBuffer().put(1.0f); rvmInputs!![rvmIdxRatio] = rvmRatioBuffer!!
            }
            if (rvmIdxFgr != -1) rvmOutputFgr = ByteBuffer.allocateDirect(rvmInterpreter!!.getOutputTensor(rvmIdxFgr).numBytes()).order(ByteOrder.nativeOrder())
            if (rvmIdxPha != -1) rvmOutputPha = ByteBuffer.allocateDirect(rvmInterpreter!!.getOutputTensor(rvmIdxPha).numBytes()).order(ByteOrder.nativeOrder())

            val count = rvmH * rvmW
            rvmOutputFloatArray = FloatArray(count)
            rvmInputArr = FloatArray(count * 3); rvmNchwFloatArray = FloatArray(count * 3)
            rvmByteArray = ByteArray(count); reusableMaskPixels = ByteBuffer.allocateDirect(count).order(ByteOrder.nativeOrder())
            rvmMaskBitmap = Bitmap.createBitmap(rvmW, rvmH, Bitmap.Config.ALPHA_8)
            rvmOutputs = mutableMapOf<Int, Any>()
            rvmImageProcessor = ImageProcessor.Builder().add(ResizeOp(rvmH, rvmW, ResizeOp.ResizeMethod.BILINEAR)).add(NormalizeOp(0f, 255f)).build()
            rvmTensorImage = TensorImage(DataType.FLOAT32)
            if (isRvmNCHW) rvmNchwBuffer = ByteBuffer.allocateDirect(1 * 3 * rvmW * rvmH * 4).order(ByteOrder.nativeOrder())
            addLog(">> RVM Ready ($actualDelegate) ${rvmW}x${rvmH}")
        } catch (e: Exception) { addLog("RVM Init Error: ${e.message}") }
    }

    private fun startStream() { if (isRtspMode) startRtsp() else startCamera() }

    private fun startCamera() {
        stopRtsp()
        _binding?.viewFinder?.visibility = View.INVISIBLE; _binding?.playerView?.visibility = View.GONE
        ProcessCameraProvider.getInstance(requireContext()).addListener({
            val provider = try { ProcessCameraProvider.getInstance(requireContext()).get() } catch (e: Exception) { return@addListener }
            val targetSize = Size(1280, 720)
            val analyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .setTargetResolution(targetSize).build().also { it.setAnalyzer(cameraExecutor!!) { proxy -> processFrame(proxy) } }
            try {
                provider.unbindAll(); provider.bindToLifecycle(viewLifecycleOwner, CameraSelector.DEFAULT_FRONT_CAMERA, analyzer)
                addLog("Camera started with ${targetSize.width}x${targetSize.height}")
            } catch (e: Exception) {}
        }, ContextCompat.getMainExecutor(requireContext()))
    }

    @OptIn(UnstableApi::class)
    private fun startRtsp() {
        stopRtsp(); stopCamera()
        _binding?.viewFinder?.visibility = View.GONE; _binding?.playerView?.visibility = View.VISIBLE
        val sharedPref = activity?.getSharedPreferences("AIfredoPrefs", Context.MODE_PRIVATE) ?: return
        val ip = (sharedPref.getString("rtsp_ip", "") ?: "").trim().removePrefix("rtsp://")
        val id = sharedPref.getString("rtsp_id", "") ?: ""; val pw = sharedPref.getString("rtsp_pw", "") ?: ""
        if (ip.isEmpty()) return
        val streamPath = if (rtspQuality == "Low") "stream2" else "stream1"
        val rtspUrl = "rtsp://${if (id.isNotEmpty()) "$id:$pw@" else ""}$ip${if (!ip.contains("/")) ":554/$streamPath" else ""}"
        val loadControl = DefaultLoadControl.Builder().setBufferDurationsMs(500, 1000, 250, 500).build()
        exoPlayer = ExoPlayer.Builder(requireContext()).setLoadControl(loadControl).build().apply {
            trackSelectionParameters = trackSelectionParameters.buildUpon().setTrackTypeDisabled(androidx.media3.common.C.TRACK_TYPE_AUDIO, true).build()
            setMediaSource(RtspMediaSource.Factory().setForceUseRtpTcp(true).setTimeoutMs(4000).createMediaSource(MediaItem.fromUri(rtspUrl)))
            prepare(); playWhenReady = true
        }
        binding.playerView.player = exoPlayer; rtspFrameHandler.post(rtspFrameRunnable)
    }

    @OptIn(UnstableApi::class)
    private fun extractFrameFromPlayer() {
        val textureView = _binding?.playerView?.videoSurfaceView as? TextureView
        if (textureView == null || !textureView.isAvailable) { isProcessing.set(false); return }

        // MediaPipe GPU 정렬 이슈 방지를 위해 너비/높이를 16의 배수로 강제 조정
        val targetW = (textureView.width / 16) * 16
        val targetH = (textureView.height / 16) * 16
        if (targetW <= 0 || targetH <= 0) { isProcessing.set(false); return }

        val frame = synchronized(segmenterLock) {
            if (rtspBitmapBuffer == null || rtspBitmapBuffer!!.width != targetW || rtspBitmapBuffer!!.height != targetH) {
                rtspBitmapBuffer?.recycle()
                rtspBitmapBuffer = Bitmap.createBitmap(targetW, targetH, Bitmap.Config.ARGB_8888)
            }
            try {
                textureView.getBitmap(rtspBitmapBuffer!!)
                // Copy를 통해 하드웨어 가속 비트맵에서 표준 소프트웨어 비트맵으로 변환 (연속적인 메모리 구조 보장)
                val cleanBitmap = Bitmap.createBitmap(targetW, targetH, Bitmap.Config.ARGB_8888)
                Canvas(cleanBitmap).drawBitmap(rtspBitmapBuffer!!, 0f, 0f, null)
                cleanBitmap
            } catch (e: Exception) { null }
        }
        if (frame != null) processFrameBitmap(frame) else isProcessing.set(false)
    }

    private fun stopCamera() { try { ProcessCameraProvider.getInstance(requireContext()).get().unbindAll() } catch (e: Exception) {} }

    private fun stopRtsp() {
        rtspFrameHandler.removeCallbacks(rtspFrameRunnable)
        exoPlayer?.release(); exoPlayer = null; _binding?.playerView?.player = null
        rtspBitmapBuffer?.recycle(); rtspBitmapBuffer = null
    }

    private fun processFrame(imageProxy: ImageProxy) {
        if (isRtspMode || isProcessing.get()) { imageProxy.close(); return }
        val width = imageProxy.width; val height = imageProxy.height; val rotation = imageProxy.imageInfo.rotationDegrees

        val bitmap = try {
            val buffer = imageProxy.planes[0].buffer
            val pixelStride = imageProxy.planes[0].pixelStride
            val rowStride = imageProxy.planes[0].rowStride
            val rowPadding = rowStride - pixelStride * width

            val b = if (rowPadding == 0) {
                val tempB = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
                buffer.rewind(); tempB.copyPixelsFromBuffer(buffer); tempB
            } else {
                val tempW = rowStride / pixelStride
                val tempB = Bitmap.createBitmap(tempW, height, Bitmap.Config.ARGB_8888)
                buffer.rewind(); tempB.copyPixelsFromBuffer(buffer)
                val cropped = Bitmap.createBitmap(tempB, 0, 0, width, height)
                tempB.recycle()
                cropped
            }
            b
        } catch (e: Exception) { null }
        imageProxy.close()

        if (bitmap != null) {
            cameraExecutor?.execute {
                val matrix = Matrix().apply {
                    postRotate(rotation.toFloat())
                    postScale(-1f, 1f, width / 2f, height / 2f)
                }
                val rotated = try { Bitmap.createBitmap(bitmap, 0, 0, width, height, matrix, true) } catch (e: Exception) { bitmap }
                if (rotated != bitmap) bitmap.recycle()
                processFrameBitmap(rotated)
            }
        }
    }

    private fun getContiguousBitmap(bitmap: Bitmap): Bitmap {
        if (bitmap.rowBytes == bitmap.width * 4 && bitmap.config == Bitmap.Config.ARGB_8888) {
            return bitmap
        }
        val pixels = IntArray(bitmap.width * bitmap.height)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        return Bitmap.createBitmap(pixels, bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)
    }

    private fun fallbackToEmptyMask(bitmap: Bitmap) {
        val emptyMask = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ALPHA_8)
        activity?.runOnUiThread {
            _binding?.bodyOverlay?.updateData(emptyMask, bitmap, startColor, endColor, isMirrorMode)
            isProcessing.set(false)
        }
    }

    private fun processFrameBitmap(originalBitmap: Bitmap) {
        synchronized(segmenterLock) {
            if (isInitializing) {
                originalBitmap.recycle()
                isProcessing.set(false)
                return
            }

            val safeBitmap = getContiguousBitmap(originalBitmap)
            if (originalBitmap !== safeBitmap) {
                originalBitmap.recycle()
            }

            when (selectedModel) {
                "MediaPipe Pose" -> processMediaPipePose(safeBitmap)
                "YOLACT" -> processYolact(safeBitmap)
                "yolo26n-seg" -> processYolo26nSeg(safeBitmap)
                "YOLOX + RVM" -> processYoloxHybrid(safeBitmap)
                "YOLOX tiny" -> processYoloxTiny(safeBitmap)
                "MODNet" -> processModNet(safeBitmap)
                "RVM 192x320", "RVM 720x1280", "RVM" -> processRvm(safeBitmap)
                "ML Kit" -> processMlKit(safeBitmap)
                else -> fallbackToEmptyMask(safeBitmap)
            }
        }
    }

    private fun processMediaPipePose(bitmap: Bitmap) {
        val pose = poseLandmarker
        if (pose != null) {
            try {
                // 엄격하게 증가하는 타임스탬프 생성
                var currentTimestamp = System.currentTimeMillis()
                if (currentTimestamp <= lastMediaPipeTimestamp) {
                    currentTimestamp = lastMediaPipeTimestamp + 1
                }
                lastMediaPipeTimestamp = currentTimestamp

                val mpImage = BitmapImageBuilder(bitmap).build()
                val result = pose.detectForVideo(mpImage, currentTimestamp)

                val segmentationMasks = result.segmentationMasks()
                if (segmentationMasks.isPresent && segmentationMasks.get().isNotEmpty()) {
                    val mask = segmentationMasks.get()[0]
                    val width = mask.width
                    val height = mask.height
                    
                    // GPU 출력 시 비연속적 데이터 문제 해결을 위해 예외 처리 및 추출 시도
                    val byteBuffer = try {
                        com.google.mediapipe.framework.image.ByteBufferExtractor.extract(mask)
                    } catch (e: Exception) {
                        addLog("MP Mask Extract Error: ${e.message}")
                        null
                    }
                    
                    if (byteBuffer == null) {
                        fallbackToEmptyMask(bitmap)
                        return
                    }

                    byteBuffer.rewind()

                    val pixels = ByteArray(width * height)
                    if (byteBuffer.capacity() >= width * height * 4) {
                        val floatBuffer = byteBuffer.asFloatBuffer()
                        for (i in 0 until width * height) {
                            pixels[i] = (if (floatBuffer.get() > 0.5f) 255 else 0).toByte()
                        }
                    } else {
                        byteBuffer.get(pixels)
                    }

                    val maskBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ALPHA_8)
                    maskBitmap.copyPixelsFromBuffer(ByteBuffer.wrap(pixels))
                    val finalMask = if (width != bitmap.width || height != bitmap.height) {
                        Bitmap.createScaledBitmap(maskBitmap, bitmap.width, bitmap.height, true).also { maskBitmap.recycle() }
                    } else maskBitmap

                    activity?.runOnUiThread {
                        _binding?.bodyOverlay?.updateData(finalMask, bitmap, startColor, endColor, isMirrorMode)
                        isProcessing.set(false)
                    }
                } else {
                    fallbackToEmptyMask(bitmap)
                }
            } catch (e: Exception) {
                addLog("MediaPipe Pose Error: ${e.message}")
                fallbackToEmptyMask(bitmap)
            }
        } else {
            fallbackToEmptyMask(bitmap)
        }
    }

    private fun processMlKit(bitmap: Bitmap) {
        val segmenter = mlKitSegmenter ?: run { fallbackToEmptyMask(bitmap); return }
        segmenter.process(InputImage.fromBitmap(bitmap, 0))
            .addOnSuccessListener { result ->
                val maskBuffer = result.buffer; val w = result.width; val h = result.height
                val pixelsArr = ByteArray(w * h); maskBuffer.rewind()
                for (i in 0 until w * h) if (maskBuffer.hasRemaining()) pixelsArr[i] = (if (maskBuffer.float > 0.45f) 255 else 0).toByte()
                val maskBitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ALPHA_8)
                maskBitmap.copyPixelsFromBuffer(ByteBuffer.wrap(pixelsArr))
                activity?.runOnUiThread { _binding?.bodyOverlay?.updateData(maskBitmap, bitmap, startColor, endColor, isMirrorMode); isProcessing.set(false) }
            }
            .addOnFailureListener {
                fallbackToEmptyMask(bitmap)
            }
    }

    private fun processYolact(bitmap: Bitmap) {
        val interpreter = yolactInterpreter ?: run { fallbackToEmptyMask(bitmap); return }
        val tImage = yolactTensorImage!!; tImage.load(bitmap)
        val inputBuffer = yolactImageProcessor?.process(tImage)?.buffer ?: run { fallbackToEmptyMask(bitmap); return }
        yolactOutputBoxes?.clear(); yolactOutputScores?.clear(); yolactOutputCoeffs?.clear(); yolactOutputProtos?.clear()
        val outputs = mapOf(yolactIdxBoxes to yolactOutputBoxes!!, yolactIdxScores to yolactOutputScores!!, yolactIdxCoeffs to yolactOutputCoeffs!!, yolactIdxProtos to yolactOutputProtos!!)
        try {
            interpreter.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputs)
        } catch (e: Exception) {
            fallbackToEmptyMask(bitmap)
            return
        }

        yolactOutputScores?.rewind(); val fbScores = yolactOutputScores?.asFloatBuffer() ?: return
        val scoresArray = reusableScoresArray!!
        if (fbScores.remaining() >= scoresArray.size) {
            fbScores.get(scoresArray)
            var bestIdx = -1; var maxScore = 0f
            for (i in 0 until 19248) { val score = scoresArray[i * 81 + 1]; if (score > maxScore) { maxScore = score; bestIdx = i } }
            if (bestIdx != -1 && maxScore > 0.15f) {
                val coeffs = FloatArray(32); yolactOutputCoeffs?.rewind()
                val fbCoeffs = yolactOutputCoeffs?.asFloatBuffer(); fbCoeffs?.position(bestIdx * 32); fbCoeffs?.get(coeffs)
                val mask = generateYolactMask(coeffs, yolactOutputProtos!!)
                val finalMask = if (mask.width != bitmap.width) Bitmap.createScaledBitmap(mask, bitmap.width, bitmap.height, true).also { mask.recycle() } else mask
                activity?.runOnUiThread { _binding?.bodyOverlay?.updateData(finalMask, bitmap, startColor, endColor, isMirrorMode); isProcessing.set(false) }
                return
            }
        }

        fallbackToEmptyMask(bitmap)
    }

    private fun processYolo26nSeg(bitmap: Bitmap) {
        val interpreter = yolo26nInterpreter ?: run { fallbackToEmptyMask(bitmap); return }
        val tImage = yolo26nTensorImage!!; tImage.load(bitmap)
        val inputBuffer = yolo26nImageProcessor?.process(tImage)?.buffer ?: run { fallbackToEmptyMask(bitmap); return }

        yolo26nOutput0?.clear(); yolo26nOutput1?.clear()
        val outputs = mapOf(0 to yolo26nOutput0!!, 1 to yolo26nOutput1!!)
        try {
            interpreter.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputs)
        } catch (e: Exception) {
            fallbackToEmptyMask(bitmap)
            return
        }

        val detectBuffer = if (yolo26nIdxDetect == 0) yolo26nOutput0 else yolo26nOutput1
        val protoBuffer = if (yolo26nIdxProto == 0) yolo26nOutput0 else yolo26nOutput1

        detectBuffer?.rewind(); val fbDet = detectBuffer?.asFloatBuffer() ?: return
        val detData = FloatArray(fbDet.remaining()); fbDet.get(detData)

        val numAnchors = 8400
        val numFeatures = detData.size / numAnchors
        if (numFeatures < 5) {
            fallbackToEmptyMask(bitmap)
            return
        }

        var maxScore = 0f; var bestIdx = -1
        for (i in 0 until numAnchors) {
            val score = detData[4 * numAnchors + i]
            if (score > maxScore) { maxScore = score; bestIdx = i }
        }

        if (bestIdx != -1 && maxScore > 0.20f) {
            val maskStartIndex = numFeatures - 32
            val coeffs = FloatArray(32)
            for (i in 0 until 32) {
                coeffs[i] = detData[(maskStartIndex + i) * numAnchors + bestIdx]
            }

            protoBuffer?.rewind(); val fbProto = protoBuffer?.asFloatBuffer() ?: return
            val protos = FloatArray(fbProto.remaining()); fbProto.get(protos)

            val maskBitmap = generateYoloMask(coeffs, protos, 160, 160)
            val finalMask = Bitmap.createScaledBitmap(maskBitmap, bitmap.width, bitmap.height, true)
            maskBitmap.recycle()
            activity?.runOnUiThread { _binding?.bodyOverlay?.updateData(finalMask, bitmap, startColor, endColor, isMirrorMode); isProcessing.set(false) }
        } else {
            fallbackToEmptyMask(bitmap)
        }
    }

    private fun generateYoloMask(coeffs: FloatArray, protos: FloatArray, maskW: Int, maskH: Int): Bitmap {
        val pixels = ByteArray(maskW * maskH)
        for (y in 0 until maskH) for (x in 0 until maskW) {
            var sum = 0f; val offset = (y * maskW + x) * 32
            for (i in 0 until 32) if (offset + i < protos.size) sum += coeffs[i] * protos[offset + i]
            val sig = 1.0f / (1.0f + exp(-sum.toDouble())).toFloat()
            pixels[y * maskW + x] = (if (sig > 0.5f) 255 else 0).toByte()
        }
        val mask = Bitmap.createBitmap(maskW, maskH, Bitmap.Config.ALPHA_8)
        mask.copyPixelsFromBuffer(ByteBuffer.wrap(pixels)); return mask
    }

    private fun generateYolactMask(coeffs: FloatArray, protos: ByteBuffer): Bitmap {
        val w = 138; val h = 138; val pixels = reusableMaskPixels!!; pixels.clear()
        val protosArray = reusableProtosArray ?: return Bitmap.createBitmap(1, 1, Bitmap.Config.ALPHA_8)
        protos.rewind(); val fbProto = protos.asFloatBuffer()
        if (fbProto.remaining() >= protosArray.size) {
            fbProto.get(protosArray)
            for (y in 0 until h) for (x in 0 until w) {
                var sum = 0f; val off = (y * w + x) * 32
                for (k in 0 until 32) sum += coeffs[k] * protosArray[off + k]
                pixels.put((if (sum > 0.5f) 255 else 0).toByte())
            }
        }
        val mask = Bitmap.createBitmap(w, h, Bitmap.Config.ALPHA_8); pixels.rewind(); mask.copyPixelsFromBuffer(pixels); return mask
    }

    private fun processModNet(bitmap: Bitmap) {
        val interpreter = modnetInterpreter ?: run { fallbackToEmptyMask(bitmap); return }
        val tImage = modnetTensorImage!!; tImage.load(bitmap)
        val inputBuffer = modnetImageProcessor?.process(tImage)?.buffer ?: run { fallbackToEmptyMask(bitmap); return }
        modnetOutputBuffer?.clear()
        try {
            interpreter.run(inputBuffer, modnetOutputBuffer)
        } catch (e: Exception) {
            fallbackToEmptyMask(bitmap)
            return
        }
        modnetOutputBuffer?.rewind(); val fb = modnetOutputBuffer?.asFloatBuffer() ?: return
        val pixels = reusableMaskPixels!!; pixels.clear()
        for (i in 0 until (256 * 256)) if (fb.hasRemaining()) pixels.put((if (fb.get() > 0.4f) 255 else 0).toByte())
        val mask = Bitmap.createBitmap(256, 256, Bitmap.Config.ALPHA_8); pixels.rewind(); mask.copyPixelsFromBuffer(pixels)
        val finalMask = if (mask.width != bitmap.width) Bitmap.createScaledBitmap(mask, bitmap.width, bitmap.height, true).also { mask.recycle() } else mask
        activity?.runOnUiThread { _binding?.bodyOverlay?.updateData(finalMask, bitmap, startColor, endColor, isMirrorMode); isProcessing.set(false) }
    }

    private fun processYoloxTiny(bitmap: Bitmap) {
        val yolox = yoloxInterpreter ?: run { fallbackToEmptyMask(bitmap); return }
        val yoloxInput = convertBitmapToByteBuffer(bitmap, yoloxW, yoloxH, isYoloxNchw)
        yoloxOutputBuffer?.clear()
        try {
            yolox.run(yoloxInput, yoloxOutputBuffer)
        } catch (e: Exception) {
            fallbackToEmptyMask(bitmap)
            return
        }
        yoloxOutputBuffer?.rewind(); val yoloxFb = yoloxOutputBuffer?.asFloatBuffer() ?: return
        val yLen = yoloxFb.remaining(); val yArr = yoloxFloatArray ?: FloatArray(yLen).also { yoloxFloatArray = it }
        yoloxFb.get(yArr)
        var maxScore = 0f; var bestBox: RectF? = null; val numAnchors = yLen / 85
        for (i in 0 until numAnchors) {
            val totalScore = yArr[i * 85 + 4] * yArr[i * 85 + 5]
            if (totalScore > maxScore) {
                maxScore = totalScore
                val cx = yArr[i * 85 + 0] / (if (yArr[i * 85 + 0] > 2f) yoloxW.toFloat() else 1f)
                val cy = yArr[i * 85 + 1] / (if (yArr[i * 85 + 1] > 2f) yoloxH.toFloat() else 1f)
                val w = yArr[i * 85 + 2] / (if (yArr[i * 85 + 2] > 2f) yoloxW.toFloat() else 1f)
                val h = yArr[i * 85 + 3] / (if (yArr[i * 85 + 3] > 2f) yoloxH.toFloat() else 1f)
                bestBox = RectF((cx - w / 2) * bitmap.width, (cy - h / 2) * bitmap.height, (cx + w / 2) * bitmap.width, (cy + h / 2) * bitmap.height)
            }
        }
        if (bestBox != null) {
            activity?.runOnUiThread { _binding?.bodyOverlay?.updateBoundingBox(bestBox, bitmap, isMirrorMode); isProcessing.set(false) }
        } else {
            fallbackToEmptyMask(bitmap)
        }
    }

    private fun processYoloxHybrid(bitmap: Bitmap) {
        val yolox = yoloxInterpreter; val rvm = rvmInterpreter
        if (yolox == null || rvm == null) { fallbackToEmptyMask(bitmap); return }
        val yoloxInput = convertBitmapToByteBuffer(bitmap, yoloxW, yoloxH, isYoloxNchw)
        yoloxOutputBuffer?.clear()
        try {
            yolox.run(yoloxInput, yoloxOutputBuffer)
        } catch (e: Exception) {
            fallbackToEmptyMask(bitmap)
            return
        }
        yoloxOutputBuffer?.rewind(); val fbYolox = yoloxOutputBuffer?.asFloatBuffer() ?: return
        val yArr = yoloxFloatArray ?: FloatArray(fbYolox.remaining()).also { yoloxFloatArray = it }
        fbYolox.get(yArr)

        var maxScore = 0f; var bestBox = RectF(0f, 0f, bitmap.width.toFloat(), bitmap.height.toFloat())
        for (i in 0 until (yArr.size / 85)) {
            val score = yArr[i * 85 + 4] * yArr[i * 85 + 5]
            if (score > maxScore) {
                maxScore = score
                val cx = yArr[i * 85 + 0] / (if (yArr[i * 85 + 0] > 2f) yoloxW.toFloat() else 1f)
                val cy = yArr[i * 85 + 1] / (if (yArr[i * 85 + 1] > 2f) yoloxH.toFloat() else 1f)
                val w = yArr[i * 85 + 2] / (if (yArr[i * 85 + 2] > 2f) yoloxW.toFloat() else 1f)
                val h = yArr[i * 85 + 3] / (if (yArr[i * 85 + 3] > 2f) yoloxH.toFloat() else 1f)
                bestBox = RectF((cx - w / 2) * bitmap.width, (cy - h / 2) * bitmap.height, (cx + w / 2) * bitmap.width, (cy + h / 2) * bitmap.height)
            }
        }
        val px = bestBox.width() * 0.15f; val py = bestBox.height() * 0.15f
        var left = (bestBox.left - px).toInt().coerceAtLeast(0); var top = (bestBox.top - py).toInt().coerceAtLeast(0)
        var right = (bestBox.right + px).toInt().coerceAtMost(bitmap.width); var bottom = (bestBox.bottom + py).toInt().coerceAtMost(bitmap.height)
        if (right <= left || bottom <= top) { left = 0; top = 0; right = bitmap.width; bottom = bitmap.height }
        val cropped = Bitmap.createBitmap(bitmap, left, top, right - left, bottom - top)
        val tImage = rvmTensorImage ?: run { cropped.recycle(); fallbackToEmptyMask(bitmap); return }
        tImage.load(cropped)
        val inputBuffer = rvmImageProcessor?.process(tImage)?.buffer ?: run { cropped.recycle(); fallbackToEmptyMask(bitmap); return }

        val totalPixels = rvmW * rvmH; val inputs = rvmInputs!!; val outputs = rvmOutputs!!
        if (isRvmNCHW) {
            val nchw = rvmNchwBuffer!!; nchw.clear()
            val fbIn = inputBuffer.asFloatBuffer(); fbIn.rewind()
            val inArr = rvmInputArr ?: FloatArray(totalPixels * 3).also { rvmInputArr = it }
            if (fbIn.remaining() >= inArr.size) {
                fbIn.get(inArr)
                val nchwArr = rvmNchwFloatArray ?: FloatArray(totalPixels * 3).also { rvmNchwFloatArray = it }
                for (i in 0 until totalPixels) {
                    val p = i * 3; nchwArr[i] = inArr[p]; nchwArr[totalPixels + i] = inArr[p + 1]; nchwArr[totalPixels * 2 + i] = inArr[p + 2]
                }
                nchw.asFloatBuffer().put(nchwArr); nchw.rewind(); inputs[rvmIdxSrc] = nchw
            } else inputs[rvmIdxSrc] = inputBuffer
        } else inputs[rvmIdxSrc] = inputBuffer

        val nextToggle = 1 - rvmStateToggle
        val statesIn = intArrayOf(rvmIdxR1i, rvmIdxR2i, rvmIdxR3i, rvmIdxR4i)
        for (i in 0..3) if (statesIn[i] != -1) { inputs[statesIn[i]] = rvmStateBuffers[i][rvmStateToggle]!!; rvmStateBuffers[i][rvmStateToggle]?.rewind() }
        if (rvmIdxRatio != -1) rvmRatioBuffer?.rewind()
        outputs.clear()
        val statesOut = intArrayOf(rvmIdxR1o, rvmIdxR2o, rvmIdxR3o, rvmIdxR4o)
        for (i in 0..3) if (statesOut[i] != -1) { outputs[statesOut[i]] = rvmStateBuffers[i][nextToggle]!!; rvmStateBuffers[i][nextToggle]?.clear() }
        if (rvmIdxFgr != -1) { outputs[rvmIdxFgr] = rvmOutputFgr!!; rvmOutputFgr?.clear() }
        if (rvmIdxPha != -1) { outputs[rvmIdxPha] = rvmOutputPha!!; rvmOutputPha?.clear() }

        try { rvm.runForMultipleInputsOutputs(inputs, outputs); rvmStateToggle = nextToggle } catch (e: Exception) { cropped.recycle(); fallbackToEmptyMask(bitmap); return }

        rvmOutputPha?.rewind(); val fbOut = rvmOutputPha?.asFloatBuffer() ?: return
        val outArr = rvmOutputFloatArray ?: FloatArray(totalPixels).also { rvmOutputFloatArray = it }
        if (fbOut.remaining() >= outArr.size) {
            fbOut.get(outArr); val byteArr = rvmByteArray!!
            for (i in 0 until totalPixels) byteArr[i] = (if (outArr[i] > 0.5f) 255 else 0).toByte()
            val pixels = reusableMaskPixels!!; pixels.clear(); pixels.put(byteArr); pixels.rewind()
            rvmMaskBitmap!!.copyPixelsFromBuffer(pixels)
            val fullMask = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ALPHA_8)
            Canvas(fullMask).drawBitmap(rvmMaskBitmap!!, null, Rect(left, top, right, bottom), null)
            activity?.runOnUiThread { _binding?.bodyOverlay?.updateData(fullMask, bitmap, startColor, endColor, isMirrorMode); isProcessing.set(false) }
        } else {
            cropped.recycle()
            fallbackToEmptyMask(bitmap)
        }
        cropped.recycle()
    }

    private fun processRvm(bitmap: Bitmap) {
        val interpreter = rvmInterpreter; val inputs = rvmInputs; val outputs = rvmOutputs
        if (interpreter == null || inputs == null || outputs == null || rvmIdxSrc == -1) { fallbackToEmptyMask(bitmap); return }
        val tImage = rvmTensorImage!!; tImage.load(bitmap)
        val inputBuffer = rvmImageProcessor?.process(tImage)?.buffer ?: run { fallbackToEmptyMask(bitmap); return }
        val totalPixels = rvmW * rvmH
        if (isRvmNCHW) {
            val nchw = rvmNchwBuffer!!; nchw.clear()
            val fbIn = inputBuffer.asFloatBuffer(); fbIn.rewind()
            val inArr = rvmInputArr ?: FloatArray(totalPixels * 3).also { rvmInputArr = it }
            if (fbIn.remaining() >= inArr.size) {
                fbIn.get(inArr)
                val nchwArr = rvmNchwFloatArray ?: FloatArray(totalPixels * 3).also { rvmNchwFloatArray = it }
                for (i in 0 until totalPixels) {
                    val p = i * 3; nchwArr[i] = inArr[p]; nchwArr[totalPixels + i] = inArr[p + 1]; nchwArr[totalPixels * 2 + i] = inArr[p + 2]
                }
                nchw.asFloatBuffer().put(nchwArr); nchw.rewind(); inputs[rvmIdxSrc] = nchw
            } else inputs[rvmIdxSrc] = inputBuffer
        } else inputs[rvmIdxSrc] = inputBuffer
        val nextToggle = 1 - rvmStateToggle
        val statesIn = intArrayOf(rvmIdxR1i, rvmIdxR2i, rvmIdxR3i, rvmIdxR4i)
        for (i in 0..3) if (statesIn[i] != -1) { inputs[statesIn[i]] = rvmStateBuffers[i][rvmStateToggle]!!; rvmStateBuffers[i][rvmStateToggle]?.rewind() }
        if (rvmIdxRatio != -1) rvmRatioBuffer?.rewind()
        outputs.clear()
        val statesOut = intArrayOf(rvmIdxR1o, rvmIdxR2o, rvmIdxR3o, rvmIdxR4o)
        for (i in 0..3) if (statesOut[i] != -1) { outputs[statesOut[i]] = rvmStateBuffers[i][nextToggle]!!; rvmStateBuffers[i][nextToggle]?.clear() }
        if (rvmIdxFgr != -1) { outputs[rvmIdxFgr] = rvmOutputFgr!!; rvmOutputFgr?.clear() }
        if (rvmIdxPha != -1) { outputs[rvmIdxPha] = rvmOutputPha!!; rvmOutputPha?.clear() }
        try {
            interpreter.runForMultipleInputsOutputs(inputs, outputs); rvmStateToggle = nextToggle
        } catch (e: Exception) {
            fallbackToEmptyMask(bitmap)
            return
        }
        rvmOutputPha?.rewind(); val fbOut = rvmOutputPha?.asFloatBuffer() ?: return
        val outArr = rvmOutputFloatArray ?: FloatArray(totalPixels).also { rvmOutputFloatArray = it }
        if (fbOut.remaining() >= outArr.size) {
            fbOut.get(outArr); val byteArr = rvmByteArray!!
            for (i in 0 until totalPixels) byteArr[i] = (if (outArr[i] > 0.5f) 255 else 0).toByte()
            val pixels = reusableMaskPixels!!; pixels.clear(); pixels.put(byteArr); pixels.rewind()
            rvmMaskBitmap!!.copyPixelsFromBuffer(pixels)
            val finalMask = if (rvmMaskBitmap!!.width != bitmap.width) Bitmap.createScaledBitmap(rvmMaskBitmap!!, bitmap.width, bitmap.height, true) else rvmMaskBitmap!!
            activity?.runOnUiThread { _binding?.bodyOverlay?.updateData(finalMask, bitmap, startColor, endColor, isMirrorMode); isProcessing.set(false) }
        } else {
            fallbackToEmptyMask(bitmap)
        }
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
        frameExtractionThread?.quitSafely(); frameExtractionThread = null
        backgroundRtspHandler = null; cameraExecutor?.shutdown(); _binding = null
    }
}
