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
import java.util.*
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
import kotlinx.coroutines.launch

class BodyChangerFragment : Fragment() {

    private var _binding: FragmentBodyChangerBinding? = null
    private val binding get() = _binding!!

    private var cameraExecutor: ExecutorService? = null
    private val isProcessing = AtomicBoolean(false)

    private var mlKitSegmenter: Segmenter? = null
    @Volatile private var poseLandmarker: PoseLandmarker? = null
    @Volatile private var yolactInterpreter: Interpreter? = null
    @Volatile private var modnetInterpreter: Interpreter? = null
    @Volatile private var rvmInterpreter: Interpreter? = null
    @Volatile private var yoloxInterpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null
    private var nnApiDelegate: NnApiDelegate? = null

    private var yolactImageProcessor: ImageProcessor? = null
    private var modnetImageProcessor: ImageProcessor? = null
    private var rvmImageProcessor: ImageProcessor? = null
    private var yoloxImageProcessor: ImageProcessor? = null
    private var yolactTensorImage: TensorImage? = null
    private var modnetTensorImage: TensorImage? = null
    private var rvmTensorImage: TensorImage? = null
    private var yoloxTensorImage: TensorImage? = null

    private var yolactOutputBoxes: ByteBuffer? = null
    private var yolactOutputScores: ByteBuffer? = null
    private var yolactOutputCoeffs: ByteBuffer? = null
    private var yolactOutputProtos: ByteBuffer? = null
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
    private var rvmFloatArray: FloatArray? = null
    private var rvmByteArray: ByteArray? = null
    private var rvmMaskBitmap: Bitmap? = null

    private var rvmH: Int = 192
    private var rvmW: Int = 320
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

    private val segmenterLock = Any()

    private var startColor: Int = Color.RED
    private var endColor: Int = Color.BLUE
    private var selectedModel: String = "MediaPipe Pose"
    private var selectedDelegate: String = "CPU"
    private var actualDelegate: String = "CPU"
    private var rtspQuality: String = "High"
    private var isMirrorMode: Boolean = false

    private val sdf = SimpleDateFormat("HH:mm:ss", Locale.getDefault())
    private val tagStr = "BodyChanger"

    private val yolactModelFile = "yolact_550x550_model_float16_quant.tflite"
    private val modnetModelFile = "MODNet_256x256_model_float16_quant.tflite"
    private val rvmModelFile = "rvm_resnet50_192x320_model_float16_quant.tflite"
    private val yoloxModelFile = "yolox_n_body_head_hand_post_0461_0.4428_1x3x256x320_float16.tflite"

    private var exoPlayer: ExoPlayer? = null
    private var isRtspMode = false
    private val rtspFrameHandler = Handler(Looper.getMainLooper())
    private val rtspFrameRunnable = object : Runnable {
        override fun run() {
            if (isRtspMode && exoPlayer?.isPlaying == true) {
                if (!isProcessing.get()) {
                    extractFrameFromPlayer()
                }
            }
            rtspFrameHandler.postDelayed(this, 16)
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
    }

    private fun setupSegmenter() {
        if (isInitializing) return
        isInitializing = true
        lifecycleScope.launch(Dispatchers.Default) {
            try {
                synchronized(segmenterLock) {
                    closeCurrentSegmenter()
                    System.gc()
                    try { Thread.sleep(200) } catch (e: Exception) {}
                    if (!isAdded) return@synchronized
                    when (selectedModel) {
                        "MediaPipe Pose" -> initMediaPipePose()
                        "YOLACT" -> initYolact()
                        "YOLOX" -> initYolox()
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
            yoloxInterpreter?.close(); yoloxInterpreter = null
            gpuDelegate?.close(); gpuDelegate = null
            nnApiDelegate?.close(); nnApiDelegate = null
            rtspBitmapBuffer?.recycle(); rtspBitmapBuffer = null
            rvmMaskBitmap?.recycle(); rvmMaskBitmap = null
            rvmInputs = null
            rvmOutputs = null
            for (i in 0..3) { rvmStateBuffers[i][0] = null; rvmStateBuffers[i][1] = null }
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

                val finalMask = if (width != originalWidth || height != originalHeight) {
                    Bitmap.createScaledBitmap(maskBitmap, originalWidth, originalHeight, true).also { maskBitmap.recycle() }
                } else maskBitmap

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
        options.setNumThreads(4)
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
                Log.e(tagStr, "GPU Delegate init failed", e)
            }
        }
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

    private fun initYolox() {
        try {
            yoloxInterpreter = Interpreter(FileUtil.loadMappedFile(requireContext(), yoloxModelFile), getInterpreterOptions())
            
            // 하드코딩 대신 모델의 실제 입력 텐서 크기를 읽어와서 세팅합니다.
            val inputShape = yoloxInterpreter!!.getInputTensor(0).shape()
            
            // 모델명이 1x3x256x320 이므로 NCHW 구조일 확률이 높습니다.
            // NCHW일 경우 shape[2]가 Height, shape[3]이 Width 입니다.
            val modelHeight = if (inputShape[1] == 3) inputShape[2] else inputShape[1]
            val modelWidth = if (inputShape[1] == 3) inputShape[3] else inputShape[2]

            yoloxImageProcessor = ImageProcessor.Builder()
                .add(ResizeOp(modelHeight, modelWidth, ResizeOp.ResizeMethod.BILINEAR))
                .add(NormalizeOp(0f, 255f))
                .build()
                
            yoloxTensorImage = TensorImage(DataType.FLOAT32)
            yoloxOutputBuffer = ByteBuffer.allocateDirect(yoloxInterpreter!!.getOutputTensor(0).numBytes()).order(ByteOrder.nativeOrder())
            addLog(">> YOLOX Ready ($actualDelegate)")
        } catch (e: Exception) { 
            addLog("YOLOX Init Error: ${e.message}") 
        }
    }

    private fun initRvm() {
        try {
            val modelFile = FileUtil.loadMappedFile(requireContext(), rvmModelFile)
            val options = getInterpreterOptions()
            rvmInterpreter = Interpreter(modelFile, options)

            rvmIdxSrc = -1; rvmIdxRatio = -1
            rvmIdxFgr = -1; rvmIdxPha = -1

            // ====================================================================
            // [해결책] 형태(Shape) 기반 텐서 매핑 (이름, 하드코딩 순서 의존 탈피)
            // ====================================================================

            // 1. 입력 텐서 매핑
            val inputStates = mutableListOf<Pair<Int, Int>>() // <Index, TotalElements>
            for (i in 0 until rvmInterpreter!!.inputTensorCount) {
                val shape = rvmInterpreter!!.getInputTensor(i).shape()
                if (shape.size == 4) {
                    // RGB 이미지는 채널이 3입니다. (NCHW 또는 NHWC 대응)
                    if (shape[1] == 3 || shape[3] == 3) {
                        rvmIdxSrc = i
                    } else {
                        // 3채널이 아닌 4D 텐서는 무조건 r1i ~ r4i 상태 텐서입니다.
                        val spatialSize = shape[1] * shape[2] * shape[3]
                        inputStates.add(Pair(i, spatialSize))
                    }
                } else if (shape.size == 1) {
                    rvmIdxRatio = i // 1D 텐서는 ratio 뿐입니다.
                }
            }

            // 상태 텐서들을 공간 크기(해상도)가 큰 순서대로 내림차순 정렬하면 정확히 r1, r2, r3, r4 순서가 됩니다.
            inputStates.sortByDescending { it.second }
            rvmIdxR1i = inputStates.getOrNull(0)?.first ?: -1
            rvmIdxR2i = inputStates.getOrNull(1)?.first ?: -1
            rvmIdxR3i = inputStates.getOrNull(2)?.first ?: -1
            rvmIdxR4i = inputStates.getOrNull(3)?.first ?: -1

            // 2. 출력 텐서 매핑
            val outputStates = mutableListOf<Pair<Int, Int>>()
            for (i in 0 until rvmInterpreter!!.outputTensorCount) {
                val shape = rvmInterpreter!!.getOutputTensor(i).shape()
                if (shape.size == 4) {
                    // 채널이 1개면 pha (알파 마스크), 3개면 fgr (전경)
                    val isPha = (shape[1] == 1 || shape[3] == 1)
                    val isFgr = (shape[1] == 3 || shape[3] == 3)

                    if (isPha) rvmIdxPha = i
                    else if (isFgr) rvmIdxFgr = i
                    else outputStates.add(Pair(i, shape[1] * shape[2] * shape[3]))
                }
            }

            outputStates.sortByDescending { it.second }
            rvmIdxR1o = outputStates.getOrNull(0)?.first ?: -1
            rvmIdxR2o = outputStates.getOrNull(1)?.first ?: -1
            rvmIdxR3o = outputStates.getOrNull(2)?.first ?: -1
            rvmIdxR4o = outputStates.getOrNull(3)?.first ?: -1

            // 3. 해상도 동적 분석
            if (rvmIdxSrc == -1) {
                addLog("RVM Init Error: 이미지 입력 텐서를 찾을 수 없습니다.")
                return
            }

            val srcShape = rvmInterpreter!!.getInputTensor(rvmIdxSrc).shape()
            isRvmNCHW = srcShape[1] == 3
            if (isRvmNCHW) { rvmH = srcShape[2]; rvmW = srcShape[3] } else { rvmH = srcShape[1]; rvmW = srcShape[2] }

            rvmImageProcessor = ImageProcessor.Builder()
                .add(ResizeOp(rvmH, rvmW, ResizeOp.ResizeMethod.BILINEAR))
                .add(NormalizeOp(0f, 255f))
                .build()
            rvmTensorImage = TensorImage(DataType.FLOAT32)

            // 4. 버퍼 할당 및 바인딩 (이제 각 인덱스가 정확하므로 numBytes()를 100% 신뢰할 수 있습니다)
            rvmStateToggle = 0
            val statesIn = intArrayOf(rvmIdxR1i, rvmIdxR2i, rvmIdxR3i, rvmIdxR4i)

            for (i in 0..3) {
                if (statesIn[i] != -1) {
                    val size = rvmInterpreter!!.getInputTensor(statesIn[i]).numBytes()
                    rvmStateBuffers[i][0] = ByteBuffer.allocateDirect(size).order(ByteOrder.nativeOrder())
                    rvmStateBuffers[i][1] = ByteBuffer.allocateDirect(size).order(ByteOrder.nativeOrder())
                }
            }

            rvmInputs = arrayOfNulls<Any>(rvmInterpreter!!.inputTensorCount)
            if (rvmIdxR1i != -1) rvmInputs!![rvmIdxR1i] = rvmStateBuffers[0][0]!!
            if (rvmIdxR2i != -1) rvmInputs!![rvmIdxR2i] = rvmStateBuffers[1][0]!!
            if (rvmIdxR3i != -1) rvmInputs!![rvmIdxR3i] = rvmStateBuffers[2][0]!!
            if (rvmIdxR4i != -1) rvmInputs!![rvmIdxR4i] = rvmStateBuffers[3][0]!!

            if (rvmIdxRatio != -1) {
                val ratioSize = rvmInterpreter!!.getInputTensor(rvmIdxRatio).numBytes()
                rvmRatioBuffer = ByteBuffer.allocateDirect(ratioSize).order(ByteOrder.nativeOrder())
                rvmRatioBuffer!!.asFloatBuffer().put(1.0f)
                rvmInputs!![rvmIdxRatio] = rvmRatioBuffer!!
            }

            // 출력 버퍼
            if (rvmIdxFgr != -1) rvmOutputFgr = ByteBuffer.allocateDirect(rvmInterpreter!!.getOutputTensor(rvmIdxFgr).numBytes()).order(ByteOrder.nativeOrder())
            if (rvmIdxPha != -1) rvmOutputPha = ByteBuffer.allocateDirect(rvmInterpreter!!.getOutputTensor(rvmIdxPha).numBytes()).order(ByteOrder.nativeOrder())

            val count = rvmH * rvmW
            rvmFloatArray = FloatArray(count)
            rvmByteArray = ByteArray(count)
            reusableMaskPixels = ByteBuffer.allocateDirect(count).order(ByteOrder.nativeOrder())
            rvmMaskBitmap = Bitmap.createBitmap(rvmW, rvmH, Bitmap.Config.ALPHA_8)
            rvmOutputs = mutableMapOf<Int, Any>()

            addLog(">> RVM Ready ($actualDelegate) ${rvmW}x${rvmH}")
        } catch (e: Exception) { addLog("RVM Init Error: ${e.message}") }
    }

    private fun startStream() {
        if (isRtspMode) startRtsp() else startCamera()
    }

    private fun startCamera() {
        stopRtsp()
        _binding?.viewFinder?.visibility = View.INVISIBLE
        _binding?.playerView?.visibility = View.GONE

        val cameraProviderFuture = ProcessCameraProvider.getInstance(requireContext())
        cameraProviderFuture.addListener({
            val cameraProvider = try { cameraProviderFuture.get() } catch (e: Exception) { return@addListener }

            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .setTargetResolution(Size(320, 192))
                .build().also { it.setAnalyzer(cameraExecutor!!) { proxy -> processFrame(proxy) } }

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(viewLifecycleOwner, CameraSelector.DEFAULT_FRONT_CAMERA, imageAnalyzer)
                addLog("Camera started")
            } catch (e: Exception) {
                Log.e(tagStr, "Camera binding failed", e)
            }
        }, ContextCompat.getMainExecutor(requireContext()))
    }

    @OptIn(UnstableApi::class)
    private fun startRtsp() {
        stopRtsp(); stopCamera()
        _binding?.viewFinder?.visibility = View.GONE
        _binding?.playerView?.visibility = View.VISIBLE
        val sharedPref = activity?.getSharedPreferences("AIfredoPrefs", Context.MODE_PRIVATE) ?: return
        val ip = (sharedPref.getString("rtsp_ip", "") ?: "").trim().removePrefix("rtsp://")
        val id = sharedPref.getString("rtsp_id", "") ?: ""
        val pw = sharedPref.getString("rtsp_pw", "") ?: ""
        if (ip.isEmpty()) return

        val streamPath = if (rtspQuality == "Low") "stream2" else "stream1"
        val rtspUrl = "rtsp://${if (id.isNotEmpty()) "$id:$pw@" else ""}$ip${if (!ip.contains("/")) ":554/$streamPath" else ""}"

        addLog("Connecting RTSP: $rtspUrl")
        val loadControl = DefaultLoadControl.Builder().setBufferDurationsMs(500, 1000, 250, 500).build()
        exoPlayer = ExoPlayer.Builder(requireContext()).setLoadControl(loadControl).build().apply {
            trackSelectionParameters = trackSelectionParameters.buildUpon().setTrackTypeDisabled(androidx.media3.common.C.TRACK_TYPE_AUDIO, true).build()
            addListener(object : Player.Listener {
                override fun onPlaybackStateChanged(playbackState: Int) {
                    if (playbackState == Player.STATE_READY && isRtspMode) addLog("RTSP Connected")
                }
            })
            setMediaSource(RtspMediaSource.Factory().setForceUseRtpTcp(true).setTimeoutMs(4000).createMediaSource(MediaItem.fromUri(rtspUrl)))
            prepare(); playWhenReady = true
        }
        binding.playerView.player = exoPlayer
        rtspFrameHandler.post(rtspFrameRunnable)
    }

    @OptIn(UnstableApi::class)
    private fun extractFrameFromPlayer() {
        if (!isRtspMode) return
        val b = _binding ?: return
        val textureView = b.playerView.videoSurfaceView as? TextureView ?: return

        if (rtspBitmapBuffer == null) {
            rtspBitmapBuffer = Bitmap.createBitmap(320, 192, Bitmap.Config.ARGB_8888)
        }

        try {
            textureView.getBitmap(rtspBitmapBuffer!!)
            val frame = rtspBitmapBuffer!!.copy(Bitmap.Config.ARGB_8888, false)
            cameraExecutor?.execute { processFrameBitmap(frame) }
        } catch (e: Exception) {
            Log.e(tagStr, "getBitmap failed", e)
        }
    }

    private fun stopCamera() { try { ProcessCameraProvider.getInstance(requireContext()).get().unbindAll() } catch (e: Exception) {} }

    private fun stopRtsp() {
        rtspFrameHandler.removeCallbacks(rtspFrameRunnable)
        exoPlayer?.release(); exoPlayer = null
        _binding?.playerView?.player = null
        rtspBitmapBuffer?.recycle(); rtspBitmapBuffer = null
    }

    private fun processFrame(imageProxy: ImageProxy) {
        if (isRtspMode || isProcessing.get()) {
            imageProxy.close()
            return
        }

        val width = imageProxy.width
        val height = imageProxy.height
        val rotation = imageProxy.imageInfo.rotationDegrees

        val bitmap = try {
            val buffer = imageProxy.planes[0].buffer
            val b = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
            buffer.rewind()
            b.copyPixelsFromBuffer(buffer)
            b
        } catch (e: Exception) {
            null
        }

        imageProxy.close()

        if (bitmap != null) {
            cameraExecutor?.execute {
                val matrix = Matrix().apply {
                    postRotate(rotation.toFloat())
                    postScale(-1f, 1f, width / 2f, height / 2f)
                }
                val rotatedBitmap = try {
                    Bitmap.createBitmap(bitmap, 0, 0, width, height, matrix, true)
                } catch (e: Exception) {
                    bitmap
                }
                if (rotatedBitmap != bitmap) bitmap.recycle()
                processFrameBitmap(rotatedBitmap)
            }
        }
    }

    private fun processFrameBitmap(bitmap: Bitmap) {
        if (isProcessing.getAndSet(true)) {
            bitmap.recycle()
            return
        }

        synchronized(segmenterLock) {
            when (selectedModel) {
                "MediaPipe Pose" -> {
                    poseLandmarker?.detectAsync(BitmapImageBuilder(bitmap).build(), System.currentTimeMillis())
                    bitmap.recycle()
                    isProcessing.set(false)
                }
                "YOLACT" -> processYolact(bitmap)
                "YOLOX" -> processYolox(bitmap)
                "MODNet" -> processModNet(bitmap)
                "RVM" -> processRvm(bitmap)
                "ML Kit" -> processMlKit(bitmap)
                else -> {
                    bitmap.recycle()
                    isProcessing.set(false)
                }
            }
        }
    }

    private fun processMlKit(bitmap: Bitmap) {
        val segmenter = mlKitSegmenter
        if (segmenter == null) {
            bitmap.recycle()
            isProcessing.set(false)
            return
        }
        segmenter.process(InputImage.fromBitmap(bitmap, 0))
            .addOnSuccessListener { result ->
                val maskBuffer = result.buffer; val w = result.width; val h = result.height
                val pixelsArr = ByteArray(w * h); maskBuffer.rewind()
                for (i in 0 until w * h) if (maskBuffer.hasRemaining()) pixelsArr[i] = (if (maskBuffer.float > 0.45f) 255 else 0).toByte()
                val maskBitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ALPHA_8)
                maskBitmap.copyPixelsFromBuffer(ByteBuffer.wrap(pixelsArr))
                activity?.runOnUiThread {
                    _binding?.bodyOverlay?.updateData(maskBitmap, bitmap, startColor, endColor, isMirrorMode)
                    isProcessing.set(false)
                }
            }
            .addOnFailureListener {
                bitmap.recycle()
                isProcessing.set(false)
            }
    }

    private fun processYolact(bitmap: Bitmap) {
        val interpreter = yolactInterpreter
        val tImage = yolactTensorImage
        if (interpreter == null || tImage == null) {
            bitmap.recycle()
            isProcessing.set(false)
            return
        }
        tImage.load(bitmap)
        val inputBuffer = yolactImageProcessor?.process(tImage)?.buffer
        if (inputBuffer == null) {
            bitmap.recycle()
            isProcessing.set(false)
            return
        }
        yolactOutputBoxes?.clear(); yolactOutputScores?.clear(); yolactOutputCoeffs?.clear(); yolactOutputProtos?.clear()
        val outputs = mapOf(yolactIdxBoxes to yolactOutputBoxes!!, yolactIdxScores to yolactOutputScores!!, yolactIdxCoeffs to yolactOutputCoeffs!!, yolactIdxProtos to yolactOutputProtos!!)
        try { interpreter.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputs) } catch (e: Exception) { bitmap.recycle(); isProcessing.set(false); return }
        yolactOutputScores?.rewind(); val scoresArray = reusableScoresArray ?: return.also { bitmap.recycle(); isProcessing.set(false) }
        yolactOutputScores?.asFloatBuffer()?.get(scoresArray)
        var bestIdx = -1; var maxScore = 0f
        for (i in 0 until 19248) { val score = scoresArray[i * 81 + 1]; if (score > maxScore) { maxScore = score; bestIdx = i } }
        if (bestIdx != -1 && maxScore > 0.15f) {
            val coeffs = FloatArray(32); yolactOutputCoeffs?.rewind(); val fb = yolactOutputCoeffs?.asFloatBuffer()
            fb?.position(bestIdx * 32); fb?.get(coeffs)
            val mask = generateYolactMask(coeffs, yolactOutputProtos!!)
            val finalMask = if (mask.width != bitmap.width) Bitmap.createScaledBitmap(mask, bitmap.width, bitmap.height, true).also { mask.recycle() } else mask
            activity?.runOnUiThread {
                _binding?.bodyOverlay?.updateData(finalMask, bitmap, startColor, endColor, isMirrorMode)
                isProcessing.set(false)
            }
        } else {
            bitmap.recycle()
            isProcessing.set(false)
        }
    }

    private fun generateYolactMask(coeffs: FloatArray, protos: ByteBuffer): Bitmap {
        val w = 138; val h = 138; val pixels = reusableMaskPixels!!; pixels.clear()
        val protosArray = reusableProtosArray ?: return Bitmap.createBitmap(1, 1, Bitmap.Config.ALPHA_8)
        protos.rewind(); protos.asFloatBuffer().get(protosArray)
        for (y in 0 until h) for (x in 0 until w) {
            var sum = 0f; val off = (y * w + x) * 32
            for (k in 0 until 32) sum += coeffs[k] * protosArray[off + k]
            pixels.put((if (sum > 0.5f) 255 else 0).toByte())
        }
        val mask = Bitmap.createBitmap(w, h, Bitmap.Config.ALPHA_8); pixels.rewind(); mask.copyPixelsFromBuffer(pixels); return mask
    }

    private fun processModNet(bitmap: Bitmap) {
        val interpreter = modnetInterpreter
        val tImage = modnetTensorImage
        if (interpreter == null || tImage == null) {
            bitmap.recycle()
            isProcessing.set(false)
            return
        }
        tImage.load(bitmap)
        val inputBuffer = modnetImageProcessor?.process(tImage)?.buffer
        if (inputBuffer == null) {
            bitmap.recycle()
            isProcessing.set(false)
            return
        }
        modnetOutputBuffer?.clear()
        try { interpreter.run(inputBuffer, modnetOutputBuffer) } catch (e: Exception) { bitmap.recycle(); isProcessing.set(false); return }
        modnetOutputBuffer?.rewind(); val fb = modnetOutputBuffer?.asFloatBuffer()
        if (fb == null) {
            bitmap.recycle()
            isProcessing.set(false)
            return
        }
        val pixels = reusableMaskPixels!!; pixels.clear()
        for (i in 0 until (256 * 256)) if (fb.hasRemaining()) pixels.put((if (fb.get() > 0.4f) 255 else 0).toByte())
        val mask = Bitmap.createBitmap(256, 256, Bitmap.Config.ALPHA_8); pixels.rewind(); mask.copyPixelsFromBuffer(pixels)
        val finalMask = if (mask.width != bitmap.width) Bitmap.createScaledBitmap(mask, bitmap.width, bitmap.height, true).also { mask.recycle() } else mask
        activity?.runOnUiThread {
            _binding?.bodyOverlay?.updateData(finalMask, bitmap, startColor, endColor, isMirrorMode)
            isProcessing.set(false)
        }
    }

    private fun processYolox(bitmap: Bitmap) {
        val interpreter = yoloxInterpreter
        val tImage = yoloxTensorImage
        if (interpreter == null || tImage == null) {
            bitmap.recycle()
            isProcessing.set(false)
            return
        }
        tImage.load(bitmap)
        val inputBuffer = yoloxImageProcessor?.process(tImage)?.buffer
        if (inputBuffer == null) {
            bitmap.recycle()
            isProcessing.set(false)
            return
        }
        yoloxOutputBuffer?.clear()
        try {
            interpreter.run(inputBuffer, yoloxOutputBuffer)
        } catch (e: Exception) {
            Log.e(tagStr, "YOLOX Inference Error", e)
            bitmap.recycle()
            isProcessing.set(false)
            return
        }
        yoloxOutputBuffer?.rewind()

        // NOTE: YOLOX는 객체 탐지 모델이므로 픽셀 단위 마스크를 출력하지 않습니다.
        // 바운딩 박스 파싱 로직이 필요하며, 기존의 마스크 처리 로직은 논리적으로 부적절하여 제거되었습니다.
        
        bitmap.recycle()
        isProcessing.set(false)
    }

    private fun processRvm(bitmap: Bitmap) {
        val interpreter = rvmInterpreter
        val tImage = rvmTensorImage
        val inputs = rvmInputs
        val outputs = rvmOutputs

        if (interpreter == null || tImage == null || inputs == null || outputs == null || rvmIdxSrc == -1) {
            bitmap.recycle()
            isProcessing.set(false)
            return
        }

        tImage.load(bitmap)
        val inputBuffer = rvmImageProcessor?.process(tImage)?.buffer
        if (inputBuffer == null) {
            bitmap.recycle()
            isProcessing.set(false)
            return
        }

        inputs[rvmIdxSrc] = inputBuffer
        (inputs[rvmIdxSrc] as ByteBuffer).rewind()

        val nextToggle = 1 - rvmStateToggle

        if (rvmIdxR1i != -1) { inputs[rvmIdxR1i] = rvmStateBuffers[0][rvmStateToggle]!!; rvmStateBuffers[0][rvmStateToggle]?.rewind() }
        if (rvmIdxR2i != -1) { inputs[rvmIdxR2i] = rvmStateBuffers[1][rvmStateToggle]!!; rvmStateBuffers[1][rvmStateToggle]?.rewind() }
        if (rvmIdxR3i != -1) { inputs[rvmIdxR3i] = rvmStateBuffers[2][rvmStateToggle]!!; rvmStateBuffers[2][rvmStateToggle]?.rewind() }
        if (rvmIdxR4i != -1) { inputs[rvmIdxR4i] = rvmStateBuffers[3][rvmStateToggle]!!; rvmStateBuffers[3][rvmStateToggle]?.rewind() }

        if (rvmIdxRatio != -1) rvmRatioBuffer?.rewind()

        for (i in inputs.indices) {
            if (inputs[i] == null) {
                Log.e(tagStr, "RVM 입력 텐서 ${i}번이 null입니다.")
                bitmap.recycle()
                isProcessing.set(false)
                return
            }
        }

        outputs.clear()

        if (rvmIdxR1o != -1) { outputs[rvmIdxR1o] = rvmStateBuffers[0][nextToggle]!!; rvmStateBuffers[0][nextToggle]?.clear() }
        if (rvmIdxR2o != -1) { outputs[rvmIdxR2o] = rvmStateBuffers[1][nextToggle]!!; rvmStateBuffers[1][nextToggle]?.clear() }
        if (rvmIdxR3o != -1) { outputs[rvmIdxR3o] = rvmStateBuffers[2][nextToggle]!!; rvmStateBuffers[2][nextToggle]?.clear() }
        if (rvmIdxR4o != -1) { outputs[rvmIdxR4o] = rvmStateBuffers[3][nextToggle]!!; rvmStateBuffers[3][nextToggle]?.clear() }

        if (rvmIdxFgr != -1) { outputs[rvmIdxFgr] = rvmOutputFgr!!; rvmOutputFgr?.clear() }
        if (rvmIdxPha != -1) { outputs[rvmIdxPha] = rvmOutputPha!!; rvmOutputPha?.clear() }

        try {
            interpreter.runForMultipleInputsOutputs(inputs, outputs)
            rvmStateToggle = nextToggle
        } catch (e: Exception) {
            Log.e(tagStr, "RVM 추론 중 예외 발생: ${e.message}", e)
            bitmap.recycle()
            isProcessing.set(false)
            return
        }

        if (rvmOutputPha == null) {
            bitmap.recycle()
            isProcessing.set(false)
            return
        }

        rvmOutputPha?.rewind()
        val fb = rvmOutputPha?.asFloatBuffer()
        if (fb == null) {
            bitmap.recycle()
            isProcessing.set(false)
            return
        }

        val count = rvmH * rvmW
        val floatArr = rvmFloatArray!!
        val byteArr = rvmByteArray!!

        fb.get(floatArr)
        for (i in 0 until count) {
            byteArr[i] = (if (floatArr[i] > 0.5f) 255 else 0).toByte()
        }

        val pixels = reusableMaskPixels!!
        pixels.clear()
        pixels.put(byteArr)
        pixels.rewind()

        val mask = rvmMaskBitmap!!
        mask.copyPixelsFromBuffer(pixels)

        val finalMask = if (mask.width != bitmap.width || mask.height != bitmap.height) {
            Bitmap.createScaledBitmap(mask, bitmap.width, bitmap.height, true)
        } else mask

        activity?.runOnUiThread {
            _binding?.bodyOverlay?.updateData(finalMask, bitmap, startColor, endColor, isMirrorMode)
            if (finalMask != mask) finalMask.recycle()
            isProcessing.set(false)
        }
    }

    private fun addLog(message: String) {
        activity?.runOnUiThread { _binding?.let { b ->
            val timestamp = sdf.format(Date())
            b.eventLog.text = "[$timestamp] $message\n${b.eventLog.text.toString().take(1000)}"
        } }
    }

    private fun allPermissionsGranted() = ContextCompat.checkSelfPermission(requireContext(), Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED

    override fun onDestroyView() { super.onDestroyView() ; cameraExecutor?.shutdown(); _binding = null }
}
