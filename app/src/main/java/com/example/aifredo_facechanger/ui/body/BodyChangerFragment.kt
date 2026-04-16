package com.example.aifredo_facechanger.ui.body

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.*
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import com.example.aifredo_facechanger.databinding.FragmentBodyChangerBinding
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.segmentation.Segmentation
import com.google.mlkit.vision.segmentation.Segmenter
import com.google.mlkit.vision.segmentation.selfie.SelfieSegmenterOptions
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

class BodyChangerFragment : Fragment() {

    private var _binding: FragmentBodyChangerBinding? = null
    private val binding get() = _binding!!

    private var cameraExecutor: ExecutorService? = null

    private var mlKitSegmenter: Segmenter? = null
    @Volatile private var yolactInterpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null
    private var nnApiDelegate: NnApiDelegate? = null

    // YOLACT 사전 할당 버퍼
    private var yolactInputBuffer: ByteBuffer? = null
    private var yolactOutputBoxes: ByteBuffer? = null
    private var yolactOutputScores: ByteBuffer? = null
    private var yolactOutputCoeffs: ByteBuffer? = null
    private var yolactOutputProtos: ByteBuffer? = null

    // 성능 최적화를 위한 전역 재사용 버퍼 (generateMask 용)
    private var reusableMaskPixels: ByteBuffer? = null
    private var reusableProtosArray: FloatArray? = null

    private val segmenterLock = Any()

    private var startColor: Int = Color.RED
    private var endColor: Int = Color.BLUE
    private var selectedModel: String = "MediaPipe"
    private var selectedDelegate: String = "CPU"
    private var actualDelegate: String = "CPU"

    private val sdf = SimpleDateFormat("HH:mm:ss", Locale.getDefault())
    private val TAG = "BodyChanger"

    private val YOLACT_MODEL_FILE = "yolact_550x550_model_float16_quant.tflite"

    companion object {
        private val sharedSegmenterExecutor: ExecutorService = Executors.newSingleThreadExecutor()
        @Volatile private var isInitializing = false
    }

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { startCamera() }

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View {
        _binding = FragmentBodyChangerBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        cameraExecutor = Executors.newSingleThreadExecutor()

        if (allPermissionsGranted()) startCamera()
        else requestPermissionLauncher.launch(arrayOf(Manifest.permission.CAMERA))
    }

    override fun onResume() {
        super.onResume()
        loadSettings()
        setupSegmenter()
    }

    private fun loadSettings() {
        val sharedPref = activity?.getSharedPreferences("AIfredoPrefs", Context.MODE_PRIVATE) ?: return

        selectedModel = sharedPref.getString("body_model", "MediaPipe") ?: "MediaPipe"
        selectedDelegate = sharedPref.getString("body_delegate", "CPU") ?: "CPU"
        val startColorStr = sharedPref.getString("body_start_color", "#FF0000") ?: "#FF0000"
        val endColorStr = sharedPref.getString("body_end_color", "#0000FF") ?: "#0000FF"

        try {
            startColor = Color.parseColor(startColorStr)
            endColor = Color.parseColor(endColorStr)
        } catch (e: Exception) {
            startColor = Color.RED
            endColor = Color.BLUE
        }
        addLog("Settings: Model=$selectedModel, Delegate=$selectedDelegate")
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
                        "YOLACT" -> initYolact()
                        "ML Kit" -> initMlKit()
                        else -> initMlKit()
                    }
                }
            } finally {
                isInitializing = false
            }
        }
    }

    private fun closeCurrentSegmenter() {
        try {
            mlKitSegmenter?.close()
            mlKitSegmenter = null
            yolactInterpreter?.close()
            yolactInterpreter = null
            gpuDelegate?.close()
            gpuDelegate = null
            nnApiDelegate?.close()
            nnApiDelegate = null
        } catch (e: Exception) {
            Log.e(TAG, "Error closing existing segmenter", e)
        }
    }

    private fun initMlKit() {
        addLog("Initializing ML Kit")
        actualDelegate = "N/A (ML Kit)"
        try {
            val options = SelfieSegmenterOptions.Builder()
                .setDetectorMode(SelfieSegmenterOptions.STREAM_MODE)
                .build()
            mlKitSegmenter = Segmentation.getClient(options)
            addLog(">> ML Kit Ready")
        } catch (e: Exception) {
            addLog("ML Kit Error: ${e.message}")
        }
    }

    private fun initYolact() {
        addLog("Initializing YOLACT with $selectedDelegate")
        actualDelegate = "CPU"
        try {
            val options = Interpreter.Options()

            when (selectedDelegate.uppercase()) {
                "GPU" -> {
                    try {
                        val compatList = CompatibilityList()
                        if (compatList.isDelegateSupportedOnThisDevice) {
                            val gpuOptions = GpuDelegate.Options().apply {
                                setInferencePreference(GpuDelegate.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED)
                            }
                            gpuDelegate = GpuDelegate(gpuOptions)
                            options.addDelegate(gpuDelegate)
                            actualDelegate = "GPU"
                            addLog("GPU Delegate applied")
                        } else {
                            addLog("GPU Delegate not supported. Using CPU.")
                        }
                    } catch (e: Exception) {
                        addLog("GPU Error: ${e.message}")
                    }
                }
                "NNAPI" -> {
                    try {
                        nnApiDelegate = NnApiDelegate()
                        options.addDelegate(nnApiDelegate)
                        actualDelegate = "NNAPI"
                        addLog("NNAPI Delegate applied")
                    } catch (e: Exception) {
                        addLog("NNAPI Error: ${e.message}")
                    }
                }
                else -> {
                    addLog("Using CPU")
                }
            }

            val modelBuffer = FileUtil.loadMappedFile(requireContext(), YOLACT_MODEL_FILE)
            val interpreter = Interpreter(modelBuffer, options)
            yolactInterpreter = interpreter

            // 입력 및 출력 버퍼 할당
            yolactInputBuffer = ByteBuffer.allocateDirect(1 * 550 * 550 * 3 * 4).apply { order(ByteOrder.nativeOrder()) }
            yolactOutputBoxes = ByteBuffer.allocateDirect(19248 * 4 * 4).apply { order(ByteOrder.nativeOrder()) }
            yolactOutputScores = ByteBuffer.allocateDirect(19248 * 81 * 4).apply { order(ByteOrder.nativeOrder()) }
            yolactOutputCoeffs = ByteBuffer.allocateDirect(19248 * 32 * 4).apply { order(ByteOrder.nativeOrder()) }
            yolactOutputProtos = ByteBuffer.allocateDirect(138 * 138 * 32 * 4).apply { order(ByteOrder.nativeOrder()) }

            // 메모리 재할당 방지를 위한 마스크 연산용 버퍼 초기화
            reusableMaskPixels = ByteBuffer.allocateDirect(138 * 138).apply { order(ByteOrder.nativeOrder()) }
            reusableProtosArray = FloatArray(138 * 138 * 32)

            addLog(">> YOLACT Ready ($actualDelegate)")
        } catch (e: Exception) {
            addLog("YOLACT Init Error: ${e.message}")
            Log.e(TAG, "YOLACT Init Error", e)
        }
    }

    private fun startCamera() {
        val context = context ?: return
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        cameraProviderFuture.addListener({
            val cameraProvider = try { cameraProviderFuture.get() } catch (e: Exception) { return@addListener }

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(binding.viewFinder.surfaceProvider)
            }

            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .setTargetResolution(Size(640, 480))
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor!!) { imageProxy ->
                        processFrame(imageProxy)
                    }
                }

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(viewLifecycleOwner, CameraSelector.DEFAULT_FRONT_CAMERA, preview, imageAnalyzer)
                addLog("Camera started")
            } catch (e: Exception) {
                Log.e(TAG, "Camera binding failed", e)
                addLog("Camera error: ${e.message}")
            }
        }, ContextCompat.getMainExecutor(context))
    }

    private fun processFrame(imageProxy: ImageProxy) {
        val bitmap = processImageProxy(imageProxy)
        if (bitmap == null) {
            imageProxy.close()
            return
        }

        synchronized(segmenterLock) {
            if (selectedModel == "YOLACT") {
                if (yolactInterpreter != null) {
                    processYolact(bitmap)
                }
            } else if (mlKitSegmenter != null) {
                processMlKit(bitmap)
            }
        }
        imageProxy.close()
    }

    private fun processMlKit(bitmap: Bitmap) {
        val startTime = System.currentTimeMillis()
        val inputImage = InputImage.fromBitmap(bitmap, 0)
        mlKitSegmenter?.process(inputImage)
            ?.addOnSuccessListener { result ->
                val inferenceTime = System.currentTimeMillis() - startTime
                val maskBuffer = result.buffer
                maskBuffer.rewind()

                val width = result.width
                val height = result.height
                val maskBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ALPHA_8)
                val pixels = ByteArray(width * height)

                // Float 형식을 정확히 Byte 로 변환 처리
                for (i in 0 until width * height) {
                    val confidence = maskBuffer.float
                    pixels[i] = (confidence * 255).toInt().toByte()
                }

                maskBitmap.copyPixelsFromBuffer(ByteBuffer.wrap(pixels))

                activity?.runOnUiThread {
                    _binding?.bodyOverlay?.updateData(maskBitmap, bitmap, startColor, endColor)
                }
                
                if (Random().nextInt(100) < 5) {
                    addLog("ML Kit: Time=${inferenceTime}ms")
                }
            }
    }

    private fun processYolact(bitmap: Bitmap) {
        val interpreter = yolactInterpreter ?: return
        val inputBuffer = yolactInputBuffer ?: return
        val outBoxes = yolactOutputBoxes ?: return
        val outScores = yolactOutputScores ?: return
        val outCoeffs = yolactOutputCoeffs ?: return
        val outProtos = yolactOutputProtos ?: return

        val startTime = System.currentTimeMillis()

        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, 550, 550, true)
        inputBuffer.rewind()

        val intValues = IntArray(550 * 550)
        scaledBitmap.getPixels(intValues, 0, 550, 0, 0, 550, 550)
        for (pixelValue in intValues) {
            inputBuffer.putFloat(((pixelValue shr 16 and 0xFF) - 123.675f) / 58.395f)
            inputBuffer.putFloat(((pixelValue shr 8 and 0xFF) - 116.28f) / 57.12f)
            inputBuffer.putFloat(((pixelValue and 0xFF) - 103.53f) / 57.375f)
        }
        inputBuffer.rewind()

        outBoxes.rewind()
        outScores.rewind()
        outCoeffs.rewind()
        outProtos.rewind()

        val outputs = mutableMapOf<Int, Any>()
        for (i in 0 until interpreter.outputTensorCount) {
            val shape = interpreter.getOutputTensor(i).shape()
            val totalSize = shape.fold(1) { acc, size -> acc * size }
            when (totalSize * 4) {
                outBoxes.capacity() -> outputs[i] = outBoxes
                outScores.capacity() -> outputs[i] = outScores
                outCoeffs.capacity() -> outputs[i] = outCoeffs
                outProtos.capacity() -> outputs[i] = outProtos
            }
        }

        try {
            interpreter.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputs)
        } catch (e: Exception) {
            Log.e(TAG, "YOLACT Inference error", e)
            addLog("Inference error: ${e.message}")
            return
        }

        val inferenceTime = System.currentTimeMillis() - startTime

        // 에러 방지용 리와인드
        outScores.rewind()
        outCoeffs.rewind()
        outProtos.rewind()

        val scoresFloatBuffer = outScores.asFloatBuffer()
        val coeffsFloatBuffer = outCoeffs.asFloatBuffer()

        // 데이터가 없으면 즉시 종료 (IndexOutOfBoundsException 방지)
        if (scoresFloatBuffer.limit() == 0) return

        var bestIdx = -1
        var maxScore = 0f

        val numDetections = minOf(19248, scoresFloatBuffer.limit() / 81)
        for (i in 0 until numDetections) {
            val score = scoresFloatBuffer.get(i * 81 + 1)
            if (score > maxScore) {
                maxScore = score
                bestIdx = i
            }
        }

        if (bestIdx != -1 && maxScore > 0.15f) {
            val coeffs = FloatArray(32)
            coeffsFloatBuffer.position(bestIdx * 32)
            coeffsFloatBuffer.get(coeffs)

            val maskBitmap = generateMask(coeffs, outProtos)
            val finalMask = Bitmap.createScaledBitmap(maskBitmap, bitmap.width, bitmap.height, true)

            activity?.runOnUiThread {
                _binding?.bodyOverlay?.updateData(finalMask, bitmap, startColor, endColor)
            }
            if (Random().nextInt(100) < 5) {
                addLog("YOLACT ($actualDelegate): Score=${String.format("%.2f", maxScore)}, Time=${inferenceTime}ms")
            }
        } else {
            if (Random().nextInt(100) < 5) {
                addLog("YOLACT ($actualDelegate): No person (${inferenceTime}ms)")
            }
        }
    }

    private fun generateMask(coeffs: FloatArray, protos: ByteBuffer): Bitmap {
        val width = 138
        val height = 138
        val maskBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ALPHA_8)

        val pixels = reusableMaskPixels ?: return maskBitmap
        val protosArray = reusableProtosArray ?: return maskBitmap

        protos.rewind()
        val protosFloatBuffer = protos.asFloatBuffer()

        // 반복문 밖에서 통째로 복사하여 속도 극대화
        protosFloatBuffer.get(protosArray)

        pixels.rewind()
        for (y in 0 until height) {
            for (x in 0 until width) {
                var sum = 0f
                val offset = (y * width + x) * 32
                for (k in 0 until 32) {
                    sum += coeffs[k] * protosArray[offset + k]
                }
                val prob = 1.0f / (1.0f + Math.exp(-sum.toDouble())).toFloat()
                val alpha = if (prob > 0.5f) 255 else 0
                pixels.put(alpha.toByte())
            }
        }
        pixels.rewind()
        maskBitmap.copyPixelsFromBuffer(pixels)
        return maskBitmap
    }

    private fun processImageProxy(imageProxy: ImageProxy): Bitmap? {
        return try {
            val bitmap = Bitmap.createBitmap(imageProxy.width, imageProxy.height, Bitmap.Config.ARGB_8888)
            bitmap.copyPixelsFromBuffer(imageProxy.planes[0].buffer)
            val matrix = Matrix().apply {
                postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())
                postScale(-1f, 1f)
            }
            Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        } catch (e: Exception) {
            null
        }
    }

    private fun addLog(message: String) {
        activity?.runOnUiThread {
            _binding?.let { b ->
                val timestamp = sdf.format(Date())
                val currentLog = b.eventLog.text.toString()
                b.eventLog.text = "[$timestamp] $message\n${currentLog.take(1000)}"
            }
        }
    }

    private fun allPermissionsGranted() = ContextCompat.checkSelfPermission(requireContext(), Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED

    override fun onPause() {
        super.onPause()
        sharedSegmenterExecutor.execute {
            synchronized(segmenterLock) {
                closeCurrentSegmenter()
            }
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        cameraExecutor?.shutdown()
        _binding = null
    }
}