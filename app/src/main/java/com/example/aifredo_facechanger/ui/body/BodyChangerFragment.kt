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
import org.tensorflow.lite.nnapi.NnApiDelegate
import org.tensorflow.lite.support.common.FileUtil
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.exp

class BodyChangerFragment : Fragment() {

    private var _binding: FragmentBodyChangerBinding? = null
    private val binding get() = _binding!!

    private var cameraExecutor: ExecutorService? = null

    private var mlKitSegmenter: Segmenter? = null
    @Volatile private var yolactInterpreter: Interpreter? = null
    private var nnApiDelegate: NnApiDelegate? = null

    // Pre-allocated buffers for YOLACT
    private var yolactInputBuffer: ByteBuffer? = null
    private var yolactOutputBoxes: ByteBuffer? = null
    private var yolactOutputScores: ByteBuffer? = null
    private var yolactOutputCoeffs: ByteBuffer? = null
    private var yolactOutputProtos: ByteBuffer? = null

    private val segmenterLock = Any()

    private var startColor: Int = Color.RED
    private var endColor: Int = Color.BLUE
    private var selectedModel: String = "MediaPipe"
    private var selectedDelegate: String = "CPU"

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

                    // 네이티브 메모리 안정화 대기
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
            nnApiDelegate?.close()
            nnApiDelegate = null
        } catch (e: Exception) {
            Log.e(TAG, "Error closing existing segmenter", e)
        }
    }

    private fun initMlKit() {
        addLog("Initializing ML Kit")
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
        addLog("Initializing YOLACT ($selectedDelegate)")
        try {
            val options = Interpreter.Options()
            if (selectedDelegate == "NNAPI") {
                try {
                    nnApiDelegate = NnApiDelegate()
                    options.addDelegate(nnApiDelegate)
                    addLog("NNAPI Delegate added successfully")
                } catch (e: Exception) {
                    addLog("NNAPI Error: ${e.message}. Using CPU.")
                    Log.e(TAG, "NNAPI init failed", e)
                }
            }
            
            val modelBuffer = FileUtil.loadMappedFile(requireContext(), YOLACT_MODEL_FILE)
            val interpreter = Interpreter(modelBuffer, options)
            yolactInterpreter = interpreter
            
            // Log output tensor shapes for debugging
            for (i in 0 until interpreter.outputTensorCount) {
                val tensor = interpreter.getOutputTensor(i)
                Log.d(TAG, "Output Tensor $i: name=${tensor.name()}, shape=${tensor.shape().contentToString()}")
            }

            // Pre-allocate buffers for YOLACT 550x550
            // Assuming default YOLACT ResNet-50 shapes:
            // Boxes: [1, 19248, 4]
            // Scores: [1, 19248, 81]
            // Coeffs: [1, 19248, 32]
            // Protos: [1, 138, 138, 32]
            
            yolactInputBuffer = ByteBuffer.allocateDirect(1 * 550 * 550 * 3 * 4).apply {
                order(ByteOrder.nativeOrder())
            }
            yolactOutputBoxes = ByteBuffer.allocateDirect(19248 * 4 * 4).apply { order(ByteOrder.nativeOrder()) }
            yolactOutputScores = ByteBuffer.allocateDirect(19248 * 81 * 4).apply { order(ByteOrder.nativeOrder()) }
            yolactOutputCoeffs = ByteBuffer.allocateDirect(19248 * 32 * 4).apply { order(ByteOrder.nativeOrder()) }
            yolactOutputProtos = ByteBuffer.allocateDirect(138 * 138 * 32 * 4).apply { order(ByteOrder.nativeOrder()) }
            
            addLog(">> YOLACT Ready (Inputs/Outputs allocated)")
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
        val inputImage = InputImage.fromBitmap(bitmap, 0)
        mlKitSegmenter?.process(inputImage)
            ?.addOnSuccessListener { result ->
                val maskBuffer = result.buffer
                maskBuffer.rewind()
                val maskBitmap = Bitmap.createBitmap(result.width, result.height, Bitmap.Config.ALPHA_8)
                val byteBuffer = ByteBuffer.allocateDirect(result.width * result.height)
                while (maskBuffer.hasRemaining()) {
                    byteBuffer.put((maskBuffer.float * 255).toInt().toByte())
                }
                byteBuffer.rewind()
                maskBitmap.copyPixelsFromBuffer(byteBuffer)
                _binding?.bodyOverlay?.updateData(maskBitmap, bitmap, startColor, endColor)
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

        // 1. Preprocessing: 550x550 Resize & Normalization
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

        // 2. Prepare Outputs
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

        // 3. Inference
        try {
            interpreter.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputs)
        } catch (e: Exception) {
            Log.e(TAG, "YOLACT Inference error", e)
            addLog("Inference error: ${e.message}")
            return
        }

        val inferenceTime = System.currentTimeMillis() - startTime

        // 4. Post-processing
        // Rewind ByteBuffers to read from the start via FloatBuffers
        outScores.rewind()
        outCoeffs.rewind()
        outProtos.rewind()

        val scoresFloatBuffer = outScores.asFloatBuffer()
        val coeffsFloatBuffer = outCoeffs.asFloatBuffer()
        
        var bestIdx = -1
        var maxScore = 0f
        
        // Find best 'person' (class 1)
        for (i in 0 until 19248) {
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
                addLog("YOLACT: MaxScore=${String.format("%.2f", maxScore)}, Time=${inferenceTime}ms")
            }
        } else {
            if (Random().nextInt(100) < 5) {
                addLog("YOLACT: No person (MaxScore=${String.format("%.2f", maxScore)})")
            }
        }
    }

    private fun generateMask(coeffs: FloatArray, protos: ByteBuffer): Bitmap {
        val width = 138
        val height = 138
        val maskBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ALPHA_8)
        val pixels = ByteBuffer.allocateDirect(width * height)
        
        protos.rewind()
        val protosFloatBuffer = protos.asFloatBuffer()
        
        for (y in 0 until height) {
            for (x in 0 until width) {
                var sum = 0f
                val offset = (y * width + x) * 32
                for (k in 0 until 32) {
                    sum += coeffs[k] * protosFloatBuffer.get(offset + k)
                }
                val prob = 1.0f / (1.0f + exp(-sum).toFloat())
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
