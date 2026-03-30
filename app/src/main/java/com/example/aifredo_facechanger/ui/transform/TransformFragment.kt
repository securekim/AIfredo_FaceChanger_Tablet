package com.example.aifredo_facechanger.ui.transform

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.content.res.AssetManager
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
import com.example.aifredo_facechanger.databinding.FragmentTransformBinding
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.BitmapExtractor
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facestylizer.FaceStylizer
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.max

class TransformFragment : Fragment() {

    private var _binding: FragmentTransformBinding? = null
    private val binding get() = _binding!!

    private var cameraExecutor: ExecutorService? = null
    private var stylizerExecutor: ExecutorService? = null
    private var backgroundExecutor: ExecutorService? = null
    
    private var faceLandmarker: FaceLandmarker? = null
    private var poseLandmarker: PoseLandmarker? = null
    private var faceStylizer: FaceStylizer? = null
    private var tfliteInterpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null
    
    // Optimization: Reuse ImageProcessor and Buffers
    private var imageProcessor: ImageProcessor? = null
    private var modelInputWidth = 512
    private var modelInputHeight = 512
    
    @Volatile private var lastStylizedBitmap: Bitmap? = null
    @Volatile private var lastStylizedFaceResult: FaceLandmarkerResult? = null 
    private var inputBitmap: Bitmap? = null
    private var modelOutputBuffer: java.nio.ByteBuffer? = null
    private var outputPixels: IntArray? = null

    private val frameCache = ConcurrentHashMap<Long, Bitmap>()
    private var lastPoseResult: PoseLandmarkerResult? = null
    private var lastFaceResult: FaceLandmarkerResult? = null
    @Volatile private var isStylizing = false

    private val TAG = "AIfredo_Transform"
    private var currentModelPref: String = "CartoonGAN_Default"
    private var renderMode: String = "Face_Only"
    private val sdf = SimpleDateFormat("HH:mm:ss", Locale.getDefault())

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { startCamera() }

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View {
        _binding = FragmentTransformBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        cameraExecutor = Executors.newSingleThreadExecutor()
        stylizerExecutor = Executors.newSingleThreadExecutor()
        backgroundExecutor = Executors.newSingleThreadExecutor()
        loadSettings()
        backgroundExecutor?.execute { if (isAdded) setupAnalyzers() }
        if (allPermissionsGranted()) startCamera()
        else requestPermissionLauncher.launch(arrayOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO))
    }

    private fun loadSettings() {
        val sharedPref = activity?.getSharedPreferences("AIfredoPrefs", Context.MODE_PRIVATE)
        currentModelPref = sharedPref?.getString("selected_model", "CartoonGAN_Default") ?: "CartoonGAN_Default"
        renderMode = sharedPref?.getString("render_mode", "Face_Only") ?: "Face_Only"
    }

    private fun setupAnalyzers() {
        val context = context ?: return
        addLog("Initializing AI...")
        
        val faceModel = "face_landmarker.task" 
        val poseModel = "pose_landmarker_lite.task"

        try {
            faceLandmarker?.close()
            faceLandmarker = FaceLandmarker.createFromOptions(context, FaceLandmarker.FaceLandmarkerOptions.builder()
                .setBaseOptions(BaseOptions.builder().setDelegate(Delegate.GPU).setModelAssetPath(faceModel).build())
                .setRunningMode(RunningMode.LIVE_STREAM)
                .setResultListener { result, _ -> processLandmarkResult(result) }
                .build())
            
            poseLandmarker?.close()
            poseLandmarker = PoseLandmarker.createFromOptions(context, PoseLandmarker.PoseLandmarkerOptions.builder()
                .setBaseOptions(BaseOptions.builder().setDelegate(Delegate.GPU).setModelAssetPath(poseModel).build())
                .setRunningMode(RunningMode.LIVE_STREAM)
                .setResultListener { result, _ -> lastPoseResult = result }
                .build())
            addLog("Landmarkers Ready (GPU)")
        } catch (e: Exception) { 
            addLog("Landmarker Warning: Using CPU")
            try {
                faceLandmarker = FaceLandmarker.createFromOptions(context, FaceLandmarker.FaceLandmarkerOptions.builder()
                    .setBaseOptions(BaseOptions.builder().setDelegate(Delegate.CPU).setModelAssetPath(faceModel).build())
                    .setRunningMode(RunningMode.LIVE_STREAM)
                    .setResultListener { result, _ -> processLandmarkResult(result) }
                    .build())
            } catch (e2: Exception) {
                addLog("Landmarker Error: ${e2.message}")
            }
        }
        setupStylizer()
    }

    private fun setupStylizer() {
        val context = context ?: return
        faceStylizer?.close(); tfliteInterpreter?.close(); gpuDelegate?.close()
        faceStylizer = null; tfliteInterpreter = null; gpuDelegate = null

        val modelName = when (currentModelPref) {
            "MediaPipe_Default" -> "face_stylizer.task"
            "CartoonGAN_Default" -> "whitebox_cartoon_gan_fp16.tflite"
            else -> "whitebox_cartoon_gan_fp16.tflite"
        }
        
        if (modelName.endsWith(".tflite")) {
            try {
                val modelBuffer = loadModelFile(context.assets, modelName)
                val compatList = CompatibilityList()

                val options = Interpreter.Options().apply {
                    if (compatList.isDelegateSupportedOnThisDevice) {
                        val gpuOptions = GpuDelegate.Options().apply {
                            setPrecisionLossAllowed(true)
                            setInferencePreference(GpuDelegate.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED)
                        }
                        gpuDelegate = GpuDelegate(gpuOptions)
                        addDelegate(gpuDelegate)
                        addLog("TFLite Stylizer: GPU Mode")
                    } else {
                        setNumThreads(4)
                        setUseXNNPACK(true)
                        addLog("TFLite Stylizer: CPU+XNNPACK")
                    }
                }
                
                tfliteInterpreter = Interpreter(modelBuffer, options)

                tfliteInterpreter?.let {
                    val inputShape = it.getInputTensor(0).shape()
                    modelInputHeight = inputShape[1]
                    modelInputWidth = inputShape[2]
                    
                    // Pre-initialize ImageProcessor
                    imageProcessor = ImageProcessor.Builder()
                        .add(ResizeOp(modelInputHeight, modelInputWidth, ResizeOp.ResizeMethod.BILINEAR))
                        .add(NormalizeOp(0f, 255f)) // Adjust if model expects [-1, 1]
                        .build()
                }
            } catch (e: Exception) {
                addLog("Model Error: ${e.message}")
            }
        } else {
            try {
                faceStylizer = FaceStylizer.createFromOptions(context, FaceStylizer.FaceStylizerOptions.builder()
                    .setBaseOptions(BaseOptions.builder().setDelegate(Delegate.GPU).setModelAssetPath(modelName).build()).build())
                addLog("MediaPipe Stylizer Ready (GPU)")
            } catch (e: Exception) {
                addLog("Stylizer Error: ${e.message}")
            }
        }
    }

    private fun loadModelFile(assets: AssetManager, path: String): MappedByteBuffer {
        assets.openFd(path).use { fd ->
            FileInputStream(fd.fileDescriptor).use { inputStream ->
                return inputStream.channel.map(
                    FileChannel.MapMode.READ_ONLY,
                    fd.startOffset,
                    fd.declaredLength
                )
            }
        }
    }

    private fun processLandmarkResult(result: FaceLandmarkerResult) {
        lastFaceResult = result
        val ts = result.timestampMs()
        val capturedBmp = frameCache[ts] ?: return
        
        if (!isStylizing && result.faceLandmarks().isNotEmpty()) {
            performStylization(capturedBmp, result)
        }
    }

    private fun performStylization(targetBmp: Bitmap, faceRes: FaceLandmarkerResult) {
        isStylizing = true
        
        stylizerExecutor?.execute {
            try {
                if (targetBmp.isRecycled) return@execute
                val faceLandmarks = faceRes.faceLandmarks()[0]
                
                // Optimized crop calculation
                var minX = 1.0f; var maxX = 0.0f; var minY = 1.0f; var maxY = 0.0f
                for (lm in faceLandmarks) {
                    if (lm.x() < minX) minX = lm.x()
                    if (lm.x() > maxX) maxX = lm.x()
                    if (lm.y() < minY) minY = lm.y()
                    if (lm.y() > maxY) maxY = lm.y()
                }
                
                val width = maxX - minX
                val height = maxY - minY
                val centerX = minX + width / 2f
                val centerY = minY + height / 2f
                val size = max(width, height) * 1.5f
                
                val left = ((centerX - size / 2f) * targetBmp.width).toInt().coerceIn(0, targetBmp.width - 1)
                val top = ((centerY - size / 2f) * targetBmp.height).toInt().coerceIn(0, targetBmp.height - 1)
                val rectW = (size * targetBmp.width).toInt().coerceAtMost(targetBmp.width - left)
                val rectH = (size * targetBmp.height).toInt().coerceAtMost(targetBmp.height - top)
                
                if (rectW > 10 && rectH > 10) {
                    val faceBmp = Bitmap.createBitmap(targetBmp, left, top, rectW, rectH)
                    val scaledFace = Bitmap.createScaledBitmap(faceBmp, modelInputWidth, modelInputHeight, true)
                    faceBmp.recycle()
                    
                    if (tfliteInterpreter != null && imageProcessor != null) {
                        val tensorImage = TensorImage.fromBitmap(scaledFace)
                        val inputBuffer = imageProcessor!!.process(tensorImage).buffer
                        
                        val outputTensor = tfliteInterpreter!!.getOutputTensor(0)
                        val outputShape = outputTensor.shape()
                        val outH = outputShape[1]
                        val outW = outputShape[2]
                        val pixelCount = outW * outH
                        
                        // Buffer and Array Reuse
                        if (modelOutputBuffer == null || modelOutputBuffer!!.capacity() != pixelCount * 3 * 4) {
                            modelOutputBuffer = java.nio.ByteBuffer.allocateDirect(pixelCount * 3 * 4).order(java.nio.ByteOrder.nativeOrder())
                        }
                        if (outputPixels == null || outputPixels!!.size != pixelCount) {
                            outputPixels = IntArray(pixelCount)
                        }
                        
                        val outputBuffer = modelOutputBuffer!!
                        outputBuffer.rewind()
                        tfliteInterpreter!!.run(inputBuffer, outputBuffer)
                        outputBuffer.rewind()
                        
                        val floatBuffer = outputBuffer.asFloatBuffer()
                        val pixels = outputPixels!!
                        
                        // Optimized Pixel Conversion Loop
                        for (i in 0 until pixelCount) {
                            val r = (floatBuffer.get() * 255f).toInt().coerceIn(0, 255)
                            val g = (floatBuffer.get() * 255f).toInt().coerceIn(0, 255)
                            val b = (floatBuffer.get() * 255f).toInt().coerceIn(0, 255)
                            pixels[i] = -0x1000000 or (r shl 16) or (g shl 8) or b
                        }
                        
                        val resultBmp = Bitmap.createBitmap(outW, outH, Bitmap.Config.ARGB_8888)
                        resultBmp.setPixels(pixels, 0, outW, 0, 0, outW, outH)
                        
                        activity?.runOnUiThread { 
                            val old = lastStylizedBitmap
                            lastStylizedBitmap = resultBmp 
                            lastStylizedFaceResult = faceRes
                            old?.recycle()
                        }
                    } else if (faceStylizer != null) {
                        val stylizedResult = faceStylizer?.stylize(BitmapImageBuilder(scaledFace).build())
                        stylizedResult?.stylizedImage()?.let { optional ->
                            if (optional.isPresent) {
                                val stylizedBmp = BitmapExtractor.extract(optional.get())
                                activity?.runOnUiThread { 
                                    val old = lastStylizedBitmap
                                    lastStylizedBitmap = stylizedBmp 
                                    lastStylizedFaceResult = faceRes
                                    old?.recycle()
                                }
                            }
                        }
                    }
                    scaledFace.recycle()
                }
            } catch (e: Exception) {
                Log.e(TAG, "Stylize Error", e)
            } finally {
                isStylizing = false
            }
        }
    }

    private fun analyzeFrame(imageProxy: ImageProxy) {
        if (!isAdded || _binding == null) { imageProxy.close(); return }
        
        // Reuse inputBitmap
        if (inputBitmap == null || inputBitmap!!.width != imageProxy.width || inputBitmap!!.height != imageProxy.height) {
            inputBitmap?.recycle()
            inputBitmap = Bitmap.createBitmap(imageProxy.width, imageProxy.height, Bitmap.Config.ARGB_8888)
        }
        inputBitmap!!.copyPixelsFromBuffer(imageProxy.planes[0].buffer)
        
        val matrix = Matrix().apply { 
            postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())
            postScale(-1f, 1f) 
        }
        // Note: Bitmap.createBitmap with matrix still creates a new bitmap.
        // For further optimization, consider using a single reusable bitmap and Canvas to draw into it.
        val newFrame = Bitmap.createBitmap(inputBitmap!!, 0, 0, inputBitmap!!.width, inputBitmap!!.height, matrix, true)
        
        val ts = System.currentTimeMillis()
        frameCache[ts] = newFrame
        
        // Cleanup cache more aggressively if needed
        val iterator = frameCache.keys.iterator()
        while (iterator.hasNext()) {
            val key = iterator.next()
            if (key < ts - 200) { // Reduced from 300 to 200ms
                frameCache[key]?.let { if (!it.isRecycled) it.recycle() }
                iterator.remove()
            }
        }
        
        val mpImage = BitmapImageBuilder(newFrame).build()
        try { faceLandmarker?.detectAsync(mpImage, ts) } catch (e: Exception) {}
        try { poseLandmarker?.detectAsync(mpImage, ts) } catch (e: Exception) {}
        
        activity?.runOnUiThread {
            if (_binding != null) {
                binding.faceOverlay.updateFrame(newFrame, lastStylizedBitmap, lastStylizedFaceResult, lastFaceResult, lastPoseResult, renderMode)
            }
        }
        imageProxy.close()
    }

    private fun startCamera() {
        val context = context ?: return
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().also { it.setSurfaceProvider(binding.viewFinder.surfaceProvider) }
            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .setTargetResolution(Size(640, 480)) // Low resolution helps performance
                .build().also { it.setAnalyzer(cameraExecutor!!) { proxy -> analyzeFrame(proxy) } }
            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(viewLifecycleOwner, CameraSelector.DEFAULT_FRONT_CAMERA, preview, imageAnalyzer)
            } catch (e: Exception) { }
        }, ContextCompat.getMainExecutor(context))
    }

    private fun addLog(message: String) {
        activity?.runOnUiThread {
            _binding?.let { b ->
                val timestamp = sdf.format(Date())
                val currentLog = b.eventLog.text.toString()
                b.eventLog.text = "[$timestamp] $message\n${currentLog.take(300)}"
                b.vlmStatus.text = "Status: $message"
            }
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        cameraExecutor?.shutdownNow(); stylizerExecutor?.shutdownNow(); backgroundExecutor?.shutdownNow()
        faceLandmarker?.close(); poseLandmarker?.close(); faceStylizer?.close(); tfliteInterpreter?.close(); gpuDelegate?.close()
        
        inputBitmap?.recycle(); inputBitmap = null
        lastStylizedBitmap?.recycle(); lastStylizedBitmap = null
        frameCache.values.forEach { if (!it.isRecycled) it.recycle() }
        frameCache.clear()

        _binding = null
    }

    private fun allPermissionsGranted() = arrayOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO)
        .all { ContextCompat.checkSelfPermission(requireContext(), it) == PackageManager.PERMISSION_GRANTED }
}
