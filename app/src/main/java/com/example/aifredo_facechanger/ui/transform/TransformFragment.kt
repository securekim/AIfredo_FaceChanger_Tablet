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
import com.example.aifredo_facechanger.utils.OneEuroFilter
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
import org.tensorflow.lite.DataType
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
import kotlin.math.sqrt

class TransformFragment : Fragment() {

    private var _binding: FragmentTransformBinding? = null
    private val binding get() = _binding!!

    private var cameraExecutor: ExecutorService? = null
    private var stylizerExecutor: ExecutorService? = null
    private var backgroundExecutor: ExecutorService? = null
    
    private var faceLandmarker: FaceLandmarker? = null
    private var poseLandmarker: PoseLandmarker? = null
    
    @Volatile private var faceStylizer: FaceStylizer? = null
    @Volatile private var tfliteInterpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null
    
    private var modelInputWidth = 256
    private var modelInputHeight = 256

    @Volatile private var lastStylizedBitmap: Bitmap? = null
    @Volatile private var lastStylizedCenterX = 0f
    @Volatile private var lastStylizedCenterY = 0f
    @Volatile private var lastStylizedSize = 0f
    
    private var inputBitmap: Bitmap? = null
    private var modelOutputBuffer: java.nio.ByteBuffer? = null

    private val frameCache = ConcurrentHashMap<Long, Bitmap>()
    private var lastPoseResult: PoseLandmarkerResult? = null
    private var lastFaceResult: FaceLandmarkerResult? = null
    @Volatile private var isStylizing = false

    private enum class LandmarkMode { FACE, POSE }
    private var currentLandmarkMode = LandmarkMode.POSE
    private var modeSwitchCounter = 0
    private val MODE_SWITCH_THRESHOLD = 5

    private var lastValidFaceResult: FaceLandmarkerResult? = null
    private var faceLossCounter = 0
    private val FACE_HOLD_MAX_FRAMES = 10

    private val filterX = OneEuroFilter(minCutoff = 0.2, beta = 0.01)
    private val filterY = OneEuroFilter(minCutoff = 0.2, beta = 0.01)
    private val filterSize = OneEuroFilter(minCutoff = 0.1, beta = 0.005)

    @Volatile private var filteredCenterX = 0f
    @Volatile private var filteredCenterY = 0f
    @Volatile private var filteredSize = 0f

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
        
        val resolution = sharedPref?.getInt("model_resolution", 256) ?: 256
        modelInputWidth = resolution
        modelInputHeight = resolution
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
        val context = context?.applicationContext ?: return
        val modelPref = currentModelPref
        val resW = modelInputWidth
        val resH = modelInputHeight

        stylizerExecutor?.execute {
            try {
                val modelName = when (modelPref) {
                    "MediaPipe_Default" -> "face_stylizer.task"
                    "CartoonGAN_Default" -> "whitebox_cartoon_gan_int8.tflite"
                    else -> "whitebox_cartoon_gan_int8.tflite"
                }

                if (modelName.endsWith(".tflite")) {
                    val modelBuffer = loadModelFile(context.assets, modelName)
                    
                    var finalInterpreter: Interpreter? = null
                    var finalDelegate: GpuDelegate? = null

                    fun tryInit(useGpu: Boolean, useResize: Boolean): Boolean {
                        var tempInterpreter: Interpreter? = null
                        var tempDelegate: GpuDelegate? = null
                        try {
                            val options = Interpreter.Options().apply {
                                if (useGpu) {
                                    val gpuOptions = GpuDelegate.Options().apply {
                                        setPrecisionLossAllowed(true)
                                        setInferencePreference(GpuDelegate.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED)
                                    }
                                    tempDelegate = GpuDelegate(gpuOptions)
                                    addDelegate(tempDelegate)
                                } else {
                                    setNumThreads(4)
                                    setUseXNNPACK(true)
                                }
                            }
                            tempInterpreter = Interpreter(modelBuffer, options)
                            if (useResize) {
                                tempInterpreter.resizeInput(0, intArrayOf(1, resH, resW, 3))
                            }
                            tempInterpreter.allocateTensors()
                            
                            finalInterpreter = tempInterpreter
                            finalDelegate = tempDelegate
                            return true
                        } catch (e: Exception) {
                            Log.w(TAG, "Init failed (gpu=$useGpu, resize=$useResize): ${e.message}")
                            tempInterpreter?.close()
                            tempDelegate?.close()
                            return false
                        }
                    }

                    val gpuSupported = CompatibilityList().isDelegateSupportedOnThisDevice
                    var success = false
                    
                    if (gpuSupported) success = tryInit(true, true)
                    if (!success) success = tryInit(false, true)
                    if (!success && gpuSupported) success = tryInit(true, false)
                    if (!success) success = tryInit(false, false)

                    if (success && finalInterpreter != null) {
                        val interp = finalInterpreter!!
                        val shape = interp.getInputTensor(0).shape()
                        if (shape.size >= 4) {
                            modelInputHeight = shape[1]
                            modelInputWidth = shape[2]
                        }
                        
                        val oldInt = tfliteInterpreter
                        val oldDel = gpuDelegate
                        val oldStylizer = faceStylizer
                        
                        tfliteInterpreter = interp
                        gpuDelegate = finalDelegate
                        faceStylizer = null
                        
                        oldInt?.close()
                        oldDel?.close()
                        oldStylizer?.close()
                        
                        activity?.runOnUiThread { 
                            addLog("TFLite Ready (${if (finalDelegate != null) "GPU" else "CPU"}) ${modelInputWidth}x${modelInputHeight}") 
                        }
                    } else {
                        Log.e(TAG, "All Stylizer initialization attempts failed")
                        activity?.runOnUiThread { addLog("Stylizer Load Error") }
                    }
                } else {
                    val newStylizer = FaceStylizer.createFromOptions(context, FaceStylizer.FaceStylizerOptions.builder()
                        .setBaseOptions(BaseOptions.builder().setDelegate(Delegate.GPU).setModelAssetPath(modelName).build()).build())
                    
                    val oldInt = tfliteInterpreter
                    val oldDel = gpuDelegate
                    val oldStylizer = faceStylizer
                    
                    faceStylizer = newStylizer
                    tfliteInterpreter = null
                    gpuDelegate = null
                    
                    oldInt?.close()
                    oldDel?.close()
                    oldStylizer?.close()
                    
                    activity?.runOnUiThread { addLog("MediaPipe Ready (GPU)") }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Setup Stylizer Error", e)
                activity?.runOnUiThread { addLog("Model Error: ${e.message}") }
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
        
        val faceDetected = result.faceLandmarks().isNotEmpty()
        if (faceDetected) {
            lastValidFaceResult = result
            faceLossCounter = 0
        } else {
            faceLossCounter++
        }

        // Mode switching
        if (currentLandmarkMode == LandmarkMode.POSE) {
            if (faceDetected) {
                modeSwitchCounter++
                if (modeSwitchCounter >= MODE_SWITCH_THRESHOLD) {
                    currentLandmarkMode = LandmarkMode.FACE
                    modeSwitchCounter = 0
                }
            } else modeSwitchCounter = 0
        } else {
            if (!faceDetected) {
                modeSwitchCounter++
                if (modeSwitchCounter >= MODE_SWITCH_THRESHOLD) {
                    currentLandmarkMode = LandmarkMode.POSE
                    modeSwitchCounter = 0
                }
            } else modeSwitchCounter = 0
        }

        val (rawX, rawY, rawS) = calculateStableCrop(result, lastPoseResult, capturedBmp.width, capturedBmp.height)
        
        if (rawS > 0) {
            filteredCenterX = filterX.filter(rawX.toDouble(), ts).toFloat()
            filteredCenterY = filterY.filter(rawY.toDouble(), ts).toFloat()
            filteredSize = filterSize.filter(rawS.toDouble(), ts).toFloat()
        }

        if (!isStylizing && filteredSize > 10) {
            performStylization(capturedBmp, filteredCenterX, filteredCenterY, filteredSize)
        }
    }

    private fun calculateStableCrop(faceRes: FaceLandmarkerResult?, poseRes: PoseLandmarkerResult?, imgW: Int, imgH: Int): Triple<Float, Float, Float> {
        var rawX = 0f
        var rawY = 0f
        var rawS = 0f
        
        val faceDetected = faceRes?.faceLandmarks()?.isNotEmpty() == true
        val holdFace = !faceDetected && lastValidFaceResult != null && faceLossCounter <= FACE_HOLD_MAX_FRAMES
        val poseDetected = poseRes?.landmarks()?.isNotEmpty() == true

        if (faceDetected || holdFace) {
            val res = if (faceDetected) faceRes!! else lastValidFaceResult!!
            val landmarks = res.faceLandmarks()[0]
            
            // Average of inner eye corners and nose tip for stable center
            val nose = landmarks[4]
            val leftEye = landmarks[33]
            val rightEye = landmarks[263]
            
            rawX = (leftEye.x() + rightEye.x() + nose.x()) / 3f * imgW
            rawY = (leftEye.y() + rightEye.y() + nose.y()) / 3f * imgH
            
            val minY = landmarks.minOf { it.y() }
            val maxY = landmarks.maxOf { it.y() }
            val faceH = (maxY - minY) * imgH
            
            rawY -= (faceH * 0.10f) // Center slightly above eyes
            rawS = faceH * 1.85f // Tuned scale
        } 
        else if (poseDetected) {
            val landmarks = poseRes!!.landmarks()[0]
            val headPoints = landmarks.take(11) 
            
            rawX = headPoints.map { it.x() }.average().toFloat() * imgW
            rawY = headPoints.map { it.y() }.average().toFloat() * imgH
            
            val lShoulder = landmarks[11]
            val rShoulder = landmarks[12]
            val shoulderDist = sqrt(Math.pow((rShoulder.x() - lShoulder.x()).toDouble(), 2.0) + 
                                    Math.pow((rShoulder.y() - lShoulder.y()).toDouble(), 2.0)).toFloat()
            
            val headHeightEst = shoulderDist * 0.45f * imgH
            rawY -= (headHeightEst * 0.15f)
            rawS = headHeightEst * 1.9f 
        }
        
        return Triple(rawX, rawY, rawS)
    }

    private fun performStylization(targetBmp: Bitmap, centerXPx: Float, centerYPx: Float, sizePx: Float) {
        isStylizing = true
        stylizerExecutor?.execute {
            try {
                if (targetBmp.isRecycled || sizePx <= 1) return@execute

                val intSize = sizePx.toInt().coerceAtLeast(1)
                val left = centerXPx - sizePx / 2f
                val top = centerYPx - sizePx / 2f
                
                val faceBmp = Bitmap.createBitmap(intSize, intSize, Bitmap.Config.ARGB_8888)
                val canvas = Canvas(faceBmp)
                // Use Matrix translation to handle edge cropping correctly without stretching
                val matrix = Matrix()
                matrix.postTranslate(-left, -top)
                canvas.drawBitmap(targetBmp, matrix, null)
                
                val interpreter = tfliteInterpreter
                val stylizer = faceStylizer

                if (interpreter != null) {
                    val inputTensor = interpreter.getInputTensor(0)
                    val h = inputTensor.shape()[1]; val w = inputTensor.shape()[2]
                    val scaledFace = Bitmap.createScaledBitmap(faceBmp, w, h, true)
                    faceBmp.recycle()
                    
                    val tensorImage = TensorImage(inputTensor.dataType())
                    tensorImage.load(scaledFace)
                    
                    val processor = ImageProcessor.Builder()
                        .add(ResizeOp(h, w, ResizeOp.ResizeMethod.BILINEAR))
                        .apply { if (inputTensor.dataType() == DataType.FLOAT32) add(NormalizeOp(0f, 255f)) }
                        .build()
                    
                    val inputBuffer = processor.process(tensorImage).buffer
                    val outputTensor = interpreter.getOutputTensor(0)
                    val outH = outputTensor.shape()[1]; val outW = outputTensor.shape()[2]
                    val isFloatOutput = outputTensor.dataType() == DataType.FLOAT32
                    
                    val bufferSize = outH * outW * 3 * (if (isFloatOutput) 4 else 1)
                    if (modelOutputBuffer == null || modelOutputBuffer!!.capacity() != bufferSize) {
                        modelOutputBuffer = java.nio.ByteBuffer.allocateDirect(bufferSize).order(java.nio.ByteOrder.nativeOrder())
                    }
                    val outputBuffer = modelOutputBuffer!!
                    outputBuffer.rewind()

                    try { interpreter.run(inputBuffer, outputBuffer) } catch (e: Exception) {
                        interpreter.allocateTensors(); outputBuffer.rewind(); interpreter.run(inputBuffer, outputBuffer)
                    }
                    
                    outputBuffer.rewind()
                    val pixelCount = outW * outH; val pixels = IntArray(pixelCount)
                    
                    if (isFloatOutput) {
                        val outputFloats = FloatArray(pixelCount * 3)
                        outputBuffer.asFloatBuffer().get(outputFloats)
                        for (i in 0 until pixelCount) {
                            val r = (outputFloats[i * 3] * 255).toInt().coerceIn(0, 255)
                            val g = (outputFloats[i * 3 + 1] * 255).toInt().coerceIn(0, 255)
                            val b = (outputFloats[i * 3 + 2] * 255).toInt().coerceIn(0, 255)
                            pixels[i] = -0x1000000 or (r shl 16) or (g shl 8) or b
                        }
                    } else {
                        val outputBytes = ByteArray(pixelCount * 3)
                        outputBuffer.get(outputBytes)
                        for (i in 0 until pixelCount) {
                            val r = outputBytes[i * 3].toInt() and 0xFF
                            val g = outputBytes[i * 3 + 1].toInt() and 0xFF
                            val b = outputBytes[i * 3 + 2].toInt() and 0xFF
                            pixels[i] = -0x1000000 or (r shl 16) or (g shl 8) or b
                        }
                    }
                    
                    val resultBmp = Bitmap.createBitmap(outW, outH, Bitmap.Config.ARGB_8888)
                    resultBmp.setPixels(pixels, 0, outW, 0, 0, outW, outH)
                    
                    activity?.runOnUiThread { 
                        val old = lastStylizedBitmap
                        lastStylizedBitmap = resultBmp 
                        lastStylizedCenterX = centerXPx
                        lastStylizedCenterY = centerYPx
                        lastStylizedSize = sizePx
                        old?.recycle()
                    }
                    scaledFace.recycle()
                } else if (stylizer != null) {
                    val scaledFace = Bitmap.createScaledBitmap(faceBmp, modelInputWidth, modelInputHeight, true)
                    faceBmp.recycle()
                    val stylizedResult = stylizer.stylize(BitmapImageBuilder(scaledFace).build())
                    stylizedResult?.stylizedImage()?.let { optional ->
                        if (optional.isPresent) {
                            val stylizedBmp = BitmapExtractor.extract(optional.get())
                            activity?.runOnUiThread {
                                val old = lastStylizedBitmap
                                lastStylizedBitmap = stylizedBmp
                                lastStylizedCenterX = centerXPx
                                lastStylizedCenterY = centerYPx
                                lastStylizedSize = sizePx
                                old?.recycle()
                            }
                        }
                    }
                    scaledFace.recycle()
                } else { faceBmp.recycle() }
            } catch (e: Exception) { Log.e(TAG, "Stylize Error", e) } finally { isStylizing = false }
        }
    }

    private fun analyzeFrame(imageProxy: ImageProxy) {
        if (!isAdded || _binding == null) { imageProxy.close(); return }
        if (inputBitmap == null || inputBitmap!!.width != imageProxy.width || imageProxy.height != inputBitmap!!.height) {
            inputBitmap?.recycle()
            inputBitmap = Bitmap.createBitmap(imageProxy.width, imageProxy.height, Bitmap.Config.ARGB_8888)
        }
        inputBitmap!!.copyPixelsFromBuffer(imageProxy.planes[0].buffer)
        val matrix = Matrix().apply { postRotate(imageProxy.imageInfo.rotationDegrees.toFloat()); postScale(-1f, 1f) }
        val newFrame = Bitmap.createBitmap(inputBitmap!!, 0, 0, inputBitmap!!.width, inputBitmap!!.height, matrix, true)
        val ts = System.currentTimeMillis()
        frameCache[ts] = newFrame
        
        val iterator = frameCache.keys.iterator()
        while (iterator.hasNext()) {
            val key = iterator.next()
            if (key < ts - 500) { 
                frameCache[key]?.let { if (!it.isRecycled) it.recycle() }
                iterator.remove() 
            }
        }

        val mpImage = BitmapImageBuilder(newFrame).build()
        try { faceLandmarker?.detectAsync(mpImage, ts) } catch (e: Exception) {}
        try { poseLandmarker?.detectAsync(mpImage, ts) } catch (e: Exception) {}

        activity?.runOnUiThread {
            if (_binding != null) {
                binding.faceOverlay.updateFrame(
                    original = newFrame, 
                    stylized = lastStylizedBitmap, 
                    sCenter = PointF(lastStylizedCenterX, lastStylizedCenterY),
                    sSize = lastStylizedSize,
                    curFace = lastFaceResult, 
                    curPose = lastPoseResult,
                    mode = renderMode,
                    isFaceActive = currentLandmarkMode == LandmarkMode.FACE
                )
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
                .setTargetResolution(Size(640, 480))
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
        inputBitmap?.recycle(); inputBitmap = null; lastStylizedBitmap?.recycle(); lastStylizedBitmap = null
        frameCache.values.forEach { if (!it.isRecycled) it.recycle() }; frameCache.clear(); _binding = null
    }

    private fun allPermissionsGranted() = arrayOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO)
        .all { ContextCompat.checkSelfPermission(requireContext(), it) == PackageManager.PERMISSION_GRANTED }
}
