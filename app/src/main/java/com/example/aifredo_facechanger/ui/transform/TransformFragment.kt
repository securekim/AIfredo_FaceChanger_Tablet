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
import java.util.concurrent.TimeUnit
import kotlin.math.sqrt

class TransformFragment : Fragment() {

    private var _binding: FragmentTransformBinding? = null
    private val binding get() = _binding!!

    private var cameraExecutor: ExecutorService? = null
    private var stylizerExecutor: ExecutorService? = null
    private var backgroundExecutor: ExecutorService? = null

    private val landmarkLock = Any()
    @Volatile private var faceLandmarker: FaceLandmarker? = null
    @Volatile private var poseLandmarker: PoseLandmarker? = null

    @Volatile private var faceStylizer: FaceStylizer? = null
    @Volatile private var tfliteInterpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null

    private var modelInputWidth = 256
    private var modelInputHeight = 256

    @Volatile private var lastStylizedBitmap: Bitmap? = null
    @Volatile private var lastStylizedCenterX = 0f
    @Volatile private var lastStylizedCenterY = 0f
    @Volatile private var lastStylizedSize = 0f

    private val inputBitmapLock = Any()
    private var inputBitmap: Bitmap? = null
    private var modelOutputBuffer: java.nio.ByteBuffer? = null

    private val frameCache = ConcurrentHashMap<Long, Bitmap>()
    @Volatile private var lastPoseResult: PoseLandmarkerResult? = null
    @Volatile private var lastFaceResult: FaceLandmarkerResult? = null
    @Volatile private var isStylizing = false
    @Volatile private var isReady = false

    private var lastValidFaceResult: FaceLandmarkerResult? = null
    private var faceLossCounter = 0
    private val FACE_HOLD_MAX_FRAMES = 10

    // Stability & Transition
    private var faceDetectionStartTime: Long = 0
    private var lastTransitionUpdateTime: Long = 0
    private var rawTransitionRatio: Float = 0f
    @Volatile private var currentCornerRatio: Float = 0f // Start from 0 (Circle)

    private val filterX = OneEuroFilter(minCutoff = 1.0, beta = 0.02)
    private val filterY = OneEuroFilter(minCutoff = 1.0, beta = 0.02)
    private val filterSize = OneEuroFilter(minCutoff = 0.5, beta = 0.01)

    @Volatile private var filteredCenterX = 0f
    @Volatile private var filteredCenterY = 0f
    @Volatile private var filteredSize = 0f

    private val TAG = "AIfredo_Transform"
    private var currentModelPref: String = "AnimeGAN_Hayao"
    private var renderMode: String = "Face_Only"
    private var useFaceLandmarkPref: Boolean = true
    private var faceDelegatePref: String = "CPU"
    private var poseDelegatePref: String = "CPU"
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

    override fun onResume() {
        super.onResume()
        val oldModel = currentModelPref
        val oldFaceLandmark = useFaceLandmarkPref
        val oldFaceDelegate = faceDelegatePref
        val oldPoseDelegate = poseDelegatePref
        
        loadSettings()
        
        if (oldModel != currentModelPref || oldFaceLandmark != useFaceLandmarkPref || 
            oldFaceDelegate != faceDelegatePref || oldPoseDelegate != poseDelegatePref) {
            backgroundExecutor?.execute { if (isAdded) setupAnalyzers() }
        }
    }

    private fun loadSettings() {
        val sharedPref = activity?.getSharedPreferences("AIfredoPrefs", Context.MODE_PRIVATE)
        currentModelPref = sharedPref?.getString("selected_model", "AnimeGAN_Hayao") ?: "AnimeGAN_Hayao"
        renderMode = sharedPref?.getString("render_mode", "Face_Only") ?: "Face_Only"
        val resolution = sharedPref?.getInt("model_resolution", 256) ?: 256
        modelInputWidth = resolution
        modelInputHeight = resolution
        useFaceLandmarkPref = sharedPref?.getBoolean("use_face_landmark", true) ?: true
        faceDelegatePref = sharedPref?.getString("face_delegate", "CPU") ?: "CPU"
        poseDelegatePref = sharedPref?.getString("pose_delegate", "CPU") ?: "CPU"
    }

    private fun setupAnalyzers() {
        val context = context ?: return
        isReady = false
        activity?.runOnUiThread { addLog("--- SYSTEM INITIALIZING ---") }
        
        try {
            synchronized(landmarkLock) {
                faceLandmarker?.close()
                faceLandmarker = null
                poseLandmarker?.close()
                poseLandmarker = null
                lastFaceResult = null
                lastPoseResult = null
                lastValidFaceResult = null
                faceLossCounter = 0
                faceDetectionStartTime = 0
                lastTransitionUpdateTime = 0
                rawTransitionRatio = 0f
                currentCornerRatio = 0f
            }

            val faceModel = "face_landmarker.task"
            val poseModel = "pose_landmarker_lite.task"
            val faceDelegate = if (faceDelegatePref == "GPU") Delegate.GPU else Delegate.CPU
            val poseDelegate = if (poseDelegatePref == "GPU") Delegate.GPU else Delegate.CPU

            val newFaceLandmarker = if (useFaceLandmarkPref) {
                FaceLandmarker.createFromOptions(context, FaceLandmarker.FaceLandmarkerOptions.builder()
                    .setBaseOptions(BaseOptions.builder().setDelegate(faceDelegate).setModelAssetPath(faceModel).build())
                    .setRunningMode(RunningMode.LIVE_STREAM)
                    .setResultListener { result, _ -> 
                        lastFaceResult = result
                        val currentTime = System.currentTimeMillis()
                        
                        val deltaTime = if (lastTransitionUpdateTime == 0L) 0L else currentTime - lastTransitionUpdateTime
                        lastTransitionUpdateTime = currentTime
                        val step = deltaTime / 100f // 0.1s duration

                        if (result.faceLandmarks().isNotEmpty()) {
                            lastValidFaceResult = result
                            faceLossCounter = 0
                            
                            if (faceDetectionStartTime == 0L) {
                                faceDetectionStartTime = currentTime
                            } else if (currentTime - faceDetectionStartTime > 100L) {
                                // Stability reached, proceed with transition
                                rawTransitionRatio = (rawTransitionRatio + step).coerceIn(0f, 1f)
                            }
                        } else {
                            faceLossCounter++
                            if (faceLossCounter > FACE_HOLD_MAX_FRAMES) {
                                faceDetectionStartTime = 0
                                // Slowly transition back to 0 (Pose)
                                rawTransitionRatio = (rawTransitionRatio - step).coerceIn(0f, 1f)
                            }
                        }
                        
                        // Apply Cubic Ease-In for "sucking in" feel
                        val t = rawTransitionRatio
                        currentCornerRatio = t * t * t 
                    }
                    .build())
            } else null
            
            val newPoseLandmarker = PoseLandmarker.createFromOptions(context, PoseLandmarker.PoseLandmarkerOptions.builder()
                .setBaseOptions(BaseOptions.builder().setDelegate(poseDelegate).setModelAssetPath(poseModel).build())
                .setRunningMode(RunningMode.LIVE_STREAM)
                .setResultListener { result, _ -> 
                    lastPoseResult = result
                    processFrame(result.timestampMs())
                }
                .build())
            
            synchronized(landmarkLock) {
                faceLandmarker = newFaceLandmarker
                poseLandmarker = newPoseLandmarker
            }

            activity?.runOnUiThread { 
                addLog("1. Face Landmarker : ${if (useFaceLandmarkPref) faceDelegatePref else "DISABLED"}")
                addLog("2. Pose Landmarker : $poseDelegatePref")
            }
            
            setupStylizerSync()
            
            isReady = true
            activity?.runOnUiThread { addLog("--- ALL SYSTEMS READY ---") }
        } catch (e: Exception) { 
            activity?.runOnUiThread { addLog("Initialization Error: ${e.message}") }
            Log.e(TAG, "Initialization Error", e)
        }
    }

    private fun setupStylizerSync() {
        val context = context?.applicationContext ?: return
        val modelPref = currentModelPref
        val resW = modelInputWidth
        val resH = modelInputHeight

        try {
            val modelName = when (modelPref) {
                "AnimeGAN_Hayao" -> "animeganv2_hayao_256x256_float16_quant.tflite"
                "AnimeGAN_Paprika" -> "animeganv2_paprika_256x256_float16_quant.tflite"
                "MediaPipe_Default" -> "face_stylizer.task"
                "CartoonGAN_Default" -> "whitebox_cartoon_gan_fp16.tflite"
                "SEMI_Filter" -> "NONE"
                else -> "animeganv2_hayao_256x256_float16_quant.tflite"
            }

            if (modelName == "NONE") {
                activity?.runOnUiThread { addLog("3. SEMI-Filter Mode (Non-AI)") }
                val oldInt = tfliteInterpreter; val oldDel = gpuDelegate; val oldStylizer = faceStylizer
                tfliteInterpreter = null; gpuDelegate = null; faceStylizer = null
                oldInt?.close(); oldDel?.close(); oldStylizer?.close()
                return
            }

            activity?.runOnUiThread { addLog("3. Loading Model : $modelName") }

            if (modelName.endsWith(".tflite")) {
                val modelBuffer = loadModelFile(context.assets, modelName)
                var finalInterpreter: Interpreter? = null
                var finalDelegate: GpuDelegate? = null
                var success = false
                
                fun tryInit(useGpu: Boolean, useResize: Boolean): Boolean {
                    return try {
                        val options = Interpreter.Options().apply {
                            if (useGpu) {
                                finalDelegate = GpuDelegate(GpuDelegate.Options().apply {
                                    setPrecisionLossAllowed(true)
                                    setInferencePreference(GpuDelegate.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED)
                                })
                                addDelegate(finalDelegate)
                            } else {
                                setNumThreads(4)
                                setUseXNNPACK(true)
                            }
                        }
                        finalInterpreter = Interpreter(modelBuffer, options)
                        if (useResize) finalInterpreter!!.resizeInput(0, intArrayOf(1, resH, resW, 3))
                        finalInterpreter!!.allocateTensors()
                        true
                    } catch (e: Exception) {
                        finalInterpreter?.close(); finalDelegate?.close()
                        false
                    }
                }

                success = tryInit(useGpu = true, useResize = true)
                if (!success) success = tryInit(useGpu = true, useResize = false)
                if (!success) success = tryInit(useGpu = false, useResize = true)
                if (!success) success = tryInit(useGpu = false, useResize = false)

                if (success && finalInterpreter != null) {
                    val shape = finalInterpreter!!.getInputTensor(0).shape()
                    if (shape.size >= 4) { modelInputHeight = shape[1]; modelInputWidth = shape[2] }
                    
                    val oldInt = tfliteInterpreter; val oldDel = gpuDelegate; val oldStylizer = faceStylizer
                    tfliteInterpreter = finalInterpreter; gpuDelegate = finalDelegate; faceStylizer = null
                    oldInt?.close(); oldDel?.close(); oldStylizer?.close()
                    
                    activity?.runOnUiThread { 
                        addLog("4. Face Stylizer  : ${if (finalDelegate != null) "GPU \uD83D\uDE80" else "CPU \uD83D\uDCBB"}")
                        addLog("   -> Resolution: ${modelInputWidth}x${modelInputHeight}")
                    }
                }
            } else {
                val newStylizer = FaceStylizer.createFromOptions(context, FaceStylizer.FaceStylizerOptions.builder()
                    .setBaseOptions(BaseOptions.builder().setDelegate(Delegate.GPU).setModelAssetPath(modelName).build()).build())
                val oldInt = tfliteInterpreter; val oldDel = gpuDelegate; val oldStylizer = faceStylizer
                faceStylizer = newStylizer; tfliteInterpreter = null; gpuDelegate = null
                oldInt?.close(); oldDel?.close(); oldStylizer?.close()
                activity?.runOnUiThread { addLog("4. Face Stylizer  : GPU \uD83D\uDE80 (MediaPipe)") }
            }
        } catch (e: Exception) { 
            activity?.runOnUiThread { addLog("Stylizer Load Error: ${e.message}") }
        }
    }

    private fun loadModelFile(assets: AssetManager, path: String): MappedByteBuffer {
        assets.openFd(path).use { fd ->
            FileInputStream(fd.fileDescriptor).use { inputStream ->
                return inputStream.channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
            }
        }
    }

    private fun processFrame(ts: Long) {
        if (!isReady) return
        val capturedBmp = frameCache[ts] ?: return
        if (capturedBmp.isRecycled) return
        
        val faceResult = if (useFaceLandmarkPref) lastFaceResult else null
        val (rawX, rawY, rawS) = calculateStableCrop(faceResult, lastPoseResult, capturedBmp.width, capturedBmp.height)
        
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
        val TARGET_MULTIPLIER = 2.45f
        
        // 1. Calculate Pose-based values (baseline)
        var pX = 0f; var pY = 0f; var pS = 0f
        val poseDetected = poseRes?.landmarks()?.isNotEmpty() == true
        if (poseDetected) {
            val landmarks = poseRes!!.landmarks()[0]
            val headPoints = landmarks.take(11)
            pX = headPoints.map { it.x() }.average().toFloat() * imgW
            pY = headPoints.map { it.y() }.average().toFloat() * imgH
            
            val lShoulder = landmarks[11]; val rShoulder = landmarks[12]
            val shoulderDist = Math.sqrt(Math.pow((rShoulder.x() - lShoulder.x()).toDouble(), 2.0) + Math.pow((rShoulder.y() - lShoulder.y()).toDouble(), 2.0)).toFloat()
            val headHeightEst = shoulderDist * 0.40f * imgH
            pS = headHeightEst * TARGET_MULTIPLIER
        }

        // 2. Calculate Face-based values
        var fX = 0f; var fY = 0f; var fS = 0f
        val faceDetected = faceRes?.faceLandmarks()?.isNotEmpty() == true
        val holdFace = !faceDetected && lastValidFaceResult != null && faceLossCounter <= FACE_HOLD_MAX_FRAMES
        if (faceDetected || holdFace) {
            val res = if (faceDetected) faceRes!! else lastValidFaceResult!!
            val landmarks = res.faceLandmarks()[0]
            val minX = landmarks.minOf { it.x() }; val maxX = landmarks.maxOf { it.x() }
            val minY = landmarks.minOf { it.y() }; val maxY = landmarks.maxOf { it.y() }
            fX = (minX + maxX) / 2f * imgW
            fY = (minY + maxY) / 2f * imgH
            val faceH = (maxY - minY) * imgH
            fS = if (pS > 0) pS else faceH * TARGET_MULTIPLIER
            fY -= (faceH * 0.05f)
        }

        // 3. Decide and Interpolate (currentCornerRatio: 0 -> Pose, 1 -> Stably Face)
        val ratio = currentCornerRatio
        
        return when {
            ratio > 0f -> {
                if (poseDetected && (faceDetected || holdFace)) {
                    Triple(
                        pX * (1 - ratio) + fX * ratio,
                        pY * (1 - ratio) + fY * ratio,
                        pS * (1 - ratio) + fS * ratio
                    )
                } else if (faceDetected || holdFace) {
                    Triple(fX, fY, fS)
                } else if (poseDetected) {
                    Triple(pX * (1 - ratio) + lastStylizedCenterX * ratio, 
                           pY * (1 - ratio) + lastStylizedCenterY * ratio, 
                           pS * (1 - ratio) + lastStylizedSize * ratio)
                } else {
                    Triple(fX, fY, fS)
                }
            }
            poseDetected -> {
                Triple(pX, pY, pS)
            }
            faceDetected || holdFace -> {
                Triple(fX, fY, fS)
            }
            else -> Triple(0f, 0f, 0f)
        }
    }

    private fun applySemiFilter(src: Bitmap): Bitmap {
        val w = src.width; val h = src.height
        val result = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(result)
        val paint = Paint(Paint.FILTER_BITMAP_FLAG)
        val cm = ColorMatrix().apply { setSaturation(1.6f); postConcat(ColorMatrix(floatArrayOf(1.3f, 0f, 0f, 0f, -20f, 0f, 1.3f, 0f, 0f, -20f, 0f, 0f, 1.3f, 0f, -20f, 0f, 0f, 0f, 1f, 0f))) }
        paint.colorFilter = ColorMatrixColorFilter(cm)
        val scale = 8
        val scaled = Bitmap.createScaledBitmap(src, (w / scale).coerceAtLeast(1), (h / scale).coerceAtLeast(1), true)
        canvas.drawBitmap(scaled, null, Rect(0, 0, w, h), paint)
        scaled.recycle()
        val edgeBmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val eCanvas = Canvas(edgeBmp); val ePaint = Paint()
        val gcm = ColorMatrix().apply { setSaturation(0f); postConcat(ColorMatrix(floatArrayOf(5f, 0f, 0f, 0f, -600f, 0f, 5f, 0f, 0f, -600f, 0f, 0f, 5f, 0f, -600f, 0f, 0f, 0f, 1f, 0f))) }
        ePaint.colorFilter = ColorMatrixColorFilter(gcm)
        eCanvas.drawBitmap(src, 0f, 0f, ePaint)
        paint.colorFilter = null; paint.xfermode = PorterDuffXfermode(PorterDuff.Mode.MULTIPLY); paint.alpha = 200
        canvas.drawBitmap(edgeBmp, 0f, 0f, paint); edgeBmp.recycle()
        return result
    }

    private fun performStylization(targetBmp: Bitmap, centerXPx: Float, centerYPx: Float, sizePx: Float) {
        if (!isReady || targetBmp.isRecycled) return
        isStylizing = true
        stylizerExecutor?.execute {
            try {
                if (!isReady || targetBmp.isRecycled || sizePx <= 1) return@execute
                val intSize = sizePx.toInt().coerceAtLeast(1)
                val left = centerXPx - sizePx / 2f
                val top = centerYPx - sizePx / 2f
                val faceBmp = Bitmap.createBitmap(intSize, intSize, Bitmap.Config.ARGB_8888)
                val canvas = Canvas(faceBmp)
                val shader = BitmapShader(targetBmp, Shader.TileMode.CLAMP, Shader.TileMode.CLAMP)
                val paint = Paint().apply { isFilterBitmap = true; setShader(shader) }
                val m = Matrix(); m.postTranslate(-left, -top); shader.setLocalMatrix(m)
                canvas.drawRect(0f, 0f, intSize.toFloat(), intSize.toFloat(), paint)
                
                var resultBmp: Bitmap? = null
                
                if (currentModelPref == "SEMI_Filter") {
                    resultBmp = applySemiFilter(faceBmp); faceBmp.recycle()
                } else if (tfliteInterpreter != null) {
                    val interpreter = tfliteInterpreter!!
                    val inputTensor = interpreter.getInputTensor(0)
                    val h = inputTensor.shape()[1]; val w = inputTensor.shape()[2]
                    val scaledFace = Bitmap.createScaledBitmap(faceBmp, w, h, true); faceBmp.recycle()
                    val tensorImage = TensorImage(inputTensor.dataType()); tensorImage.load(scaledFace)
                    val processor = ImageProcessor.Builder().add(ResizeOp(h, w, ResizeOp.ResizeMethod.BILINEAR)).apply { if (inputTensor.dataType() == DataType.FLOAT32) add(NormalizeOp(0f, 255f)) }.build()
                    val inputBuffer = processor.process(tensorImage).buffer
                    val outputTensor = interpreter.getOutputTensor(0); val outH = outputTensor.shape()[1]; val outW = outputTensor.shape()[2]
                    val isFloatOutput = outputTensor.dataType() == DataType.FLOAT32
                    val bufferSize = outH * outW * 3 * (if (isFloatOutput) 4 else 1)
                    if (modelOutputBuffer == null || modelOutputBuffer!!.capacity() != bufferSize) modelOutputBuffer = java.nio.ByteBuffer.allocateDirect(bufferSize).order(java.nio.ByteOrder.nativeOrder())
                    val outputBuffer = modelOutputBuffer!!; outputBuffer.rewind()
                    try { interpreter.run(inputBuffer, outputBuffer) } catch (e: Exception) { interpreter.allocateTensors(); outputBuffer.rewind(); interpreter.run(inputBuffer, outputBuffer) }
                    outputBuffer.rewind(); val pixelCount = outW * outH; val pixels = IntArray(pixelCount)
                    if (isFloatOutput) {
                        val outputFloats = FloatArray(pixelCount * 3); outputBuffer.asFloatBuffer().get(outputFloats)
                        for (i in 0 until pixelCount) {
                            val r = (outputFloats[i * 3] * 255).toInt().coerceIn(0, 255)
                            val g = (outputFloats[i * 3 + 1] * 255).toInt().coerceIn(0, 255)
                            val b = (outputFloats[i * 3 + 2] * 255).toInt().coerceIn(0, 255)
                            pixels[i] = -0x1000000 or (r shl 16) or (g shl 8) or b
                        }
                    } else {
                        val outputBytes = ByteArray(pixelCount * 3); outputBuffer.get(outputBytes)
                        for (i in 0 until pixelCount) {
                            val r = outputBytes[i * 3].toInt() and 0xFF; val g = outputBytes[i * 3 + 1].toInt() and 0xFF; val b = outputBytes[i * 3 + 2].toInt() and 0xFF
                            pixels[i] = -0x1000000 or (r shl 16) or (g shl 8) or b
                        }
                    }
                    resultBmp = Bitmap.createBitmap(outW, outH, Bitmap.Config.ARGB_8888); resultBmp.setPixels(pixels, 0, outW, 0, 0, outW, outH)
                    scaledFace.recycle()
                } else if (faceStylizer != null) {
                    val scaledFace = Bitmap.createScaledBitmap(faceBmp, modelInputWidth, modelInputHeight, true); faceBmp.recycle()
                    val stylizedResult = faceStylizer!!.stylize(BitmapImageBuilder(scaledFace).build())
                    stylizedResult?.stylizedImage()?.let { optional -> if (optional.isPresent) {
                        resultBmp = BitmapExtractor.extract(optional.get())
                    } }
                    scaledFace.recycle()
                } else faceBmp.recycle()

                resultBmp?.let { res ->
                    activity?.runOnUiThread {
                        lastStylizedBitmap = res
                        lastStylizedCenterX = centerXPx; lastStylizedCenterY = centerYPx; lastStylizedSize = sizePx
                    }
                }
            } catch (e: Exception) { Log.e(TAG, "Stylize Error", e) } finally { isStylizing = false }
        }
    }

    private fun analyzeFrame(imageProxy: ImageProxy) {
        if (!isAdded || _binding == null || !isReady) { imageProxy.close(); return }
        
        val newFrame = try {
            synchronized(inputBitmapLock) {
                if (inputBitmap == null || inputBitmap!!.width != imageProxy.width || imageProxy.height != inputBitmap!!.height) {
                    inputBitmap?.recycle()
                    inputBitmap = Bitmap.createBitmap(imageProxy.width, imageProxy.height, Bitmap.Config.ARGB_8888)
                }
                inputBitmap!!.copyPixelsFromBuffer(imageProxy.planes[0].buffer)
                val matrix = Matrix().apply { postRotate(imageProxy.imageInfo.rotationDegrees.toFloat()); postScale(-1f, 1f) }
                Bitmap.createBitmap(inputBitmap!!, 0, 0, inputBitmap!!.width, inputBitmap!!.height, matrix, true)
            }
        } catch (e: Exception) { null }

        if (newFrame == null) { imageProxy.close(); return }

        val ts = System.currentTimeMillis()
        frameCache[ts] = newFrame
        
        val iterator = frameCache.keys.iterator()
        while (iterator.hasNext()) {
            val key = iterator.next()
            if (key < ts - 1500) { frameCache[key]?.let { if (!it.isRecycled) it.recycle() }; iterator.remove() }
        }

        try {
            val mpImage = BitmapImageBuilder(newFrame).build()
            synchronized(landmarkLock) {
                if (isReady) {
                    faceLandmarker?.detectAsync(mpImage, ts)
                    poseLandmarker?.detectAsync(mpImage, ts)
                }
            }
        } catch (e: Exception) { Log.e(TAG, "Detection trigger error", e) }

        activity?.runOnUiThread {
            if (_binding != null) {
                binding.faceOverlay.updateFrame(
                    original = newFrame, stylized = lastStylizedBitmap, sCenter = PointF(filteredCenterX, filteredCenterY),
                    sSize = filteredSize, curFace = lastFaceResult, curPose = lastPoseResult, mode = renderMode, 
                    isFaceActive = useFaceLandmarkPref, shapeProgress = currentCornerRatio
                )
            }
        }
        imageProxy.close()
    }

    private fun startCamera() {
        val context = context ?: return
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        cameraProviderFuture.addListener({
            val cameraProvider = try { cameraProviderFuture.get() } catch (e: Exception) { return@addListener }
            val preview = Preview.Builder().build().also { p -> p.setSurfaceProvider(binding.viewFinder.surfaceProvider) }
            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .setTargetResolution(Size(640, 480))
                .build().also { a -> a.setAnalyzer(cameraExecutor!!) { proxy -> analyzeFrame(proxy) } }
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
                b.eventLog.text = "[$timestamp] $message\n${currentLog.take(500)}"
                b.vlmStatus.text = "Status: $message"
            }
        }
    }

    override fun onDestroyView() {
        isReady = false
        super.onDestroyView()
        cameraExecutor?.shutdown()
        stylizerExecutor?.shutdown()
        backgroundExecutor?.shutdown()
        try {
            cameraExecutor?.awaitTermination(200, TimeUnit.MILLISECONDS)
            stylizerExecutor?.awaitTermination(200, TimeUnit.MILLISECONDS)
        } catch (e: Exception) {}
        
        synchronized(landmarkLock) {
            faceLandmarker?.close(); faceLandmarker = null
            poseLandmarker?.close(); poseLandmarker = null
        }
        faceStylizer?.close(); faceStylizer = null
        tfliteInterpreter?.close(); tfliteInterpreter = null
        gpuDelegate?.close(); gpuDelegate = null
        
        synchronized(inputBitmapLock) { inputBitmap?.recycle(); inputBitmap = null }
        lastStylizedBitmap?.recycle(); lastStylizedBitmap = null
        frameCache.values.forEach { if (!it.isRecycled) it.recycle() }; frameCache.clear()
        _binding = null
    }

    private fun allPermissionsGranted() = arrayOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO)
        .all { ContextCompat.checkSelfPermission(requireContext(), it) == PackageManager.PERMISSION_GRANTED }
}
