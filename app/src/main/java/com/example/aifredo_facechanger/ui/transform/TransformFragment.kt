package com.example.aifredo_facechanger.ui.transform

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.content.res.AssetManager
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
import androidx.media3.exoplayer.rtsp.RtspMediaSource
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
import kotlin.math.*

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

    private var trackOffsetX = 0f
    private var trackOffsetY = 0f
    private var hasValidTrackOffset = false
    private val TARGET_MULTIPLIER = 1.4f

    private var lastRawFx = 0f
    private var lastRawFy = 0f
    private var lastRawFs = 0f
    private var unstableFaceCounter = 0
    private var faceStableCounter = 0

    private var lastTransitionUpdateTime: Long = 0
    @Volatile private var rawTransitionRatio: Float = 0f
    @Volatile private var targetTransitionRatio: Float = 0f
    @Volatile private var currentCornerRatio: Float = 0f

    private val filterX = OneEuroFilter(minCutoff = 5.0, beta = 0.6)
    private val filterY = OneEuroFilter(minCutoff = 5.0, beta = 0.6)
    private val filterSize = OneEuroFilter(minCutoff = 2.0, beta = 0.2)

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

    // RTSP
    private var exoPlayer: ExoPlayer? = null
    private var isRtspMode = false
    private val rtspFrameHandler = Handler(Looper.getMainLooper())
    private val rtspFrameRunnable = object : Runnable {
        override fun run() {
            if (isRtspMode && exoPlayer?.isPlaying == true) {
                extractFrameFromPlayer()
            }
            rtspFrameHandler.postDelayed(this, 40) // ~25fps for analysis
        }
    }

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { startStream() }

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

        if (allPermissionsGranted()) startStream()
        else requestPermissionLauncher.launch(arrayOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO))
    }

    override fun onResume() {
        super.onResume()
        val oldRtspMode = isRtspMode
        loadSettings()

        if (oldRtspMode != isRtspMode) {
            startStream()
        }
        
        backgroundExecutor?.execute { if (isAdded) setupAnalyzers() }
    }

    private fun loadSettings() {
        val sharedPref = activity?.getSharedPreferences("AIfredoPrefs", Context.MODE_PRIVATE) ?: return
        currentModelPref = sharedPref.getString("selected_model", "AnimeGAN_Hayao") ?: "AnimeGAN_Hayao"
        renderMode = sharedPref.getString("render_mode", "Face_Only") ?: "Face_Only"
        val resolution = sharedPref.getInt("model_resolution", 256)
        modelInputWidth = resolution
        modelInputHeight = resolution
        useFaceLandmarkPref = sharedPref.getBoolean("use_face_landmark", true)
        faceDelegatePref = sharedPref.getString("face_delegate", "CPU") ?: "CPU"
        poseDelegatePref = sharedPref.getString("pose_delegate", "CPU") ?: "CPU"
        isRtspMode = sharedPref.getString("cam_source", "Embedded") == "RTSP"
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
                lastTransitionUpdateTime = 0
                rawTransitionRatio = 0f
                targetTransitionRatio = 0f
                currentCornerRatio = 0f

                trackOffsetX = 0f
                trackOffsetY = 0f
                hasValidTrackOffset = false

                lastRawFx = 0f
                lastRawFy = 0f
                lastRawFs = 0f
                unstableFaceCounter = 0
                faceStableCounter = 0
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

                        val step = (deltaTime / 1000f) * 3.5f

                        if (result.faceLandmarks().isNotEmpty()) {
                            lastValidFaceResult = result
                            faceLossCounter = 0
                            if (faceStableCounter >= 20) {
                                targetTransitionRatio = 1f
                            }
                        } else {
                            faceLossCounter++
                            if (faceLossCounter > 3) {
                                targetTransitionRatio = 0f
                            }
                        }

                        if (rawTransitionRatio < targetTransitionRatio) {
                            rawTransitionRatio = (rawTransitionRatio + step).coerceAtMost(targetTransitionRatio)
                        } else if (rawTransitionRatio > targetTransitionRatio) {
                            rawTransitionRatio = (rawTransitionRatio - step).coerceAtLeast(targetTransitionRatio)
                        }

                        val t = rawTransitionRatio
                        currentCornerRatio = t * t * (3 - 2 * t)
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
                        addLog("4. Face Stylizer  : ${if (finalDelegate != null) "GPU" else "CPU"}")
                        addLog("   -> Resolution: ${modelInputWidth}x${modelInputHeight}")
                    }
                }
            } else {
                val newStylizer = FaceStylizer.createFromOptions(context, FaceStylizer.FaceStylizerOptions.builder()
                    .setBaseOptions(BaseOptions.builder().setDelegate(Delegate.GPU).setModelAssetPath(modelName).build()).build())
                val oldInt = tfliteInterpreter; val oldDel = gpuDelegate; val oldStylizer = faceStylizer
                faceStylizer = newStylizer; tfliteInterpreter = null; gpuDelegate = null
                oldInt?.close(); oldDel?.close(); oldStylizer?.close()
                activity?.runOnUiThread { addLog("4. Face Stylizer  : GPU (MediaPipe)") }
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
        val poseDetected = poseRes?.landmarks()?.isNotEmpty() == true
        var pCx = 0f; var pCy = 0f; var pSize = 0f

        if (poseDetected) {
            val landmarks = poseRes!!.landmarks()[0]
            val nose = landmarks[0]
            val earL = landmarks[7]
            val earR = landmarks[8]
            val dL = sqrt(((nose.x() - earL.x()) * imgW).pow(2) + ((nose.y() - earL.y()) * imgH).pow(2))
            val dR = sqrt(((nose.x() - earR.x()) * imgW).pow(2) + ((nose.y() - earR.y()) * imgH).pow(2))
            val refDist = max(dL, dR)
            val closeness = (refDist / imgW).coerceIn(0f, 1f)
            val adaptiveMultiplier = 2.2f + (closeness * 1.0f) 

            pCx = nose.x() * imgW
            pCy = (nose.y() * imgH) - (refDist * 0.35f) 
            pSize = refDist * adaptiveMultiplier
        }

        val faceDetected = faceRes?.faceLandmarks()?.isNotEmpty() == true
        var fCx = 0f; var fCy = 0f; var fSize = 0f
        var isValidFace = false

        if (faceDetected) {
            val landmarks = faceRes!!.faceLandmarks()[0]
            val minFx = landmarks.minOf { it.x() } * imgW
            val maxFx = landmarks.maxOf { it.x() } * imgW
            val minFy = landmarks.minOf { it.y() } * imgH
            val maxFy = landmarks.maxOf { it.y() } * imgH

            val faceW = maxFx - minFx
            val faceH = maxFy - minFy
            val rawCx = (minFx + maxFx) / 2f
            val rawCy = (minFy + maxFy) / 2f - (faceH * 0.10f)
            val rawSize = max(faceW, faceH) * TARGET_MULTIPLIER

            var faceValidThisFrame = true
            if (faceH > 0 && (faceW / faceH) < 0.7f) faceValidThisFrame = false
            if (poseDetected && pSize > 0) {
                if (faceW < pSize * 0.4f || faceH < pSize * 0.4f) faceValidThisFrame = false
                val centerDist = sqrt((rawCx - pCx).pow(2) + (rawCy - pCy).pow(2))
                if (centerDist > pSize * 0.6f) { faceValidThisFrame = false; faceStableCounter = 0 }
            }
            if (landmarks.size > 362) {
                val eyeL = landmarks[133]; val eyeR = landmarks[362]
                val eyeDist = sqrt(((eyeL.x() - eyeR.x()) * imgW).pow(2) + ((eyeL.y() - eyeR.y()) * imgH).pow(2))
                if (eyeDist < faceW * 0.15f) faceValidThisFrame = false
            }
            if (faceValidThisFrame) faceStableCounter++ else faceStableCounter = 0

            if (faceStableCounter >= 20) {
                if (lastRawFs > 0f) {
                    val sizeJump = abs(rawSize - lastRawFs) / lastRawFs
                    val posJump = sqrt((rawCx - lastRawFx).pow(2) + (rawCy - lastRawFy).pow(2)) / lastRawFs
                    if (sizeJump < 0.15f && posJump < 0.15f) { isValidFace = true; unstableFaceCounter = 0 }
                    else { unstableFaceCounter++; if (unstableFaceCounter > 4) { isValidFace = true; unstableFaceCounter = 0 } }
                } else { isValidFace = true; unstableFaceCounter = 0 }
            }
            if (isValidFace) { fCx = rawCx; fCy = rawCy; fSize = rawSize; lastRawFx = rawCx; lastRawFy = rawCy; lastRawFs = rawSize }
        } else faceStableCounter = 0

        return when {
            isValidFace && poseDetected -> { trackOffsetX = fCx - pCx; trackOffsetY = fCy - pCy; hasValidTrackOffset = true; Triple(fCx, fCy, fSize) }
            poseDetected && hasValidTrackOffset -> Triple(pCx + trackOffsetX, pCy + trackOffsetY, lastRawFs)
            isValidFace -> Triple(fCx, fCy, fSize)
            poseDetected -> Triple(pCx, pCy, pSize * TARGET_MULTIPLIER)
            else -> Triple(lastStylizedCenterX, lastStylizedCenterY, lastStylizedSize)
        }
    }

    private fun applySemiFilter(src: Bitmap): Bitmap {
        val originalW = src.width; val originalH = src.height
        val targetSize = 160
        val needsScaling = originalW > targetSize || originalH > targetSize
        val workingBitmap = if (needsScaling) Bitmap.createScaledBitmap(src, targetSize, targetSize, true) else src
        val w = workingBitmap.width; val h = workingBitmap.height
        val pixels = IntArray(w * h); workingBitmap.getPixels(pixels, 0, w, 0, 0, w, h)
        val smoothPixels = applyBilateralFilter(pixels, w, h, 1, 1.5, 25.0)
        for (i in smoothPixels.indices) {
            val c = smoothPixels[i]; val r = (c shr 16) and 0xE0; val g = (c shr 8) and 0xE0; val b = c and 0xE0
            smoothPixels[i] = 0xFF000000.toInt() or (r shl 16) or (g shl 8) or b
        }
        val edges = extractEdges(pixels, w, h); val hsv = FloatArray(3)
        for (i in smoothPixels.indices) {
            val color = smoothPixels[i]; val edge = edges[i]; var r = (color shr 16) and 0xFF; var g = (color shr 8) and 0xFF; var b = color and 0xFF
            if (edge == 0) { r = r shr 1; g = g shr 1; b = b shr 1 }
            Color.RGBToHSV(r, g, b, hsv); hsv[1] = hsv[1] * 0.45f; hsv[2] = (hsv[2] * 1.3f).coerceAtMost(1.0f); smoothPixels[i] = Color.HSVToColor(hsv)
        }
        val result = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888); result.setPixels(smoothPixels, 0, w, 0, 0, w, h)
        if (needsScaling && workingBitmap != src) workingBitmap.recycle()
        return if (needsScaling) { val finalRes = Bitmap.createScaledBitmap(result, originalW, originalH, true); result.recycle(); finalRes } else result
    }

    private fun applyBilateralFilter(pixels: IntArray, w: Int, h: Int, radius: Int, sigmaSpatial: Double, sigmaColor: Double): IntArray {
        val output = IntArray(pixels.size); val size = 2 * radius + 1; val spatialWeights = FloatArray(size * size)
        for (dy in -radius..radius) for (dx in -radius..radius) spatialWeights[(dy + radius) * size + (dx + radius)] = exp(-(dx * dx + dy * dy) / (2.0 * sigmaSpatial * sigmaSpatial)).toFloat()
        val colorWeightTable = FloatArray(256 * 256 * 3 + 1); val colorDenom = 2.0 * sigmaColor * sigmaColor
        for (i in colorWeightTable.indices) colorWeightTable[i] = exp(-i / colorDenom).toFloat()
        for (y in 0 until h) {
            val yOffset = y * w; val yMin = (y - radius).coerceAtLeast(0); val yMax = (y + radius).coerceAtMost(h - 1)
            for (x in 0 until w) {
                val centerColor = pixels[yOffset + x]; val cr = (centerColor shr 16) and 0xFF; val cg = (centerColor shr 8) and 0xFF; val cb = centerColor and 0xFF
                var sumR = 0f; var sumG = 0f; var sumB = 0f; var totalWeight = 0f
                for (ny in yMin..yMax) {
                    val neighborYOffset = ny * w; val dy = ny - y; val spatialRowOffset = (dy + radius) * size
                    val xMin = (x - radius).coerceAtLeast(0); val xMax = (x + radius).coerceAtMost(w - 1)
                    for (nx in xMin..xMax) {
                        val neighborColor = pixels[neighborYOffset + nx]; val nr = (neighborColor shr 16) and 0xFF; val ng = (neighborColor shr 8) and 0xFF; val nb = neighborColor and 0xFF
                        val distSq = (cr - nr) * (cr - nr) + (cg - ng) * (cg - ng) + (cb - nb) * (cb - nb)
                        val weight = spatialWeights[spatialRowOffset + (nx - x + radius)] * colorWeightTable[distSq]
                        sumR += nr * weight; sumG += ng * weight; sumB += nb * weight; totalWeight += weight
                    }
                }
                output[yOffset + x] = 0xFF000000.toInt() or ((sumR / totalWeight).toInt() shl 16) or ((sumG / totalWeight).toInt() shl 8) or (sumB / totalWeight).toInt()
            }
        }
        return output
    }

    private fun extractEdges(pixels: IntArray, w: Int, h: Int): IntArray {
        val gray = IntArray(pixels.size)
        for (i in pixels.indices) { val c = pixels[i]; gray[i] = ((c shr 16 and 0xFF) * 77 + (c shr 8 and 0xFF) * 150 + (c and 0xFF) * 29) shr 8 }
        val blurred = IntArray(gray.size); val window = IntArray(9)
        for (y in 1 until h - 1) {
            val rowOffset = y * w
            for (x in 1 until w - 1) {
                window[0] = gray[rowOffset - w + x - 1]; window[1] = gray[rowOffset - w + x]; window[2] = gray[rowOffset - w + x + 1]
                window[3] = gray[rowOffset + x - 1]; window[4] = gray[rowOffset + x]; window[5] = gray[rowOffset + x + 1]
                window[6] = gray[rowOffset + w + x - 1]; window[7] = gray[rowOffset + w + x]; window[8] = gray[rowOffset + w + x + 1]
                window.sort(); blurred[rowOffset + x] = window[4]
            }
        }
        val blockSize = 7; val radius = blockSize / 2; val C = 5; val temp = IntArray(gray.size)
        for (y in 0 until h) {
            var currentSum = 0; val rowOffset = y * w
            for (i in -radius..radius) currentSum += blurred[rowOffset + i.coerceIn(0, w - 1)]
            for (x in 0 until w) { temp[rowOffset + x] = currentSum; currentSum += blurred[rowOffset + (x + radius + 1).coerceIn(0, w - 1)] - blurred[rowOffset + (x - radius).coerceIn(0, w - 1)] }
        }
        val edges = IntArray(gray.size); val count = blockSize * blockSize
        for (x in 0 until w) {
            var currentSum = 0; for (i in -radius..radius) currentSum += temp[i.coerceIn(0, h - 1) * w + x]
            for (y in 0 until h) {
                val idx = y * w + x; edges[idx] = if (blurred[idx] < (currentSum / count) - C) 0 else 255
                currentSum += temp[(y + radius + 1).coerceIn(0, h - 1) * w + x] - temp[(y - radius).coerceIn(0, h - 1) * w + x]
            }
        }
        return edges
    }

    private fun performStylization(targetBmp: Bitmap, centerXPx: Float, centerYPx: Float, sizePx: Float) {
        if (!isReady || targetBmp.isRecycled) return
        isStylizing = true
        stylizerExecutor?.execute {
            try {
                if (!isReady || targetBmp.isRecycled || sizePx <= 1) return@execute
                val intSize = sizePx.toInt().coerceAtLeast(1)
                val left = centerXPx - sizePx / 2f; val top = centerYPx - sizePx / 2f
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
                    val interpreter = tfliteInterpreter!!; val inputTensor = interpreter.getInputTensor(0)
                    val h = inputTensor.shape()[1]; val w = inputTensor.shape()[2]
                    val scaledFace = Bitmap.createScaledBitmap(faceBmp, w, h, true); faceBmp.recycle()
                    val tensorImage = TensorImage(inputTensor.dataType()); tensorImage.load(scaledFace)
                    val processor = ImageProcessor.Builder().add(ResizeOp(h, w, ResizeOp.ResizeMethod.BILINEAR)).apply { if (inputTensor.dataType() == DataType.FLOAT32) add(NormalizeOp(0f, 255f)) }.build()
                    val inputBuffer = processor.process(tensorImage).buffer
                    val outputTensor = interpreter.getOutputTensor(0); val outH = outputTensor.shape()[1]; val outW = outputTensor.shape()[2]
                    val isFloatOutput = outputTensor.dataType() == DataType.FLOAT32; val bufferSize = outH * outW * 3 * (if (isFloatOutput) 4 else 1)
                    if (modelOutputBuffer == null || modelOutputBuffer!!.capacity() != bufferSize) modelOutputBuffer = java.nio.ByteBuffer.allocateDirect(bufferSize).order(java.nio.ByteOrder.nativeOrder())
                    val outputBuffer = modelOutputBuffer!!; outputBuffer.rewind()
                    try { interpreter.run(inputBuffer, outputBuffer) } catch (e: Exception) { interpreter.allocateTensors(); outputBuffer.rewind(); interpreter.run(inputBuffer, outputBuffer) }
                    outputBuffer.rewind(); val pixelCount = outW * outH; val pixels = IntArray(pixelCount)
                    if (isFloatOutput) {
                        val outputFloats = FloatArray(pixelCount * 3); outputBuffer.asFloatBuffer().get(outputFloats)
                        for (i in 0 until pixelCount) {
                            val r = (outputFloats[i * 3] * 255).toInt().coerceIn(0, 255); val g = (outputFloats[i * 3 + 1] * 255).toInt().coerceIn(0, 255); val b = (outputFloats[i * 3 + 2] * 255).toInt().coerceIn(0, 255)
                            pixels[i] = -0x1000000 or (r shl 16) or (g shl 8) or b
                        }
                    } else {
                        val outputBytes = ByteArray(pixelCount * 3); outputBuffer.get(outputBytes)
                        for (i in 0 until pixelCount) {
                            val r = outputBytes[i * 3].toInt() and 0xFF; val g = outputBytes[i * 3 + 1].toInt() and 0xFF; val b = outputBytes[i * 3 + 2].toInt() and 0xFF
                            pixels[i] = -0x1000000 or (r shl 16) or (g shl 8) or b
                        }
                    }
                    resultBmp = Bitmap.createBitmap(outW, outH, Bitmap.Config.ARGB_8888); resultBmp.setPixels(pixels, 0, outW, 0, 0, outW, outH); scaledFace.recycle()
                } else if (faceStylizer != null) {
                    val scaledFace = Bitmap.createScaledBitmap(faceBmp, modelInputWidth, modelInputHeight, true); faceBmp.recycle()
                    val stylizedResult = faceStylizer!!.stylize(BitmapImageBuilder(scaledFace).build())
                    stylizedResult?.stylizedImage()?.let { optional -> if (optional.isPresent) resultBmp = BitmapExtractor.extract(optional.get()) }
                    scaledFace.recycle()
                } else faceBmp.recycle()
                resultBmp?.let { res -> activity?.runOnUiThread { lastStylizedBitmap = res; lastStylizedCenterX = centerXPx; lastStylizedCenterY = centerYPx; lastStylizedSize = sizePx } }
            } catch (e: Exception) { Log.e(TAG, "Stylize Error", e) } finally { isStylizing = false }
        }
    }

    private fun analyzeFrame(imageProxy: ImageProxy) {
        if (!isAdded || _binding == null || !isReady) { imageProxy.close(); return }
        val newFrame = try {
            synchronized(inputBitmapLock) {
                if (inputBitmap == null || inputBitmap!!.width != imageProxy.width || imageProxy.height != inputBitmap!!.height) {
                    inputBitmap?.recycle(); inputBitmap = Bitmap.createBitmap(imageProxy.width, imageProxy.height, Bitmap.Config.ARGB_8888)
                }
                inputBitmap!!.copyPixelsFromBuffer(imageProxy.planes[0].buffer)
                val matrix = Matrix().apply { postRotate(imageProxy.imageInfo.rotationDegrees.toFloat()); postScale(-1f, 1f) }
                Bitmap.createBitmap(inputBitmap!!, 0, 0, inputBitmap!!.width, inputBitmap!!.height, matrix, true)
            }
        } catch (e: Exception) { null }
        if (newFrame == null) { imageProxy.close(); return }
        analyzeBitmap(newFrame)
        imageProxy.close()
    }

    private fun analyzeBitmap(newFrame: Bitmap) {
        if (!isAdded || _binding == null || !isReady) return
        val ts = System.currentTimeMillis(); frameCache[ts] = newFrame
        val iterator = frameCache.keys.iterator()
        while (iterator.hasNext()) { val key = iterator.next(); if (key < ts - 1500) { frameCache[key]?.let { if (!it.isRecycled) it.recycle() }; iterator.remove() } }
        try {
            val mpImage = BitmapImageBuilder(newFrame).build()
            synchronized(landmarkLock) { if (isReady) { faceLandmarker?.detectAsync(mpImage, ts); poseLandmarker?.detectAsync(mpImage, ts) } }
        } catch (e: Exception) { Log.e(TAG, "Detection trigger error", e) }
        activity?.runOnUiThread {
            _binding?.faceOverlay?.updateFrame(
                original = newFrame, stylized = lastStylizedBitmap, sCenter = PointF(filteredCenterX, filteredCenterY),
                sSize = filteredSize, curFace = lastFaceResult, curPose = lastPoseResult, mode = renderMode,
                isFaceActive = useFaceLandmarkPref, shapeProgress = currentCornerRatio
            )
        }
    }

    private fun startStream() {
        if (isRtspMode) startRtsp() else startCamera()
    }

    private fun startCamera() {
        stopRtsp()
        _binding?.viewFinder?.visibility = View.VISIBLE
        _binding?.playerView?.visibility = View.GONE
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
                addLog("Camera started")
            } catch (e: Exception) { addLog("Camera bind failed: ${e.message}") }
        }, ContextCompat.getMainExecutor(context))
    }

    @OptIn(UnstableApi::class)
    private fun startRtsp() {
        stopCamera()
        _binding?.viewFinder?.visibility = View.GONE
        _binding?.playerView?.visibility = View.VISIBLE
        val sharedPref = activity?.getSharedPreferences("AIfredoPrefs", Context.MODE_PRIVATE) ?: return
        val ip = sharedPref.getString("rtsp_ip", "") ?: ""
        val id = sharedPref.getString("rtsp_id", "") ?: ""
        val pw = sharedPref.getString("rtsp_pw", "") ?: ""
        if (ip.isEmpty()) { 
            addLog("RTSP IP is empty. Please set it in Settings.")
            return 
        }

        // Sanitize URL: remove prefix if already included and trim spaces
        val cleanIp = ip.trim().removePrefix("rtsp://")
        val rtspUrl = if (id.isNotEmpty() && pw.isNotEmpty()) {
            "rtsp://$id:$pw@$cleanIp"
        } else {
            "rtsp://$cleanIp"
        }

        addLog("Connecting to RTSP: $rtspUrl")
        
        exoPlayer = ExoPlayer.Builder(requireContext()).build().apply {
            addListener(object : Player.Listener {
                override fun onPlayerError(error: androidx.media3.common.PlaybackException) {
                    val message = error.cause?.message ?: error.message
                    addLog("Playback Error: $message")
                    Log.e(TAG, "ExoPlayer Error", error)
                }
            })
            
            // Using TCP is generally more stable for mobile apps to avoid firewall issues
            val mediaSource = RtspMediaSource.Factory()
                .setForceUseRtpTcp(true)
                .createMediaSource(MediaItem.fromUri(rtspUrl))
            
            setMediaSource(mediaSource)
            prepare()
            playWhenReady = true
        }
        binding.playerView?.player = exoPlayer
        rtspFrameHandler.post(rtspFrameRunnable)
    }

    private fun extractFrameFromPlayer() {
        val b = _binding ?: return
        val textureView = b.playerView?.videoSurfaceView as? TextureView ?: return
        val bitmap = textureView.getBitmap() ?: return
        cameraExecutor?.execute { analyzeBitmap(bitmap) }
    }

    private fun stopCamera() { try { ProcessCameraProvider.getInstance(requireContext()).get().unbindAll() } catch (e: Exception) {} }

    private fun stopRtsp() { rtspFrameHandler.removeCallbacks(rtspFrameRunnable); exoPlayer?.release(); exoPlayer = null; _binding?.playerView?.player = null }

    private fun addLog(message: String) {
        activity?.runOnUiThread {
            _binding?.let { b ->
                val timestamp = sdf.format(Date())
                b.eventLog.text = "[$timestamp] $message\n${b.eventLog.text.toString().take(500)}"
                b.vlmStatus.text = "Status: $message"
            }
        }
    }

    override fun onDestroyView() {
        isReady = false; stopRtsp(); super.onDestroyView()
        cameraExecutor?.shutdown(); stylizerExecutor?.shutdown(); backgroundExecutor?.shutdown()
        try { cameraExecutor?.awaitTermination(200, TimeUnit.MILLISECONDS); stylizerExecutor?.awaitTermination(200, TimeUnit.MILLISECONDS) } catch (e: Exception) {}
        synchronized(landmarkLock) { faceLandmarker?.close(); faceLandmarker = null; poseLandmarker?.close(); poseLandmarker = null }
        faceStylizer?.close(); faceStylizer = null; tfliteInterpreter?.close(); tfliteInterpreter = null; gpuDelegate?.close(); gpuDelegate = null
        synchronized(inputBitmapLock) { inputBitmap?.recycle(); inputBitmap = null }
        lastStylizedBitmap?.recycle(); lastStylizedBitmap = null
        frameCache.values.forEach { if (!it.isRecycled) it.recycle() }; frameCache.clear(); _binding = null
    }

    private fun allPermissionsGranted() = arrayOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO)
        .all { ContextCompat.checkSelfPermission(requireContext(), it) == PackageManager.PERMISSION_GRANTED }
}
