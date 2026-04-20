package com.example.aifredo_facechanger.ui.transform

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
import com.example.aifredo_facechanger.databinding.FragmentTransformBinding
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facestylizer.FaceStylizer
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import java.nio.ByteBuffer
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.RejectedExecutionException
import java.util.concurrent.TimeUnit
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.DataType

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
    private var modelOutputBuffer: ByteBuffer? = null

    private val landmarkLock = Any()
    private val inputBitmapLock = Any()
    private var inputBitmap: Bitmap? = null
    private var lastFaceResult: FaceLandmarkerResult? = null
    private var lastPoseResult: PoseLandmarkerResult? = null
    @Volatile private var lastStylizedBitmap: Bitmap? = null

    private var filteredCenterX = 0f
    private var filteredCenterY = 0f
    private var filteredSize = 0f
    private var lastStylizedCenterX = 0f
    private var lastStylizedCenterY = 0f
    private var lastStylizedSize = 0f

    @Volatile private var isStylizing = false
    @Volatile private var isReady = false
    private var renderMode = "Face_Only"
    private var useFaceLandmarkPref = true
    @Volatile private var currentCornerRatio = 0f

    private val frameCache = ConcurrentHashMap<Long, Bitmap>()
    private val sdf = SimpleDateFormat("HH:mm:ss", Locale.getDefault())
    private val TAG = "TransformFragment"

    private var modelInputWidth = 256
    private var modelInputHeight = 256

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

        if (allPermissionsGranted()) startStream()
        else requestPermissionLauncher.launch(arrayOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO))
    }

    override fun onResume() {
        super.onResume()
        loadPreferences()
        startStream()
        setupModels()
    }

    override fun onPause() {
        super.onPause()
        stopRtsp()
        stopCamera()
    }

    private fun loadPreferences() {
        val sharedPref = activity?.getSharedPreferences("AIfredoPrefs", Context.MODE_PRIVATE) ?: return
        renderMode = sharedPref.getString("render_mode", "Face_Only") ?: "Face_Only"
        useFaceLandmarkPref = sharedPref.getBoolean("use_face_landmark", true)
        isRtspMode = sharedPref.getString("cam_source", "Embedded") == "RTSP"
    }

    private fun setupModels() {
        val bgExec = backgroundExecutor
        if (bgExec == null || bgExec.isShutdown) return

        bgExec.execute {
            synchronized(landmarkLock) {
                isReady = false
                val sharedPref = activity?.getSharedPreferences("AIfredoPrefs", Context.MODE_PRIVATE) ?: return@execute
                val faceDel = sharedPref.getString("face_delegate", "CPU") ?: "CPU"
                val poseDel = sharedPref.getString("pose_delegate", "CPU") ?: "CPU"
                val selectedModel = sharedPref.getString("selected_model", "AnimeGAN_Hayao") ?: "AnimeGAN_Hayao"

                try {
                    faceLandmarker?.close()
                    val faceBase = BaseOptions.builder().setModelAssetPath("face_landmarker.task")
                    if (faceDel == "GPU") faceBase.setDelegate(Delegate.GPU)
                    faceLandmarker = FaceLandmarker.createFromOptions(requireContext(), FaceLandmarker.FaceLandmarkerOptions.builder()
                        .setBaseOptions(faceBase.build()).setRunningMode(RunningMode.LIVE_STREAM)
                        .setResultListener { result, image -> 
                            lastFaceResult = result
                            processStylization(image.width, image.height, result.timestampMs()) 
                        }.build())

                    poseLandmarker?.close()
                    val poseBase = BaseOptions.builder().setModelAssetPath("pose_landmarker_lite.task")
                    if (poseDel == "GPU") poseBase.setDelegate(Delegate.GPU)
                    poseLandmarker = PoseLandmarker.createFromOptions(requireContext(), PoseLandmarker.PoseLandmarkerOptions.builder()
                        .setBaseOptions(poseBase.build()).setRunningMode(RunningMode.LIVE_STREAM)
                        .setResultListener { result, _ -> lastPoseResult = result }.build())

                    val sExec = stylizerExecutor
                    if (sExec != null && !sExec.isShutdown) {
                        sExec.execute {
                            faceStylizer?.close(); faceStylizer = null
                            tfliteInterpreter?.close(); tfliteInterpreter = null
                            gpuDelegate?.close(); gpuDelegate = null
                            try {
                                initStylizer(selectedModel)
                                isReady = true
                                addLog("Models Ready: $selectedModel")
                                Log.d(TAG, "Stylizer initialized: $selectedModel")
                            } catch (e: Exception) {
                                addLog("Stylizer Init Error: ${e.message}")
                                Log.e(TAG, "Stylizer Init Error", e)
                            }
                        }
                    }
                } catch (e: Exception) { 
                    addLog("Landmarker Init Error: ${e.message}")
                    Log.e(TAG, "Landmarker Init Error", e)
                }
            }
        }
    }

    private fun initStylizer(modelName: String) {
        val context = requireContext()
        // SEMI_Filter: Use Face Shape (hull) with AnimeGAN_Hayao style
        val effectiveModel = if (modelName == "SEMI_Filter") {
            currentCornerRatio = 1.0f
            "AnimeGAN_Hayao"
        } else {
            currentCornerRatio = 0.0f
            modelName
        }

        when (effectiveModel) {
            "MediaPipe_Default" -> {
                val base = BaseOptions.builder().setModelAssetPath("face_stylizer.task").build()
                faceStylizer = FaceStylizer.createFromOptions(context, FaceStylizer.FaceStylizerOptions.builder().setBaseOptions(base).build())
            }
            "AnimeGAN_Hayao", "AnimeGAN_Paprika", "CartoonGAN_Default" -> {
                val assetName = when (effectiveModel) {
                    "AnimeGAN_Hayao" -> "animeganv2_hayao_256x256_float16_quant.tflite"
                    "AnimeGAN_Paprika" -> "animeganv2_paprika_256x256_float16_quant.tflite"
                    else -> "whitebox_cartoon_gan_fp16.tflite"
                }
                val options = Interpreter.Options()
                try {
                    if (CompatibilityList().isDelegateSupportedOnThisDevice) {
                        gpuDelegate = GpuDelegate()
                        options.addDelegate(gpuDelegate)
                    }
                } catch (e: Exception) { Log.e(TAG, "GPU Delegate init failed", e) }
                
                tfliteInterpreter = Interpreter(FileUtil.loadMappedFile(context, assetName), options)
                modelInputWidth = 256; modelInputHeight = 256
            }
        }
    }

    private fun processStylization(originalWidth: Int, originalHeight: Int, timestamp: Long) {
        if (isStylizing || !isReady) return
        val face = lastFaceResult ?: return
        if (face.faceLandmarks().isEmpty()) return

        val landmarks = face.faceLandmarks()[0]
        var minX = 1f; var maxX = 0f; var minY = 1f; var maxY = 0f
        for (lm in landmarks) {
            if (lm.x() < minX) minX = lm.x(); if (lm.x() > maxX) maxX = lm.x()
            if (lm.y() < minY) minY = lm.y(); if (lm.y() > maxY) maxY = lm.y()
        }

        val centerX = (minX + maxX) / 2f
        val centerY = (minY + maxY) / 2f
        val width = maxX - minX
        val height = maxY - minY
        val size = maxOf(width, height) * 1.5f

        filteredCenterX = filteredCenterX * 0.7f + centerX * 0.3f
        filteredCenterY = filteredCenterY * 0.7f + centerY * 0.3f
        filteredSize = filteredSize * 0.7f + size * 0.3f

        val centerXPx = filteredCenterX * originalWidth
        val centerYPx = filteredCenterY * originalHeight
        val sizePx = filteredSize * maxOf(originalWidth, originalHeight)

        val sExec = stylizerExecutor
        if (sExec == null || sExec.isShutdown) return

        sExec.execute {
            if (isStylizing) return@execute
            isStylizing = true
            try {
                val left = (centerXPx - sizePx / 2).toInt().coerceAtLeast(0)
                val top = (centerYPx - sizePx / 2).toInt().coerceAtLeast(0)
                val right = (centerXPx + sizePx / 2).toInt().coerceAtMost(originalWidth)
                val bottom = (centerYPx + sizePx / 2).toInt().coerceAtMost(originalHeight)
                if (right <= left || bottom <= top) return@execute

                val faceBmp = synchronized(frameCache) {
                    val currentFrame = frameCache[timestamp] ?: frameCache.values.firstOrNull()
                    if (currentFrame == null || currentFrame.isRecycled) return@execute
                    Bitmap.createBitmap(currentFrame, left, top, right - left, bottom - top)
                } ?: return@execute

                var resultBmp: Bitmap? = null
                val interpreter = tfliteInterpreter
                if (interpreter != null) {
                    val inputTensor = interpreter.getInputTensor(0)
                    val w = inputTensor.shape()[1]; val h = inputTensor.shape()[2]
                    val scaledFace = Bitmap.createScaledBitmap(faceBmp, w, h, true); faceBmp.recycle()
                    
                    val tensorImage = TensorImage(inputTensor.dataType())
                    tensorImage.load(scaledFace)
                    
                    val processor = ImageProcessor.Builder()
                        .add(ResizeOp(h, w, ResizeOp.ResizeMethod.BILINEAR))
                        .apply { if (inputTensor.dataType() == DataType.FLOAT32) add(NormalizeOp(127.5f, 127.5f)) }
                        .build()
                    
                    val inputBuffer = processor.process(tensorImage).buffer
                    val outputTensor = interpreter.getOutputTensor(0)
                    val outH = outputTensor.shape()[1]; val outW = outputTensor.shape()[2]
                    val isFloatOutput = outputTensor.dataType() == DataType.FLOAT32
                    val bufferSize = outH * outW * 3 * (if (isFloatOutput) 4 else 1)
                    
                    if (modelOutputBuffer == null || modelOutputBuffer!!.capacity() != bufferSize) {
                        modelOutputBuffer = ByteBuffer.allocateDirect(bufferSize).order(java.nio.ByteOrder.nativeOrder())
                    }
                    val outputBuffer = modelOutputBuffer!!; outputBuffer.rewind()
                    
                    try { 
                        interpreter.run(inputBuffer, outputBuffer) 
                    } catch (e: Exception) { 
                        interpreter.allocateTensors()
                        outputBuffer.rewind()
                        interpreter.run(inputBuffer, outputBuffer) 
                    }
                    
                    outputBuffer.rewind()
                    val pixelCount = outW * outH
                    val pixels = IntArray(pixelCount)
                    if (isFloatOutput) {
                        val outputFloats = FloatArray(pixelCount * 3)
                        outputBuffer.asFloatBuffer().get(outputFloats)
                        for (i in 0 until pixelCount) {
                            val r = ((outputFloats[i * 3] + 1f) * 127.5f).toInt().coerceIn(0, 255)
                            val g = ((outputFloats[i * 3 + 1] + 1f) * 127.5f).toInt().coerceIn(0, 255)
                            val b = ((outputFloats[i * 3 + 2] + 1f) * 127.5f).toInt().coerceIn(0, 255)
                            pixels[i] = -0x1000000 or (r shl 16) or (g shl 8) or b
                        }
                    } else {
                        val outputBytes = ByteArray(pixelCount * 3); outputBuffer.get(outputBytes)
                        for (i in 0 until pixelCount) {
                            val r = outputBytes[i * 3].toInt() and 0xFF; val g = outputBytes[i * 3 + 1].toInt() and 0xFF; val b = outputBytes[i * 3 + 2].toInt() and 0xFF
                            pixels[i] = -0x1000000 or (r shl 16) or (g shl 8) or b
                        }
                    }
                    resultBmp = Bitmap.createBitmap(outW, outH, Bitmap.Config.ARGB_8888)
                    resultBmp.setPixels(pixels, 0, outW, 0, 0, outW, outH)
                    scaledFace.recycle()
                } else if (faceStylizer != null) {
                    val scaledFace = Bitmap.createScaledBitmap(faceBmp, modelInputWidth, modelInputHeight, true); faceBmp.recycle()
                    val stylizedResult = faceStylizer!!.stylize(BitmapImageBuilder(scaledFace).build())
                    stylizedResult?.stylizedImage()?.let { optional -> if (optional.isPresent) resultBmp = com.google.mediapipe.framework.image.BitmapExtractor.extract(optional.get()) }
                    scaledFace.recycle()
                } else faceBmp.recycle()
                
                resultBmp?.let { res -> 
                    activity?.runOnUiThread { 
                        // Do NOT recycle here. FaceOverlayView will handle it.
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
        if (!isAdded || _binding == null || !isReady) { newFrame.recycle(); return }
        val ts = System.currentTimeMillis()
        synchronized(frameCache) {
            frameCache[ts] = newFrame
            val iterator = frameCache.keys.iterator()
            while (iterator.hasNext()) {
                val key = iterator.next()
                if (key < ts - 1000) { frameCache[key]?.let { if (!it.isRecycled) it.recycle() }; iterator.remove() }
            }
        }
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

    private fun startStream() { if (isRtspMode) startRtsp() else startCamera() }

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
            } catch (e: Exception) { Log.e(TAG, "Camera bind failed", e) }
        }, ContextCompat.getMainExecutor(context))
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
        val rtspQuality = sharedPref.getString("rtsp_quality", "High") ?: "High"
        if (ip.isEmpty()) { addLog("RTSP IP is empty."); return }

        val streamPath = if (rtspQuality == "Low") "stream2" else "stream1"
        val rtspUrl = buildString {
            append("rtsp://")
            if (id.isNotEmpty() && pw.isNotEmpty()) append("$id:$pw@")
            append(ip)
            if (!ip.contains("/")) { if (!ip.contains(":")) append(":554"); append("/$streamPath") }
        }

        addLog("Connecting RTSP: $rtspUrl")
        val loadControl = DefaultLoadControl.Builder().setBufferDurationsMs(500, 1000, 250, 500).build()
        val mediaItem = MediaItem.Builder().setUri(rtspUrl).setLiveConfiguration(MediaItem.LiveConfiguration.Builder().setTargetOffsetMs(500).build()).build()
        exoPlayer = ExoPlayer.Builder(requireContext()).setLoadControl(loadControl).build().apply {
            trackSelectionParameters = trackSelectionParameters.buildUpon().setTrackTypeDisabled(androidx.media3.common.C.TRACK_TYPE_AUDIO, true).build()
            addListener(object : Player.Listener {
                override fun onPlayerError(error: androidx.media3.common.PlaybackException) { addLog("Playback Error: ${error.message}") }
                override fun onPlaybackStateChanged(playbackState: Int) { if (playbackState == Player.STATE_READY && isRtspMode) addLog("RTSP Connected") }
            })
            setMediaSource(RtspMediaSource.Factory().setForceUseRtpTcp(false).setTimeoutMs(4000).createMediaSource(mediaItem))
            prepare(); playWhenReady = true
        }
        binding.playerView.player = exoPlayer
        rtspFrameHandler.post(rtspFrameRunnable)
    }

    private fun extractFrameFromPlayer() {
        val b = _binding ?: return
        val textureView = b.playerView.videoSurfaceView as? TextureView ?: return
        if (!isReady) return
        val originalBitmap = textureView.getBitmap() ?: return

        val cExec = cameraExecutor
        if (cExec != null && !cExec.isShutdown) {
            cExec.execute {
                val targetWidth = 640
                val targetHeight = (originalBitmap.height * (targetWidth.toFloat() / originalBitmap.width)).toInt()
                val bitmap = try { Bitmap.createScaledBitmap(originalBitmap, targetWidth, targetHeight, true) } catch (e: Exception) { originalBitmap }
                if (bitmap != originalBitmap) originalBitmap.recycle()
                analyzeBitmap(bitmap)
            }
        }
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
        isReady = false
        stopRtsp()

        // Capture models to close them on background thread
        val fStylizer = faceStylizer
        val tInterpreter = tfliteInterpreter
        val gDelegate = gpuDelegate

        // Submit close task BEFORE shutdown
        stylizerExecutor?.let { exec ->
            if (!exec.isShutdown) {
                try {
                    exec.execute {
                        fStylizer?.close()
                        tInterpreter?.close()
                        gDelegate?.close()
                    }
                } catch (e: RejectedExecutionException) {
                    fStylizer?.close()
                    tInterpreter?.close()
                    gDelegate?.close()
                }
            } else {
                fStylizer?.close()
                tInterpreter?.close()
                gDelegate?.close()
            }
        }

        faceStylizer = null
        tfliteInterpreter = null
        gpuDelegate = null

        cameraExecutor?.shutdown()
        stylizerExecutor?.shutdown()
        backgroundExecutor?.shutdown()

        synchronized(landmarkLock) { 
            faceLandmarker?.close()
            faceLandmarker = null
            poseLandmarker?.close()
            poseLandmarker = null 
        }

        synchronized(inputBitmapLock) { 
            inputBitmap?.recycle()
            inputBitmap = null 
        }

        lastStylizedBitmap = null
        synchronized(frameCache) { 
            frameCache.values.forEach { if (!it.isRecycled) it.recycle() }
            frameCache.clear() 
        }

        _binding = null
        super.onDestroyView()
    }

    private fun allPermissionsGranted() = arrayOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO)
        .all { ContextCompat.checkSelfPermission(requireContext(), it) == PackageManager.PERMISSION_GRANTED }
}
