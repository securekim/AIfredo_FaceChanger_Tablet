package com.example.aifredo_facechanger.ui.transform

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.*
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.video.*
import androidx.camera.video.VideoCapture
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import com.example.aifredo_facechanger.databinding.FragmentTransformBinding
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.BitmapExtractor
import com.google.mediapipe.framework.image.ByteBufferExtractor
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facedetector.FaceDetector
import com.google.mediapipe.tasks.vision.facedetector.FaceDetectorResult
import com.google.mediapipe.tasks.vision.facestylizer.FaceStylizer
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import java.io.File
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class TransformFragment : Fragment() {

    private var _binding: FragmentTransformBinding? = null
    private val binding get() = _binding!!

    private lateinit var cameraExecutor: ExecutorService
    private lateinit var stylizerExecutor: ExecutorService
    
    private var faceDetector: FaceDetector? = null
    private var faceStylizer: FaceStylizer? = null
    private var poseLandmarker: PoseLandmarker? = null
    
    private val sdf = SimpleDateFormat("HH:mm:ss", Locale.getDefault())
    
    private var inputBitmap: Bitmap? = null
    @Volatile private var currentFrameBitmap: Bitmap? = null
    private var lastStylizedBitmap: Bitmap? = null
    private var lastPoseResult: PoseLandmarkerResult? = null
    private var isStylizing = false

    private var smoothedRect: RectF? = null
    private var lastStylizeTime: Long = 0
    private val STYLIZE_INTERVAL = 1000L 

    // 화면 깜빡임 방지를 위한 카운터
    private var faceNotFoundFrames = 0
    private val MAX_LOST_FRAMES = 10

    private var fallTriggered = false
    private var fallStartTime: Long = 0
    private val FALL_CONFIRM_MS = 2000L

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
        view.postDelayed({ if (isAdded) setupAnalyzers() }, 500)
        if (allPermissionsGranted()) startCamera()
        else requestPermissionLauncher.launch(arrayOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO))
    }

    private fun setupAnalyzers() {
        val context = context ?: return
        val baseOptions = BaseOptions.builder().setDelegate(Delegate.GPU)
        
        try {
            faceDetector = FaceDetector.createFromOptions(context, FaceDetector.FaceDetectorOptions.builder()
                .setBaseOptions(baseOptions.setModelAssetPath("face_detector.task").build())
                .setRunningMode(RunningMode.LIVE_STREAM)
                .setMinDetectionConfidence(0.3f)
                .setResultListener { result, _ -> processFaceResult(result) }
                .build())
            addLog("AI: Face GPU")
        } catch (e: Exception) { addLog("AI: Face CPU") }

        try {
            poseLandmarker = PoseLandmarker.createFromOptions(context, PoseLandmarker.PoseLandmarkerOptions.builder()
                .setBaseOptions(baseOptions.setModelAssetPath("pose_landmarker_lite.task").build())
                .setRunningMode(RunningMode.LIVE_STREAM)
                .setResultListener { result, _ -> lastPoseResult = result; processPoseResult(result) }
                .build())
            addLog("AI: Pose GPU")
        } catch (e: Exception) { addLog("AI: Pose CPU") }

        try {
            faceStylizer = FaceStylizer.createFromOptions(context, FaceStylizer.FaceStylizerOptions.builder()
                .setBaseOptions(BaseOptions.builder().setDelegate(Delegate.CPU).setModelAssetPath("face_stylizer.task").build())
                .build())
            addLog("AI: Stylizer Ready")
        } catch (e: Exception) { addLog("AI: Stylizer Error") }
    }

    private fun processFaceResult(result: FaceDetectorResult) {
        val detections = result.detections()
        val currentTime = System.currentTimeMillis()

        if (detections.isNullOrEmpty()) {
            faceNotFoundFrames++
            if (faceNotFoundFrames > MAX_LOST_FRAMES) {
                smoothedRect = null
            }
            return
        }

        faceNotFoundFrames = 0
        val detection = detections[0]
        val boundingBox = detection.boundingBox()

        // 1. 좌표 안정화
        val currentBmp = currentFrameBitmap ?: return
        val rawRect = calculateSquareCrop(currentBmp.width, currentBmp.height, boundingBox)
        
        if (smoothedRect == null) smoothedRect = rawRect
        else {
            val alpha = 0.15f
            smoothedRect = RectF(
                smoothedRect!!.left * (1 - alpha) + rawRect.left * alpha,
                smoothedRect!!.top * (1 - alpha) + rawRect.top * alpha,
                smoothedRect!!.right * (1 - alpha) + rawRect.right * alpha,
                smoothedRect!!.bottom * (1 - alpha) + rawRect.bottom * alpha
            )
        }

        // 2. 변환 실행 (크래시 방지: 원본 비트맵 직접 사용)
        if (!isStylizing && faceStylizer != null && (currentTime - lastStylizeTime > STYLIZE_INTERVAL)) {
            val faceBitmap = try {
                val left = smoothedRect!!.left.toInt().coerceIn(0, currentBmp.width - 1)
                val top = smoothedRect!!.top.toInt().coerceIn(0, currentBmp.height - 1)
                val width = smoothedRect!!.width().toInt().coerceAtMost(currentBmp.width - left)
                val height = smoothedRect!!.height().toInt().coerceAtMost(currentBmp.height - top)
                if (width > 20 && height > 20) Bitmap.createBitmap(currentBmp, left, top, width, height) else null
            } catch (e: Exception) { null }

            if (faceBitmap != null) {
                isStylizing = true
                lastStylizeTime = currentTime
                stylizerExecutor.execute {
                    try {
                        val stylizedResult = faceStylizer?.stylize(BitmapImageBuilder(faceBitmap).build())
                        stylizedResult?.stylizedImage()?.let { optional ->
                            if (optional.isPresent) {
                                val mpImage = optional.get()
                                val stylizedBmp = try {
                                    BitmapExtractor.extract(mpImage)
                                } catch (e: IllegalArgumentException) {
                                    // Extracting Bitmap from a MPImage created by objects other than Bitmap is not supported
                                    // fall back to ByteBuffer extraction
                                    val buffer = ByteBufferExtractor.extract(mpImage, MPImage.IMAGE_FORMAT_RGBA)
                                    val bmp = Bitmap.createBitmap(mpImage.width, mpImage.height, Bitmap.Config.ARGB_8888)
                                    buffer.rewind()
                                    bmp.copyPixelsFromBuffer(buffer)
                                    bmp
                                }

                                if (stylizedBmp != null) {
                                    val old = lastStylizedBitmap
                                    lastStylizedBitmap = stylizedBmp
                                    old?.recycle()
                                }
                            }
                        }
                    } catch (e: Exception) {
                        Log.e("TransformFragment", "Stylize Error", e)
                    } finally {
                        isStylizing = false
                        faceBitmap.recycle()
                    }
                }
            }
        }
    }

    private fun processPoseResult(result: PoseLandmarkerResult) {
        val landmarksList = result.landmarks()
        if (landmarksList.isEmpty()) return
        val landmarks = landmarksList[0]
        if (landmarks.size < 29) return

        val minX = landmarks.minOf { it.x() }; val maxX = landmarks.maxOf { it.x() }
        val minY = landmarks.minOf { it.y() }; val maxY = landmarks.maxOf { it.y() }
        val bodyWidth = maxX - minX; val bodyHeight = maxY - minY
        val nose = landmarks[0]
        val verticalSpan = Math.abs(nose.y() - (landmarks[27].y() + landmarks[28].y()) / 2f)
        val shoulderWidth = Math.abs(landmarks[11].x() - landmarks[12].x())

        val isLyingDown = (bodyWidth > bodyHeight * 0.9f) && (verticalSpan < shoulderWidth * 1.5f) && (maxY > 0.6f)
        if (isLyingDown) {
            if (!fallTriggered) { fallTriggered = true; fallStartTime = System.currentTimeMillis() }
            else if (System.currentTimeMillis() - fallStartTime > FALL_CONFIRM_MS) confirmFall()
        } else { fallTriggered = false }
    }

    private fun analyzeFrame(imageProxy: ImageProxy) {
        if (!isAdded || _binding == null) { imageProxy.close(); return }
        
        if (inputBitmap == null || inputBitmap!!.width != imageProxy.width || inputBitmap!!.height != imageProxy.height) {
            inputBitmap = Bitmap.createBitmap(imageProxy.width, imageProxy.height, Bitmap.Config.ARGB_8888)
        }
        
        inputBitmap!!.copyPixelsFromBuffer(imageProxy.planes[0].buffer)
        val matrix = Matrix().apply { postRotate(imageProxy.imageInfo.rotationDegrees.toFloat()); postScale(-1f, 1f) }
        
        val newFrame = Bitmap.createBitmap(inputBitmap!!, 0, 0, inputBitmap!!.width, inputBitmap!!.height, matrix, true)
        val oldFrame = currentFrameBitmap
        currentFrameBitmap = newFrame
        // 주의: oldFrame을 즉시 recycle하면 뷰 그리기와 충돌할 수 있으므로 GC에 맡기거나 교체 주기를 조절
        // 여기서는 안전을 위해 즉시 recycle을 피합니다.

        val mpImage = BitmapImageBuilder(newFrame).build()
        val timestamp = System.currentTimeMillis()
        
        faceDetector?.detectAsync(mpImage, timestamp)
        poseLandmarker?.detectAsync(mpImage, timestamp)
        
        activity?.runOnUiThread {
            if (_binding != null) {
                binding.faceOverlay.updateFrame(newFrame, lastStylizedBitmap, smoothedRect, lastPoseResult)
            }
        }
        imageProxy.close()
    }

    private fun calculateSquareCrop(imgW: Int, imgH: Int, boundingBox: RectF): RectF {
        val centerX = boundingBox.centerX(); val centerY = boundingBox.centerY()
        var size = Math.max(boundingBox.width(), boundingBox.height()) * 1.6f
        val maxPossibleSize = Math.min(Math.min(centerX, imgW.toFloat() - centerX), Math.min(centerY, imgH.toFloat() - centerY)) * 2f
        size = Math.min(size, maxPossibleSize)
        return RectF(centerX - size / 2f, centerY - size / 2f, centerX + size / 2f, centerY + size / 2f)
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
                .setTargetResolution(Size(1280, 720)) 
                .build().also { it.setAnalyzer(cameraExecutor) { proxy -> analyzeFrame(proxy) } }
            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(viewLifecycleOwner, CameraSelector.DEFAULT_FRONT_CAMERA, preview, imageAnalyzer)
            } catch (e: Exception) { Log.e("Transform", "Bind Error", e) }
        }, ContextCompat.getMainExecutor(context))
    }

    private fun confirmFall() {
        if (!fallTriggered) return
        fallTriggered = false
        activity?.runOnUiThread {
            addLog("FALL DETECTED!")
            Toast.makeText(context, "Fall Detected!", Toast.LENGTH_LONG).show()
        }
    }

    private fun addLog(message: String) {
        activity?.runOnUiThread {
            _binding?.let { b ->
                val timestamp = sdf.format(Date())
                b.eventLog.text = "[$timestamp] $message\n${b.eventLog.text.toString().take(200)}"
                b.vlmStatus.text = "Status: $message"
            }
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        cameraExecutor.shutdown(); stylizerExecutor.shutdown()
        faceDetector?.close(); poseLandmarker?.close(); faceStylizer?.close()
        _binding = null
    }

    private fun allPermissionsGranted() = arrayOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO)
        .all { ContextCompat.checkSelfPermission(requireContext(), it) == PackageManager.PERMISSION_GRANTED }
}
