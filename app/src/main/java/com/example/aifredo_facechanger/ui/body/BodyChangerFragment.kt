package com.example.aifredo_facechanger.ui.body

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
import com.example.aifredo_facechanger.databinding.FragmentBodyChangerBinding
import com.google.mediapipe.framework.image.BitmapExtractor
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.imagesegmenter.ImageSegmenter
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.segmentation.Segmentation
import com.google.mlkit.vision.segmentation.Segmenter
import com.google.mlkit.vision.segmentation.selfie.SelfieSegmenterOptions
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class BodyChangerFragment : Fragment() {

    private var _binding: FragmentBodyChangerBinding? = null
    private val binding get() = _binding!!

    private var cameraExecutor: ExecutorService? = null
    private var imageSegmenter: ImageSegmenter? = null
    private var mlKitSegmenter: Segmenter? = null
    private var yoloInterpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null
    
    private var bodyModel: String = "MediaPipe"
    private var bodyDelegate: String = "CPU"
    private var startColor: Int = Color.RED
    private var endColor: Int = Color.BLUE

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
        
        loadSettings()
        setupSegmenter()

        if (allPermissionsGranted()) startCamera()
        else requestPermissionLauncher.launch(arrayOf(Manifest.permission.CAMERA))
    }

    private fun loadSettings() {
        val sharedPref = activity?.getSharedPreferences("AIfredoPrefs", Context.MODE_PRIVATE) ?: return
        bodyModel = sharedPref.getString("body_model", "MediaPipe") ?: "MediaPipe"
        bodyDelegate = sharedPref.getString("body_delegate", "CPU") ?: "CPU"
        
        val startColorStr = sharedPref.getString("body_start_color", "#FF0000") ?: "#FF0000"
        val endColorStr = sharedPref.getString("body_end_color", "#0000FF") ?: "#0000FF"
        
        try {
            startColor = Color.parseColor(startColorStr)
            endColor = Color.parseColor(endColorStr)
        } catch (e: Exception) {
            startColor = Color.RED
            endColor = Color.BLUE
        }
    }

    private fun setupSegmenter() {
        imageSegmenter?.close(); imageSegmenter = null
        mlKitSegmenter?.close(); mlKitSegmenter = null
        yoloInterpreter?.close(); yoloInterpreter = null
        gpuDelegate?.close(); gpuDelegate = null

        when (bodyModel) {
            "MediaPipe" -> {
                try {
                    val baseOptionsBuilder = BaseOptions.builder()
                        // assets에 있는 실제 파일명으로 수정
                        .setModelAssetPath("mediapipe-meet-segmentation_model_float16_quant.tflite")
                    
                    if (bodyDelegate == "GPU") {
                        baseOptionsBuilder.setDelegate(Delegate.GPU)
                    } else {
                        baseOptionsBuilder.setDelegate(Delegate.CPU)
                    }

                    val options = ImageSegmenter.ImageSegmenterOptions.builder()
                        .setBaseOptions(baseOptionsBuilder.build())
                        .setRunningMode(RunningMode.LIVE_STREAM)
                        .setResultListener { result, image ->
                            val masks = result.confidenceMasks()
                            if (masks.isPresent && masks.get().isNotEmpty()) {
                                // MediaPipe의 마스크를 비트맵으로 추출
                                val mask = BitmapExtractor.extract(masks.get()[0])
                                binding.bodyOverlay.updateData(mask, null, startColor, endColor)
                            }
                        }
                        .build()
                    
                    imageSegmenter = ImageSegmenter.createFromOptions(requireContext(), options)
                    Log.d("BodyChanger", "MediaPipe Segmenter initialized")
                } catch (e: Exception) {
                    Log.e("BodyChanger", "MediaPipe init error: ${e.message}", e)
                }
            }
            "ML Kit" -> {
                val options = SelfieSegmenterOptions.Builder()
                    .setDetectorMode(SelfieSegmenterOptions.STREAM_MODE)
                    .build()
                mlKitSegmenter = Segmentation.getClient(options)
            }
            "YOLO" -> {
                try {
                    val options = Interpreter.Options()
                    if (bodyDelegate == "GPU") {
                        gpuDelegate = GpuDelegate()
                        options.addDelegate(gpuDelegate)
                    } else {
                        options.setNumThreads(4)
                    }
                    val modelBuffer = loadModelFile(requireContext().assets, "AIfredo_epoch150_Loss28.task")
                    yoloInterpreter = Interpreter(modelBuffer, options)
                } catch (e: Exception) {
                    Log.e("BodyChanger", "YOLO init error", e)
                }
            }
        }
    }

    private fun loadModelFile(assets: AssetManager, path: String): MappedByteBuffer {
        assets.openFd(path).use { fd ->
            FileInputStream(fd.fileDescriptor).use { inputStream ->
                return inputStream.channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
            }
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(requireContext())
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(binding.viewFinder.surfaceProvider)
            }

            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor!!) { imageProxy ->
                        val bitmap = processImageProxy(imageProxy)
                        if (bitmap != null) {
                            when (bodyModel) {
                                "MediaPipe" -> {
                                    val mpImage = BitmapImageBuilder(bitmap).build()
                                    imageSegmenter?.segmentAsync(mpImage, System.currentTimeMillis())
                                }
                                "ML Kit" -> {
                                    val inputImage = InputImage.fromBitmap(bitmap, 0)
                                    mlKitSegmenter?.process(inputImage)
                                        ?.addOnSuccessListener { result ->
                                            val maskBuffer = result.buffer
                                            maskBuffer.rewind()
                                            
                                            // ML Kit은 Float mask를 제공하므로 변환 필요
                                            val maskBitmap = Bitmap.createBitmap(result.width, result.height, Bitmap.Config.ALPHA_8)
                                            val byteBuffer = ByteBuffer.allocateDirect(result.width * result.height)
                                            while (maskBuffer.hasRemaining()) {
                                                val confidence = maskBuffer.float
                                                byteBuffer.put((confidence * 255).toInt().toByte())
                                            }
                                            byteBuffer.rewind()
                                            maskBitmap.copyPixelsFromBuffer(byteBuffer)
                                            
                                            binding.bodyOverlay.updateData(maskBitmap, bitmap, startColor, endColor)
                                        }
                                }
                                "YOLO" -> {
                                    processYolo(bitmap)
                                }
                            }
                        }
                        imageProxy.close()
                    }
                }

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(viewLifecycleOwner, CameraSelector.DEFAULT_FRONT_CAMERA, preview, imageAnalyzer)
            } catch (e: Exception) {
                Log.e("BodyChanger", "Camera binding failed", e)
            }
        }, ContextCompat.getMainExecutor(requireContext()))
    }

    private fun processYolo(bitmap: Bitmap) {
        val interpreter = yoloInterpreter ?: return
        
        val inputShape = interpreter.getInputTensor(0).shape()
        val inputH = inputShape[1]
        val inputW = inputShape[2]
        
        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, inputW, inputH, true)
        val inputBuffer = ByteBuffer.allocateDirect(1 * inputH * inputW * 3 * 4).order(ByteOrder.nativeOrder())
        
        val intValues = IntArray(inputW * inputH)
        scaledBitmap.getPixels(intValues, 0, inputW, 0, 0, inputW, inputH)
        for (pixelValue in intValues) {
            inputBuffer.putFloat(((pixelValue shr 16) and 0xFF) / 255.0f)
            inputBuffer.putFloat(((pixelValue shr 8) and 0xFF) / 255.0f)
            inputBuffer.putFloat((pixelValue and 0xFF) / 255.0f)
        }
        
        val outputShape = interpreter.getOutputTensor(0).shape()
        val outH = outputShape[1]
        val outW = outputShape[2]
        val outputBuffer = ByteBuffer.allocateDirect(1 * outH * outW * 1 * 4).order(ByteOrder.nativeOrder())
        
        interpreter.run(inputBuffer, outputBuffer)
        outputBuffer.rewind()
        
        val maskBitmap = Bitmap.createBitmap(outW, outH, Bitmap.Config.ALPHA_8)
        val maskPixels = ByteArray(outW * outH)
        for (i in 0 until outW * outH) {
            val confidence = outputBuffer.float
            maskPixels[i] = if (confidence > 0.5f) 255.toByte() else 0.toByte()
        }
        maskBitmap.copyPixelsFromBuffer(ByteBuffer.wrap(maskPixels))
        
        activity?.runOnUiThread {
            binding.bodyOverlay.updateData(maskBitmap, bitmap, startColor, endColor)
        }
    }

    private fun processImageProxy(imageProxy: ImageProxy): Bitmap? {
        val buffer = imageProxy.planes[0].buffer
        val bitmap = Bitmap.createBitmap(imageProxy.width, imageProxy.height, Bitmap.Config.ARGB_8888)
        bitmap.copyPixelsFromBuffer(buffer)
        
        val matrix = Matrix().apply {
            postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())
            postScale(-1f, 1f)
        }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    private fun allPermissionsGranted() = ContextCompat.checkSelfPermission(requireContext(), Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED

    override fun onDestroyView() {
        super.onDestroyView()
        cameraExecutor?.shutdown()
        imageSegmenter?.close()
        mlKitSegmenter?.close()
        yoloInterpreter?.close()
        gpuDelegate?.close()
        _binding = null
    }
}