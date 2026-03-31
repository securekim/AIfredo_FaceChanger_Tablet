package com.example.aifredo_facechanger.ui.transform

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import com.example.aifredo_facechanger.utils.OneEuroFilter
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import kotlin.math.max
import kotlin.math.sqrt

class FaceOverlayView(context: Context, attrs: AttributeSet?) : View(context, attrs) {

    private var originalBitmap: Bitmap? = null
    private var stylizedFullFrame: Bitmap? = null
    private var currentFaceResult: FaceLandmarkerResult? = null
    private var currentPoseResult: PoseLandmarkerResult? = null
    private var isFaceModeActive: Boolean = false

    private val poseFiltersX = mutableMapOf<Int, OneEuroFilter>()
    private val poseFiltersY = mutableMapOf<Int, OneEuroFilter>()
    private var filteredPoseLandmarks: List<PointF>? = null

    private val paint = Paint().apply { isAntiAlias = true; isFilterBitmap = true }
    private val pointPaint = Paint().apply { color = Color.GREEN; style = Paint.Style.FILL }
    private val facePath = Path()

    fun updateFrame(original: Bitmap, stylized: Bitmap?, stylizedFace: FaceLandmarkerResult?, stylizedPose: PoseLandmarkerResult?,
                    currentFace: FaceLandmarkerResult?, currentPose: PoseLandmarkerResult?, mode: String, isFaceMode: Boolean) {
        this.originalBitmap = original
        this.stylizedFullFrame = stylized
        this.currentFaceResult = currentFace
        this.currentPoseResult = currentPose
        this.isFaceModeActive = isFaceMode

        applyPoseFilter(currentPose)
        invalidate()
    }

    private fun applyPoseFilter(result: PoseLandmarkerResult?) {
        val landmarks = result?.landmarks()?.getOrNull(0) ?: run { filteredPoseLandmarks = null; return }
        val timestamp = System.currentTimeMillis()
        filteredPoseLandmarks = landmarks.mapIndexed { index, landmark ->
            val filterX = poseFiltersX.getOrPut(index) { OneEuroFilter(minCutoff = 10.0, beta = 1.0) }
            val filterY = poseFiltersY.getOrPut(index) { OneEuroFilter(minCutoff = 10.0, beta = 1.0) }
            PointF(filterX.filter(landmark.x().toDouble(), timestamp).toFloat(), filterY.filter(landmark.y().toDouble(), timestamp).toFloat())
        }
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val bitmap = originalBitmap ?: return
        if (bitmap.isRecycled) return

        val midX = width / 2f
        val scale = max(midX / bitmap.width, height.toFloat() / bitmap.height)
        val drawW = bitmap.width * scale
        val drawH = bitmap.height * scale
        val offsetX = (midX - drawW) / 2f
        val offsetY = (height.toFloat() - drawH) / 2f
        val offX = midX + offsetX

        // 1. 좌측 원본 렌더링
        canvas.save()
        canvas.clipRect(0f, 0f, midX, height.toFloat())
        canvas.drawBitmap(bitmap, null, RectF(offsetX, offsetY, offsetX + drawW, offsetY + drawH), paint)
        
        val curFace = currentFaceResult?.faceLandmarks()?.getOrNull(0)
        curFace?.forEach { canvas.drawCircle(offsetX + it.x() * drawW, offsetY + it.y() * drawH, 3f, pointPaint) }
        filteredPoseLandmarks?.forEach { canvas.drawCircle(offsetX + it.x * drawW, offsetY + it.y * drawH, 3f, pointPaint) }
        canvas.restore()

        // 2. 우측 합성 결과 렌더링
        canvas.save()
        canvas.clipRect(midX, 0f, width.toFloat(), height.toFloat())
        canvas.drawBitmap(bitmap, null, RectF(offX, offsetY, offX + drawW, offsetY + drawH), paint)

        val stylized = stylizedFullFrame
        if (stylized != null && !stylized.isRecycled) {
            var centerXPx = 0f; var centerYPx = 0f; var sizePx = 0f
            var found = false

            if (isFaceModeActive && curFace != null) {
                val minYPx = curFace.minOf { it.y() } * drawH
                val maxYPx = curFace.maxOf { it.y() } * drawH
                val faceHPx = maxYPx - minYPx
                
                centerXPx = curFace.map { it.x() }.average().toFloat() * drawW
                centerYPx = curFace.map { it.y() }.average().toFloat() * drawH
                centerYPx -= (faceHPx * 0.08f) // 보정치 하향 (0.15 -> 0.08)
                sizePx = faceHPx * 1.8f // 배율 하향 (2.2 -> 1.8)

                facePath.reset()
                facePath.addCircle(offX + centerXPx, offsetY + centerYPx, sizePx * 0.45f, Path.Direction.CW)
                canvas.save(); canvas.clipPath(facePath)
                found = true
            } else {
                val poseLandmarks = currentPoseResult?.landmarks()?.getOrNull(0)
                if (poseLandmarks != null) {
                    val headPart = poseLandmarks.take(11)
                    val lShoulder = poseLandmarks[11]; val rShoulder = poseLandmarks[12]
                    val shoulderWidth = sqrt(Math.pow((rShoulder.x() - lShoulder.x()).toDouble(), 2.0) + Math.pow((rShoulder.y() - lShoulder.y()).toDouble(), 2.0)).toFloat()
                    val headPoints = headPart.filterIndexed { i, _ -> i in 0..8 }
                    
                    val finalHeadWidthPx = max((headPart.maxOf { it.x() } - headPart.minOf { it.x() }) * drawW, shoulderWidth * 0.40f * drawW)
                    
                    centerXPx = headPoints.map { it.x() }.average().toFloat() * drawW
                    centerYPx = headPoints.map { it.y() }.average().toFloat() * drawH
                    centerYPx -= (finalHeadWidthPx * 0.10f) 
                    sizePx = finalHeadWidthPx * 1.8f

                    facePath.reset()
                    facePath.addCircle(offX + centerXPx, offsetY + centerYPx, sizePx * 0.45f, Path.Direction.CW)
                    canvas.save(); canvas.clipPath(facePath)
                    found = true
                }
            }

            if (found) {
                val destRect = RectF(offX + centerXPx - sizePx / 2f, offsetY + centerYPx - sizePx / 2f, offX + centerXPx + sizePx / 2f, offsetY + centerYPx + sizePx / 2f)
                canvas.drawBitmap(stylized, null, destRect, paint)
                canvas.restore()
            }
        }
        canvas.restore()
        canvas.drawLine(midX, 0f, midX, height.toFloat(), Paint().apply { color = Color.WHITE; strokeWidth = 3f })
    }
}
