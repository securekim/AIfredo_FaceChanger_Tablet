package com.example.aifredo_facechanger.ui.transform

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import com.example.aifredo_facechanger.utils.OneEuroFilter
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import kotlin.math.max

class FaceOverlayView(context: Context, attrs: AttributeSet?) : View(context, attrs) {

    private var originalBitmap: Bitmap? = null
    private var stylizedFullFrame: Bitmap? = null
    private var lastStylizedFaceResult: FaceLandmarkerResult? = null
    private var currentFaceResult: FaceLandmarkerResult? = null
    private var currentPoseResult: PoseLandmarkerResult? = null
    private var renderMode: String = "Face_Only"

    // OneEuroFilter instances for Pose landmarks to reduce jitter
    private val poseFiltersX = mutableMapOf<Int, OneEuroFilter>()
    private val poseFiltersY = mutableMapOf<Int, OneEuroFilter>()
    private var filteredPoseLandmarks: List<PointF>? = null

    private val paint = Paint().apply { isAntiAlias = true; isFilterBitmap = true }
    private val pointPaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.FILL
    }
    private val facePath = Path()

    private val faceOutlineIndices = intArrayOf(
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
        152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
    )

    fun updateFrame(original: Bitmap, stylized: Bitmap?, stylizedFace: FaceLandmarkerResult?,
                    currentFace: FaceLandmarkerResult?, currentPose: PoseLandmarkerResult?, mode: String) {
        this.originalBitmap = original
        this.stylizedFullFrame = stylized
        this.lastStylizedFaceResult = stylizedFace
        this.currentFaceResult = currentFace
        this.currentPoseResult = currentPose
        this.renderMode = mode

        // Apply jitter reduction filter to Pose landmarks
        applyPoseFilter(currentPose)

        invalidate()
    }

    private fun applyPoseFilter(result: PoseLandmarkerResult?) {
        val landmarks = result?.landmarks()?.getOrNull(0)
        if (landmarks == null) {
            filteredPoseLandmarks = null
            return
        }

        val timestamp = System.currentTimeMillis()
        filteredPoseLandmarks = landmarks.mapIndexed { index, landmark ->
            // Apply filtering to ALL Pose landmarks (including face/head points in Pose)
            val filterX = poseFiltersX.getOrPut(index) { OneEuroFilter(minCutoff = 2.0, beta = 0.005) }
            val filterY = poseFiltersY.getOrPut(index) { OneEuroFilter(minCutoff = 2.0, beta = 0.005) }

            PointF(
                filterX.filter(landmark.x().toDouble(), timestamp).toFloat(),
                filterY.filter(landmark.y().toDouble(), timestamp).toFloat()
            )
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

        // 1. 좌측 원본
        canvas.save()
        canvas.clipRect(0f, 0f, midX, height.toFloat())
        canvas.drawBitmap(bitmap, null, RectF(offsetX, offsetY, offsetX + drawW, offsetY + drawH), paint)
        
        val curFace = currentFaceResult?.faceLandmarks()?.getOrNull(0)
        
        // Draw Face landmarks on the LEFT (Original) side
        curFace?.forEach {
            canvas.drawCircle(offsetX + it.x() * drawW, offsetY + it.y() * drawH, 3f, pointPaint)
        }

        // Draw Pose landmarks on the LEFT (Original) side
        filteredPoseLandmarks?.forEach {
            canvas.drawCircle(offsetX + it.x * drawW, offsetY + it.y * drawH, 3f, pointPaint)
        }
        canvas.restore()

        // 2. 우측 (합성 결과물)
        canvas.save()
        canvas.clipRect(midX, 0f, width.toFloat(), height.toFloat())
        canvas.drawBitmap(bitmap, null, RectF(offX, offsetY, offX + drawW, offsetY + drawH), paint)

        val stylized = stylizedFullFrame

        if (stylized != null && !stylized.isRecycled && curFace != null) {
            // 얼굴 윤곽 패스 생성
            facePath.reset()
            for (i in faceOutlineIndices.indices) {
                val p = curFace[faceOutlineIndices[i]]
                val px = offX + (p.x() * drawW)
                val py = offsetY + (p.y() * drawH)
                if (i == 0) facePath.moveTo(px, py) else facePath.lineTo(px, py)
            }
            facePath.close()

            canvas.save()
            canvas.clipPath(facePath)

            // Calculate exact crop area in pixels, matching TransformFragment's logic
            val minXPx = curFace.minOf { it.x() } * drawW
            val maxXPx = curFace.maxOf { it.x() } * drawW
            val minYPx = curFace.minOf { it.y() } * drawH
            val maxYPx = curFace.maxOf { it.y() } * drawH

            val widthPx = maxXPx - minXPx
            val heightPx = maxYPx - minYPx
            val centerXPx = (minXPx + maxXPx) / 2f
            val centerYPx = (minYPx + maxYPx) / 2f
            
            // Use same 1.5x square logic as TransformFragment to maintain alignment
            val sizePx = max(widthPx, heightPx) * 1.5f
            
            val destRect = RectF(
                offX + centerXPx - sizePx / 2f,
                offsetY + centerYPx - sizePx / 2f,
                offX + centerXPx + sizePx / 2f,
                offsetY + centerYPx + sizePx / 2f
            )
            
            canvas.drawBitmap(stylized, null, destRect, paint)
            canvas.restore()
        }
        canvas.restore()

        canvas.drawLine(midX, 0f, midX, height.toFloat(), Paint().apply { color = Color.WHITE; strokeWidth = 3f })
    }
}
