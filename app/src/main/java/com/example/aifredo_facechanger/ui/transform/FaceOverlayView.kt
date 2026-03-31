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
    private var lastStylizedFaceResult: FaceLandmarkerResult? = null
    private var lastStylizedPoseResult: PoseLandmarkerResult? = null
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

    fun updateFrame(original: Bitmap, stylized: Bitmap?, stylizedFace: FaceLandmarkerResult?, stylizedPose: PoseLandmarkerResult?,
                    currentFace: FaceLandmarkerResult?, currentPose: PoseLandmarkerResult?, mode: String) {
        this.originalBitmap = original
        this.stylizedFullFrame = stylized
        this.lastStylizedFaceResult = stylizedFace
        this.lastStylizedPoseResult = stylizedPose
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
            val filterX = poseFiltersX.getOrPut(index) { OneEuroFilter(minCutoff = 10.0, beta = 1.0) }
            val filterY = poseFiltersY.getOrPut(index) { OneEuroFilter(minCutoff = 10.0, beta = 1.0) }

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
        val poseLandmarks = currentPoseResult?.landmarks()?.getOrNull(0)

        if (stylized != null && !stylized.isRecycled) {
            var centerXPx = 0f; var centerYPx = 0f; var sizePx = 0f
            var found = false

            if (curFace != null) {
                val minXPx = curFace.minOf { it.x() } * drawW
                val maxXPx = curFace.maxOf { it.x() } * drawW
                val minYPx = curFace.minOf { it.y() } * drawH
                val maxYPx = curFace.maxOf { it.y() } * drawH
                
                centerXPx = (minXPx + maxXPx) / 2f
                centerYPx = (minYPx + maxYPx) / 2f
                sizePx = max(maxXPx - minXPx, maxYPx - minYPx) * 1.5f

                // Detailed path clipping
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
                found = true
            } else if (poseLandmarks != null) {
                // Estimation using Pose
                val headPart = poseLandmarks.take(11)
                val lShoulder = poseLandmarks[11]
                val rShoulder = poseLandmarks[12]
                
                val shoulderWidth = sqrt(Math.pow((rShoulder.x() - lShoulder.x()).toDouble(), 2.0) + 
                                        Math.pow((rShoulder.y() - lShoulder.y()).toDouble(), 2.0)).toFloat()
                
                val estimatedHeadSize = shoulderWidth * 0.9f
                
                val detectedMinXPx = headPart.minOf { it.x() } * drawW
                val detectedMaxXPx = headPart.maxOf { it.x() } * drawW
                val detectedMinYPx = headPart.minOf { it.y() } * drawH
                val detectedMaxYPx = headPart.maxOf { it.y() } * drawH
                
                val detectedWPx = (detectedMaxXPx - detectedMinXPx)
                val finalHeadWidthPx = max(detectedWPx, estimatedHeadSize * drawW)
                
                centerXPx = (detectedMinXPx + detectedMaxXPx) / 2f
                centerYPx = (detectedMinYPx + detectedMaxYPx) / 2f
                
                // Adjust vertical position to cover the forehead more
                centerYPx -= (finalHeadWidthPx * 0.25f) 

                sizePx = finalHeadWidthPx * 1.8f

                // Circular clipping
                facePath.reset()
                facePath.addCircle(offX + centerXPx, offsetY + centerYPx, sizePx * 0.45f, Path.Direction.CW)
                canvas.save()
                canvas.clipPath(facePath)
                found = true
            }

            if (found) {
                val destRect = RectF(
                    offX + centerXPx - sizePx / 2f,
                    offsetY + centerYPx - sizePx / 2f,
                    offX + centerXPx + sizePx / 2f,
                    offsetY + centerYPx + sizePx / 2f
                )
                
                canvas.drawBitmap(stylized, null, destRect, paint)
                canvas.restore()
            }
        }
        canvas.restore()

        canvas.drawLine(midX, 0f, midX, height.toFloat(), Paint().apply { color = Color.WHITE; strokeWidth = 3f })
    }
}
