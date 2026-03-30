package com.example.aifredo_facechanger.ui.transform

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
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

    private val paint = Paint().apply { isAntiAlias = true; isFilterBitmap = true }
    private val pointPaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.FILL
    }
    private val facePath = Path()
    private val alignmentMatrix = Matrix()

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
        invalidate()
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
        canvas.restore()

        // 2. 우측
        canvas.save()
        canvas.clipRect(midX, 0f, width.toFloat(), height.toFloat())
        canvas.drawBitmap(bitmap, null, RectF(offX, offsetY, offX + drawW, offsetY + drawH), paint)

        val stylized = stylizedFullFrame
        val curFace = currentFaceResult?.faceLandmarks()?.getOrNull(0)

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
            
            // 왜곡(회전, 틸팅, 비정상적 확대)을 방지하기 위해 정방형 사각형 영역에 드로잉
            // TransformFragment의 크롭 로직과 동일하게 1.5배 영역을 계산
            val minX = curFace.minOf { it.x() }
            val maxX = curFace.maxOf { it.x() }
            val minY = curFace.minOf { it.y() }
            val maxY = curFace.maxOf { it.y() }
            
            val centerX = (minX + maxX) / 2f
            val centerY = (minY + maxY) / 2f
            val fW = maxX - minX
            val fH = maxY - minY
            val size = max(fW, fH) * 1.5f
            
            val left = centerX - size / 2f
            val top = centerY - size / 2f
            
            val destRect = RectF(
                offX + left * drawW,
                offsetY + top * drawH,
                offX + (left + size) * drawW,
                offsetY + (top + size) * drawH
            )
            
            // 행렬 변환 없이 사각형 영역에 그대로 그려 회전과 틸팅을 제거함
            canvas.drawBitmap(stylized, null, destRect, paint)
            canvas.restore()
        }

        // 랜드마크 점 그리기 (초록색) - 기존 로직 유지
        curFace?.forEach {
            canvas.drawCircle(offX + it.x() * drawW, offsetY + it.y() * drawH, 3f, pointPaint)
        }
        currentPoseResult?.landmarks()?.getOrNull(0)?.forEach {
            canvas.drawCircle(offX + it.x() * drawW, offsetY + it.y() * drawH, 3f, pointPaint)
        }

        canvas.restore()
        canvas.drawLine(midX, 0f, midX, height.toFloat(), Paint().apply { color = Color.WHITE; strokeWidth = 3f })
    }
}
