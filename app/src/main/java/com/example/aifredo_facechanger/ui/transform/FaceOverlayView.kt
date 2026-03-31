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
    private var stylizedBitmap: Bitmap? = null
    private var stylizedCenter = PointF(0f, 0f)
    private var stylizedSize = 0f
    
    private var currentFaceResult: FaceLandmarkerResult? = null
    private var currentPoseResult: PoseLandmarkerResult? = null
    private var isFaceActive: Boolean = false

    private val poseFiltersX = mutableMapOf<Int, OneEuroFilter>()
    private val poseFiltersY = mutableMapOf<Int, OneEuroFilter>()
    private var filteredPoseLandmarks: List<PointF>? = null

    private val paint = Paint().apply { isAntiAlias = true; isFilterBitmap = true }
    private val pointPaint = Paint().apply { color = Color.GREEN; style = Paint.Style.FILL }
    private val facePath = Path()

    fun updateFrame(
        original: Bitmap, 
        stylized: Bitmap?, 
        sCenter: PointF,
        sSize: Float,
        curFace: FaceLandmarkerResult?, 
        curPose: PoseLandmarkerResult?, 
        mode: String,
        isFaceActive: Boolean
    ) {
        this.originalBitmap = original
        this.stylizedBitmap = stylized
        this.stylizedCenter = sCenter
        this.stylizedSize = sSize
        this.currentFaceResult = curFace
        this.currentPoseResult = curPose
        this.isFaceActive = isFaceActive

        applyPoseFilter(curPose)
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
        
        if (isFaceActive) {
            currentFaceResult?.faceLandmarks()?.getOrNull(0)?.forEach { 
                canvas.drawCircle(offsetX + it.x() * drawW, offsetY + it.y() * drawH, 2f, pointPaint) 
            }
        } else {
            filteredPoseLandmarks?.forEach { 
                canvas.drawCircle(offsetX + it.x * drawW, offsetY + it.y * drawH, 2f, pointPaint) 
            }
        }
        canvas.restore()

        // 2. 우측 합성 결과 렌더링
        canvas.save()
        canvas.clipRect(midX, 0f, width.toFloat(), height.toFloat())
        canvas.drawBitmap(bitmap, null, RectF(offX, offsetY, offX + drawW, offsetY + drawH), paint)

        val stylized = stylizedBitmap
        if (stylized != null && !stylized.isRecycled && stylizedSize > 0) {
            // 변환된 당시의 좌표를 현재 뷰 스케일에 맞게 변환
            val sCenterX = stylizedCenter.x * scale
            val sCenterY = stylizedCenter.y * scale
            val sSizePx = stylizedSize * scale

            facePath.reset()
            // 원형 마스크가 변환 이미지를 꽉 채우도록 비율 설정 (0.485f)
            facePath.addCircle(offX + sCenterX, offsetY + sCenterY, sSizePx * 0.485f, Path.Direction.CW)
            
            canvas.save()
            canvas.clipPath(facePath)
            val destRect = RectF(
                offX + sCenterX - sSizePx / 2f, 
                offsetY + sCenterY - sSizePx / 2f, 
                offX + sCenterX + sSizePx / 2f, 
                offsetY + sCenterY + sSizePx / 2f
            )
            canvas.drawBitmap(stylized, null, destRect, paint)
            canvas.restore()
        }
        canvas.restore()
        
        // 중앙 분리선
        canvas.drawLine(midX, 0f, midX, height.toFloat(), Paint().apply { color = Color.WHITE; strokeWidth = 3f })
    }
}
