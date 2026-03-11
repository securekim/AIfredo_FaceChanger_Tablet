package com.example.aifredo_facechanger.ui.transform

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult

class FaceOverlayView(context: Context, attrs: AttributeSet?) : View(context, attrs) {

    private var originalBitmap: Bitmap? = null
    private var stylizedFace: Bitmap? = null
    private var faceRect: RectF? = null
    private var poseResult: PoseLandmarkerResult? = null

    // 깜빡임 방지를 위한 캐시
    private var lastStylizedFace: Bitmap? = null
    private var lastFaceRect: RectF? = null
    
    private val paint = Paint().apply {
        isAntiAlias = true
        isFilterBitmap = true
    }

    // 실물 얼굴을 가리기 위한 마스크용 페인트 (배경과 색상을 맞춰서 자연스럽게 가림)
    private val maskPaint = Paint().apply {
        color = Color.BLACK
        style = Paint.Style.FILL
    }

    private val linePaint = Paint().apply {
        color = Color.parseColor("#00FFFF")
        strokeWidth = 6f
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
    }

    private val pointPaint = Paint().apply {
        color = Color.YELLOW
        style = Paint.Style.FILL
    }

    private val POSE_CONNECTIONS = listOf(
        Pair(0, 1), Pair(1, 2), Pair(2, 3), Pair(3, 7),
        Pair(0, 4), Pair(4, 5), Pair(5, 6), Pair(6, 8),
        Pair(9, 10),
        Pair(11, 12), Pair(11, 13), Pair(13, 15), Pair(12, 14), Pair(14, 16),
        Pair(11, 23), Pair(12, 24), Pair(23, 24)
    )

    private val destRectLeft = RectF()
    private val destRectRight = RectF()
    private val facePath = Path()

    fun updateFrame(original: Bitmap, stylized: Bitmap?, rect: RectF?, pose: PoseLandmarkerResult?) {
        this.originalBitmap = original
        
        // 새로운 데이터가 오면 캐시 업데이트, 아니면 이전 캐시 유지
        if (stylized != null && !stylized.isRecycled) {
            this.stylizedFace = stylized
            this.lastStylizedFace = stylized
        } else {
            this.stylizedFace = lastStylizedFace
        }
        
        if (rect != null) {
            this.faceRect = rect
            this.lastFaceRect = rect
        } else {
            this.faceRect = lastFaceRect
        }
        
        this.poseResult = pose
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        
        val bitmap = originalBitmap ?: return
        if (bitmap.isRecycled) return

        val midX = width / 2f
        val screenHeight = height.toFloat()
        
        // 스케일 계산 (중앙 정렬)
        val scale = Math.max((width / 2f) / bitmap.width, screenHeight / bitmap.height)
        val drawWidth = bitmap.width * scale
        val drawHeight = bitmap.height * scale
        val offsetX = ((width / 2f) - drawWidth) / 2f
        val offsetY = (screenHeight - drawHeight) / 2f

        destRectLeft.set(offsetX, offsetY, offsetX + drawWidth, offsetY + drawHeight)
        destRectRight.set(midX + offsetX, offsetY, midX + offsetX + drawWidth, offsetY + drawHeight)

        // 1. 좌측 원본 영상
        canvas.save()
        canvas.clipRect(0f, 0f, midX, screenHeight)
        canvas.drawBitmap(bitmap, null, destRectLeft, paint)
        canvas.restore()

        // 2. 우측 영상 영역
        canvas.save()
        canvas.clipRect(midX, 0f, width.toFloat(), screenHeight)
        
        // 우측 배경 그리기 (원본 밝기 그대로 유지)
        canvas.drawBitmap(bitmap, null, destRectRight, paint)

        // 얼굴 영역 처리 (실제 얼굴이 절대 안 보이도록 함)
        val activeRect = faceRect
        val activeFace = stylizedFace

        if (activeRect != null) {
            val rightFaceRect = RectF(
                midX + offsetX + (activeRect.left * scale),
                offsetY + (activeRect.top * scale),
                midX + offsetX + (activeRect.right * scale),
                offsetY + (activeRect.bottom * scale)
            )

            // 원본 얼굴 위치를 가림 (배경색과 유사하게 처리하거나 검정으로 마스킹)
            // 여기서는 실물이 절대 안 보이게 검정으로 가린 후 위에 그림
            canvas.drawOval(rightFaceRect, maskPaint)

            // 3. 변형된 얼굴 그리기 (마스크 위에 그림)
            if (activeFace != null && !activeFace.isRecycled) {
                facePath.reset()
                facePath.addOval(rightFaceRect, Path.Direction.CW)
                canvas.save()
                canvas.clipPath(facePath)
                canvas.drawBitmap(activeFace, null, rightFaceRect, paint)
                canvas.restore()
            }
        }

        // 4. 관절 표시 (얼굴 위에 그려짐)
        poseResult?.let { result ->
            for (landmarks in result.landmarks()) {
                // 선 그리기
                for (connection in POSE_CONNECTIONS) {
                    if (connection.first < landmarks.size && connection.second < landmarks.size) {
                        val start = landmarks[connection.first]
                        val end = landmarks[connection.second]
                        val sx = midX + offsetX + (start.x() * drawWidth)
                        val sy = offsetY + (start.y() * drawHeight)
                        val ex = midX + offsetX + (end.x() * drawWidth)
                        val ey = offsetY + (end.y() * drawHeight)
                        canvas.drawLine(sx, sy, ex, ey, linePaint)
                    }
                }
                // 점 그리기
                for (landmark in landmarks) {
                    val px = midX + offsetX + (landmark.x() * drawWidth)
                    val py = offsetY + (landmark.y() * drawHeight)
                    canvas.drawCircle(px, py, 6f, pointPaint)
                }
            }
        }
        canvas.restore()

        // 5. 중앙 구분선
        canvas.drawLine(midX, 0f, midX, screenHeight, linePaint.apply { color = Color.WHITE; strokeWidth = 3f })
    }
}
