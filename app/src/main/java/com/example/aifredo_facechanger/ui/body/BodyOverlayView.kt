package com.example.aifredo_facechanger.ui.body

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View

/**
 * RVM 및 기타 모델의 결과를 원본 프레임과 함께 그리는 커스텀 뷰.
 * Point 1: 원본 프레임과 마스크를 '동시에' 그려서 엇박자를 해결함.
 */
class BodyOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private var maskBitmap: Bitmap? = null
    private var backgroundBitmap: Bitmap? = null
    private var startColor: Int = Color.RED
    private var endColor: Int = Color.BLUE
    private var isMirrorMode: Boolean = false

    private val gradientPaint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val backgroundPaint = Paint(Paint.FILTER_BITMAP_FLAG)

    private var lastViewHeight = -1f
    private var lastStartColor = 0
    private var lastEndColor = 0

    /**
     * 원본 프레임과 마스크를 한 번에 업데이트함.
     */
    fun updateData(mask: Bitmap?, original: Bitmap?, startCol: Int, endCol: Int, isMirror: Boolean = false) {
        val oldMask = maskBitmap
        val oldBg = backgroundBitmap
        
        maskBitmap = mask
        backgroundBitmap = original
        
        // 이전 비트맵 자원 해제
        if (oldMask != null && oldMask != mask && !oldMask.isRecycled) {
            oldMask.recycle()
        }
        if (oldBg != null && oldBg != original && !oldBg.isRecycled) {
            oldBg.recycle()
        }
        
        this.startColor = startCol
        this.endColor = endCol
        this.isMirrorMode = isMirror
        
        postInvalidate()
    }

    fun updateMaskOnly(mask: Bitmap?, startCol: Int, endCol: Int, isMirror: Boolean = false) {
        updateData(mask, null, startCol, endCol, isMirror)
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        
        val bg = backgroundBitmap
        val mask = maskBitmap
        
        val vw = width.toFloat()
        val vh = height.toFloat()
        if (vw <= 0 || vh <= 0) return

        // 1. 원본 배경 그리기 (추론 스레드에서 가져온 프레임)
        if (bg != null && !bg.isRecycled) {
            drawScaledBitmap(canvas, bg, vw, vh, backgroundPaint)
        }

        if (mask == null || mask.isRecycled) return

        // Gradient 업데이트 (필요할 때만)
        if (vh != lastViewHeight || startColor != lastStartColor || endColor != lastEndColor) {
            gradientPaint.shader = LinearGradient(0f, 0f, 0f, vh, startColor, endColor, Shader.TileMode.CLAMP)
            lastViewHeight = vh
            lastStartColor = startColor
            lastEndColor = endColor
        }

        val halfWidth = vw / 2f
        canvas.save()
        
        // 2. 설정된 모드에 따라 마스크 영역 클리핑 (거울 모드 등)
        if (isMirrorMode) {
            canvas.clipRect(0f, 0f, halfWidth, vh)
        } else {
            canvas.clipRect(halfWidth, 0f, vw, vh)
        }

        // 3. 마스크 그리기 (배경 위에 완벽히 밀착됨)
        drawScaledBitmap(canvas, mask, vw, vh, gradientPaint)

        canvas.restore()
    }

    private fun drawScaledBitmap(canvas: Canvas, bitmap: Bitmap, vw: Float, vh: Float, paint: Paint) {
        val bw = bitmap.width.toFloat()
        val bh = bitmap.height.toFloat()

        val scale: Float
        val dx: Float
        val dy: Float

        val viewRatio = vw / vh
        val bitmapRatio = bw / bh

        // Center Crop 방식의 스케일링
        if (bitmapRatio > viewRatio) {
            scale = vh / bh
            dx = (vw - bw * scale) / 2f
            dy = 0f
        } else {
            scale = vw / bw
            dx = 0f
            dy = (vh - bh * scale) / 2f
        }

        val drawRect = RectF(dx, dy, dx + bw * scale, dy + bh * scale)
        canvas.drawBitmap(bitmap, null, drawRect, paint)
    }
}