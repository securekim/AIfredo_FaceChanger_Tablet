package com.example.aifredo_facechanger.ui.body

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View

class BodyOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private var maskBitmap: Bitmap? = null
    private var startColor: Int = Color.RED
    private var endColor: Int = Color.BLUE
    private var isMirrorMode: Boolean = false

    private val maskPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        xfermode = PorterDuffXfermode(PorterDuff.Mode.DST_IN)
    }
    private val gradientPaint = Paint(Paint.ANTI_ALIAS_FLAG)

    fun updateData(mask: Bitmap?, original: Bitmap?, startCol: Int, endCol: Int, isMirror: Boolean = false) {
        maskBitmap = mask
        startColor = startCol
        endColor = endCol
        isMirrorMode = isMirror
        postInvalidate()
    }

    fun updateMaskOnly(mask: Bitmap?, startCol: Int, endCol: Int, isMirror: Boolean = false) {
        maskBitmap = mask
        startColor = startCol
        endColor = endCol
        isMirrorMode = isMirror
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val mask = maskBitmap
        if (mask == null || mask.isRecycled) return

        val viewWidth = width.toFloat()
        val viewHeight = height.toFloat()
        if (viewWidth <= 0 || viewHeight <= 0) return

        val maskWidth = mask.width.toFloat()
        val maskHeight = mask.height.toFloat()

        val scale: Float
        val dx: Float
        val dy: Float

        val viewRatio = viewWidth / viewHeight
        val maskRatio = maskWidth / maskHeight

        if (maskRatio > viewRatio) {
            scale = viewHeight / maskHeight
            dx = (viewWidth - maskWidth * scale) / 2f
            dy = 0f
        } else {
            scale = viewWidth / maskWidth
            dx = 0f
            dy = (viewHeight - maskHeight * scale) / 2f
        }

        val drawRect = RectF(dx, dy, dx + maskWidth * scale, dy + maskHeight * scale)
        val halfWidth = viewWidth / 2f
        val clipRect = if (isMirrorMode) RectF(0f, 0f, halfWidth, viewHeight) else RectF(halfWidth, 0f, viewWidth, viewHeight)

        canvas.save()
        canvas.clipRect(clipRect)

        val saveCount = canvas.saveLayer(0f, 0f, viewWidth, viewHeight, null)

        gradientPaint.shader = LinearGradient(
            0f, 0f, 0f, viewHeight,
            startColor, endColor, Shader.TileMode.CLAMP
        )
        canvas.drawRect(0f, 0f, viewWidth, viewHeight, gradientPaint)

        try {
            if (!mask.isRecycled) {
                canvas.drawBitmap(mask, null, drawRect, maskPaint)
            }
        } catch (e: Exception) {}

        canvas.restoreToCount(saveCount)
        canvas.restore()
    }
}