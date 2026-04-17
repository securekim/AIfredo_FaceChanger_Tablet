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
    private var originalBitmap: Bitmap? = null
    private var startColor: Int = Color.RED
    private var endColor: Int = Color.BLUE

    private val maskPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        // Keeps the destination (gradient) where the source (mask) is present
        xfermode = PorterDuffXfermode(PorterDuff.Mode.DST_IN)
    }
    private val gradientPaint = Paint(Paint.ANTI_ALIAS_FLAG)

    fun updateData(mask: Bitmap?, original: Bitmap?, startCol: Int, endCol: Int) {
        maskBitmap = mask
        originalBitmap = original
        startColor = startCol
        endColor = endCol
        postInvalidate()
    }

    fun updateMaskOnly(mask: Bitmap?, startCol: Int, endCol: Int) {
        maskBitmap = mask
        startColor = startCol
        endColor = endCol
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val mask = maskBitmap ?: return

        val viewWidth = width.toFloat()
        val viewHeight = height.toFloat()
        if (viewWidth <= 0 || viewHeight <= 0) return

        val maskWidth = mask.width.toFloat()
        val maskHeight = mask.height.toFloat()

        // Calculate aspect ratio scaling to match PreviewView's default FILL_CENTER
        val scale: Float
        val dx: Float
        val dy: Float

        val viewRatio = viewWidth / viewHeight
        val maskRatio = maskWidth / maskHeight

        if (maskRatio > viewRatio) {
            // Mask is wider than view (relatively) -> match height, crop width
            scale = viewHeight / maskHeight
            dx = (viewWidth - maskWidth * scale) / 2f
            dy = 0f
        } else {
            // View is wider than mask (relatively) -> match width, crop height
            scale = viewWidth / maskWidth
            dx = 0f
            dy = (viewHeight - maskHeight * scale) / 2f
        }

        val drawRect = RectF(dx, dy, dx + maskWidth * scale, dy + maskHeight * scale)

        // Split screen: Only draw on the right half
        val halfWidth = viewWidth / 2f
        canvas.save()
        canvas.clipRect(halfWidth, 0f, viewWidth, viewHeight)

        // Use a layer to apply the PorterDuff xfermode correctly
        val saveCount = canvas.saveLayer(0f, 0f, viewWidth, viewHeight, null)

        // 1. Draw Gradient (Destination)
        gradientPaint.shader = LinearGradient(
            0f, 0f, 0f, viewHeight,
            startColor, endColor, Shader.TileMode.CLAMP
        )
        // We draw the gradient over the whole view (it's clipped by the canvas anyway)
        canvas.drawRect(0f, 0f, viewWidth, viewHeight, gradientPaint)

        // 2. Draw Mask (Source) with DST_IN
        // This will keep the gradient only where the mask has alpha (the person)
        // Drawing it on drawRect ensures it aligns with the background camera preview
        canvas.drawBitmap(mask, null, drawRect, maskPaint)

        canvas.restoreToCount(saveCount)
        canvas.restore()
    }
}