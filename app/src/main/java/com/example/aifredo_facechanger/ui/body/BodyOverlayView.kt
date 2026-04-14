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

    private val paint = Paint(Paint.ANTI_ALIAS_FLAG)
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

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val mask = maskBitmap ?: return

        val viewWidth = width.toFloat()
        val viewHeight = height.toFloat()
        val destRect = RectF(0f, 0f, viewWidth, viewHeight)

        // We don't draw the original bitmap here because PreviewView is already showing it.
        // This view acts as a transparent overlay that only draws the segmented effect.

        // Save layer for masking
        val saveCount = canvas.saveLayer(0f, 0f, viewWidth, viewHeight, null)

        // Draw Gradient over the entire view
        gradientPaint.shader = LinearGradient(
            0f, 0f, 0f, viewHeight,
            startColor, endColor, Shader.TileMode.CLAMP
        )
        canvas.drawRect(destRect, gradientPaint)

        // Apply Mask (Segmentation Result)
        // The mask defines where the gradient should be visible (the person)
        canvas.drawBitmap(mask, null, destRect, maskPaint)

        canvas.restoreToCount(saveCount)
    }
}