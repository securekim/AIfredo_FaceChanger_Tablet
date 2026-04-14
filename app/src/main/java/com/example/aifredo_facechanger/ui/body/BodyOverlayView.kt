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
        xfermode = PorterDuffXfermode(PorterDuff.Mode.DST_IN)
    }
    private val gradientPaint = Paint(Paint.ANTI_ALIAS_FLAG)

    fun updateData(mask: Bitmap?, original: Bitmap?, startCol: Int, endCol: Int) {
        maskBitmap = mask
        originalBitmap = original
        startColor = startCol
        endColor = endCol
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val original = originalBitmap ?: return
        val mask = maskBitmap ?: return

        val viewWidth = width.toFloat()
        val viewHeight = height.toFloat()

        // Draw original frame first
        val destRect = RectF(0f, 0f, viewWidth, viewHeight)
        canvas.drawBitmap(original, null, destRect, paint)

        // Create a layer for the right side effect
        val saveCount = canvas.saveLayer(0f, 0f, viewWidth, viewHeight, null)

        // Restrict to right half for the effect
        canvas.clipRect(viewWidth / 2f, 0f, viewWidth, viewHeight)

        // Draw Gradient
        gradientPaint.shader = LinearGradient(
            viewWidth / 2f, 0f, viewWidth, viewHeight,
            startColor, endColor, Shader.TileMode.CLAMP
        )
        canvas.drawRect(viewWidth / 2f, 0f, viewWidth, viewHeight, gradientPaint)

        // Apply Mask (Segmentation Result)
        // The mask should be scaled to match the view
        canvas.drawBitmap(mask, null, destRect, maskPaint)

        canvas.restoreToCount(saveCount)
        
        // Draw divider
        paint.color = Color.WHITE
        paint.strokeWidth = 5f
        canvas.drawLine(viewWidth / 2f, 0f, viewWidth / 2f, viewHeight, paint)
    }
}