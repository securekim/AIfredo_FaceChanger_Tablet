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
        
        // Right side only: from width/2 to width
        val halfWidth = viewWidth / 2f
        val destRect = RectF(halfWidth, 0f, viewWidth, viewHeight)
        
        // We only want to draw the mask/gradient on the right side.
        // We need to clip the canvas to the right half.
        canvas.save()
        canvas.clipRect(halfWidth, 0f, viewWidth, viewHeight)

        // Save layer for masking within the clipped area
        val saveCount = canvas.saveLayer(halfWidth, 0f, viewWidth, viewHeight, null)

        // Draw Gradient over the entire view (it will be clipped)
        gradientPaint.shader = LinearGradient(
            halfWidth, 0f, halfWidth, viewHeight,
            startColor, endColor, Shader.TileMode.CLAMP
        )
        canvas.drawRect(destRect, gradientPaint)

        // Apply Mask (Segmentation Result)
        // The mask defines where the gradient should be visible (the person)
        // We use destRect so it stretches to cover the right half. 
        // Note: The mask itself represents the full person, so stretching it to half might look weird if not handled.
        // Usually, the mask matches the aspect ratio of the camera.
        // If the camera is full screen, the mask is full screen. 
        // We should draw the FULL mask but it will be clipped to the right half.
        val fullRect = RectF(0f, 0f, viewWidth, viewHeight)
        canvas.drawBitmap(mask, null, fullRect, maskPaint)

        canvas.restoreToCount(saveCount)
        canvas.restore()
    }
}