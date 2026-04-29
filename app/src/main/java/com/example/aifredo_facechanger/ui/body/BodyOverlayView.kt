package com.example.aifredo_facechanger.ui.body

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import java.util.concurrent.ConcurrentLinkedQueue

/**
 * RVM 및 기타 모델의 결과를 원본 프레임과 함께 그리는 커스텀 뷰.
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
    private var isHalfBoundary: Boolean = true
    private var boundingBox: RectF? = null

    private val gradientPaint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val backgroundPaint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 8f
    }

    private var lastViewHeight = -1f
    private var lastStartColor = 0
    private var lastEndColor = 0

    // [핵심 변경] 사용이 끝난 비트맵을 즉시 날리지 않고, 그리기(onDraw)가 끝날 때까지 대기시키는 안전 큐
    private val recycleQueue = ConcurrentLinkedQueue<Bitmap>()

    fun updateData(mask: Bitmap?, original: Bitmap?, startCol: Int, endCol: Int, isMirror: Boolean = false, isHalf: Boolean = true) {
        // 새 데이터가 들어오면, 기존 데이터는 재활용 큐로 밀어넣음
        if (maskBitmap != null && maskBitmap != mask) {
            recycleQueue.add(maskBitmap)
        }
        if (backgroundBitmap != null && backgroundBitmap != original) {
            recycleQueue.add(backgroundBitmap)
        }

        maskBitmap = mask
        backgroundBitmap = original
        boundingBox = null // Clear box if drawing mask

        this.startColor = startCol
        this.endColor = endCol
        this.isMirrorMode = isMirror
        this.isHalfBoundary = isHalf

        postInvalidate()
    }

    fun updateBoundingBox(box: RectF?, original: Bitmap?, isMirror: Boolean = false, isHalf: Boolean = true) {
        if (backgroundBitmap != null && backgroundBitmap != original) {
            recycleQueue.add(backgroundBitmap)
        }
        backgroundBitmap = original
        boundingBox = box
        this.isMirrorMode = isMirror
        this.isHalfBoundary = isHalf
        postInvalidate()
    }

    fun updateMaskOnly(mask: Bitmap?, startCol: Int, endCol: Int, isMirror: Boolean = false, isHalf: Boolean = true) {
        updateData(mask, null, startCol, endCol, isMirror, isHalf)
    }

    fun clearMemory() {
        if (maskBitmap != null) recycleQueue.add(maskBitmap)
        if (backgroundBitmap != null) recycleQueue.add(backgroundBitmap)
        maskBitmap = null
        backgroundBitmap = null
        boundingBox = null
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val bg = backgroundBitmap
        val mask = maskBitmap
        val box = boundingBox

        val vw = width.toFloat()
        val vh = height.toFloat()
        if (vw <= 0 || vh <= 0) return

        // 1. 원본 배경 그리기
        if (bg != null && !bg.isRecycled) {
            drawScaledBitmap(canvas, bg, vw, vh, backgroundPaint)
        }

        // 2. 마스크 그리기 (그라데이션)
        if (mask != null && !mask.isRecycled) {
            if (vh != lastViewHeight || startColor != lastStartColor || endColor != lastEndColor) {
                gradientPaint.shader = LinearGradient(0f, 0f, 0f, vh, startColor, endColor, Shader.TileMode.CLAMP)
                lastViewHeight = vh
                lastStartColor = startColor
                lastEndColor = endColor
            }

            canvas.save()

            if (isHalfBoundary) {
                val halfWidth = vw / 2f
                if (isMirrorMode) {
                    canvas.clipRect(0f, 0f, halfWidth, vh)
                } else {
                    canvas.clipRect(halfWidth, 0f, vw, vh)
                }
            }

            drawScaledBitmap(canvas, mask, vw, vh, gradientPaint)
            canvas.restore()
        }

        // 3. 바운딩 박스 그리기
        if (box != null && bg != null && !bg.isRecycled) {
            canvas.save()
            if (isHalfBoundary) {
                val halfWidth = vw / 2f
                if (isMirrorMode) {
                    canvas.clipRect(0f, 0f, halfWidth, vh)
                } else {
                    canvas.clipRect(halfWidth, 0f, vw, vh)
                }
            }
            
            val bw = bg.width.toFloat()
            val bh = bg.height.toFloat()
            val scale: Float
            val dx: Float
            val dy: Float
            val viewRatio = vw / vh
            val bitmapRatio = bw / bh

            if (bitmapRatio > viewRatio) {
                scale = vh / bh
                dx = (vw - bw * scale) / 2f
                dy = 0f
            } else {
                scale = vw / bw
                dx = 0f
                dy = (vh - bh * scale) / 2f
            }

            val drawBox = RectF(
                box.left * scale + dx,
                box.top * scale + dy,
                box.right * scale + dx,
                box.bottom * scale + dy
            )
            canvas.drawRect(drawBox, boxPaint)
            canvas.restore()
        }

        // [핵심 변경] 그리기(onDraw)가 완전히 끝난 직후, 대기 중이던 옛날 비트맵들을 안전하게 파기
        while (recycleQueue.isNotEmpty()) {
            val bmp = recycleQueue.poll()
            if (bmp != null && !bmp.isRecycled && bmp != maskBitmap && bmp != backgroundBitmap) {
                bmp.recycle()
            }
        }
    }

    private fun drawScaledBitmap(canvas: Canvas, bitmap: Bitmap, vw: Float, vh: Float, paint: Paint) {
        val bw = bitmap.width.toFloat()
        val bh = bitmap.height.toFloat()

        val scale: Float
        val dx: Float
        val dy: Float

        val viewRatio = vw / vh
        val bitmapRatio = bw / bh

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