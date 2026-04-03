package com.example.aifredo_facechanger.ui.transform

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import com.example.aifredo_facechanger.utils.OneEuroFilter
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import kotlin.math.max
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin

class FaceOverlayView(context: Context, attrs: AttributeSet?) : View(context, attrs) {

    private var originalBitmap: Bitmap? = null
    
    // Cross-fade state for stylized overlay
    private var currentStylizedBitmap: Bitmap? = null
    private var currentStylizedCenter = PointF(0f, 0f)
    private var currentStylizedSize = 0f
    
    private var prevStylizedBitmap: Bitmap? = null
    private var prevStylizedCenter = PointF(0f, 0f)
    private var prevStylizedSize = 0f
    
    private var transitionStartTime = 0L
    private val FADE_DURATION = 100f // 0.1s in ms

    private var shapeProgress = 0f
    
    private var currentFaceResult: FaceLandmarkerResult? = null
    private var lastValidFaceResult: FaceLandmarkerResult? = null
    private var currentPoseResult: PoseLandmarkerResult? = null
    private var isFaceActive: Boolean = false

    private val poseFiltersX = mutableMapOf<Int, OneEuroFilter>()
    private val poseFiltersY = mutableMapOf<Int, OneEuroFilter>()
    private var filteredPoseLandmarks: List<PointF>? = null

    private val paint = Paint().apply { isAntiAlias = true; isFilterBitmap = true }
    
    private val facePointPaint = Paint().apply { 
        color = Color.GREEN
        style = Paint.Style.FILL
        alpha = 180
    }
    
    private val posePointPaint = Paint().apply { 
        color = Color.CYAN
        style = Paint.Style.FILL
        alpha = 200
    }
    
    private val facePath = Path()
    private val stylizedPaint = Paint().apply {
        isAntiAlias = true
        isFilterBitmap = true
    }

    fun updateFrame(
        original: Bitmap, 
        stylized: Bitmap?, 
        sCenter: PointF,
        sSize: Float,
        curFace: FaceLandmarkerResult?, 
        curPose: PoseLandmarkerResult?, 
        mode: String,
        isFaceActive: Boolean,
        shapeProgress: Float = 0f
    ) {
        this.originalBitmap = original
        
        // Detect new stylized bitmap and trigger cross-fade
        if (stylized != null && stylized !== this.currentStylizedBitmap) {
            if (this.currentStylizedBitmap != null) {
                // If we already have a previous one still fading, recycle it first
                prevStylizedBitmap?.let { if (!it.isRecycled) it.recycle() }
                
                prevStylizedBitmap = this.currentStylizedBitmap
                prevStylizedCenter.set(this.currentStylizedCenter)
                prevStylizedSize = this.currentStylizedSize
                transitionStartTime = System.currentTimeMillis()
            }
            
            this.currentStylizedBitmap = stylized
            this.currentStylizedCenter.set(sCenter)
            this.currentStylizedSize = sSize
            
            if (prevStylizedBitmap == null) transitionStartTime = 0L
        } else if (stylized == null) {
            // Optional: Handle fade out if needed, but for now just clear
            currentStylizedBitmap = null
        } else {
            // Even if bitmap is the same instance, update its position/size (interpolation)
            this.currentStylizedCenter.set(sCenter)
            this.currentStylizedSize = sSize
        }

        this.currentFaceResult = curFace
        if (curFace?.faceLandmarks()?.isNotEmpty() == true) {
            this.lastValidFaceResult = curFace
        }
        this.currentPoseResult = curPose
        this.isFaceActive = isFaceActive
        this.shapeProgress = shapeProgress

        applyPoseFilter(curPose)
        invalidate()
    }

    private fun applyPoseFilter(result: PoseLandmarkerResult?) {
        val landmarks = result?.landmarks()?.getOrNull(0) ?: run { filteredPoseLandmarks = null; return }
        val timestamp = System.currentTimeMillis()
        filteredPoseLandmarks = landmarks.mapIndexed { index, landmark ->
            val filterX = poseFiltersX.getOrPut(index) { OneEuroFilter(minCutoff = 5.0, beta = 0.5) }
            val filterY = poseFiltersY.getOrPut(index) { OneEuroFilter(minCutoff = 5.0, beta = 0.5) }
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
        
        filteredPoseLandmarks?.forEach { 
            canvas.drawCircle(offsetX + it.x * drawW, offsetY + it.y * drawH, 6f, posePointPaint) 
        }

        if (isFaceActive) {
            currentFaceResult?.faceLandmarks()?.getOrNull(0)?.forEach { 
                canvas.drawCircle(offsetX + it.x() * drawW, offsetY + it.y() * drawH, 3f, facePointPaint) 
            }
        }
        canvas.restore()

        // 2. 우측 합성 결과 렌더링
        canvas.save()
        canvas.clipRect(midX, 0f, width.toFloat(), height.toFloat())
        canvas.drawBitmap(bitmap, null, RectF(offX, offsetY, offX + drawW, offsetY + drawH), paint)

        val currentTime = System.currentTimeMillis()
        val elapsed = currentTime - transitionStartTime
        val alpha = if (transitionStartTime == 0L || FADE_DURATION <= 0) 1f else (elapsed / FADE_DURATION).coerceIn(0f, 1f)

        // Draw Previous (Fading out)
        if (alpha < 1f && prevStylizedBitmap != null && !prevStylizedBitmap!!.isRecycled) {
            drawStylizedArea(canvas, prevStylizedBitmap!!, prevStylizedCenter, prevStylizedSize, 1f - alpha, scale, offX, offsetY, drawW, drawH)
        }

        // Draw Current (Fading in)
        if (currentStylizedBitmap != null && !currentStylizedBitmap!!.isRecycled && currentStylizedSize > 0) {
            drawStylizedArea(canvas, currentStylizedBitmap!!, currentStylizedCenter, currentStylizedSize, alpha, scale, offX, offsetY, drawW, drawH)
        }

        // Cleanup
        if (alpha >= 1f && prevStylizedBitmap != null) {
            prevStylizedBitmap?.let { if (!it.isRecycled) it.recycle() }
            prevStylizedBitmap = null
        }

        // Keep animating if transition is active
        if (alpha < 1f) invalidate()

        canvas.restore()
        
        canvas.drawLine(midX, 0f, midX, height.toFloat(), Paint().apply { color = Color.WHITE; strokeWidth = 4f })
    }

    private fun drawStylizedArea(
        canvas: Canvas, 
        stylized: Bitmap, 
        center: PointF, 
        size: Float, 
        alpha: Float,
        scale: Float, 
        offX: Float, 
        offsetY: Float, 
        drawW: Float, 
        drawH: Float
    ) {
        val sCenterX = center.x * scale
        val sCenterY = center.y * scale
        val sSizePx = size * scale
        
        val centerX = offX + sCenterX
        val centerY = offsetY + sCenterY

        facePath.reset()
        val landmarks = (currentFaceResult ?: lastValidFaceResult)?.faceLandmarks()?.getOrNull(0)

        if (landmarks != null) {
            val allPoints = landmarks.map { PointF(offX + it.x() * drawW, offsetY + it.y() * drawH) }
            val hull = getConvexHull(allPoints)
            
            var maxDistSq = 0f
            for (p in hull) {
                val dx = p.x - centerX
                val dy = p.y - centerY
                val distSq = dx * dx + dy * dy
                if (distSq > maxDistSq) maxDistSq = distSq
            }
            val targetRadius = kotlin.math.sqrt(maxDistSq) * 1.05f
            
            if (shapeProgress > 0f) {
                for (i in hull.indices) {
                    val hp = hull[i]
                    val angle = atan2((hp.y - centerY).toDouble(), (hp.x - centerX).toDouble())
                    val initX = centerX + cos(angle).toFloat() * targetRadius
                    val initY = centerY + sin(angle).toFloat() * targetRadius
                    val px = initX + (hp.x - initX) * shapeProgress
                    val py = initY + (hp.y - initY) * shapeProgress
                    if (i == 0) facePath.moveTo(px, py) else facePath.lineTo(px, py)
                }
                facePath.close()
            } else {
                facePath.addCircle(centerX, centerY, targetRadius, Path.Direction.CW)
            }
            
            canvas.save()
            val shader = BitmapShader(stylized, Shader.TileMode.CLAMP, Shader.TileMode.CLAMP)
            val m = Matrix()
            m.postScale(sSizePx / stylized.width, sSizePx / stylized.height)
            m.postTranslate(centerX - sSizePx / 2f, centerY - sSizePx / 2f)
            shader.setLocalMatrix(m)
            stylizedPaint.shader = shader
            stylizedPaint.alpha = (alpha * 255).toInt()
            canvas.drawPath(facePath, stylizedPaint)
            canvas.restore()
            
        } else {
            facePath.addCircle(centerX, centerY, sSizePx * 0.485f, Path.Direction.CW)
            canvas.save()
            canvas.clipPath(facePath)
            val destRect = RectF(centerX - sSizePx/2, centerY - sSizePx/2, centerX + sSizePx/2, centerY + sSizePx/2)
            paint.alpha = (alpha * 255).toInt()
            canvas.drawBitmap(stylized, null, destRect, paint)
            paint.alpha = 255
            canvas.restore()
        }
    }

    private fun getConvexHull(points: List<PointF>): List<PointF> {
        if (points.size <= 2) return points
        val sorted = points.sortedWith(Comparator { p1, p2 ->
            if (p1.x != p2.x) p1.x.compareTo(p2.x) else p1.y.compareTo(p2.y)
        })
        val lower = mutableListOf<PointF>()
        for (p in sorted) {
            while (lower.size >= 2) {
                if (crossProduct(lower[lower.size - 2], lower.last(), p) <= 0) lower.removeAt(lower.size - 1)
                else break
            }
            lower.add(p)
        }
        val upper = mutableListOf<PointF>()
        for (i in sorted.indices.reversed()) {
            val p = sorted[i]
            while (upper.size >= 2) {
                if (crossProduct(upper[upper.size - 2], upper.last(), p) <= 0) upper.removeAt(upper.size - 1)
                else break
            }
            upper.add(p)
        }
        lower.removeAt(lower.size - 1)
        upper.removeAt(upper.size - 1)
        return lower + upper
    }

    private fun crossProduct(a: PointF, b: PointF, c: PointF): Float {
        return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
    }
}
