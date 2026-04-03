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
    private var stylizedBitmap: Bitmap? = null
    private var stylizedCenter = PointF(0f, 0f)
    private var stylizedSize = 0f
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
        this.stylizedBitmap = stylized
        this.stylizedCenter = sCenter
        this.stylizedSize = sSize
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

        val stylized = stylizedBitmap
        if (stylized != null && !stylized.isRecycled && stylizedSize > 0) {
            val sCenterX = stylizedCenter.x * scale
            val sCenterY = stylizedCenter.y * scale
            val sSizePx = stylizedSize * scale
            
            val centerX = offX + sCenterX
            val centerY = offsetY + sCenterY

            facePath.reset()
            val landmarks = (currentFaceResult ?: lastValidFaceResult)?.faceLandmarks()?.getOrNull(0)

            if (landmarks != null) {
                // Use all landmarks to find the absolute outer boundary (includes nose, chin, forehead, etc.)
                val allPoints = landmarks.map { PointF(offX + it.x() * drawW, offsetY + it.y() * drawH) }
                
                // Get the convex hull of ALL landmarks to ensure no part (like nose) is cut off
                val hull = getConvexHull(allPoints)
                
                // Determine the radius of a circle that would encapsulate this hull
                var maxDistSq = 0f
                for (p in hull) {
                    val dx = p.x - centerX
                    val dy = p.y - centerY
                    val distSq = dx * dx + dy * dy
                    if (distSq > maxDistSq) maxDistSq = distSq
                }
                val targetRadius = kotlin.math.sqrt(maxDistSq) * 1.05f // Slight margin
                
                if (shapeProgress > 0f) {
                    for (i in hull.indices) {
                        val hp = hull[i]
                        // Find corresponding point on the circle for smooth morphing
                        val angle = atan2((hp.y - centerY).toDouble(), (hp.x - centerX).toDouble())
                        val initX = centerX + cos(angle).toFloat() * targetRadius
                        val initY = centerY + sin(angle).toFloat() * targetRadius
                        
                        // Interpolate from Circle -> Convex Hull
                        val px = initX + (hp.x - initX) * shapeProgress
                        val py = initY + (hp.y - initY) * shapeProgress
                        
                        if (i == 0) facePath.moveTo(px, py) else facePath.lineTo(px, py)
                    }
                    facePath.close()
                } else {
                    facePath.addCircle(centerX, centerY, targetRadius, Path.Direction.CW)
                }
                
                // Draw with high quality using BitmapShader
                canvas.save()
                val shader = BitmapShader(stylized, Shader.TileMode.CLAMP, Shader.TileMode.CLAMP)
                val m = Matrix()
                m.postScale(sSizePx / stylized.width, sSizePx / stylized.height)
                m.postTranslate(centerX - sSizePx / 2f, centerY - sSizePx / 2f)
                shader.setLocalMatrix(m)
                stylizedPaint.shader = shader
                canvas.drawPath(facePath, stylizedPaint)
                canvas.restore()
                
            } else {
                facePath.addCircle(centerX, centerY, sSizePx * 0.485f, Path.Direction.CW)
                canvas.save()
                canvas.clipPath(facePath)
                val destRect = RectF(centerX - sSizePx/2, centerY - sSizePx/2, centerX + sSizePx/2, centerY + sSizePx/2)
                canvas.drawBitmap(stylized, null, destRect, paint)
                canvas.restore()
            }
        }
        canvas.restore()
        
        canvas.drawLine(midX, 0f, midX, height.toFloat(), Paint().apply { color = Color.WHITE; strokeWidth = 4f })
    }

    private fun getConvexHull(points: List<PointF>): List<PointF> {
        if (points.size <= 2) return points
        
        // Sort points by X then Y
        val sorted = points.sortedWith(Comparator { p1, p2 ->
            if (p1.x != p2.x) p1.x.compareTo(p2.x) else p1.y.compareTo(p2.y)
        })

        val lower = mutableListOf<PointF>()
        for (p in sorted) {
            while (lower.size >= 2) {
                if (crossProduct(lower[lower.size - 2], lower.last(), p) <= 0) {
                    lower.removeAt(lower.size - 1)
                } else break
            }
            lower.add(p)
        }

        val upper = mutableListOf<PointF>()
        for (i in sorted.indices.reversed()) {
            val p = sorted[i]
            while (upper.size >= 2) {
                if (crossProduct(upper[upper.size - 2], upper.last(), p) <= 0) {
                    upper.removeAt(upper.size - 1)
                } else break
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
