package com.example.aifredo_facechanger.utils

import kotlin.math.abs

/**
 * OneEuroFilter implementation to reduce jitter in landmark tracking.
 * This is tuned for very slight and weak filtering as requested.
 */
class OneEuroFilter(
    private val minCutoff: Double = 1.0,
    private val beta: Double = 0.01,
    private val dCutoff: Double = 1.0
) {
    private var xPrev: Double? = null
    private var dxPrev: Double = 0.0
    private var tPrev: Long? = null

    fun filter(value: Double, timestamp: Long): Double {
        if (tPrev == null || xPrev == null) {
            tPrev = timestamp
            xPrev = value
            return value
        }

        val dt = (timestamp - tPrev!!).toDouble() / 1000.0
        if (dt <= 0) return xPrev!!

        val dx = (value - xPrev!!) / dt
        val edx = lowPassFilter(dx, dxPrev, alpha(dt, dCutoff))
        dxPrev = edx

        val cutoff = minCutoff + beta * abs(edx)
        val x = lowPassFilter(value, xPrev!!, alpha(dt, cutoff))
        
        xPrev = x
        tPrev = timestamp
        
        return x
    }

    private fun alpha(dt: Double, cutoff: Double): Double {
        val tau = 1.0 / (2.0 * Math.PI * cutoff)
        return 1.0 / (1.0 + tau / dt)
    }

    private fun lowPassFilter(value: Double, prevValue: Double, alpha: Double): Double {
        return alpha * value + (1.0 - alpha) * prevValue
    }
}
