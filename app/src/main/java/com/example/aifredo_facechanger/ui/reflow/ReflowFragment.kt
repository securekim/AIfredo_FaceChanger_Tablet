package com.example.aifredo_facechanger.ui.reflow

import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.fragment.app.Fragment
import androidx.lifecycle.ViewModelProvider
import com.example.aifredo_facechanger.databinding.FragmentReflowBinding
import java.util.*

class ReflowFragment : Fragment() {

    private var _binding: FragmentReflowBinding? = null
    private val binding get() = _binding!!

    // Performance Monitoring
    private var maxMem = 0L
    private var maxCpu = 0.0
    private var lastCpuTime = 0L
    private var lastSampleTime = 0L
    private val perfHandler = Handler(Looper.getMainLooper())
    private val perfRunnable = object : Runnable {
        override fun run() {
            updatePerformanceMetrics()
            perfHandler.postDelayed(this, 1000)
        }
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        val reflowViewModel =
            ViewModelProvider(this).get(ReflowViewModel::class.java)

        _binding = FragmentReflowBinding.inflate(inflater, container, false)
        val root: View = binding.root

        val textView: TextView = binding.textReflow
        reflowViewModel.text.observe(viewLifecycleOwner) {
            textView.text = it
        }
        return root
    }

    override fun onResume() {
        super.onResume()
        perfHandler.post(perfRunnable)
    }

    override fun onPause() {
        super.onPause()
        perfHandler.removeCallbacks(perfRunnable)
    }

    private fun updatePerformanceMetrics() {
        if (_binding == null) return
        val runtime = Runtime.getRuntime()
        val usedMem = (runtime.totalMemory() - runtime.freeMemory()) / 1024 / 1024
        if (usedMem > maxMem) maxMem = usedMem

        val currentCpuTime = android.os.Process.getElapsedCpuTime()
        val currentTime = System.currentTimeMillis()
        var cpuUsage = 0.0
        if (lastSampleTime > 0) {
            val cpuDiff = currentCpuTime - lastCpuTime
            val timeDiff = currentTime - lastSampleTime
            if (timeDiff > 0) {
                cpuUsage = (cpuDiff.toDouble() / timeDiff.toDouble() / Runtime.getRuntime().availableProcessors()) * 100.0
                if (cpuUsage > maxCpu) maxCpu = cpuUsage
            }
        }
        lastCpuTime = currentCpuTime
        lastSampleTime = currentTime

        binding.perfText.text = String.format(
            Locale.getDefault(),
            "CPU: %.1f%% (Peak: %.1f%%)\nMEM: %dMB (Peak: %dMB)",
            cpuUsage, maxCpu, usedMem, maxMem
        )
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}