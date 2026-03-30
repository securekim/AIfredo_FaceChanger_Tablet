package com.example.aifredo_facechanger.ui.settings

import android.content.Context
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import com.example.aifredo_facechanger.R
import com.example.aifredo_facechanger.databinding.FragmentSettingsBinding

class SettingsFragment : Fragment() {

    private var _binding: FragmentSettingsBinding? = null
    private val binding get() = _binding!!

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentSettingsBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        
        val sharedPref = activity?.getSharedPreferences("AIfredoPrefs", Context.MODE_PRIVATE) ?: return
        
        // --- Model Selection ---
        val currentModel = sharedPref.getString("selected_model", "MediaPipe_Default")
        when (currentModel) {
            "MediaPipe_Default" -> binding.radioMediapipeDefault.isChecked = true
            "MediaPipe_AIfredo" -> binding.radioMediapipeAifredo.isChecked = true
            "CartoonGAN_Default" -> binding.radioCartoonganDefault.isChecked = true
            else -> binding.radioMediapipeDefault.isChecked = true
        }

        binding.radioGroupModel.setOnCheckedChangeListener { _, checkedId ->
            val selectedModel = when (checkedId) {
                R.id.radio_mediapipe_default -> "MediaPipe_Default"
                R.id.radio_mediapipe_aifredo -> "MediaPipe_AIfredo"
                R.id.radio_cartoongan_default -> "CartoonGAN_Default"
                else -> "MediaPipe_Default"
            }
            with(sharedPref.edit()) {
                putString("selected_model", selectedModel)
                apply()
            }
        }

        // --- Rendering Mode Selection ---
        val currentRenderMode = sharedPref.getString("render_mode", "Face_Only")
        when (currentRenderMode) {
            "Face_Only" -> binding.radioFaceOnly.isChecked = true
            "Full_Frame" -> binding.radioFullFrame.isChecked = true
            else -> binding.radioFaceOnly.isChecked = true
        }

        binding.radioGroupRenderMode.setOnCheckedChangeListener { _, checkedId ->
            val selectedMode = when (checkedId) {
                R.id.radio_face_only -> "Face_Only"
                R.id.radio_full_frame -> "Full_Frame"
                else -> "Face_Only"
            }
            with(sharedPref.edit()) {
                putString("render_mode", selectedMode)
                apply()
            }
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}