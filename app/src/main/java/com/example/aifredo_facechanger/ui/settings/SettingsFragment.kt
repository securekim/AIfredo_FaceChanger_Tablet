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
        val currentModel = sharedPref.getString("selected_model", "AnimeGAN_Hayao")
        when (currentModel) {
            "AnimeGAN_Hayao" -> binding.radioAnimeganHayao.isChecked = true
            "AnimeGAN_Paprika" -> binding.radioAnimeganPaprika.isChecked = true
            "MediaPipe_Default" -> binding.radioMediapipeDefault.isChecked = true
            "CartoonGAN_Default" -> binding.radioCartoonganDefault.isChecked = true
            "SEMI_Filter" -> binding.radioSemiFilter.isChecked = true
            else -> binding.radioAnimeganHayao.isChecked = true
        }

        binding.radioGroupModel.setOnCheckedChangeListener { _, checkedId ->
            val selectedModel = when (checkedId) {
                R.id.radio_animegan_hayao -> "AnimeGAN_Hayao"
                R.id.radio_animegan_paprika -> "AnimeGAN_Paprika"
                R.id.radio_mediapipe_default -> "MediaPipe_Default"
                R.id.radio_cartoongan_default -> "CartoonGAN_Default"
                R.id.radio_semi_filter -> "SEMI_Filter"
                else -> "AnimeGAN_Hayao"
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

        // --- Resolution Selection ---
        val currentResolution = sharedPref.getInt("model_resolution", 256)
        when (currentResolution) {
            256 -> binding.radioRes256.isChecked = true
            512 -> binding.radioRes512.isChecked = true
            else -> binding.radioRes256.isChecked = true
        }

        binding.radioGroupResolution.setOnCheckedChangeListener { _, checkedId ->
            val selectedRes = when (checkedId) {
                R.id.radio_res_256 -> 256
                R.id.radio_res_512 -> 512
                else -> 256
            }
            with(sharedPref.edit()) {
                putInt("model_resolution", selectedRes)
                apply()
            }
        }

        // --- Landmark Options ---
        val useFaceLandmark = sharedPref.getBoolean("use_face_landmark", true)
        binding.switchUseFaceLandmark.isChecked = useFaceLandmark
        binding.switchUseFaceLandmark.setOnCheckedChangeListener { _, isChecked ->
            with(sharedPref.edit()) {
                putBoolean("use_face_landmark", isChecked)
                apply()
            }
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}