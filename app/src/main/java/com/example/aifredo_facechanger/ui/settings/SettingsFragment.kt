package com.example.aifredo_facechanger.ui.settings

import android.content.Context
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import com.example.aifredo_facechanger.R
import com.example.aifredo_facechanger.databinding.FragmentSettingsBinding
import com.google.android.material.tabs.TabLayout

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
        
        setupTabLayout()
        
        // Restore last selected tab
        val lastTab = sharedPref.getInt("last_settings_tab", 0)
        binding.tabLayout.getTabAt(lastTab)?.select()
        updateTabVisibility(lastTab)

        // --- Face Settings ---
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
            sharedPref.edit().putString("selected_model", selectedModel).apply()
        }

        val faceDelegate = sharedPref.getString("face_delegate", "CPU")
        if (faceDelegate == "GPU") binding.radioFaceGpu.isChecked = true else binding.radioFaceCpu.isChecked = true
        binding.radioGroupFaceDelegate.setOnCheckedChangeListener { _, checkedId ->
            val selected = if (checkedId == R.id.radio_face_gpu) "GPU" else "CPU"
            sharedPref.edit().putString("face_delegate", selected).apply()
        }

        val poseDelegate = sharedPref.getString("pose_delegate", "CPU")
        if (poseDelegate == "GPU") binding.radioPoseGpu.isChecked = true else binding.radioPoseCpu.isChecked = true
        binding.radioGroupPoseDelegate.setOnCheckedChangeListener { _, checkedId ->
            val selected = if (checkedId == R.id.radio_pose_gpu) "GPU" else "CPU"
            sharedPref.edit().putString("pose_delegate", selected).apply()
        }

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
            sharedPref.edit().putString("render_mode", selectedMode).apply()
        }

        binding.switchUseFaceLandmark.isChecked = sharedPref.getBoolean("use_face_landmark", true)
        binding.switchUseFaceLandmark.setOnCheckedChangeListener { _, isChecked ->
            sharedPref.edit().putBoolean("use_face_landmark", isChecked).apply()
        }

        // --- Body Settings ---
        val currentBodyModel = sharedPref.getString("body_model", "MediaPipe Pose")
        when (currentBodyModel) {
            "MediaPipe Pose" -> binding.radioBodyMediapipePose.isChecked = true
            "ML Kit" -> binding.radioBodyMlkit.isChecked = true
            "YOLACT" -> binding.radioBodyYolact.isChecked = true
            "MODNet" -> binding.radioBodyModnet.isChecked = true
            else -> binding.radioBodyMediapipePose.isChecked = true
        }

        binding.radioGroupBodyModel.setOnCheckedChangeListener { _, checkedId ->
            val selected = when (checkedId) {
                R.id.radio_body_mediapipe_pose -> "MediaPipe Pose"
                R.id.radio_body_mlkit -> "ML Kit"
                R.id.radio_body_yolact -> "YOLACT"
                R.id.radio_body_modnet -> "MODNet"
                else -> "MediaPipe Pose"
            }
            sharedPref.edit().putString("body_model", selected).apply()
        }

        val bodyDelegate = sharedPref.getString("body_delegate", "CPU")
        when (bodyDelegate) {
            "GPU" -> binding.radioBodyGpu.isChecked = true
            "NNAPI" -> binding.radioBodyNnapi.isChecked = true
            else -> binding.radioBodyCpu.isChecked = true
        }
        binding.radioGroupBodyDelegate.setOnCheckedChangeListener { _, checkedId ->
            val selected = when (checkedId) {
                R.id.radio_body_gpu -> "GPU"
                R.id.radio_body_nnapi -> "NNAPI"
                else -> "CPU"
            }
            sharedPref.edit().putString("body_delegate", selected).apply()
        }

        binding.editBodyStartColor.setText(sharedPref.getString("body_start_color", "#FF0000"))
        binding.editBodyEndColor.setText(sharedPref.getString("body_end_color", "#0000FF"))
    }

    private fun setupTabLayout() {
        binding.tabLayout.addOnTabSelectedListener(object : TabLayout.OnTabSelectedListener {
            override fun onTabSelected(tab: TabLayout.Tab?) {
                val position = tab?.position ?: 0
                updateTabVisibility(position)
                activity?.getSharedPreferences("AIfredoPrefs", Context.MODE_PRIVATE)
                    ?.edit()?.putInt("last_settings_tab", position)?.apply()
            }
            override fun onTabUnselected(tab: TabLayout.Tab?) {}
            override fun onTabReselected(tab: TabLayout.Tab?) {}
        })
    }

    private fun updateTabVisibility(position: Int) {
        when (position) {
            0 -> {
                binding.layoutFaceSettings.visibility = View.VISIBLE
                binding.layoutBodySettings.visibility = View.GONE
            }
            1 -> {
                binding.layoutFaceSettings.visibility = View.GONE
                binding.layoutBodySettings.visibility = View.VISIBLE
            }
        }
    }

    override fun onPause() {
        super.onPause()
        val sharedPref = activity?.getSharedPreferences("AIfredoPrefs", Context.MODE_PRIVATE) ?: return
        sharedPref.edit().apply {
            putString("body_start_color", binding.editBodyStartColor.text.toString())
            putString("body_end_color", binding.editBodyEndColor.text.toString())
            apply()
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}