package com.example.aifredo_facechanger

import android.content.Context
import android.os.Bundle
import android.view.Menu
import android.view.MenuItem
import androidx.appcompat.app.AppCompatActivity
import androidx.navigation.findNavController
import androidx.navigation.fragment.NavHostFragment
import androidx.navigation.navOptions
import androidx.navigation.ui.AppBarConfiguration
import androidx.navigation.ui.navigateUp
import androidx.navigation.ui.onNavDestinationSelected
import androidx.navigation.ui.setupActionBarWithNavController
import androidx.navigation.ui.setupWithNavController
import com.example.aifredo_facechanger.databinding.ActivityMainBinding

class MainActivity : AppCompatActivity() {

    private lateinit var appBarConfiguration: AppBarConfiguration
    private lateinit var binding: ActivityMainBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        setSupportActionBar(binding.appBarMain.toolbar)

        val navHostFragment =
            supportFragmentManager.findFragmentById(R.id.nav_host_fragment_content_main) as NavHostFragment
        val navController = navHostFragment.navController

        // 상단 바에서 햄버거 메뉴가 나타날 최상위 목적지들 정의
        val topLevelDestinations = setOf(
            R.id.nav_transform, R.id.nav_body_changer, R.id.nav_reflow, 
            R.id.nav_slideshow, R.id.nav_voice, R.id.nav_settings
        )
        
        appBarConfiguration = AppBarConfiguration(topLevelDestinations, binding.drawerLayout)
        setupActionBarWithNavController(navController, appBarConfiguration)

        // Drawer 및 Bottom Navigation 연결 (ID가 nav graph와 일치해야 함)
        binding.navView?.setupWithNavController(navController)
        binding.appBarMain.contentMain.bottomNavView?.setupWithNavController(navController)

        // 마지막으로 사용했던 메뉴 복원 로직 개선
        val sharedPref = getSharedPreferences("AIfredoPrefs", Context.MODE_PRIVATE)
        val lastNavName = sharedPref.getString("last_nav_name", null)
        val lastNavId = if (lastNavName != null) {
            resources.getIdentifier(lastNavName, "id", packageName)
        } else {
            sharedPref.getInt("last_nav_id", R.id.nav_transform)
        }
        
        binding.root.post {
            if (navController.currentDestination?.id != lastNavId && lastNavId != 0) {
                try {
                    if (navController.graph.findNode(lastNavId) != null) {
                        // popUpTo를 사용하여 백스택이 쌓이는 것을 방지
                        navController.navigate(lastNavId, null, navOptions {
                            popUpTo(navController.graph.startDestinationId) {
                                saveState = true
                            }
                            launchSingleTop = true
                            restoreState = true
                        })
                    }
                } catch (e: Exception) {
                    // 목적지가 유효하지 않을 경우 무시
                }
            }
        }

        navController.addOnDestinationChangedListener { _, destination, _ ->
            try {
                val entryName = resources.getResourceEntryName(destination.id)
                sharedPref.edit()
                    .putInt("last_nav_id", destination.id)
                    .putString("last_nav_name", entryName)
                    .apply()
            } catch (e: Exception) {
                sharedPref.edit().putInt("last_nav_id", destination.id).apply()
            }
        }
    }

    override fun onCreateOptionsMenu(menu: Menu): Boolean {
        // Drawer가 없는 레이아웃 구성일 때만 overflow 메뉴 인플레이트
        if (binding.navView == null) {
            menuInflater.inflate(R.menu.overflow, menu)
        }
        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        val navController = findNavController(R.id.nav_host_fragment_content_main)
        // onNavDestinationSelected를 사용하여 메뉴 ID와 목적지 ID가 같으면 자동 이동
        return item.onNavDestinationSelected(navController) || super.onOptionsItemSelected(item)
    }

    override fun onSupportNavigateUp(): Boolean {
        val navController = findNavController(R.id.nav_host_fragment_content_main)
        return navController.navigateUp(appBarConfiguration) || super.onSupportNavigateUp()
    }
}
