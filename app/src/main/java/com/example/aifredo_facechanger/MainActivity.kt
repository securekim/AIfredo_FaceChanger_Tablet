package com.example.aifredo_facechanger

import android.content.Context
import android.os.Bundle
import android.view.Menu
import android.view.MenuItem
import com.google.android.material.snackbar.Snackbar
import com.google.android.material.navigation.NavigationView
import androidx.navigation.findNavController
import androidx.navigation.fragment.NavHostFragment
import androidx.navigation.ui.AppBarConfiguration
import androidx.navigation.ui.navigateUp
import androidx.navigation.ui.setupActionBarWithNavController
import androidx.navigation.ui.setupWithNavController
import androidx.appcompat.app.AppCompatActivity
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
            (supportFragmentManager.findFragmentById(R.id.nav_host_fragment_content_main) as NavHostFragment?)!!
        val navController = navHostFragment.navController

        binding.navView?.let {
            appBarConfiguration = AppBarConfiguration(
                setOf(
                    R.id.nav_transform, R.id.nav_body_changer, R.id.nav_reflow, R.id.nav_slideshow, R.id.nav_voice, R.id.nav_settings
                ),
                binding.drawerLayout
            )
            setupActionBarWithNavController(navController, appBarConfiguration)
            it.setupWithNavController(navController)
        }

        binding.appBarMain.contentMain.bottomNavView?.let {
            // If both drawer and bottom nav are used, we should ideally merge the top-level destinations
            val topLevelDestinations = setOf(
                R.id.nav_transform, R.id.nav_body_changer, R.id.nav_reflow, R.id.nav_slideshow, R.id.nav_voice, R.id.nav_settings
            )
            appBarConfiguration = AppBarConfiguration(topLevelDestinations, binding.drawerLayout)
            setupActionBarWithNavController(navController, appBarConfiguration)
            it.setupWithNavController(navController)
        }

        // Restore last menu
        val sharedPref = getSharedPreferences("AIfredoPrefs", Context.MODE_PRIVATE)
        // Using resource name instead of ID for persistence to avoid issues with shifting resource IDs between builds
        val lastNavName = sharedPref.getString("last_nav_name", null)
        val lastNavId = if (lastNavName != null) {
            resources.getIdentifier(lastNavName, "id", packageName)
        } else {
            sharedPref.getInt("last_nav_id", R.id.nav_transform)
        }
        
        // Use post to ensure navigation happens after setup
        binding.root.post {
            if (navController.currentDestination?.id != lastNavId && lastNavId != 0) {
                try {
                    // Check if the destination exists in the navigation graph before attempting to navigate
                    if (navController.graph.findNode(lastNavId) != null) {
                        navController.navigate(lastNavId)
                    }
                } catch (e: Exception) {
                    // If navigation fails (e.g. invalid destination for current state), stay at current/start
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
        val result = super.onCreateOptionsMenu(menu)
        val navView: NavigationView? = findViewById(R.id.nav_view)
        if (navView == null) {
            menuInflater.inflate(R.menu.overflow, menu)
        }
        return result
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        when (item.itemId) {
            R.id.nav_settings -> {
                val navController = findNavController(R.id.nav_host_fragment_content_main)
                navController.navigate(R.id.nav_settings)
            }
        }
        return super.onOptionsItemSelected(item)
    }

    override fun onSupportNavigateUp(): Boolean {
        val navController = findNavController(R.id.nav_host_fragment_content_main)
        return navController.navigateUp(appBarConfiguration) || super.onSupportNavigateUp()
    }
}