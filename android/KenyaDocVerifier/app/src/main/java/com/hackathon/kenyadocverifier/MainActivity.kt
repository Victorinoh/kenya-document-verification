package com.hackathon.kenyadocverifier

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.view.View
import android.widget.ArrayAdapter
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.lifecycle.lifecycleScope
import com.bumptech.glide.Glide
import com.hackathon.kenyadocverifier.databinding.ActivityMainBinding
import kotlinx.coroutines.launch
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.asRequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private var selectedImageUri: Uri? = null
    private var cameraImageUri: Uri? = null

    private val docTypes = listOf(
        "Auto Detect",
        "National ID",
        "KCSE Certificate",
        "Passport"
    )
    private val docTypeValues = listOf(
        "",
        "national_id",
        "kcse_certificate",
        "passport"
    )

    private val galleryLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == Activity.RESULT_OK) {
            result.data?.data?.let { uri ->
                selectedImageUri = uri
                showImagePreview(uri)
            }
        }
    }

    private val cameraLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == Activity.RESULT_OK) {
            cameraImageUri?.let { uri ->
                selectedImageUri = uri
                showImagePreview(uri)
            }
        }
    }

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val allGranted = permissions.values.all { it }
        if (allGranted) showImageSourceDialog()
        else Toast.makeText(this, "Permissions required to select documents", Toast.LENGTH_LONG).show()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        setupDocTypeDropdown()
        setupClickListeners()
    }

    private fun setupDocTypeDropdown() {
        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, docTypes)
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        binding.spinnerDocType.adapter = adapter
    }

    private fun setupClickListeners() {
        binding.btnSelectImage.setOnClickListener {
            checkPermissionsAndShowDialog()
        }
        binding.btnVerify.setOnClickListener {
            if (selectedImageUri == null) {
                Toast.makeText(this, "Please select a document image first", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            verifyDocument()
        }
    }

    private fun checkPermissionsAndShowDialog() {
        val permissions = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            arrayOf(Manifest.permission.CAMERA, Manifest.permission.READ_MEDIA_IMAGES)
        } else {
            arrayOf(Manifest.permission.CAMERA, Manifest.permission.READ_EXTERNAL_STORAGE)
        }
        val notGranted = permissions.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }
        if (notGranted.isEmpty()) showImageSourceDialog()
        else permissionLauncher.launch(notGranted.toTypedArray())
    }

    private fun showImageSourceDialog() {
        AlertDialog.Builder(this)
            .setTitle("Select Document")
            .setItems(arrayOf("📷  Take Photo", "🖼️  Choose from Gallery")) { _, which ->
                when (which) {
                    0 -> openCamera()
                    1 -> openGallery()
                }
            }
            .show()
    }

    private fun openGallery() {
        val intent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
        intent.type = "image/*"
        galleryLauncher.launch(intent)
    }

    private fun openCamera() {
        val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
        val imageFile = File(getExternalFilesDir(Environment.DIRECTORY_PICTURES), "DOC_$timestamp.jpg")
        cameraImageUri = FileProvider.getUriForFile(this, "$packageName.fileprovider", imageFile)
        val intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        intent.putExtra(MediaStore.EXTRA_OUTPUT, cameraImageUri)
        cameraLauncher.launch(intent)
    }

    private fun showImagePreview(uri: Uri) {
        binding.imgPreview.visibility = View.VISIBLE
        Glide.with(this).load(uri).into(binding.imgPreview)
        binding.btnVerify.isEnabled = true
        binding.tvSelectHint.text = "Document selected ✓ — tap Verify to continue"
    }

    private fun verifyDocument() {
        val uri = selectedImageUri ?: return
        val file = uriToFile(uri) ?: run {
            Toast.makeText(this, "Could not read image file", Toast.LENGTH_SHORT).show()
            return
        }
        val selectedIndex = binding.spinnerDocType.selectedItemPosition
        val docTypeValue = docTypeValues[selectedIndex]
        setLoadingState(true)

        lifecycleScope.launch {
            try {
                val requestFile = file.asRequestBody("image/jpeg".toMediaType())
                val filePart = MultipartBody.Part.createFormData("file", file.name, requestFile)
                val docTypePart = if (docTypeValue.isNotEmpty())
                    docTypeValue.toRequestBody("text/plain".toMediaType())
                else null

                val response = RetrofitClient.instance.verifyDocument(filePart, docTypePart)

                if (response.isSuccessful && response.body() != null) {
                    val result = response.body()!!
                    val intent = Intent(this@MainActivity, ResultActivity::class.java).apply {
                        putExtra("verdict",        result.verdict)
                        putExtra("score",          result.score)
                        putExtra("confidence",     result.confidence)
                        putExtra("doc_type",       result.document_type)
                        putExtra("duration_ms",    result.duration_ms)
                        putExtra("classifier",     result.layers.classifier.score)
                        putExtra("detector",       result.layers.detector.score)
                        putExtra("ocr",            result.layers.ocr.score)
                        putExtra("ocr_confidence", result.ocr_summary.confidence)
                        putExtra("fields_found",   result.ocr_summary.fields_found)
                        val checks = result.validation.map { it.check }.toTypedArray()
                        val passed = result.validation.map { it.passed }.toBooleanArray()
                        putExtra("checks", checks)
                        putExtra("passed", passed)
                    }
                    startActivity(intent)
                } else {
                    Toast.makeText(
                        this@MainActivity,
                        "API Error: ${response.code()} — ${response.message()}",
                        Toast.LENGTH_LONG
                    ).show()
                }
            } catch (e: Exception) {
                Toast.makeText(
                    this@MainActivity,
                    "Connection failed: ${e.message}\nCheck your IP in ApiService.kt",
                    Toast.LENGTH_LONG
                ).show()
            } finally {
                setLoadingState(false)
            }
        }
    }

    private fun uriToFile(uri: Uri): File? {
        return try {
            val inputStream = contentResolver.openInputStream(uri) ?: return null
            val tempFile = File(cacheDir, "temp_document.jpg")
            FileOutputStream(tempFile).use { output -> inputStream.copyTo(output) }
            tempFile
        } catch (e: Exception) { null }
    }

    private fun setLoadingState(loading: Boolean) {
        binding.btnVerify.isEnabled = !loading
        binding.btnSelectImage.isEnabled = !loading
        binding.progressBar.visibility = if (loading) View.VISIBLE else View.GONE
        binding.btnVerify.text = if (loading) "Verifying..." else "Verify Document"
    }
}