package com.hackathon.kenyadocverifier

import android.graphics.Color
import android.os.Bundle
import android.view.View
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.hackathon.kenyadocverifier.databinding.ActivityResultBinding

class ResultActivity : AppCompatActivity() {

    private lateinit var binding: ActivityResultBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityResultBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val verdict     = intent.getStringExtra("verdict")      ?: "UNKNOWN"
        val score       = intent.getFloatExtra("score", 0f)
        val confidence  = intent.getStringExtra("confidence")   ?: ""
        val docType     = intent.getStringExtra("doc_type")     ?: ""
        val durationMs  = intent.getIntExtra("duration_ms", 0)
        val classifier  = intent.getFloatExtra("classifier", 0f)
        val detector    = intent.getFloatExtra("detector", 0f)
        val ocr         = intent.getFloatExtra("ocr", 0f)
        val ocrConf     = intent.getFloatExtra("ocr_confidence", 0f)
        val fieldsFound = intent.getStringExtra("fields_found") ?: ""
        val checks      = intent.getStringArrayExtra("checks")  ?: emptyArray()
        val passed      = intent.getBooleanArrayExtra("passed") ?: BooleanArray(0)

        displayVerdict(verdict, score, confidence)
        displayLayerBars(classifier, detector, ocr)
        displayDetails(docType, ocrConf, fieldsFound, durationMs)
        displayChecklist(checks, passed)

        binding.btnTryAgain.setOnClickListener { finish() }
    }

    private fun displayVerdict(verdict: String, score: Float, confidence: String) {
        val colour = when (verdict) {
            "GENUINE"   -> Color.parseColor("#2d7a2d")
            "UNCERTAIN" -> Color.parseColor("#b08000")
            "FAKE"      -> Color.parseColor("#9b2335")
            else        -> Color.GRAY
        }
        val emoji = when (verdict) {
            "GENUINE"   -> "✅"
            "UNCERTAIN" -> "⚠️"
            "FAKE"      -> "❌"
            else        -> "⚠️"
        }
        binding.verdictBanner.setBackgroundColor(colour)
        binding.tvVerdict.text = "$emoji  $verdict"
        binding.scoreCircle.setBackgroundColor(colour)
        binding.tvScore.text = "${score.toInt()}%"
        binding.tvConfidence.text = confidence
    }

    private fun displayLayerBars(classifier: Float, detector: Float, ocr: Float) {
        binding.tvClassifierScore.text = "${classifier.toInt()}%"
        binding.tvDetectorScore.text   = "${detector.toInt()}%"
        binding.tvOcrScore.text        = "${ocr.toInt()}%"
        binding.root.post {
            animateBar(binding.barClassifier, classifier)
            animateBar(binding.barDetector,   detector)
            animateBar(binding.barOcr,        ocr)
        }
    }

    private fun animateBar(barView: View, percentage: Float) {
        val parent = barView.parent as LinearLayout
        val maxWidth = parent.width
        val targetWidth = (maxWidth * (percentage / 100f)).toInt()
        val params = barView.layoutParams
        params.width = targetWidth
        barView.layoutParams = params
    }

    private fun displayDetails(docType: String, ocrConf: Float, fieldsFound: String, durationMs: Int) {
        binding.tvDocType.text        = docType
        binding.tvOcrConfidence.text  = "${"%.1f".format(ocrConf)}%"
        binding.tvFieldsFound.text    = fieldsFound
        binding.tvProcessingTime.text = "${durationMs}ms"
    }

    private fun displayChecklist(checks: Array<String>, passed: BooleanArray) {
        binding.checklistContainer.removeAllViews()
        checks.forEachIndexed { index, checkText ->
            val isPassed = if (index < passed.size) passed[index] else false
            val tv = TextView(this).apply {
                text = if (isPassed) "✅  $checkText" else "❌  $checkText"
                textSize = 14f
                setTextColor(
                    if (isPassed) Color.parseColor("#1A7A3C")
                    else Color.parseColor("#9b2335")
                )
                setPadding(0, 12, 0, 12)
            }
            binding.checklistContainer.addView(tv)
            val divider = View(this@ResultActivity).apply {
                layoutParams = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT, 1
                )
                setBackgroundColor(Color.parseColor("#EEEEEE"))
            }
            binding.checklistContainer.addView(divider)
        }
    }
}
