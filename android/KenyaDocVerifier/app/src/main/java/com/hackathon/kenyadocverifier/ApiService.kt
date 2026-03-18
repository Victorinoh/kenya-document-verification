package com.hackathon.kenyadocverifier

import okhttp3.MultipartBody
import okhttp3.RequestBody
import retrofit2.Response
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import retrofit2.http.Multipart
import retrofit2.http.POST
import retrofit2.http.Part

data class VerifyResponse(
    val verdict: String,
    val score: Float,
    val confidence: String,
    val document_type: String,
    val duration_ms: Int,
    val layers: Layers,
    val validation: List<ValidationCheck>,
    val ocr_summary: OcrSummary
)

data class Layers(
    val classifier: LayerScore,
    val detector: LayerScore,
    val ocr: LayerScore
)

data class LayerScore(
    val score: Float,
    val label: String = ""
)

data class ValidationCheck(
    val check: String,
    val passed: Boolean
)

data class OcrSummary(
    val confidence: Float,
    val fields_found: String
)

interface ApiService {
    @Multipart
    @POST("verify")
    suspend fun verifyDocument(
        @Part file: MultipartBody.Part,
        @Part("doc_type") docType: RequestBody?
    ): Response<VerifyResponse>
}

object RetrofitClient {

    // ⚠️ Replace with your laptop IP from ipconfig
    private const val BASE_URL = "http://10.0.2.2:8000/"

    val instance: ApiService by lazy {
        Retrofit.Builder()
            .baseUrl(BASE_URL)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
            .create(ApiService::class.java)
    }
}