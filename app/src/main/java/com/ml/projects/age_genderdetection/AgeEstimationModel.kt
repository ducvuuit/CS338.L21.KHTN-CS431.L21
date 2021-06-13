package com.ml.projects.age_genderdetection

import android.graphics.Bitmap
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp

// Class đoán tuổi
class AgeEstimationModel {

    private val inputImageSize = 200

    // Xử lý ảnh đầu vào
    // Resize + Normalize
    private val inputImageProcessor =
            ImageProcessor.Builder()
                    .add(ResizeOp(inputImageSize, inputImageSize, ResizeOp.ResizeMethod.BILINEAR))
                    .add(NormalizeOp(0f, 255f))
                    .build()

    // Normalize tuổi bằng cách chia cho 116 (tuổi lớn nhất trong dataset)
    private val p = 116

    // Tính thời gian
    var inferenceTime : Long = 0

    var interpreter : Interpreter? = null

    // Từ ảnh đầu vào, return ra tuổi
    suspend fun predictAge(image: Bitmap) = withContext( Dispatchers.Main ) {
        val start = System.currentTimeMillis()
        // Chuyển ảnh về shape: [ 1 , 200 , 200 , 3 ]
        val tensorInputImage = TensorImage.fromBitmap(image)
        // ouput có dạng: [ 1 , 1 ]
        val ageOutputArray = Array(1){ FloatArray(1) }
        val processedImageBuffer = inputImageProcessor.process(tensorInputImage).buffer
        interpreter?.run(
                processedImageBuffer,
                ageOutputArray
        )
        inferenceTime = System.currentTimeMillis() - start
        return@withContext ageOutputArray[0][0] * p
    }
}



