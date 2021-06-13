package com.ml.projects.age_genderdetection

import android.graphics.Bitmap
import kotlinx.coroutines.*
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp

// Tạo class đoán giới tính
class GenderClassificationModel {

    private val inputImageSize = 128

    // Xử lý ảnh đầu vào
    // Resize + Normalize
    private val inputImageProcessor =
            ImageProcessor.Builder()
                    .add( ResizeOp( inputImageSize , inputImageSize , ResizeOp.ResizeMethod.BILINEAR ) )
                    .add( NormalizeOp( 0f , 255f ) )
                    .build()

    // tạo biến tính thời gian suy luận
    var inferenceTime : Long = 0

    var interpreter : Interpreter? = null

    // Đầu vào 1 bức hình, đầu ra 0 hoặc 1
    suspend fun predictGender( image : Bitmap ) = withContext( Dispatchers.Default ) {
        val start = System.currentTimeMillis()
        // shape -> [ 1 , 128 , 128 , 3 ]
        val tensorInputImage = TensorImage.fromBitmap( image )
        // [ 0 , 1 ]
        val genderOutputArray = Array( 1 ){ FloatArray( 2 ) }
        val processedImageBuffer = inputImageProcessor.process( tensorInputImage ).buffer
        interpreter?.run(
            processedImageBuffer,
            genderOutputArray
        )
        inferenceTime = System.currentTimeMillis() - start
        return@withContext genderOutputArray[ 0 ]
    }


}