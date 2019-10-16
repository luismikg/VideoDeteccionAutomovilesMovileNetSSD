package com.example.cameramobilenetssdkotlin

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.io.IOException
import java.lang.Exception
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.*

class MobileNetSSD(context: Context) {

    val modelName =  "MobileNetSSD_Swish.tflite"
    //val modelName =  "MobileNetSSD_ReLU.tflite"
    lateinit var tflite: Interpreter
    var context:Context

    init{
        this.context = context
    }

    /**
     * Load TensorFlow model.
     */
    fun loadCNN() :String{
        val model = this.modelName

        try {
            this.tflite = Interpreter( this.loadModelFile( model ) )
            println("listo")
        } catch (e:Exception){
            println("error: "+e.message)
        }
        return model
    }

    /**
     * Load tensorflow lite file
     * @return
     * @throws IOException
     */
    private fun loadModelFile( model: String ): MappedByteBuffer {

        val fileDescriptor = this.context.getAssets().openFd(model)
        val inputStream = FileInputStream(fileDescriptor.getFileDescriptor())
        val fileChannel = inputStream.getChannel()

        val startOffset = fileDescriptor.getStartOffset()
        val declaredLength = fileDescriptor.getDeclaredLength()
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    /**
     * Method to init detections.
     * @param img: Original img
     */
    fun detection( img:Bitmap?, confidence:Double, iou:Double ): Array<Any> {

        var img300x300 = Bitmap.createScaledBitmap( img, 300, 300, false)
        var input = prepocessImg( img300x300 )
        var output = Array(1) { Array(1917) { FloatArray(14) } }

        //--------------------------------------------
        var startDetection = System.currentTimeMillis()
        this.tflite.run(input, output)
        var endDetection = System.currentTimeMillis()
        //--------------------------------------------

        //********************************************
        var startDecode = System.currentTimeMillis()
        var boxes = Decode().decode( output , confidence, iou )
        var endDecode = System.currentTimeMillis()

        var timeDetection = endDetection - startDetection
        var timeDecodification = endDecode - startDecode

        return arrayOf( boxes, timeDetection, timeDecodification )
        //********************************************
    }

    fun prepocessImg( img:Bitmap ): ByteBuffer{

        var mean = arrayOf(103.939, 116.779, 123.68)

        // Specify the input size like [1][300][300][3]
        val DIM_BATCH_SIZE = 1
        val DIM_IMG_SIZE_X = 300
        val DIM_IMG_SIZE_Y = 300
        val DIM_PIXEL_SIZE = 3
        // Number of bytes to hold a float (32 bits / float) / (8 bits / byte) = 4 bytes / float
        val BYTE_SIZE_OF_FLOAT = 4

        var inputBuffer = ByteBuffer.allocateDirect(BYTE_SIZE_OF_FLOAT * DIM_BATCH_SIZE * DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE);
        inputBuffer.order( ByteOrder.nativeOrder() );

        val pixels = IntArray(300 * 300)
        img.getPixels(pixels, 0, 300, 0, 0, 300, 300)
        for (pixel in pixels) {
            //CAFFE image mode
            inputBuffer.putFloat(((pixel and 0xFF) - mean[0]).toFloat())
            inputBuffer.putFloat(((pixel shr 8 and 0xFF) - mean[1]).toFloat())
            inputBuffer.putFloat(((pixel shr 16 and 0xFF) - mean[2]).toFloat())
        }

        return  inputBuffer
    }

    class Decode{

        // Data from CNN-output
        private lateinit var boxLoc: ArrayList<FloatArray>
        private lateinit var variances: ArrayList<FloatArray>
        private lateinit var priorBox: ArrayList<FloatArray>
        private lateinit var boxConf: ArrayList<FloatArray>

        fun decode( prediction:Array<Array<FloatArray>>, confidence: Double, iou:Double ): List<Box> {
            //Divide all predictions to different arrays containers (boxLoc, variances, priorBox, boxConf)
            this.getSpecificData( prediction, confidence )
            var boxes = this.detectionsOut( iou )
            return boxes
        }

        /**
         * //Divide all predictions to different arrays containers (boxLoc, variances, priorBox, boxConf)
         * @param predictions All results fron tensorflow model
         */
        private fun getSpecificData(predictions: Array<Array<FloatArray>>, confidence: Double) {

            val boxData = ArrayList<FloatArray>() //Array(predictions.size) { Array(predictions[0].size) { FloatArray(4) } }
            val variancesData = ArrayList<FloatArray>() //Array(predictions.size) { Array(predictions[0].size) { FloatArray(4) } }
            val priorBoxData = ArrayList<FloatArray>() //Array(predictions.size) { Array(predictions[0].size) { FloatArray(4) } }
            val boxConfData = ArrayList<FloatArray>() // Array(predictions.size) { Array(predictions[0].size) { FloatArray(predictions[0].size - 4 - 8) } }
            var i = 0
            for (prediction in predictions) {
                for (predictionData in prediction) {
                    var conf = Arrays.copyOfRange(predictionData, 4, predictionData.size - 8)
                    if( conf[1] >= confidence) {
                        boxData.add( Arrays.copyOfRange(predictionData, 0, 4) )
                        variancesData.add( Arrays.copyOfRange(predictionData, predictionData.size - 4, predictionData.size) )
                        priorBoxData.add( Arrays.copyOfRange(predictionData, predictionData.size - 8, predictionData.size - 4) )
                        boxConfData.add( Arrays.copyOfRange(predictionData, 4, predictionData.size - 8) )
                    }
                }
            }

            this.boxLoc = boxData
            this.variances = variancesData
            this.priorBox = priorBoxData
            this.boxConf = boxConfData
        }

        /**
         * This method return all bounding boxes which has a higher confidence than parameter "confidenceThreshold"
         * @param predictions all predictions from tensorflow model
         */
        private fun detectionsOut( iou:Double ):List<Box>  {

            var boxes = LinkedList<Box>()

            for (i in 0 until this.boxLoc.size) {
                val decodeBox = this.decodeBoxes(this.boxLoc[i], this.priorBox[i], this.variances[i])

                val box = Box()
                box.setxMin(decodeBox[0])
                box.setyMin(decodeBox[1])
                box.setxMax(decodeBox[2])
                box.setyMax(decodeBox[3])
                box.setConfidence( this.boxConf[i][1].toDouble() )
                box.setArea((box.getxMax() - box.getxMin() + 1) * (box.getyMax() - box.getyMin() + 1))
                boxes.add(box)
            }

            val nonMaximumSuppression = NonMaximumSuppression()
            boxes = nonMaximumSuppression.nms(boxes, iouThresh = iou) as LinkedList<Box>
            return boxes
        }

        /**
         * Get coordinates from "boxLoc" array as a function of its variances and priorBoxes
         * @param boxesLoc
         * @param priorBoxes
         * @param variances
         * @return
         */
        private fun decodeBoxes( boxLoc: FloatArray, priorBox: FloatArray, variance: FloatArray ): DoubleArray {

            val priorWidth = priorBox[2] - priorBox[0]
            val priorHeight = priorBox[3] - priorBox[1]
            val priorCenterX = 0.5f * (priorBox[2] + priorBox[0])
            val priorCenterY = 0.5f * (priorBox[3] + priorBox[1])

            var decodeBoxCenterX = boxLoc[0] * priorWidth * variance[0]
            decodeBoxCenterX = decodeBoxCenterX + priorCenterX

            var decodeBoxCenterY =
                boxLoc[1] * priorHeight * variance[1]
            decodeBoxCenterY = decodeBoxCenterY + priorCenterY

            var decodeBoxWidth = Math.exp((boxLoc[2] * variance[2]).toDouble())
            decodeBoxWidth = decodeBoxWidth * priorWidth

            var decodeBoxHeight = Math.exp((boxLoc[3] * variance[3]).toDouble())
            decodeBoxHeight = decodeBoxHeight * priorHeight

            val decodeBoxXMin = decodeBoxCenterX - 0.5 * decodeBoxWidth
            val decodeBoxYMin = decodeBoxCenterY - 0.5 * decodeBoxHeight
            val decodeBoxXmax = decodeBoxCenterX + 0.5 * decodeBoxWidth
            val decodeBoxYmax = decodeBoxCenterY + 0.5 * decodeBoxHeight

            val decodeBox = doubleArrayOf(
                Math.min(Math.max(decodeBoxXMin, 0.0), 1.0),
                Math.min(Math.max(decodeBoxYMin, 0.0), 1.0),
                Math.min(Math.max(decodeBoxXmax, 0.0), 1.0), Math.min(Math.max(decodeBoxYmax, 0.0), 1.0)
            )

            return decodeBox
        }


        class  Box {
            private var xMin: Double = 0.0
            private var yMin: Double = 0.0

            private var xMax: Double = 0.0
            private var yMax: Double = 0.0
            private var confidence: Double = 0.0
            private var area: Double = 0.0

            private var timeDetccion: Double = 0.0
            private var timeDecodification: Double = 0.0

            ////////////////////////
            fun geTimeDetection(): Double {
                return timeDetccion
            }

            fun setTimeDetection(time: Double) {
                this.timeDetccion = time
            }

            fun geTimeDecodification(): Double {
                return timeDecodification
            }

            fun setTimeDecodification(time: Double) {
                this.timeDecodification = time
            }
            ////////////////////////
            fun getxMin(): Double {
                return xMin
            }

            fun setxMin(xMin: Double) {
                this.xMin = xMin
            }

            fun getyMin(): Double {
                return yMin
            }

            fun setyMin(yMin: Double) {
                this.yMin = yMin
            }

            fun getxMax(): Double {
                return xMax
            }

            fun setxMax(xMax: Double) {
                this.xMax = xMax
            }

            fun getyMax(): Double {
                return yMax
            }

            fun setyMax(yMax: Double) {
                this.yMax = yMax
            }

            fun getConfidence(): Double {
                return confidence
            }

            fun setConfidence(confidence: Double) {
                this.confidence = confidence
            }

            fun getArea(): Double {
                return area
            }

            fun setArea(area: Double) {
                this.area = area
            }
        }

        internal class NonMaximumSuppression {
            @JvmOverloads
            fun nms(boxesToProcess: List<Box>, iouThresh: Double = 0.45): List<Box> {

                val idxs = LinkedList<Int>()
                val boxes = LinkedList<Box>()

                if (boxesToProcess.size == 0) {
                    return boxes
                }

                for (i in boxesToProcess.indices) {
                    idxs.add(i)
                }

                Collections.sort(idxs) { o1, o2 ->
                    val box1 = boxesToProcess[o1!!.toInt()]
                    val box2 = boxesToProcess[o2!!.toInt()]

                    var returned = 0
                    if (box1.getConfidence() < box2.getConfidence()) {
                        returned = -1
                    }
                    if (box1.getConfidence() > box2.getConfidence()) {
                        returned = 1
                    }

                    returned
                }

                //LinkedList pick = new LinkedList();
                while (!idxs.isEmpty()) {

                    val last = idxs.size - 1
                    val i = idxs[last]
                    //pick.add(i);
                    boxes.add(boxesToProcess[i])
                    // List of boxes we want to ignore
                    val suppress = LinkedList<Int>()
                    suppress.add(last)

                    for (pos in 0 until last) {
                        val j = idxs[pos]

                        val boxOne = boxesToProcess[i]
                        val boxTwo = boxesToProcess[j]
                        val iou = this.getIntersectionOverUnion(boxOne, boxTwo)
                        // if iou's box(j) is high with box(i), just get rid, because it probably correspond to the same object.
                        if (iou > iouThresh) {
                            suppress.add(pos)
                        }
                    }

                    Collections.sort(suppress, Collections.reverseOrder())
                    for (idxToRemove in suppress) {
                        idxs.removeAt(idxToRemove)
                    }
                }
                return boxes
            }

            private fun getIntersectionOverUnion(boxOne: Box, boxTwo: Box): Double {
                var iou = 0.0

                //Area of overlap
                val xMin = Math.max(boxOne.getxMin(), boxTwo.getxMin())
                val yMin = Math.max(boxOne.getyMin(), boxTwo.getyMin())
                val xMax = Math.min(boxOne.getxMax(), boxTwo.getxMax())
                val yMax = Math.min(boxOne.getyMax(), boxTwo.getyMax())

                var a = Math.max(0.0, xMax - xMin).toFloat()
                var b = Math.max(0.0, yMax - yMin).toFloat()
                if (a > 0) {
                    a = a + 1
                }
                if (b > 0) {
                    b = b + 1
                }

                val areaOver = a * b

                //Get iou
                iou = areaOver / (boxOne.getArea() + boxTwo.getArea() - areaOver)
                return iou
            }
        }
    }

}