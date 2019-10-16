package com.example.cameramobilenetssdkotlin

import android.Manifest
import android.app.Activity
import android.content.Context
import android.content.pm.ActivityInfo
import android.content.pm.PackageManager
import android.content.res.Configuration
import android.graphics.*
import android.hardware.Camera
import android.os.Bundle
import android.support.v4.app.ActivityCompat
import android.support.v4.content.ContextCompat
import android.view.View
import kotlinx.android.synthetic.main.content_main.*
import android.util.DisplayMetrics
import android.graphics.Bitmap
import android.view.MotionEvent
import java.math.RoundingMode
import java.text.DecimalFormat
import android.os.Vibrator

class MainActivity : Activity() {

    var flagStop = false

    //Neural Network Configuration:
    var CONFIDENCE = 0.41
    var IoU = 0.1

    //Camera configuration:
    var CAMERA_REQUEST_CODE = 10
    var cameraThread : Thread? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        this.hideSystemUI()
        this.initBtnEvents()
        this.initApp()
    }

    private fun hideSystemUI() {
        val decorView = window.decorView
        decorView.systemUiVisibility = (View.SYSTEM_UI_FLAG_IMMERSIVE
                or View.SYSTEM_UI_FLAG_LAYOUT_STABLE
                or View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
                or View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
                // Hide the nav bar and status bar
                or View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                or View.SYSTEM_UI_FLAG_FULLSCREEN)
    }

    private fun initBtnEvents(){
        btnConfidenceMore.setOnTouchListener( View.OnTouchListener { view, motionEvent ->

            when (motionEvent.action){
                MotionEvent.ACTION_DOWN -> {
                    this.vibrate()
                }
            }

            this@MainActivity.CONFIDENCE = this@MainActivity.CONFIDENCE + 0.0001
            this@MainActivity.CONFIDENCE = this@MainActivity.fixFormat( this@MainActivity.CONFIDENCE )
            txtConfidence.text = ""+this@MainActivity.CONFIDENCE
            true
        })

        btnConfidenceLess.setOnTouchListener( View.OnTouchListener { view, motionEvent ->

            when (motionEvent.action){
                MotionEvent.ACTION_DOWN -> {
                    this.vibrate()
                }
            }

            this@MainActivity.CONFIDENCE = this@MainActivity.CONFIDENCE - 0.0001
            this@MainActivity.CONFIDENCE = this@MainActivity.fixFormat( this@MainActivity.CONFIDENCE )
            txtConfidence.text = ""+this@MainActivity.CONFIDENCE
            true
        })
    }

    private fun initZoomButtons( camera:Camera, params:Camera.Parameters ){
        btnZoomLess.setOnClickListener{
            this.vibrate()

            val params = camera.getParameters()
            params.setZoom(Math.max(params.getZoom() - 1, 0))
            camera.setParameters(params)
        }

        btnZoomMore.setOnClickListener{
            this.vibrate()

            val params = camera.getParameters()
            params.setZoom(Math.min(params.getZoom() + 1, params.getMaxZoom()))
            camera.setParameters(params)
        }
    }

    private fun initApp(){
        txtConfidence.text = ""+this@MainActivity.CONFIDENCE

        val orientation = resources.configuration.orientation
        if (orientation == Configuration.ORIENTATION_LANDSCAPE) {
            // In landscape
            this.checkPermissions()
        } else {
            // In portrait
            requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE
        }
    }

    private fun checkPermissions(){
        if ( (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) ){
            startCamara()
        }else{
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), CAMERA_REQUEST_CODE)
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        if (grantResults.isEmpty() or (grantResults[0] != PackageManager.PERMISSION_GRANTED)) {
            checkPermissions()
        } else {
            startCamara()
        }
    }

    private fun startCamara(){
        //Check camara
        var isCamara = checkCameraHardware(this)
        if ( isCamara ) {

            var camera:Camera? = getCameraInstance()
            val params:Camera.Parameters = camera!!.parameters

            val displayMetrics = DisplayMetrics()
            windowManager.defaultDisplay.getMetrics(displayMetrics)

            var width = displayMetrics.widthPixels
            var height = displayMetrics.heightPixels

            var sizes = params?.let {
                it.getSupportedPictureSizes()
            }
            var bestSize = getOptimalPreviewSize(sizes, width, height)

            val cameraPreview = CameraPreview(this, this@MainActivity, camera!!, camera!!.getParameters().getSupportedPreviewSizes() )
            camera_preview.addView( cameraPreview )

            camera.startPreview()

            params.setPictureSize(bestSize!!.width, bestSize!!.height);
            params.setFocusMode(Camera.Parameters.FOCUS_MODE_CONTINUOUS_VIDEO)

            this.setConfigurationCamera(camera!!, params)
        }
    }

    private fun setConfigurationCamera( camera:Camera, params:Camera.Parameters ){
        //set color efects to none
        params.setColorEffect(Camera.Parameters.EFFECT_NONE);

        //set antibanding to none
        if (params.getAntibanding() != null) {
            params.setAntibanding(Camera.Parameters.ANTIBANDING_OFF);
        }

        // set white ballance
        if (params.getWhiteBalance() != null) {
            params.setWhiteBalance(Camera.Parameters.WHITE_BALANCE_CLOUDY_DAYLIGHT);
        }

        //set flash
        if (params.getFlashMode() != null) {
            params.setFlashMode(Camera.Parameters.FLASH_MODE_OFF);
        }

        //set zoom
        if (params.isZoomSupported()) {
            params.setZoom(0);
        }

        camera.setParameters(params)

        this.initZoomButtons(camera, params)
    }


    fun startCampturePictures( camera:Camera ){

        //Read neural network model:
        var mobileNetSSD = MobileNetSSD(this@MainActivity)
        mobileNetSSD.loadCNN()

        val displayMetrics = DisplayMetrics()
        windowManager.defaultDisplay.getMetrics(displayMetrics)

        var width = displayMetrics.widthPixels
        var height = displayMetrics.heightPixels

        var bit = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        bit = makeTransparent(bit)
        imageBase.setImageBitmap(bit)

        val cameraReady = Camera.PictureCallback { data, _ ->
            Thread( Runnable {
                val bitmap = BitmapFactory.decodeByteArray(data, 0, data.size)
                var boxesReturn = mobileNetSSD.detection(bitmap, confidence = this@MainActivity.CONFIDENCE, iou = this@MainActivity.IoU)
                var boxes = boxesReturn[0] as List<MobileNetSSD.Decode.Box>

                this.drawRectangle(bit, boxes)
                this@MainActivity.runOnUiThread {
                    imageBase.invalidate()
                }
                cameraThread?.let {
                    it.start()
                }
            }).start()
        }


        cameraThread = Thread( Runnable {
                this@MainActivity.runOnUiThread {
                    camera.takePicture(null, null, cameraReady)
                }
        })
        cameraThread?.let {
            it.start()
        }
    }

    /** Check if this device has a camera */
    private fun checkCameraHardware(context: Context): Boolean {
        return context.packageManager.hasSystemFeature(PackageManager.FEATURE_CAMERA)
    }

    /** A safe way to get an instance of the Camera object. */
    fun getCameraInstance(): Camera? {
        return try {
            Camera.open(Camera.CameraInfo.CAMERA_FACING_BACK) // attempt to get a Camera instance
        } catch (e: Exception) {
            // Camera is not available (in use or does not exist)
            null // returns null if camera is unavailable
        }
    }

    private fun getOptimalPreviewSize(sizes: List<Camera.Size>?, w: Int, h: Int): Camera.Size? {
        val ASPECT_TOLERANCE = 0.1
        val targetRatio = h.toDouble() / w

        if (sizes == null)
            return null

        var optimalSize: Camera.Size? = null
        var minDiff = java.lang.Double.MAX_VALUE

        for (size in sizes) {
            val ratio = size.height.toDouble() / size.width.toDouble()
            if (Math.abs(ratio - targetRatio) > ASPECT_TOLERANCE)
                continue

            if (Math.abs(size.height - h) < minDiff) {
                optimalSize = size
                minDiff = Math.abs(size.height - h).toDouble()
            }
        }

        if (optimalSize == null) {
            minDiff = java.lang.Double.MAX_VALUE
            for (size in sizes) {
                if (Math.abs(size.height - h) < minDiff) {
                    optimalSize = size
                    minDiff = Math.abs(size.height - h).toDouble()
                }
            }
        }

        return optimalSize
    }

    fun  drawRectangle(bmp:Bitmap, boxes:List<MobileNetSSD.Decode.Box>):Bitmap{

        var widthImg = bmp.width
        var heightImg = bmp.height

        val canvas = Canvas(bmp)
        canvas.drawColor(Color.TRANSPARENT,PorterDuff.Mode.CLEAR);

        val paint = Paint(Paint.ANTI_ALIAS_FLAG)
        paint.setStyle(Paint.Style.STROKE)
        paint.setStrokeWidth(5.0f)
        paint.setColor(Color.RED)

        for(box in boxes){
            box.getxMin()*widthImg
            box.getyMin()*heightImg
            box.getxMax()*widthImg
            box.getyMax()*heightImg

            canvas.drawRect(
                (box.getxMin() * widthImg).toFloat(),
                (box.getyMin() * heightImg).toFloat(),
                (box.getxMax() * widthImg).toFloat(),
                (box.getyMax() * heightImg).toFloat(), paint
            )
        }
        return bmp
    }

    // Convert transparentColor to be transparent in a Bitmap.
    fun makeTransparent(bit: Bitmap): Bitmap {
        val width = bit.width
        val height = bit.height
        val myBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val allpixels = IntArray(myBitmap.height * myBitmap.width)
        bit.getPixels(allpixels, 0, myBitmap.width, 0, 0, myBitmap.width, myBitmap.height)
        myBitmap.setPixels(allpixels, 0, width, 0, 0, width, height)

        for (i in 0 until myBitmap.height * myBitmap.width) {
                allpixels[i] = Color.alpha(Color.TRANSPARENT)
        }

        myBitmap.setPixels(allpixels, 0, myBitmap.width, 0, 0, myBitmap.width, myBitmap.height)
        return myBitmap
    }

    fun fixFormat( x:Double ):Double {
        val df = DecimalFormat("#.####")
        df.roundingMode = RoundingMode.CEILING
        return df.format(x).toDouble()
    }

    private fun vibrate() {
        val vibrator = getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
        vibrator.vibrate(50)
    }
}
