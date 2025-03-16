#if !UNITY_WSA_10_0

using OpenCVForUnity.CoreModule;
using OpenCVForUnity.ImgprocModule;
using OpenCVForUnity.UnityUtils;
using OpenCVForUnity.UnityUtils.Helper;
using OpenCVForUnityExample.DnnModel;
using System.Threading;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;  

namespace OpenCVForUnityExample
{
    /// <summary>
    /// Global score holder.
    /// </summary>
    // public static class GlobalScore
    // {
    //     public static int score = 0;
    // }

    /// <summary>
    /// Object Detection YOLOX Example
    /// An example of using OpenCV dnn module with YOLOX Object Detection.
    /// Referring to https://github.com/opencv/opencv_zoo/tree/master/models/object_detection_yolox
    /// https://github.com/Megvii-BaseDetection/YOLOX
    /// https://github.com/Megvii-BaseDetection/YOLOX/tree/main/demo/ONNXRuntime
    /// 
    /// [Tested Models]
    /// yolox_nano.onnx https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx
    /// yolox_tiny.onnx https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx
    /// yolox_s.onnx https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx
    /// </summary>
    [RequireComponent(typeof(MultiSource2MatHelper))]
    public class ObjectDetectionYOLOXExample : MonoBehaviour
    {
        [Header("Output")]
        /// <summary>
        /// The RawImage for previewing the result.
        /// </summary>
        public RawImage resultPreview;

        [Header("Score")]
        /// <summary>
        /// Public integer to track the score.
        /// </summary>
        public int scoreCounter = 0;

        [Space(10)]
        [Tooltip("Path to a binary file of model contains trained weights. It could be a file with extensions .caffemodel (Caffe), .pb (TensorFlow), .t7 or .net (Torch), .weights (Darknet).")]
        public string model = "yolox_tiny.onnx";

        [Tooltip("Path to a text file of model contains network configuration. It could be a file with extensions .prototxt (Caffe), .pbtxt (TensorFlow), .cfg (Darknet).")]
        public string config = "";

        [Tooltip("Optional path to a text file with names of classes to label detected objects.")]
        public string classes = "coco.names";

        [Tooltip("Confidence threshold.")]
        public float confThreshold = 0.25f;

        [Tooltip("Non-maximum suppression threshold.")]
        public float nmsThreshold = 0.45f;

        [Tooltip("Maximum detections per image.")]
        public int topK = 1000;

        [Tooltip("Preprocess input image by resizing to a specific width.")]
        public int inpWidth = 416;

        [Tooltip("Preprocess input image by resizing to a specific height.")]
        public int inpHeight = 416;

        protected string classes_filepath;
        protected string config_filepath;
        protected string model_filepath;

        /// <summary>
        /// The texture.
        /// </summary>
        Texture2D texture;

        /// <summary>
        /// The multi source to mat helper.
        /// </summary>
        MultiSource2MatHelper multiSource2MatHelper;

        /// <summary>
        /// The BGR Mat.
        /// </summary>
        Mat bgrMat;

        /// <summary>
        /// The YOLOX ObjectDetector.
        /// </summary>
        YOLOXObjectDetector objectDetector;

        /// <summary>
        /// The FPS monitor.
        /// </summary>
        FpsMonitor fpsMonitor;

        /// <summary>
        /// The CancellationTokenSource.
        /// </summary>
        CancellationTokenSource cts = new CancellationTokenSource();

        async void Start()
        {
            fpsMonitor = GetComponent<FpsMonitor>();

            multiSource2MatHelper = gameObject.GetComponent<MultiSource2MatHelper>();
            multiSource2MatHelper.outputColorFormat = Source2MatHelperColorFormat.RGBA;

            // Asynchronously retrieves the readable file path from the StreamingAssets directory.
            if (fpsMonitor != null)
                fpsMonitor.consoleText = "Preparing file access...";

            if (!string.IsNullOrEmpty(classes))
            {
                classes_filepath = await Utils.getFilePathAsyncTask("OpenCVForUnity/dnn/" + classes, cancellationToken: cts.Token);
                if (string.IsNullOrEmpty(classes_filepath))
                    Debug.Log("The file:" + classes + " did not exist in the folder “Assets/StreamingAssets/OpenCVForUnity/dnn”.");
            }
            if (!string.IsNullOrEmpty(config))
            {
                config_filepath = await Utils.getFilePathAsyncTask("OpenCVForUnity/dnn/" + config, cancellationToken: cts.Token);
                if (string.IsNullOrEmpty(config_filepath))
                    Debug.Log("The file:" + config + " did not exist in the folder “Assets/StreamingAssets/OpenCVForUnity/dnn”.");
            }
            if (!string.IsNullOrEmpty(model))
            {
                model_filepath = await Utils.getFilePathAsyncTask("OpenCVForUnity/dnn/" + model, cancellationToken: cts.Token);
                if (string.IsNullOrEmpty(model_filepath))
                    Debug.Log("The file:" + model + " did not exist in the folder “Assets/StreamingAssets/OpenCVForUnity/dnn”.");
            }

            if (fpsMonitor != null)
                fpsMonitor.consoleText = "";

            Run();
        }

        void Run()
        {
            // Enable native side OpenCV debug log.
            Utils.setDebugMode(true);

            if (string.IsNullOrEmpty(model_filepath))
            {
                Debug.LogError("model: " + model + " or " + "config: " + config + " or " + "classes: " + classes + " is not loaded.");
            }
            else
            {
                objectDetector = new YOLOXObjectDetector(model_filepath, config_filepath, classes_filepath, new Size(inpWidth, inpHeight), confThreshold, nmsThreshold, topK);
            }

            multiSource2MatHelper.Initialize();
        }

        public void OnSourceToMatHelperInitialized()
        {
            Debug.Log("OnSourceToMatHelperInitialized");

            Mat rgbaMat = multiSource2MatHelper.GetMat();

            texture = new Texture2D(rgbaMat.cols(), rgbaMat.rows(), TextureFormat.RGBA32, false);
            Utils.matToTexture2D(rgbaMat, texture);

            resultPreview.texture = texture;
            resultPreview.GetComponent<AspectRatioFitter>().aspectRatio = (float)texture.width / texture.height;

            if (fpsMonitor != null)
            {
                fpsMonitor.Add("width", rgbaMat.width().ToString());
                fpsMonitor.Add("height", rgbaMat.height().ToString());
                fpsMonitor.Add("orientation", Screen.orientation.ToString());
            }

            bgrMat = new Mat(rgbaMat.rows(), rgbaMat.cols(), CvType.CV_8UC3);
        }

        public void OnSourceToMatHelperDisposed()
        {
            Debug.Log("OnSourceToMatHelperDisposed");

            if (bgrMat != null)
                bgrMat.Dispose();

            if (texture != null)
            {
                Texture2D.Destroy(texture);
                texture = null;
            }
        }

        public void OnSourceToMatHelperErrorOccurred(Source2MatHelperErrorCode errorCode, string message)
        {
            Debug.Log("OnSourceToMatHelperErrorOccurred " + errorCode + ":" + message);

            if (fpsMonitor != null)
            {
                fpsMonitor.consoleText = "ErrorCode: " + errorCode + ":" + message;
            }
        }

        // Update is called once per frame
        void Update()
        {
            if (multiSource2MatHelper.IsPlaying() && multiSource2MatHelper.DidUpdateThisFrame())
            {
                Mat rgbaMat = multiSource2MatHelper.GetMat();

                if (objectDetector == null)
                {
                    Imgproc.putText(rgbaMat, "model file is not loaded.", new Point(5, rgbaMat.rows() - 30),
                        Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 255, 255, 255), 2, Imgproc.LINE_AA, false);
                    Imgproc.putText(rgbaMat, "Please read console message.", new Point(5, rgbaMat.rows() - 10),
                        Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 255, 255, 255), 2, Imgproc.LINE_AA, false);
                }
                else
                {
                    Imgproc.cvtColor(rgbaMat, bgrMat, Imgproc.COLOR_RGBA2BGR);

                    // Run inference and obtain detection results.
                    Mat results = objectDetector.infer(bgrMat);

                    Imgproc.cvtColor(bgrMat, rgbaMat, Imgproc.COLOR_BGR2RGBA);

                    // Visualize detections. (This method handles drawing bounding boxes and score logic.)
                    objectDetector.visualize(rgbaMat, results, false, true);
                }

                Utils.matToTexture2D(rgbaMat, texture);
            }

            // // Increment the score every frame
            // scoreCounter++;
            // GlobalScore.score = scoreCounter;
            // Debug.Log("Score: " + GlobalScore.score);
        }

        void OnDestroy()
        {
            multiSource2MatHelper.Dispose();

            if (objectDetector != null)
                objectDetector.dispose();

            Utils.setDebugMode(false);

            if (cts != null)
                cts.Dispose();
        }

        public void OnBackButtonClick()
        {
            SceneManager.LoadScene("OpenCVForUnityExample");
        }

        public void OnPlayButtonClick()
        {
            multiSource2MatHelper.Play();
        }

        public void OnPauseButtonClick()
        {
            multiSource2MatHelper.Pause();
        }

        public void OnStopButtonClick()
        {
            multiSource2MatHelper.Stop();
        }

        public void OnChangeCameraButtonClick()
        {
            multiSource2MatHelper.requestedIsFrontFacing = !multiSource2MatHelper.requestedIsFrontFacing;
        }
    }
}
#endif
