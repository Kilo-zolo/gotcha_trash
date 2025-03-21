using OpenCVForUnity.CoreModule;
using OpenCVForUnity.ImgprocModule;
using OpenCVForUnity.UnityUtils;
using OpenCVForUnity.UnityUtils.Helper;
using OpenCVForUnity.VideoModule;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.SceneManagement;

namespace OpenCVForUnityExample
{
    /// <summary>
    /// CamShift Example
    /// An example of object tracking using the Video.Camshift function.
    /// Referring to http://www.computervisiononline.com/blog/tutorial-using-camshift-track-objects-video.
    /// http://docs.opencv.org/3.2.0/db/df8/tutorial_py_meanshift.html
    /// </summary>
    [RequireComponent(typeof(MultiSource2MatHelper))]
    public class CamShiftExample : MonoBehaviour
    {
        /// <summary>
        /// The texture.
        /// </summary>
        Texture2D texture;

        /// <summary>
        /// The roi point list.
        /// </summary>
        List<Point> roiPointList;

        /// <summary>
        /// The roi rect.
        /// </summary>
        OpenCVForUnity.CoreModule.Rect roiRect;

        /// <summary>
        /// The hsv mat.
        /// </summary>
        Mat hsvMat;

        /// <summary>
        /// The roi hist mat.
        /// </summary>
        Mat roiHistMat;

        /// <summary>
        /// The termination.
        /// </summary>
        TermCriteria termination;

        /// <summary>
        /// The multi source to mat helper.
        /// </summary>
        MultiSource2MatHelper multiSource2MatHelper;

        /// <summary>
        /// The stored touch point.
        /// </summary>
        Point storedTouchPoint;

        /// <summary>
        /// The flag for requesting the start of the CamShift Method.
        /// </summary>
        bool shouldStartCamShift = false;

        /// <summary>
        /// The FPS monitor.
        /// </summary>
        FpsMonitor fpsMonitor;

        // Use this for initialization
        void Start()
        {
            fpsMonitor = GetComponent<FpsMonitor>();

            roiPointList = new List<Point>();
            termination = new TermCriteria(TermCriteria.EPS | TermCriteria.COUNT, 10, 1);

            multiSource2MatHelper = gameObject.GetComponent<MultiSource2MatHelper>();
            multiSource2MatHelper.outputColorFormat = Source2MatHelperColorFormat.RGBA;
            multiSource2MatHelper.Initialize();
        }

        /// <summary>
        /// Raises the source to mat helper initialized event.
        /// </summary>
        public void OnSourceToMatHelperInitialized()
        {
            Debug.Log("OnSourceToMatHelperInitialized");

            Mat rgbaMat = multiSource2MatHelper.GetMat();

            texture = new Texture2D(rgbaMat.cols(), rgbaMat.rows(), TextureFormat.RGBA32, false);
            Utils.matToTexture2D(rgbaMat, texture);

            // Set the Texture2D as the main texture of the Renderer component attached to the game object
            gameObject.GetComponent<Renderer>().material.mainTexture = texture;

            // Adjust the scale of the game object to match the dimensions of the texture
            gameObject.transform.localScale = new Vector3(rgbaMat.cols(), rgbaMat.rows(), 1);
            Debug.Log("Screen.width " + Screen.width + " Screen.height " + Screen.height + " Screen.orientation " + Screen.orientation);

            // Adjust the orthographic size of the main Camera to fit the aspect ratio of the image
            float width = rgbaMat.width();
            float height = rgbaMat.height();
            float widthScale = (float)Screen.width / width;
            float heightScale = (float)Screen.height / height;
            if (widthScale < heightScale)
            {
                Camera.main.orthographicSize = (width * (float)Screen.height / (float)Screen.width) / 2;
            }
            else
            {
                Camera.main.orthographicSize = height / 2;
            }


            if (fpsMonitor != null)
            {
                fpsMonitor.Add("width", rgbaMat.width().ToString());
                fpsMonitor.Add("height", rgbaMat.height().ToString());
                fpsMonitor.Add("orientation", Screen.orientation.ToString());
            }
            if (fpsMonitor != null)
            {
                fpsMonitor.consoleText = "Please touch the 4 points surrounding the tracking object.";
            }

            hsvMat = new Mat(rgbaMat.rows(), rgbaMat.cols(), CvType.CV_8UC3);
        }

        /// <summary>
        /// Raises the source to mat helper disposed event.
        /// </summary>
        public void OnSourceToMatHelperDisposed()
        {
            Debug.Log("OnSourceToMatHelperDisposed");

            hsvMat.Dispose();
            if (roiHistMat != null)
                roiHistMat.Dispose();
            roiPointList.Clear();

            if (texture != null)
            {
                Texture2D.Destroy(texture);
                texture = null;
            }
        }

        /// <summary>
        /// Raises the source to mat helper error occurred event.
        /// </summary>
        /// <param name="errorCode">Error code.</param>
        /// <param name="message">Message.</param>
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
#if ((UNITY_ANDROID || UNITY_IOS) && !UNITY_EDITOR)
            //Touch
            int touchCount = Input.touchCount;
            if (touchCount == 1)
            {
                Touch t = Input.GetTouch(0);
                if(t.phase == TouchPhase.Ended && !EventSystem.current.IsPointerOverGameObject (t.fingerId)) {
                    storedTouchPoint = new Point (t.position.x, t.position.y);
                    //Debug.Log ("touch X " + t.position.x);
                    //Debug.Log ("touch Y " + t.position.y);
                }
            }
#else
            //Mouse
            if (Input.GetMouseButtonUp(0) && !EventSystem.current.IsPointerOverGameObject())
            {
                storedTouchPoint = new Point(Input.mousePosition.x, Input.mousePosition.y);
                //Debug.Log ("mouse X " + Input.mousePosition.x);
                //Debug.Log ("mouse Y " + Input.mousePosition.y);
            }
#endif

            if (multiSource2MatHelper.IsPlaying() && multiSource2MatHelper.DidUpdateThisFrame())
            {

                Mat rgbaMat = multiSource2MatHelper.GetMat();

                Imgproc.cvtColor(rgbaMat, hsvMat, Imgproc.COLOR_RGBA2RGB);
                Imgproc.cvtColor(hsvMat, hsvMat, Imgproc.COLOR_RGB2HSV);

                if (storedTouchPoint != null)
                {
                    ConvertScreenPointToTexturePoint(storedTouchPoint, storedTouchPoint, gameObject, rgbaMat.cols(), rgbaMat.rows());
                    OnTouch(rgbaMat, storedTouchPoint);
                    storedTouchPoint = null;
                }

                Point[] points = roiPointList.ToArray();

                if (shouldStartCamShift)
                {
                    shouldStartCamShift = false;

                    using (MatOfPoint roiPointMat = new MatOfPoint(points))
                    {
                        roiRect = Imgproc.boundingRect(roiPointMat);
                    }

                    if (roiHistMat != null)
                    {
                        roiHistMat.Dispose();
                        roiHistMat = null;
                    }
                    roiHistMat = new Mat();

                    using (Mat roiHSVMat = new Mat(hsvMat, roiRect))
                    using (Mat maskMat = new Mat())
                    {
                        Imgproc.calcHist(new List<Mat>(new Mat[] { roiHSVMat }), new MatOfInt(0), maskMat, roiHistMat, new MatOfInt(16), new MatOfFloat(0, 180));
                        Core.normalize(roiHistMat, roiHistMat, 0, 255, Core.NORM_MINMAX);

                        //Debug.Log ("roiHist " + roiHistMat.ToString ());
                    }
                }
                else if (points.Length == 4)
                {
                    using (Mat backProj = new Mat())
                    {
                        Imgproc.calcBackProject(new List<Mat>(new Mat[] { hsvMat }), new MatOfInt(0), roiHistMat, backProj, new MatOfFloat(0, 180), 1.0);

                        RotatedRect r = Video.CamShift(backProj, roiRect, termination);
                        r.points(points);
                    }
                }

                if (points.Length < 4)
                {
                    for (int i = 0; i < points.Length; i++)
                    {
                        Imgproc.circle(rgbaMat, points[i], 6, new Scalar(0, 0, 255, 255), 2);
                    }

                }
                else
                {
                    for (int i = 0; i < 4; i++)
                    {
                        Imgproc.line(rgbaMat, points[i], points[(i + 1) % 4], new Scalar(255, 0, 0, 255), 2);
                    }

                    Imgproc.rectangle(rgbaMat, roiRect.tl(), roiRect.br(), new Scalar(0, 255, 0, 255), 2);
                }

                //Imgproc.putText (rgbaMat, "W:" + rgbaMat.width () + " H:" + rgbaMat.height () + " SO:" + Screen.orientation, new Point (5, rgbaMat.rows () - 10), Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar (255, 255, 255, 255), 2, Imgproc.LINE_AA, false);

                Utils.matToTexture2D(rgbaMat, texture);
            }
        }

        private void OnTouch(Mat img, Point touchPoint)
        {
            if (roiPointList.Count == 4)
            {
                roiPointList.Clear();
            }

            if (roiPointList.Count < 4)
            {
                roiPointList.Add(touchPoint);

                if (!(new OpenCVForUnity.CoreModule.Rect(0, 0, img.width(), img.height()).contains(roiPointList[roiPointList.Count - 1])))
                {
                    roiPointList.RemoveAt(roiPointList.Count - 1);
                }

                if (roiPointList.Count == 4)
                {
                    shouldStartCamShift = true;
                }
            }
        }

        /// <summary>
        /// Converts the screen point to texture point.
        /// </summary>
        /// <param name="screenPoint">Screen point.</param>
        /// <param name="dstPoint">Dst point.</param>
        /// <param name="texturQuad">Texture quad.</param>
        /// <param name="textureWidth">Texture width.</param>
        /// <param name="textureHeight">Texture height.</param>
        /// <param name="camera">Camera.</param>
        private void ConvertScreenPointToTexturePoint(Point screenPoint, Point dstPoint, GameObject textureQuad, int textureWidth = -1, int textureHeight = -1, Camera camera = null)
        {
            if (textureWidth < 0 || textureHeight < 0)
            {
                Renderer r = textureQuad.GetComponent<Renderer>();
                if (r != null && r.material != null && r.material.mainTexture != null)
                {
                    textureWidth = r.material.mainTexture.width;
                    textureHeight = r.material.mainTexture.height;
                }
                else
                {
                    textureWidth = (int)textureQuad.transform.localScale.x;
                    textureHeight = (int)textureQuad.transform.localScale.y;
                }
            }

            if (camera == null)
                camera = Camera.main;

            Vector3 quadPosition = textureQuad.transform.localPosition;
            Vector3 quadScale = textureQuad.transform.localScale;

            Vector2 tl = camera.WorldToScreenPoint(new Vector3(quadPosition.x - quadScale.x / 2, quadPosition.y + quadScale.y / 2, quadPosition.z));
            Vector2 tr = camera.WorldToScreenPoint(new Vector3(quadPosition.x + quadScale.x / 2, quadPosition.y + quadScale.y / 2, quadPosition.z));
            Vector2 br = camera.WorldToScreenPoint(new Vector3(quadPosition.x + quadScale.x / 2, quadPosition.y - quadScale.y / 2, quadPosition.z));
            Vector2 bl = camera.WorldToScreenPoint(new Vector3(quadPosition.x - quadScale.x / 2, quadPosition.y - quadScale.y / 2, quadPosition.z));

            using (Mat srcRectMat = new Mat(4, 1, CvType.CV_32FC2))
            using (Mat dstRectMat = new Mat(4, 1, CvType.CV_32FC2))
            {
                srcRectMat.put(0, 0, tl.x, tl.y, tr.x, tr.y, br.x, br.y, bl.x, bl.y);
                dstRectMat.put(0, 0, 0, 0, quadScale.x, 0, quadScale.x, quadScale.y, 0, quadScale.y);

                using (Mat perspectiveTransform = Imgproc.getPerspectiveTransform(srcRectMat, dstRectMat))
                using (MatOfPoint2f srcPointMat = new MatOfPoint2f(screenPoint))
                using (MatOfPoint2f dstPointMat = new MatOfPoint2f())
                {
                    Core.perspectiveTransform(srcPointMat, dstPointMat, perspectiveTransform);

                    dstPoint.x = dstPointMat.get(0, 0)[0] * textureWidth / quadScale.x;
                    dstPoint.y = dstPointMat.get(0, 0)[1] * textureHeight / quadScale.y;
                }
            }
        }

        /// <summary>
        /// Raises the destroy event.
        /// </summary>
        void OnDestroy()
        {
            multiSource2MatHelper.Dispose();
        }

        /// <summary>
        /// Raises the back button click event.
        /// </summary>
        public void OnBackButtonClick()
        {
            SceneManager.LoadScene("OpenCVForUnityExample");
        }

        /// <summary>
        /// Raises the play button click event.
        /// </summary>
        public void OnPlayButtonClick()
        {
            multiSource2MatHelper.Play();
        }

        /// <summary>
        /// Raises the pause button click event.
        /// </summary>
        public void OnPauseButtonClick()
        {
            multiSource2MatHelper.Pause();
        }

        /// <summary>
        /// Raises the stop button click event.
        /// </summary>
        public void OnStopButtonClick()
        {
            multiSource2MatHelper.Stop();
        }

        /// <summary>
        /// Raises the change camera button click event.
        /// </summary>
        public void OnChangeCameraButtonClick()
        {
            multiSource2MatHelper.requestedIsFrontFacing = !multiSource2MatHelper.requestedIsFrontFacing;
        }
    }
}