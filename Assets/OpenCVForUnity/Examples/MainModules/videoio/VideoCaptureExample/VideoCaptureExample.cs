using OpenCVForUnity.CoreModule;
using OpenCVForUnity.ImgprocModule;
using OpenCVForUnity.UnityUtils;
using OpenCVForUnity.VideoioModule;
using System;
using System.Collections;
using System.Threading;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

namespace OpenCVForUnityExample
{
    /// <summary>
    /// VideoCapture Example
    /// An example of playing a video file using the VideoCapture class.
    /// http://docs.opencv.org/3.2.0/dd/d43/tutorial_py_video_display.html
    /// </summary>
    public class VideoCaptureExample : MonoBehaviour
    {
        [Header("Output")]
        /// <summary>
        /// The RawImage for previewing the result.
        /// </summary>
        public RawImage resultPreview;

        [Space(10)]

        /// <summary>
        /// The seek bar slider.
        /// </summary>
        public Slider seekBarSlider;

        Slider.SliderEvent defaultSliderEvent = new Slider.SliderEvent();

        /// <summary>
        /// The videocapture.
        /// </summary>
        VideoCapture capture;

        /// <summary>
        /// The rgb mat.
        /// </summary>
        Mat rgbMat;

        /// <summary>
        /// The texture.
        /// </summary>
        Texture2D texture;

        /// <summary>
        /// Indicates whether the video is playing.
        /// </summary>
        bool isPlaying = false;

        /// <summary>
        /// Indicates whether the video frame needs updating.
        /// </summary>
        bool shouldUpdateVideoFrame = false;

        /// <summary>
        /// The prev frame tick count.
        /// </summary>
        long prevFrameTickCount;

        /// <summary>
        /// The current frame tick count.
        /// </summary>
        long currentFrameTickCount;

        /// <summary>
        /// The FPS monitor.
        /// </summary>
        FpsMonitor fpsMonitor;

        /// <summary>
        /// VIDEO_FILENAME
        /// </summary>
        protected static readonly string VIDEO_FILENAME = "OpenCVForUnity/768x576_mjpeg.mjpeg";

        /// <summary>
        /// The CancellationTokenSource.
        /// </summary>
        CancellationTokenSource cts = new CancellationTokenSource();

        // Use this for initialization
        async void Start()
        {
            fpsMonitor = GetComponent<FpsMonitor>();

            capture = new VideoCapture();

            // Asynchronously retrieves the readable file path from the StreamingAssets directory.
            if (fpsMonitor != null)
                fpsMonitor.consoleText = "Preparing file access...";

            var video_filepath = await Utils.getFilePathAsyncTask(VIDEO_FILENAME, cancellationToken: cts.Token);

            if (fpsMonitor != null)
                fpsMonitor.consoleText = "";

            capture.open(video_filepath);
            Initialize();
        }

        private void Initialize()
        {
            rgbMat = new Mat();

            if (!capture.isOpened())
            {
                Debug.LogError(VIDEO_FILENAME + " is not opened. Please move from “OpenCVForUnity/StreamingAssets/OpenCVForUnity/” to “Assets/StreamingAssets/OpenCVForUnity/” folder.");
            }

            Debug.Log("CAP_PROP_FORMAT: " + capture.get(Videoio.CAP_PROP_FORMAT));
            Debug.Log("CAP_PROP_POS_MSEC: " + capture.get(Videoio.CAP_PROP_POS_MSEC));
            Debug.Log("CAP_PROP_POS_FRAMES: " + capture.get(Videoio.CAP_PROP_POS_FRAMES));
            Debug.Log("CAP_PROP_POS_AVI_RATIO: " + capture.get(Videoio.CAP_PROP_POS_AVI_RATIO));
            Debug.Log("CAP_PROP_FRAME_COUNT: " + capture.get(Videoio.CAP_PROP_FRAME_COUNT));
            Debug.Log("CAP_PROP_FPS: " + capture.get(Videoio.CAP_PROP_FPS));
            Debug.Log("CAP_PROP_FRAME_WIDTH: " + capture.get(Videoio.CAP_PROP_FRAME_WIDTH));
            Debug.Log("CAP_PROP_FRAME_HEIGHT: " + capture.get(Videoio.CAP_PROP_FRAME_HEIGHT));
            double ext = capture.get(Videoio.CAP_PROP_FOURCC);
            Debug.Log("CAP_PROP_FOURCC: " + (char)((int)ext & 0XFF) + (char)(((int)ext & 0XFF00) >> 8) + (char)(((int)ext & 0XFF0000) >> 16) + (char)(((int)ext & 0XFF000000) >> 24));

            if (fpsMonitor != null)
            {
                fpsMonitor.Add("CAP_PROP_FORMAT", capture.get(Videoio.CAP_PROP_FORMAT).ToString());
                fpsMonitor.Add("CAP_PROP_POS_MSEC", capture.get(Videoio.CAP_PROP_POS_MSEC).ToString());
                fpsMonitor.Add("CAP_PROP_POS_FRAMES", capture.get(Videoio.CAP_PROP_POS_FRAMES).ToString());
                fpsMonitor.Add("CAP_PROP_POS_AVI_RATIO", capture.get(Videoio.CAP_PROP_POS_AVI_RATIO).ToString());
                fpsMonitor.Add("CAP_PROP_FRAME_COUNT", capture.get(Videoio.CAP_PROP_FRAME_COUNT).ToString());
                fpsMonitor.Add("CAP_PROP_FPS", capture.get(Videoio.CAP_PROP_FPS).ToString());
                fpsMonitor.Add("CAP_PROP_FRAME_WIDTH", capture.get(Videoio.CAP_PROP_FRAME_WIDTH).ToString());
                fpsMonitor.Add("CAP_PROP_FRAME_HEIGHT", capture.get(Videoio.CAP_PROP_FRAME_HEIGHT).ToString());
                fpsMonitor.Add("CAP_PROP_FOURCC", "" + (char)((int)ext & 0XFF) + (char)(((int)ext & 0XFF00) >> 8) + (char)(((int)ext & 0XFF0000) >> 16) + (char)(((int)ext & 0XFF000000) >> 24));
                fpsMonitor.Add("STATE", "");
            }

            capture.grab();
            capture.retrieve(rgbMat);

            texture = new Texture2D(rgbMat.cols(), rgbMat.rows(), TextureFormat.RGB24, false);

            resultPreview.texture = texture;
            resultPreview.GetComponent<AspectRatioFitter>().aspectRatio = (float)texture.width / texture.height;


            capture.set(Videoio.CAP_PROP_POS_FRAMES, 0);

            StartCoroutine("WaitFrameTime");

            isPlaying = true;
        }

        // Update is called once per frame
        void Update()
        {
            if (isPlaying && shouldUpdateVideoFrame)
            {
                shouldUpdateVideoFrame = false;

                //Loop play
                if (capture.get(Videoio.CAP_PROP_POS_FRAMES) >= capture.get(Videoio.CAP_PROP_FRAME_COUNT))
                    capture.set(Videoio.CAP_PROP_POS_FRAMES, 0);

                if (capture.grab())
                {
                    capture.retrieve(rgbMat);

                    Imgproc.cvtColor(rgbMat, rgbMat, Imgproc.COLOR_BGR2RGB);

                    if (fpsMonitor != null)
                    {
                        fpsMonitor.Add("CAP_PROP_POS_MSEC", capture.get(Videoio.CAP_PROP_POS_MSEC).ToString());
                        fpsMonitor.Add("CAP_PROP_POS_FRAMES", capture.get(Videoio.CAP_PROP_POS_FRAMES).ToString());
                        fpsMonitor.Add("CAP_PROP_POS_AVI_RATIO", capture.get(Videoio.CAP_PROP_POS_AVI_RATIO).ToString());
                        fpsMonitor.Add("CAP_PROP_FRAME_COUNT", capture.get(Videoio.CAP_PROP_FRAME_COUNT).ToString());
                        int msec = (int)Math.Round(1000.0 * (currentFrameTickCount - prevFrameTickCount) / Core.getTickFrequency());
                        int fps = (int)Math.Round(1000.0 / msec);
                        fpsMonitor.Add("STATE", msec + "ms (" + fps + "fps)");
                    }

                    Utils.matToTexture2D(rgbMat, texture);

                    var tmp = seekBarSlider.onValueChanged;
                    seekBarSlider.onValueChanged = defaultSliderEvent;
                    seekBarSlider.value = (float)capture.get(Videoio.CAP_PROP_POS_AVI_RATIO);
                    seekBarSlider.onValueChanged = tmp;
                }
            }
        }

        private IEnumerator WaitFrameTime()
        {
            double videoFPS = (capture.get(Videoio.CAP_PROP_FPS) <= 0) ? 10.0 : capture.get(Videoio.CAP_PROP_FPS);
            float frameTime_sec = (float)(1000.0 / videoFPS / 1000.0);
            WaitForSeconds wait = new WaitForSeconds(frameTime_sec);
            prevFrameTickCount = currentFrameTickCount = Core.getTickCount();

            capture.grab();

            while (true)
            {
                if (isPlaying)
                {
                    shouldUpdateVideoFrame = true;

                    prevFrameTickCount = currentFrameTickCount;
                    currentFrameTickCount = Core.getTickCount();

                    yield return wait;
                }
                else
                {
                    yield return null;
                }
            }
        }

        /// <summary>
        /// Raises the destroy event.
        /// </summary>
        void OnDestroy()
        {
            StopCoroutine("WaitFrameTime");

            capture.release();

            if (rgbMat != null)
                rgbMat.Dispose();

            if (cts != null)
                cts.Dispose();
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
            isPlaying = true;
        }

        /// <summary>
        /// Raises the pause button click event.
        /// </summary>
        public void OnPauseButtonClick()
        {
            isPlaying = false;
        }

        /// <summary>
        /// Raises the jump ahead button click event.
        /// </summary>
        public void OnJumpAheadButtonClick()
        {
            int courentFrame = (int)capture.get(Videoio.CAP_PROP_POS_FRAMES) + 50;
            if (courentFrame >= capture.get(Videoio.CAP_PROP_FRAME_COUNT))
            {
                courentFrame = (int)capture.get(Videoio.CAP_PROP_FRAME_COUNT);
            }

            capture.set(Videoio.CAP_PROP_POS_FRAMES, courentFrame);
        }

        /// <summary>
        /// Raises the jump back button click event.
        /// </summary>
        public void OnJumpBackButtonClick()
        {
            int courentFrame = (int)capture.get(Videoio.CAP_PROP_POS_FRAMES) - 50;
            if (courentFrame <= 0)
            {
                courentFrame = 0;
            }

            capture.set(Videoio.CAP_PROP_POS_FRAMES, courentFrame);
        }

        /// <summary>
        /// Raises the seek bar slider value changed event.
        /// </summary>
        public void OnSeekBarSliderValueChanged()
        {
            bool supported = capture.set(Videoio.CAP_PROP_POS_AVI_RATIO, seekBarSlider.value);

            if (!supported)
            {
                capture.set(Videoio.CAP_PROP_POS_FRAMES, (int)(seekBarSlider.value * capture.get(Videoio.CAP_PROP_FRAME_COUNT)));
            }
        }
    }
}