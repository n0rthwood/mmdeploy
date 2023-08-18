using System;
using System.Collections.Generic;
using System.Diagnostics;
using OpenCvSharp;
using MMDeploy;
using static System.Console;

namespace object_detection
{
    class Program
    {
        static void CvMatToMat(OpenCvSharp.Mat[] cvMats, out MMDeploy.Mat[] mats)
        {
            mats = new MMDeploy.Mat[cvMats.Length];
            unsafe
            {
                for (int i = 0; i < cvMats.Length; i++)
                {
                    mats[i].Data = cvMats[i].DataPointer;
                    mats[i].Height = cvMats[i].Height;
                    mats[i].Width = cvMats[i].Width;
                    mats[i].Channel = cvMats[i].Dims;
                    mats[i].Format = PixelFormat.BGR;
                    mats[i].Type = DataType.Int8;
                    mats[i].Device = null;
                }
            }
        }

        static void CvWaitKey()
        {
            Cv2.WaitKey();
        }

        static void Main(string[] args)
        {
            if (args.Length != 3)
            {
                WriteLine("usage:\n  object_detection deviceName modelPath imagePath\n");
                Environment.Exit(1);
            }

            string deviceName = args[0];
            string modelPath = args[1];
            string imagePath = args[2];

            // 1. create handle
            Detector handle = new Detector(modelPath, deviceName, 0);


            WriteLine($"{imagePath}");
            var img = Cv2.ImRead(imagePath, ImreadModes.Color);
            WriteLine($"img.Dims {img.Dims}");
            WriteLine($"img.rows and cols {img.Rows} {img.Cols}");
            // 2. prepare input
            OpenCvSharp.Mat[] imgs = new OpenCvSharp.Mat[1] {img };
            CvMatToMat(imgs, out var mats);
            
            
            double totalTime = 0;
            double minTime = double.MaxValue;
            double maxTime = double.MinValue;

            int iterations = 1000;
            List<DetectorOutput> output = null;
            for (int i = 0; i < iterations; i++)
            {
                Stopwatch stopwatch = new Stopwatch();
                stopwatch.Start();

                // 3. process
                output = handle.Apply(mats);

                stopwatch.Stop();
                double elapsedMilliseconds = stopwatch.Elapsed.TotalMilliseconds;
                totalTime += elapsedMilliseconds;

                if (elapsedMilliseconds < minTime) minTime = elapsedMilliseconds;
                if (elapsedMilliseconds > maxTime) maxTime = elapsedMilliseconds;
            }

            double avgTime = totalTime / iterations;

            WriteLine($"Average Time: {avgTime} milliseconds");
            WriteLine($"Minimum Time: {minTime} milliseconds");
            WriteLine($"Maximum Time: {maxTime} milliseconds");

            // 4. show result
            foreach (var obj in output[0].Results)
            {
                if (obj.Score > 0.3)
                {
                    if (obj.HasMask)
                    {
                        OpenCvSharp.Mat imgMask = new OpenCvSharp.Mat(obj.Mask.Height, obj.Mask.Width, MatType.CV_8UC1, obj.Mask.Data);

                        // Calculate the position and size of the bounding box
                        float x0 = Math.Max((float)Math.Floor(obj.BBox.Left) - 1, 0f);
                        float y0 = Math.Max((float)Math.Floor(obj.BBox.Top) - 1, 0f);
                        int width = (int)Math.Floor(obj.BBox.Right) - (int)x0;
                        int height = (int)Math.Floor(obj.BBox.Bottom) - (int)y0;

                        // Resize the mask to fit the bounding box
                        Cv2.Resize(imgMask, imgMask, new Size(width, height));

                        // Define the ROI
                        OpenCvSharp.Rect roi = new OpenCvSharp.Rect((int)x0, (int)y0, width, height);

                        // Split the channels
                        Cv2.Split(imgs[0], out OpenCvSharp.Mat[] ch);
                        int col = 0;

                        // Apply the bitwise operation to the specific region
                        Cv2.BitwiseOr(imgMask, ch[col][roi], ch[col][roi]);
                        Cv2.Merge(ch, imgs[0]);
                    }

                    // Draw the rectangle
                    Cv2.Rectangle(imgs[0], new Point((int)obj.BBox.Left, (int)obj.BBox.Top),
                        new Point((int)obj.BBox.Right, (int)obj.BBox.Bottom), new Scalar(0, 255, 0));
                }

            }
            Cv2.NamedWindow("mmdet", WindowFlags.GuiExpanded);
            Cv2.ImShow("mmdet", imgs[0]);
            CvWaitKey();

            handle.Close();
        }
    }
}
