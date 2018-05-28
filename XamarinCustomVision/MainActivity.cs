using System;
using Android.App;
using Android.Widget;
using Android.OS;
using Android.Support.Design.Widget;
using Android.Support.V7.App;
using Android.Views;
using Android.Content;
using Android.Provider;
using Android.Graphics;
using Android.Runtime;
using Org.Tensorflow.Contrib.Android;
using System.IO;
using System.Linq;

namespace XamarinCustomVision
{
	[Activity(Label = "@string/app_name", Theme = "@style/AppTheme.NoActionBar", MainLauncher = true)]
	public class MainActivity : AppCompatActivity
	{
        ImageView imageView;

        public struct Recognition
        {
            public string Label;
            public float Confidence;

            public override string ToString()
            {
                return string.Format($"[Label: {Label}, Confidence: {Confidence}]");
            }
        }

        protected override void OnCreate(Bundle savedInstanceState)
		{
			base.OnCreate(savedInstanceState);

			SetContentView(Resource.Layout.activity_main);

            var btnCamera = FindViewById<Button>(Resource.Id.myButton);
            imageView = FindViewById<ImageView>(Resource.Id.image);

            btnCamera.Click += Take_Picture;

        }

        private void Take_Picture(object sender, EventArgs e)
        {
            Intent intent = new Intent(MediaStore.ActionImageCapture);
            StartActivityForResult(intent, 0);
        }

        protected override void OnActivityResult(int requestCode, [GeneratedEnum] Result resultCode, Intent data)
        {
            base.OnActivityResult(requestCode, resultCode, data);
            Bitmap bitmap = (Bitmap)data.Extras.Get("data");
            imageView.SetImageBitmap(bitmap);

            var assets = Application.Context.Assets;
            var inferenceInterface = new TensorFlowInferenceInterface(assets, "model.pb");
            var sr = new StreamReader(assets.Open("labels.txt"));
            var labels = sr.ReadToEnd()
                           .Split('\n')
                           .Select(s => s.Trim())
                           .Where(s => !string.IsNullOrEmpty(s))
                           .ToList();

            var resizedBitmap = Bitmap.CreateScaledBitmap(bitmap, 227, 227, false)
                              .Copy(Bitmap.Config.Argb8888, false);
            var floatValues = new float[227 * 227 * 3];
            var intValues = new int[227 * 227];
            resizedBitmap.GetPixels(intValues, 0, 227, 0, 0, 227, 227);
            for (int i = 0; i < intValues.Length; ++i)
            {
                var val = intValues[i];
                floatValues[i * 3 + 0] = ((val & 0xFF) - 104);
                floatValues[i * 3 + 1] = (((val >> 8) & 0xFF) - 117);
                floatValues[i * 3 + 2] = (((val >> 16) & 0xFF) - 123);
            }

            var outputs = new float[labels.Count];
            inferenceInterface.Feed("Placeholder", floatValues, 1, 227, 227, 3);
            inferenceInterface.Run(new[] { "loss" });
            inferenceInterface.Fetch("loss", outputs);

            var results = new Recognition[labels.Count];
            for (int i = 0; i < labels.Count; i++)
            {
                results[i] = new Recognition { Confidence = outputs[i], Label = labels[i] };
            }

            Array.Sort(results, (x, y) => y.Confidence.CompareTo(x.Confidence));
            ((TextView)FindViewById(Resource.Id.result)).Text = String.Format("Result: {0}, Confidence: {1}", results[0].Label, results[0].Confidence);

        }
    }
}