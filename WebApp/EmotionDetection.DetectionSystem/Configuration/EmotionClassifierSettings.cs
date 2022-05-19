using System;

namespace EmotionDetection.DetectionSystem.Configuration
{
    public class EmotionClassifierSettings
    {
        public string BaseApiUri { get; set; }
        public string ModelPath { get; set; }
        public string PythonPath { get; set; }
    }
}
