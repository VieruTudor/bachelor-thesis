using EmotionDetection.Data.Enums;

namespace EmotionDetection.Data.Entities
{
    public class Prediction
    {
        public Emotion Emotion { get; set; }

        public float Level { get; set; }
    }
}
