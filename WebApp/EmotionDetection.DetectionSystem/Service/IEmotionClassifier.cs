using System.Collections.Generic;
using System.Threading.Tasks;
using EmotionDetection.Data.Entities;
using EmotionDetection.Data.Enums;

namespace EmotionDetection.DetectionSystem.Service
{
    public interface IEmotionClassifier
    {
        public Task<List<Prediction>> GetPredictionFromAudio(string audioPath);
        public List<Prediction> GetPredictionFromVideo(string videoPath);
    }
}
