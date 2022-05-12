using System.Collections.Generic;
using EmotionDetection.Data.Entities;
using MediatR;

namespace EmotionDetection.Business.Files.Queries
{
    public class DetectEmotionFromAudioCommand : IRequest<List<Prediction>>
    {
        public string FilePath { get; set; }
    }
}
