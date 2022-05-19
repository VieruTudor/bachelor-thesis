using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using EmotionDetection.Business.Files.Queries;
using EmotionDetection.Data.Entities;
using EmotionDetection.DetectionSystem.Service;
using MediatR;

namespace EmotionDetection.Business.Files.Handlers
{
    public class DetectEmotionFromAudioCommandHandler : IRequestHandler<DetectEmotionFromAudioCommand, List<Prediction>>
    {
        private readonly IEmotionClassifier _emotionClassifier;

        public DetectEmotionFromAudioCommandHandler(IEmotionClassifier emotionClassifier)
        {
            _emotionClassifier = emotionClassifier;
        }

        public async Task<List<Prediction>> Handle(DetectEmotionFromAudioCommand request, CancellationToken cancellationToken)
        {
            return await _emotionClassifier.GetPredictionFromAudio(request.FilePath);
        }
    }
}
