using System.Threading.Tasks;
using EmotionDetection.API.Requests;
using EmotionDetection.Business.Files.Queries;

namespace EmotionDetection.API.Utils
{
    public static class RequestMappings
    {
        public static async Task<DetectEmotionFromAudioCommand> ToCommand(this DetectEmotionFromAudioRequest request)
        {
            return new DetectEmotionFromAudioCommand
            {
                FilePath = await FileUtils.GetTempFilePath(request.AudioFile)
            };
        }
    }
}
