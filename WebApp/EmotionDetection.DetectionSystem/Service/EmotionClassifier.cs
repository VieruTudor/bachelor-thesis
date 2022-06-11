using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using EmotionDetection.Data.Entities;
using EmotionDetection.Data.Enums;
using EmotionDetection.DetectionSystem.Configuration;
using EmotionDetection.DetectionSystem.Requests;
using EmotionDetection.DetectionSystem.Utils;
using Microsoft.Extensions.Options;
using Newtonsoft.Json;

namespace EmotionDetection.DetectionSystem.Service
{
    public class EmotionClassifier : IEmotionClassifier
    {
        private HttpClient _httpClient;
        private string _serviceBaseUri;

        public EmotionClassifier(IOptions<EmotionClassifierSettings> settings)
        {
            _httpClient = new HttpClient();
            _serviceBaseUri = settings.Value.BaseApiUri;
        }

        public async Task<List<Prediction>> GetPredictionFromAudio(string audioPath)
        {
            var requestBody = new DetectEmotionFromInferenceModelRequest { AudioPath = audioPath };
            var detectAudioUri = $"{_serviceBaseUri}/detect-emotion-from-audio";

            var fullRequestBody = JsonConvert.SerializeObject(requestBody);
            var response = await _httpClient.PostAsJsonAsync(detectAudioUri, fullRequestBody);

            var results = Deserializer.GetHttpResponseResult<Dictionary<string, Dictionary<int, float>>>(response);

            return results.Result["results"]
                .Select(kvp =>
                    new Prediction
                    {
                        Emotion = (Emotion)kvp.Key,
                        Level = kvp.Value
                    })
                .ToList();
        }

        public List<Prediction> GetPredictionFromVideo(string videoPath)
        {
            throw new NotImplementedException();
        }
    }
}